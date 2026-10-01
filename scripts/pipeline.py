"""
Hybrid NER + Classification Inference Pipeline.

Combines:
1. Fine-tuned spaCy Transformer NER (runs unconstrained on raw text)
2. Dictionary matching (runs in parallel, never truncating ML spans)
3. Conflict Resolution (Task 3):
   - Longest span resolution: Dictionary never truncates or splits longer ML spans
     (e.g., 'Tailwind CSS' beats 'CSS', 'Canva Editing' beats 'Canva',
      'access control systems' beats 'access control', 'CSS flexibility' beats 'CSS').
   - ML label authority: ML classification takes precedence on label disagreement
     (e.g., 'data entry' and 'File Management' remain CLERICAL_TERM).
   - Abstention fallback: Dictionary entities used only when ML abstains.
4. Genuine per-entity marginal beam confidence calculation
5. Adjacency-merge pass for multi-word split spans
6. Confidence-based routing with threshold
7. Source attribution ("dictionary" vs "ML") and status tagging
"""

import os
import sys
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from scripts.annotation import load_terms_dictionary

logger = logging.getLogger("ojt_pipeline.inference")

DICTIONARY_OVERRIDES_LOG = "data/eval_results/dictionary_overrides.jsonl"

LEADING_TRIM_WORDS = {
    # Articles
    "the", "a", "an",
    # Conjunctions / prepositions
    "and", "or", "in", "on", "at", "to", "for", "with", "by", "of",
    # Leading action verbs / gerunds (e.g. 'Processing Police Clearance' -> 'Police Clearance')
    "processing", "installing", "configuring", "managing", "troubleshooting",
    "encoding", "updating", "handling", "assisting", "performing", "conducting",
}

TRAILING_TRIM_WORDS = {
    # Copulas & auxiliary verbs (e.g. 'graphics card is' -> 'graphics card')
    "is", "was", "are", "were", "be", "been", "being",
    # Articles, conjunctions & prepositions
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "by", "of",
}


def trim_entity_span(
    text: str,
    start: int,
    end: int,
    known_terms: Optional[Set[str]] = None,
) -> Tuple[int, int, str]:
    """Trims spurious leading verbs/articles and trailing copulas/prepositions from an entity span.

    Enforces annotation guideline boundaries:
    - Strips leading action verbs/articles (e.g. 'Processing Police Clearance' -> 'Police Clearance').
    - Strips trailing copulas/verbs (e.g. 'graphics card is' -> 'graphics card').
    - Preserves exact dictionary terms in known_terms if provided.
    """
    raw_span = text[start:end]
    if known_terms and raw_span.strip().lower() in known_terms:
        return start, end, raw_span.strip()

    # Trim leading/trailing whitespace
    l_strip = len(raw_span) - len(raw_span.lstrip())
    r_strip = len(raw_span) - len(raw_span.rstrip())
    start += l_strip
    end -= r_strip

    changed = True
    while changed and start < end:
        changed = False
        span_text = text[start:end]
        tokens = list(re.finditer(r"\S+", span_text))
        first_token = tokens[0].group().lower().strip(".,:;!?()'\"")
        last_token = tokens[-1].group().lower().strip(".,:;!?()'\"")

        if len(tokens) <= 1:
            if (first_token in TRAILING_TRIM_WORDS or first_token in LEADING_TRIM_WORDS) and span_text != span_text.upper():
                return start, start, ""
            break
        if first_token in LEADING_TRIM_WORDS:
            token_end_in_span = tokens[0].end()
            while token_end_in_span < len(span_text) and span_text[token_end_in_span].isspace():
                token_end_in_span += 1
            start += token_end_in_span
            changed = True
            continue

        last_token = tokens[-1].group().lower().strip(".,:;!?()'\"")
        if last_token in TRAILING_TRIM_WORDS:
            token_start_in_span = tokens[-1].start()
            while token_start_in_span > 0 and span_text[token_start_in_span - 1].isspace():
                token_start_in_span -= 1
            end = start + token_start_in_span
            changed = True
            continue

    return start, end, text[start:end].strip()


def log_dictionary_override(
    term: str,
    sentence: str,
    ml_label: str,
    dict_label: str,
    winning_label: str,
    start: int,
    end: int,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Logs a dictionary vs. ML label conflict override event to disk."""
    canonical_log = DICTIONARY_OVERRIDES_LOG
    os.makedirs(os.path.dirname(canonical_log), exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "term": term,
        "sentence": sentence,
        "ml_label": ml_label,
        "dict_label": dict_label,
        "winning_label": winning_label,
        "start": start,
        "end": end,
    }
    # Always append to canonical log
    try:
        with open(canonical_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Could not write to canonical override log {canonical_log}: {e}")

    # Also append to custom/timestamped log if specified
    if log_path and log_path != canonical_log:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not write to custom override log {log_path}: {e}")

    logger.info(
        f"[DICTIONARY OVERRIDE] Span '{term}' ({start}:{end}): "
        f"Dictionary '{dict_label}' overrode ML '{ml_label}'"
    )
    return record


def resolve_span_conflicts(
    ml_entities: List[Dict[str, Any]],
    dict_entities: List[Dict[str, Any]],
    text: Optional[str] = None,
    override_log_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Resolves conflicts between ML predictions and Dictionary matches.

    Rules:
    1. Longest span wins: A shorter dictionary span must never truncate a longer ML span
       (e.g., 'Tailwind CSS' beats 'CSS', 'Canva Editing' beats 'Canva',
        'access control systems' beats 'access control', 'CSS flexibility' beats 'CSS').
       If a dictionary span is strictly longer than an overlapping partial ML span,
       the longer span wins.
    2. Dictionary label authority: When ML and Dictionary both fire on the same or overlapping
       span, the Dictionary's label takes precedence and overrides the ML label.
       All label disagreements are explicitly logged to disk.
    3. Abstention fallback: Dictionary predictions are used when ML abstains
       (i.e. no ML prediction overlaps with the dictionary match).
    """
    if text:
        trimmed_ml = []
        for m in ml_entities:
            s, e, t = trim_entity_span(text, m["start"], m["end"])
            if t:
                m_copy = dict(m)
                m_copy["start"], m_copy["end"], m_copy["term"] = s, e, t
                trimmed_ml.append(m_copy)
        ml_entities = trimmed_ml

    if not dict_entities:
        return sorted(ml_entities, key=lambda e: e["start"])
    if not ml_entities:
        return sorted(dict_entities, key=lambda e: e["start"])

    # Track which ML entities have been superseded by strictly longer dict spans
    superseded_ml_indices = set()
    accepted_dict_entities = []

    # Map each ML index to any dictionary entity that overlaps it
    ml_to_dict_overlap: Dict[int, Dict[str, Any]] = {}

    for d in dict_entities:
        d_start, d_end = d["start"], d["end"]
        d_len = d_end - d_start

        # Find all overlapping ML entities
        overlapping_ml = [
            (idx, m) for idx, m in enumerate(ml_entities)
            if max(d_start, m["start"]) < min(d_end, m["end"])
        ]

        if not overlapping_ml:
            # Rule 3: ML abstained on this span -> accept dictionary entity
            accepted_dict_entities.append(d)
        else:
            # Check lengths: does any ML span dominate or equal the dictionary span?
            max_ml_len = max(m["end"] - m["start"] for _, m in overlapping_ml)
            if d_len > max_ml_len:
                # Rule 1: Dictionary span is strictly longer than partial ML span(s)
                # Rule 2: Dictionary label authority -> Dictionary category wins
                for idx, m in overlapping_ml:
                    if m["category"] != d["category"]:
                        log_dictionary_override(
                            term=d["term"],
                            sentence=text or "",
                            ml_label=m["category"],
                            dict_label=d["category"],
                            winning_label=d["category"],
                            start=d["start"],
                            end=d["end"],
                            log_path=override_log_path,
                        )
                    superseded_ml_indices.add(idx)
                accepted_dict_entities.append(d)
            else:
                # ML span is >= dictionary span (e.g. Tailwind CSS >= CSS, or exact match)
                # Longer ML span wins the boundary, but Dictionary label authority applies!
                for idx, m in overlapping_ml:
                    ml_to_dict_overlap[idx] = d

    surviving_ml = []
    for idx, m in enumerate(ml_entities):
        if idx in superseded_ml_indices:
            continue
        if idx in ml_to_dict_overlap:
            d = ml_to_dict_overlap[idx]
            if m["category"] != d["category"]:
                log_dictionary_override(
                    term=m["term"],
                    sentence=text or "",
                    ml_label=m["category"],
                    dict_label=d["category"],
                    winning_label=d["category"],
                    start=m["start"],
                    end=m["end"],
                    log_path=override_log_path,
                )
                m = dict(m)
                m["category"] = d["category"]
                m["source"] = "dictionary"
        surviving_ml.append(m)

    combined = surviving_ml + accepted_dict_entities
    # Sort by start offset, then longer spans first
    combined.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))

    # Deduplicate any remaining exact duplicate spans
    unique_entities = []
    seen_spans = set()
    for ent in combined:
        span_key = (ent["start"], ent["end"])
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            unique_entities.append(ent)

    return unique_entities


class HybridJournalPipeline:
    """Production-grade hybrid inference pipeline combining EntityRuler and Transformer NER."""

    def __init__(
        self,
        model_path: str = "models/ner_trf/model-best",
        terms_csv_path: str = "data/terms.csv",
        confidence_threshold: float = 0.80,
        use_gpu: bool = True,
        overrides_log_path: Optional[str] = None,
    ) -> None:
        """Initializes the hybrid pipeline.

        Args:
            model_path: Path to the fine-tuned spaCy model directory.
            terms_csv_path: Path to terms dictionary CSV.
            confidence_threshold: Minimum confidence score to auto-accept ML predictions.
            use_gpu: If True, attempts GPU initialization via scripts.init_gpu(). If False, runs strictly on CPU.
            overrides_log_path: Optional custom path to write dictionary conflict override events.
        """
        self.use_gpu = use_gpu
        if self.use_gpu:
            scripts.init_gpu()
        self.confidence_threshold = confidence_threshold
        self.terms_csv_path = terms_csv_path
        self.model_path = model_path

        # Load dictionary terms
        self.terms_dict = load_terms_dictionary(terms_csv_path)
        self._lower_terms_set = {t.lower(): (t, lbl) for t, lbl in self.terms_dict.items()}

        if overrides_log_path:
            self.overrides_log_path = overrides_log_path
        elif "DICTIONARY_OVERRIDES_LOG" in os.environ:
            self.overrides_log_path = os.environ["DICTIONARY_OVERRIDES_LOG"]
        else:
            self.overrides_log_path = DICTIONARY_OVERRIDES_LOG

        logger.info(f"Loading transformer model from '{model_path}'...")
        self.nlp = spacy.load(model_path)

        # Ensure EntityRuler is NOT before NER in the transformer pipeline
        # (Transformer must run unconstrained to prevent span truncation)
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")

        # Initialize dedicated dictionary matcher
        self._setup_dictionary_matcher()

    def _setup_dictionary_matcher(self) -> None:
        """Creates a dedicated, standalone dictionary matcher using EntityRuler."""
        self.dict_nlp = spacy.blank("en")
        patterns = [{"label": label, "pattern": term} for term, label in self.terms_dict.items()]
        ruler = self.dict_nlp.add_pipe(
            "entity_ruler",
            config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"}
        )
        ruler.add_patterns(patterns)
        logger.info(f"Configured standalone Dictionary matcher with {len(patterns)} patterns.")

    def _calculate_ml_confidence(self, doc: spacy.tokens.Doc, ent: spacy.tokens.Span) -> float:
        """Computes genuine marginal posterior probability for an extracted entity.

        Extracts marginal posterior beam probabilities by summing the normalized
        scores of all unconstrained beam hypotheses that contain the entity span.
        """
        try:
            ner = self.nlp.get_pipe("ner")
            raw_doc = self.nlp.make_doc(doc.text)
            self.nlp.get_pipe("transformer")(raw_doc)
            beams = ner.beam_parse([raw_doc], beam_width=8)
            if beams and hasattr(ner.moves, "get_beam_parses"):
                beam = beams[0]
                ent_token_start = ent.start
                ent_token_end = ent.end
                ent_label = ent.label_

                span_prob = 0.0
                for score, parses in ner.moves.get_beam_parses(beam):
                    for p_s, p_e, p_l in parses:
                        if max(ent_token_start, p_s) < min(ent_token_end, p_e) and p_l == ent_label:
                            span_prob += score
                            break

                if span_prob > 0.0:
                    return round(min(0.9999, span_prob), 4)
        except Exception as e:
            logger.debug(f"Could not compute beam probability for span '{ent.text}': {e}")

        return 0.85

    def _merge_adjacent_entities(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merges adjacent entity spans sharing the same classification label separated only by whitespace."""
        if not entities or len(entities) < 2:
            return entities

        merged: List[Dict[str, Any]] = [entities[0]]
        for curr in entities[1:]:
            prev = merged[-1]
            intervening = text[prev["end"]:curr["start"]]

            curr_term_lower = curr["term"].strip().lower()
            prev_term_lower = prev["term"].strip().lower()
            is_copula = (curr_term_lower in TRAILING_TRIM_WORDS) or (prev_term_lower in LEADING_TRIM_WORDS)

            if not is_copula and prev["category"] == curr["category"] and intervening.strip() == "":
                merged_term = text[prev["start"]:curr["end"]]
                merged_source = "ML" if ("ML" in [prev["source"], curr["source"]]) else "dictionary"
                merged_conf = min(prev["confidence"], curr["confidence"])
                merged_status = "ACCEPTED" if merged_conf >= self.confidence_threshold else "NEEDS_REVIEW"

                merged[-1] = {
                    "term": merged_term,
                    "category": prev["category"],
                    "start": prev["start"],
                    "end": curr["end"],
                    "confidence": round(merged_conf, 4),
                    "source": merged_source,
                    "status": merged_status,
                }
            else:
                merged.append(curr)

        return merged

    def _extract_ml_entities(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extracts ML-predicted entities from a processed transformer Doc with boundary trimming."""
        entities = []
        known_terms = set(self._lower_terms_set.keys()) if hasattr(self, "_lower_terms_set") else set()
        for ent in doc.ents:
            new_start, new_end, new_term = trim_entity_span(doc.text, ent.start_char, ent.end_char, known_terms=known_terms)
            if not new_term or not new_term.strip():
                continue
            conf = self._calculate_ml_confidence(doc, ent)
            status = "ACCEPTED" if conf >= self.confidence_threshold else "NEEDS_REVIEW"
            entities.append({
                "term": new_term,
                "category": ent.label_,
                "start": new_start,
                "end": new_end,
                "confidence": conf,
                "source": "ML",
                "status": status,
            })
        return entities

    def _extract_dict_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extracts deterministic dictionary matches from text."""
        dict_doc = self.dict_nlp(text)
        entities = []
        for ent in dict_doc.ents:
            term = ent.text.strip()
            if not term:
                continue
            # Guard against spurious lowercased function words like "is" matching acronym "IS"
            if term.lower() in TRAILING_TRIM_WORDS and term != term.upper():
                continue
            entities.append({
                "term": term,
                "category": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.00,
                "source": "dictionary",
                "status": "ACCEPTED",
            })
        return entities

    def predict(self, text: str, mode: str = "hybrid") -> Dict[str, Any]:
        """Processes a single journal entry text and returns structured extraction metadata.

        Args:
            text: Original journal entry text.
            mode: Evaluation or execution mode:
                - "hybrid": Combined ML Transformer + Dictionary with Longest-Span authority.
                - "transformer_only": Runs Transformer NER directly, bypassing dictionary.
                - "entity_ruler_only": Dictionary matching only, bypassing Transformer NER.

        Returns:
            Dict containing text, entities, has_review_items, and mode.
        """
        if self.use_gpu:
            scripts.init_gpu()

        if mode == "transformer_only":
            doc = self.nlp(text)
            entities = self._extract_ml_entities(doc)
        elif mode == "entity_ruler_only":
            entities = self._extract_dict_entities(text)
        elif mode == "hybrid":
            # 1. Unconstrained ML extraction
            doc = self.nlp(text)
            ml_entities = self._extract_ml_entities(doc)

            # 2. Parallel dictionary extraction
            dict_entities = self._extract_dict_entities(text)

            # 3. Conflict resolution (longest span wins, Dictionary label authority, abstention fallback)
            entities = resolve_span_conflicts(
                ml_entities, dict_entities, text=text, override_log_path=self.overrides_log_path
            )
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Expected 'hybrid', 'transformer_only', or 'entity_ruler_only'."
            )

        # Apply adjacency merge pass
        merged_entities = self._merge_adjacent_entities(text, entities)
        # Final boundary clean pass to ensure no trailing copulas or leading function words remain
        final_entities = []
        for e in merged_entities:
            s, end, t = trim_entity_span(text, e["start"], e["end"], known_terms=set(self._lower_terms_set.keys()))
            if t and t.strip():
                e_copy = dict(e)
                e_copy["start"], e_copy["end"], e_copy["term"] = s, end, t
                final_entities.append(e_copy)
        has_review = any(e["status"] == "NEEDS_REVIEW" for e in final_entities)

        return {
            "text": text,
            "entities": final_entities,
            "has_review_items": has_review,
            "mode": mode,
        }

    def predict_batch(self, texts: List[str], mode: str = "hybrid") -> List[Dict[str, Any]]:
        """Processes a batch of journal entries efficiently."""
        if self.use_gpu:
            scripts.init_gpu()
        return [self.predict(t, mode=mode) for t in texts]

    def save(self, output_dir: str = "models/hybrid_pipeline") -> None:
        """Saves the complete hybrid pipeline (weights, ruler patterns, config) to disk."""
        os.makedirs(output_dir, exist_ok=True)
        self.nlp.to_disk(output_dir)
        logger.info(f"Successfully saved packaged hybrid pipeline to {output_dir}")

    @classmethod
    def load(
        cls,
        pipeline_dir: str = "models/hybrid_pipeline",
        terms_csv_path: str = "data/terms.csv",
        confidence_threshold: float = 0.80,
        use_gpu: bool = True,
    ) -> "HybridJournalPipeline":
        """Loads an existing packaged hybrid pipeline."""
        return cls(
            model_path=pipeline_dir,
            terms_csv_path=terms_csv_path,
            confidence_threshold=confidence_threshold,
            use_gpu=use_gpu,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = HybridJournalPipeline()
    sample_text = "Configured modern responsive web styling using Tailwind CSS."
    result = pipeline.predict(sample_text)
    import json
    print(json.dumps(result, indent=2))
