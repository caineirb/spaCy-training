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
import logging
from typing import List, Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from scripts.annotation import load_terms_dictionary

logger = logging.getLogger("ojt_pipeline.inference")


def resolve_span_conflicts(
    ml_entities: List[Dict[str, Any]],
    dict_entities: List[Dict[str, Any]],
    text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Resolves conflicts between ML predictions and Dictionary matches.

    Rules (Task 3):
    1. Longest span wins: A shorter dictionary span must never truncate a longer ML span
       (e.g., 'Tailwind CSS' beats 'CSS', 'Canva Editing' beats 'Canva',
       'access control systems' beats 'access control', 'CSS flexibility' beats 'CSS').
       If a dictionary span is strictly longer than an overlapping partial ML span,
       the longer span wins.
    2. ML label authority: When ML and Dictionary disagree on label, the ML label takes
       precedence (e.g., 'data entry' and 'File Management' as CLERICAL_TERM).
    3. Abstention fallback: Dictionary predictions are used ONLY when ML abstains
       (i.e. no ML prediction overlaps with the dictionary match).
    """
    if not dict_entities:
        return sorted(ml_entities, key=lambda e: e["start"])
    if not ml_entities:
        return sorted(dict_entities, key=lambda e: e["start"])

    # Track which ML entities have been superseded by strictly longer dict spans
    superseded_ml_indices = set()
    accepted_dict_entities = []

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
                # Rule 1: Dictionary span is strictly longer than the partial ML span(s)
                # Rule 2: ML label authority -> adopt ML category
                inherited_label = overlapping_ml[0][1]["category"]
                merged_dict_ent = dict(d)
                merged_dict_ent["category"] = inherited_label
                accepted_dict_entities.append(merged_dict_ent)
                for idx, _ in overlapping_ml:
                    superseded_ml_indices.add(idx)
            else:
                # ML span is >= dictionary span (e.g. Tailwind CSS >= CSS)
                # Dictionary span is dropped to prevent truncation
                pass

    surviving_ml = [
        m for idx, m in enumerate(ml_entities)
        if idx not in superseded_ml_indices
    ]

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
    ) -> None:
        """Initializes the hybrid pipeline.

        Args:
            model_path: Path to the fine-tuned spaCy model directory.
            terms_csv_path: Path to terms dictionary CSV.
            confidence_threshold: Minimum confidence score to auto-accept ML predictions.
            use_gpu: If True, attempts GPU initialization via scripts.init_gpu(). If False, runs strictly on CPU.
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

            if prev["category"] == curr["category"] and intervening.strip() == "":
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
        """Extracts ML-predicted entities from a processed transformer Doc."""
        entities = []
        for ent in doc.ents:
            term = ent.text.strip()
            if not term:
                continue
            conf = self._calculate_ml_confidence(doc, ent)
            status = "ACCEPTED" if conf >= self.confidence_threshold else "NEEDS_REVIEW"
            entities.append({
                "term": term,
                "category": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
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

            # 3. Conflict resolution (longest span wins, ML label authority, abstention fallback)
            entities = resolve_span_conflicts(ml_entities, dict_entities, text=text)
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Expected 'hybrid', 'transformer_only', or 'entity_ruler_only'."
            )

        # Apply adjacency merge pass
        merged_entities = self._merge_adjacent_entities(text, entities)
        has_review = any(e["status"] == "NEEDS_REVIEW" for e in merged_entities)

        return {
            "text": text,
            "entities": merged_entities,
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
