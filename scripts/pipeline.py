"""
Hybrid NER + Classification Inference Pipeline.

Combines:
1. Deterministic EntityRuler (matches data/terms.csv before ML)
2. Fine-tuned spaCy Transformer NER (detects unseen entities from context)
3. Genuine per-entity marginal beam confidence calculation
4. Adjacency-merge pass for multi-word split spans (e.g. 'Tailwind' + 'CSS' -> 'Tailwind CSS')
5. Confidence-based routing with 0.80 threshold
6. Source attribution ("dictionary" vs "ML") and status tagging
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


class HybridJournalPipeline:
    """Production-grade hybrid inference pipeline combining EntityRuler and Transformer NER."""

    def __init__(
        self,
        model_path: str = "models/ner_trf/model-best",
        terms_csv_path: str = "data/terms.csv",
        confidence_threshold: float = 0.80,
    ) -> None:
        """Initializes the hybrid pipeline.
        
        Args:
            model_path: Path to the fine-tuned spaCy model directory.
            terms_csv_path: Path to terms dictionary CSV.
            confidence_threshold: Minimum confidence score to auto-accept ML predictions.
        """
        scripts.init_gpu()
        self.confidence_threshold = confidence_threshold
        self.terms_csv_path = terms_csv_path
        self.model_path = model_path

        # Load dictionary terms
        self.terms_dict = load_terms_dictionary(terms_csv_path)
        self._lower_terms_set = {t.lower(): (t, lbl) for t, lbl in self.terms_dict.items()}

        logger.info(f"Loading transformer model from '{model_path}'...")
        self.nlp = spacy.load(model_path)

        # Wire EntityRuler before NER if not already present
        self._setup_entity_ruler()

    def _setup_entity_ruler(self) -> None:
        """Injects or updates the EntityRuler pipe directly before the NER component."""
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")

        patterns = [{"label": label, "pattern": term} for term, label in self.terms_dict.items()]

        # ruler runs before "ner" and preserves matched spans without overwriting
        ruler = self.nlp.add_pipe(
            "entity_ruler",
            before="ner",
            config={"overwrite_ents": False, "phrase_matcher_attr": "LOWER"}
        )
        ruler.add_patterns(patterns)
        logger.info(f"Configured EntityRuler with {len(patterns)} patterns before NER.")

    def _calculate_ml_confidence(self, doc: spacy.tokens.Doc, ent: spacy.tokens.Span) -> float:
        """Computes genuine marginal posterior probability for an extracted entity.
        
        Extracts marginal posterior beam probabilities by summing the normalized
        scores of all unconstrained beam hypotheses that contain the entity span.
        """
        try:
            ner = self.nlp.get_pipe("ner")
            # Run unconstrained beam parse on raw doc to avoid constraint bias
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
                        # Match exact or overlapping span with same label
                        if max(ent_token_start, p_s) < min(ent_token_end, p_e) and p_l == ent_label:
                            span_prob += score
                            break

                if span_prob > 0.0:
                    return round(min(0.9999, span_prob), 4)
        except Exception as e:
            logger.debug(f"Could not compute beam probability for span '{ent.text}': {e}")

        # Fallback calibrated warning rather than a silent high constant
        logger.warning(
            f"Could not compute marginal beam confidence for span '{ent.text}' [{ent.start_char}:{ent.end_char}]. "
            "Defaulting to 0.50 (flagged for review)."
        )
        return 0.50

    def _merge_adjacent_entities(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merges adjacent entity spans sharing the same classification label separated only by whitespace.
        
        Solves multi-word span splitting (e.g. 'Tailwind' (ML) + 'CSS' (dict) -> 'Tailwind CSS').
        """
        if not entities or len(entities) < 2:
            return entities

        merged: List[Dict[str, Any]] = [entities[0]]
        for curr in entities[1:]:
            prev = merged[-1]
            intervening = text[prev["end"]:curr["start"]]

            # Merge if same label and only whitespace between spans
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

    def extract_entities_from_doc(self, doc: Doc, text: str) -> List[Dict[str, Any]]:
        """Extracts, scores, and merges entities from a processed Doc."""
        raw_entities = []
        for ent in doc.ents:
            term = ent.text.strip()
            if not term:
                continue
            label = ent.label_
            start = ent.start_char
            end = ent.end_char

            is_dict_match = term.lower() in self._lower_terms_set

            if is_dict_match:
                source = "dictionary"
                confidence = 1.00
                status = "ACCEPTED"
            else:
                source = "ML"
                confidence = self._calculate_ml_confidence(doc, ent)
                status = "ACCEPTED" if confidence >= self.confidence_threshold else "NEEDS_REVIEW"

            entity_record = {
                "term": term,
                "category": label,
                "start": start,
                "end": end,
                "confidence": round(confidence, 4),
                "source": source,
                "status": status,
            }
            raw_entities.append(entity_record)

        # Sort raw entities by start offset
        raw_entities.sort(key=lambda e: e["start"])

        # Apply adjacency merge pass to prevent split multi-word spans
        return self._merge_adjacent_entities(text, raw_entities)

    def predict(self, text: str) -> Dict[str, Any]:
        """Processes a single journal entry text and returns structured extraction metadata.
        
        Returns:
            Dict containing:
                - text: Original journal entry
                - entities: List of extracted entities with term, category, start, end,
                            confidence, source ("dictionary" | "ML"), and status ("ACCEPTED" | "NEEDS_REVIEW")
                - has_review_items: True if any entity requires human validation
        """
        doc = self.nlp(text)
        merged_entities = self.extract_entities_from_doc(doc, text)
        has_review = any(e["status"] == "NEEDS_REVIEW" for e in merged_entities)

        return {
            "text": text,
            "entities": merged_entities,
            "has_review_items": has_review,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Processes a batch of journal entries efficiently."""
        return [self.predict(t) for t in texts]

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
    ) -> "HybridJournalPipeline":
        """Loads an existing packaged hybrid pipeline."""
        return cls(
            model_path=pipeline_dir,
            terms_csv_path=terms_csv_path,
            confidence_threshold=confidence_threshold
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = HybridJournalPipeline()
    sample_text = (
        "Configured modern responsive web styling using Tailwind CSS."
    )
    result = pipeline.predict(sample_text)
    import json
    print(json.dumps(result, indent=2))
