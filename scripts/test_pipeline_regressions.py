"""
Regression tests for Hybrid NER Pipeline improvements (Tasks 3, 5, 6).

Validates:
1. Longest span resolution: Dictionary must never truncate or split a longer ML span
   (Tailwind CSS, Canva Editing, access control systems, CSS flexibility).
2. Dictionary label authority: Dictionary label takes precedence on disagreement
   (reversing prior ML label authority, with audit logging).
3. Abstention fallback: Dictionary entities accepted when ML abstains.
4. Span boundary trimming: Symmetrically trims leading verbs/articles
   ('Processing Police Clearance' -> 'Police Clearance') and trailing copulas
   ('graphics card is' -> 'graphics card').
5. Override audit logging: Confirms disagreements are persisted to JSONL.
"""

import os
import sys
import json
import tempfile
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.pipeline import (
    resolve_span_conflicts,
    trim_entity_span,
    HybridJournalPipeline,
)


class TestPipelineRegressions(unittest.TestCase):
    """Regression test suite for hybrid pipeline fixes."""

    def test_longest_span_ml_over_dict(self):
        """Rule 1: Shorter dictionary span must never truncate a longer ML span."""
        ml_ents = [
            {"term": "Tailwind CSS", "category": "IT_TERM", "start": 44, "end": 56, "confidence": 0.95, "source": "ML", "status": "ACCEPTED"},
            {"term": "Canva Editing", "category": "IT_TERM", "start": 0, "end": 13, "confidence": 0.98, "source": "ML", "status": "ACCEPTED"},
            {"term": "access control systems", "category": "CLERICAL_TERM", "start": 29, "end": 51, "confidence": 0.92, "source": "ML", "status": "ACCEPTED"},
            {"term": "CSS flexibility", "category": "IT_TERM", "start": 10, "end": 25, "confidence": 0.94, "source": "ML", "status": "ACCEPTED"},
        ]

        dict_ents = [
            {"term": "CSS", "category": "IT_TERM", "start": 53, "end": 56, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
            {"term": "Canva", "category": "IT_TERM", "start": 0, "end": 5, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
            {"term": "access control", "category": "IT_TERM", "start": 29, "end": 43, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
            {"term": "CSS", "category": "IT_TERM", "start": 10, "end": 13, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
        ]

        resolved = resolve_span_conflicts(ml_ents, dict_ents)
        terms = [e["term"] for e in resolved]

        self.assertIn("Tailwind CSS", terms)
        self.assertNotIn("CSS", terms)

        self.assertIn("Canva Editing", terms)
        self.assertNotIn("Canva", terms)

        self.assertIn("access control systems", terms)
        self.assertNotIn("access control", terms)

        self.assertIn("CSS flexibility", terms)

    def test_dictionary_label_authority_on_disagreement(self):
        """Rule 2: When ML and Dictionary disagree on label, Dictionary label takes precedence."""
        ml_ents = [
            {"term": "data entry", "category": "CLERICAL_TERM", "start": 13, "end": 23, "confidence": 0.96, "source": "ML", "status": "ACCEPTED"},
            {"term": "File Management", "category": "CLERICAL_TERM", "start": 11, "end": 26, "confidence": 0.91, "source": "ML", "status": "ACCEPTED"},
        ]

        # Contradictory IT_TERM labels from dictionary
        dict_ents = [
            {"term": "data entry", "category": "IT_TERM", "start": 13, "end": 23, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
            {"term": "File Management", "category": "IT_TERM", "start": 11, "end": 26, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
        ]

        resolved = resolve_span_conflicts(ml_ents, dict_ents)
        resolved_map = {e["term"]: e["category"] for e in resolved}

        # Under dictionary label authority, dictionary labels win
        self.assertEqual(resolved_map.get("data entry"), "IT_TERM")
        self.assertEqual(resolved_map.get("File Management"), "IT_TERM")

    def test_dictionary_used_only_when_ml_abstains(self):
        """Rule 3: Dictionary provides entities when ML makes no prediction overlapping the span."""
        ml_ents = []
        dict_ents = [
            {"term": "Photoshop", "category": "IT_TERM", "start": 10, "end": 19, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
        ]

        resolved = resolve_span_conflicts(ml_ents, dict_ents)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["term"], "Photoshop")
        self.assertEqual(resolved[0]["category"], "IT_TERM")
        self.assertEqual(resolved[0]["source"], "dictionary")

    def test_dict_longer_than_partial_ml_span(self):
        """Rule 1 + 2: Longer dictionary span wins both boundary and label authority."""
        ml_ents = [
            {"term": "management", "category": "CLERICAL_TERM", "start": 16, "end": 26, "confidence": 0.88, "source": "ML", "status": "ACCEPTED"},
        ]
        dict_ents = [
            {"term": "File Management", "category": "IT_TERM", "start": 11, "end": 26, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
        ]

        resolved = resolve_span_conflicts(ml_ents, dict_ents)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["term"], "File Management")
        self.assertEqual(resolved[0]["category"], "IT_TERM")

    def test_dictionary_override_logging(self):
        """Task 6: Every label override must be logged to a reviewable file with details."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_log_path = tmp.name

        try:
            sample_sentence = "I was assigned to do data entry in the main office."
            start_idx = sample_sentence.index("data entry")
            end_idx = start_idx + len("data entry")
            ml_ents = [
                {"term": "data entry", "category": "CLERICAL_TERM", "start": start_idx, "end": end_idx, "confidence": 0.95, "source": "ML", "status": "ACCEPTED"},
            ]
            dict_ents = [
                {"term": "data entry", "category": "IT_TERM", "start": start_idx, "end": end_idx, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
            ]
            resolved = resolve_span_conflicts(
                ml_ents, dict_ents, text=sample_sentence, override_log_path=tmp_log_path
            )

            self.assertEqual(resolved[0]["category"], "IT_TERM")

            # Verify log content
            with open(tmp_log_path, "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f if line.strip()]

            self.assertGreaterEqual(len(logs), 1)
            entry = logs[0]
            self.assertEqual(entry["term"], "data entry")
            self.assertEqual(entry["sentence"], sample_sentence)
            self.assertEqual(entry["ml_label"], "CLERICAL_TERM")
            self.assertEqual(entry["dict_label"], "IT_TERM")
            self.assertEqual(entry["winning_label"], "IT_TERM")
        finally:
            if os.path.exists(tmp_log_path):
                os.remove(tmp_log_path)

    def test_trailing_copula_trimming(self):
        """Task 5: Trims trailing copulas ('graphics card is' -> 'graphics card')."""
        raw_text = "Troubleshooted the system unit and figured out that the graphic graphics card is defective"
        start, end, trimmed_term = trim_entity_span(raw_text, 64, 80)
        self.assertEqual(trimmed_term, "graphics card")
        self.assertEqual(start, 64)
        self.assertEqual(end, 77)

    def test_leading_verb_trimming(self):
        """Task 5: Trims leading action verbs ('Processing Police Clearance' -> 'Police Clearance')."""
        raw_text = "Processing Police Clearance for the department"
        start, end, trimmed_term = trim_entity_span(raw_text, 0, 27)
        self.assertEqual(trimmed_term, "Police Clearance")
        self.assertEqual(start, 11)
        self.assertEqual(end, 27)


if __name__ == "__main__":
    unittest.main()
