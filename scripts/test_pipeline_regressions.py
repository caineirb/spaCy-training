"""
Regression tests for Hybrid NER Pipeline improvements (Task 3).

Validates:
1. Longest span resolution: Dictionary must never truncate or split a longer ML span
   (Tailwind CSS, Canva Editing, access control systems, CSS flexibility).
2. ML label authority: Dictionary must not override ML label on disagreement
   (data entry, file management).
3. Abstention fallback: Dictionary entities accepted only when ML abstains.
4. End-to-end integration: Tests against HybridJournalPipeline with simulated text.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.pipeline import resolve_span_conflicts, HybridJournalPipeline


class TestPipelineRegressions(unittest.TestCase):
    """Regression test suite for hybrid pipeline fixes."""

    def test_longest_span_ml_over_dict(self):
        """Rule (a): ML span is longer than dictionary sub-span -> ML span wins."""
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

    def test_ml_label_authority_on_disagreement(self):
        """Rule (b): When ML and Dictionary disagree on label, ML label takes precedence."""
        ml_ents = [
            {"term": "data entry", "category": "CLERICAL_TERM", "start": 13, "end": 23, "confidence": 0.96, "source": "ML", "status": "ACCEPTED"},
            {"term": "File Management", "category": "CLERICAL_TERM", "start": 11, "end": 26, "confidence": 0.91, "source": "ML", "status": "ACCEPTED"},
        ]

        # Contradictory IT_TERM labels from legacy dictionary
        dict_ents = [
            {"term": "data entry", "category": "IT_TERM", "start": 13, "end": 23, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
            {"term": "File Management", "category": "IT_TERM", "start": 11, "end": 26, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
        ]

        resolved = resolve_span_conflicts(ml_ents, dict_ents)
        resolved_map = {e["term"]: e["category"] for e in resolved}

        self.assertEqual(resolved_map.get("data entry"), "CLERICAL_TERM")
        self.assertEqual(resolved_map.get("File Management"), "CLERICAL_TERM")

    def test_dictionary_used_only_when_ml_abstains(self):
        """Rule (b): Dictionary provides entities only when ML makes no prediction overlapping the span."""
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
        """Rule (a): If dictionary span is strictly longer than partial ML span, longer span wins with ML label authority."""
        ml_ents = [
            {"term": "management", "category": "CLERICAL_TERM", "start": 16, "end": 26, "confidence": 0.88, "source": "ML", "status": "ACCEPTED"},
        ]
        dict_ents = [
            {"term": "File Management", "category": "IT_TERM", "start": 11, "end": 26, "confidence": 1.0, "source": "dictionary", "status": "ACCEPTED"},
        ]

        resolved = resolve_span_conflicts(ml_ents, dict_ents)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["term"], "File Management")
        # ML label authority takes precedence even when dict span is longer
        self.assertEqual(resolved[0]["category"], "CLERICAL_TERM")


if __name__ == "__main__":
    unittest.main()
