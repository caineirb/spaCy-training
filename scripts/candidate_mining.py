"""
Candidate mining and active-learning loop module.

Scans journal entries for:
1. Novel ML-extracted entities not yet cataloged in the dictionary
2. Contextual linguistic trigger patterns (e.g., 'developed ... using [X]', 'encoded ... during [X]')

Aggregates frequencies, sample contexts, and exports a ranked `candidate,count`
CSV for human review and feedback into `data/terms.csv`.
"""

import os
import re
import sys
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
from scripts.annotation import load_terms_dictionary
from scripts.pipeline import HybridJournalPipeline

logger = logging.getLogger("ojt_pipeline.candidate_mining")

# Linguistic syntactic trigger patterns for candidate discovery
TRIGGER_PATTERNS = [
    # IT triggers
    (
        r"(?:developed|built|coded|implemented|created|refactored|designed)\s+(?:.*?)\s+(?:using|in|with)\s+([A-Z][A-Za-z0-9\.\+#_\-]+(?:\s+[A-Z][A-Za-z0-9\.\+#_\-]+){0,2})",
        "IT_TERM",
    ),
    (
        r"(?:configured|deployed|installed|setup|hosted)\s+(?:.*?)\s+(?:on|via|using)\s+([A-Z][A-Za-z0-9\.\+#_\-]+(?:\s+[A-Z][A-Za-z0-9\.\+#_\-]+){0,2})",
        "IT_TERM",
    ),
    (
        r"(?:debugged|optimized|migrated|benchmarked)\s+(?:.*?)\s+(?:in|for|on)\s+([A-Z][A-Za-z0-9\.\+#_\-]+(?:\s+[A-Z][A-Za-z0-9\.\+#_\-]+){0,2})",
        "IT_TERM",
    ),
    # Clerical triggers
    (
        r"(?:encoded|transcribed|filed|audited|organized|logged)\s+(?:.*?)\s+(?:during|for|into)\s+([A-Z][A-Za-z0-9\.\+#_\-]+(?:\s+[A-Z][A-Za-z0-9\.\+#_\-]+){0,2})",
        "CLERICAL_TERM",
    ),
    (
        r"(?:prepared|printed|distributed|sorted)\s+(?:the\s+)?([A-Z][A-Za-z0-9\.\+#_\-]+(?:\s+[A-Z][A-Za-z0-9\.\+#_\-]+){0,2})\s+(?:reports|files|records|paperwork|documents)",
        "CLERICAL_TERM",
    ),
    (
        r"(?:assisted\s+(?:the\s+)?supervisor\s+with)\s+([A-Z][A-Za-z0-9\.\+#_\-]+(?:\s+[A-Z][A-Za-z0-9\.\+#_\-]+){0,2})",
        "CLERICAL_TERM",
    ),
]


class CandidateMiner:
    """Mines novel task terms from journals and manages the active-learning loop."""

    def __init__(
        self,
        pipeline: Optional[HybridJournalPipeline] = None,
        terms_csv_path: str = "data/terms.csv",
    ) -> None:
        self.terms_csv_path = terms_csv_path
        self.terms_dict = load_terms_dictionary(terms_csv_path)
        self.known_terms_lower = {t.lower() for t in self.terms_dict.keys()}
        self.pipeline = pipeline or HybridJournalPipeline(terms_csv_path=terms_csv_path)

    def mine_from_sentences(
        self,
        sentences: List[str],
        output_csv_path: str = "data/candidates/mined_candidates.csv",
    ) -> pd.DataFrame:
        """Extracts and aggregates novel candidates from a list of journal sentences.
        
        Combines ML extractions (entities not in dictionary) and syntactic pattern triggers.
        Exports a ranked table with candidate, count, suggested label, and sample context.
        """
        candidate_counts: Counter = Counter()
        candidate_labels: Dict[str, Counter] = defaultdict(Counter)
        candidate_contexts: Dict[str, List[str]] = defaultdict(list)
        candidate_confidences: Dict[str, List[float]] = defaultdict(list)

        # 1. Mine via Hybrid Pipeline predictions (ML extractions not in dictionary)
        predictions = self.pipeline.predict_batch(sentences)
        for pred in predictions:
            text = pred["text"]
            for ent in pred["entities"]:
                term = ent["term"].strip()
                if term.lower() not in self.known_terms_lower and len(term) > 1:
                    candidate_counts[term] += 1
                    candidate_labels[term][ent["category"]] += 1
                    candidate_confidences[term].append(ent["confidence"])
                    if len(candidate_contexts[term]) < 2:
                        candidate_contexts[term].append(text)

        # 2. Mine via linguistic syntactic trigger regex patterns
        for text in sentences:
            for pattern, suggested_lbl in TRIGGER_PATTERNS:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    matched_term = match.group(1).strip()
                    # Clean trailing punctuation
                    matched_term = re.sub(r"[,\.;:]$", "", matched_term).strip()
                    if matched_term.lower() not in self.known_terms_lower and len(matched_term) > 2:
                        candidate_counts[matched_term] += 1
                        candidate_labels[matched_term][suggested_lbl] += 1
                        candidate_confidences[matched_term].append(0.85)
                        if len(candidate_contexts[matched_term]) < 2:
                            candidate_contexts[matched_term].append(text)

        # Build DataFrame
        rows = []
        for term, count in candidate_counts.most_common():
            top_label = candidate_labels[term].most_common(1)[0][0]
            confs = candidate_confidences[term]
            mean_conf = round(sum(confs) / len(confs), 2) if confs else 0.85
            contexts = " | ".join(candidate_contexts[term])

            rows.append({
                "candidate": term,
                "count": count,
                "suggested_label": top_label,
                "mean_confidence": mean_conf,
                "sample_context": contexts,
            })

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Mined {len(df)} unique candidate terms. Saved to {output_csv_path}")
        return df

    def add_validated_term(self, term: str, label: str) -> None:
        """Active learning feedback: appends human-validated candidate term into terms.csv.
        
        Args:
            term: The validated term (e.g. 'FastAPI')
            label: The category label (e.g. 'IT_TERM' or 'CLERICAL_TERM')
        """
        term = term.strip()
        label = label.strip()

        # Read existing terms CSV
        df = pd.read_csv(self.terms_csv_path)
        existing_terms = set(df["term"].str.lower())

        if term.lower() in existing_terms:
            logger.info(f"Term '{term}' is already present in {self.terms_csv_path}")
            return

        # Append new row
        raw_label = "IT_TASK" if "IT" in label else "CLERICAL"
        new_row = pd.DataFrame([{"term": term, "label": raw_label}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.terms_csv_path, index=False)
        logger.info(f"Added validated term '{term}' ({label}) to {self.terms_csv_path}")

        # Refresh in-memory dictionary & pipeline EntityRuler
        self.terms_dict = load_terms_dictionary(self.terms_csv_path)
        self.known_terms_lower.add(term.lower())
        self.pipeline._setup_entity_ruler()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    miner = CandidateMiner()
    sample_journals = [
        "I built an asynchronous REST API using FastAPI and connected it to Redis.",
        "Refactored our frontend user interface using Svelte and Tailwind CSS.",
        "Assisted the department supervisor with Student Registration paperwork.",
        "Developed a modern web service in Bun with high throughput.",
        "Encoded patient records using Airtable for team tracking.",
        "Attended the morning standup meeting with the team.",
    ]
    df_candidates = miner.mine_from_sentences(sample_journals)
    print("Mined candidates:")
    print(df_candidates[["candidate", "count", "suggested_label", "mean_confidence"]])
