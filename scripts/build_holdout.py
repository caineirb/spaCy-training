"""
Helper script to scaffold and generate draft annotations for the Real-World Holdout Test Set.

Workflow:
1. Reads unprocessed real-world OJT journal text files from data/test/raw/*.txt.
2. Segments text into sentences.
3. Runs the HybridJournalPipeline to predict candidate task entities.
4. Suggests a 'term_status' ('seen' vs. 'unseen') by cross-referencing data/terms.csv
   and the training annotations in data/data.jsonl.
5. Emits candidate records to data/test/holdout_draft.jsonl for manual audit and verification.

NOTE: This script DOES NOT overwrite data/test/holdout.jsonl. Promotion from draft
to gold holdout is strictly a manual human-in-the-loop validation step.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Set, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import spacy
from scripts.pipeline import HybridJournalPipeline

logger = logging.getLogger("ojt_pipeline.build_holdout")


def load_known_training_terms(
    terms_csv_path: str = "data/terms.csv",
    annotations_jsonl_path: str = "data/data.jsonl",
) -> Set[str]:
    """Loads all known lowercase term strings from terms.csv and training annotations.
    
    Used to suggest whether a holdout term is 'seen' in training context or 'unseen'.
    """
    known: Set[str] = set()

    # 1. From terms.csv
    if os.path.exists(terms_csv_path):
        df = pd.read_csv(terms_csv_path)
        for t in df["term"].dropna():
            clean = str(t).strip().lower()
            if clean:
                known.add(clean)

    # 2. From training annotations
    if os.path.exists(annotations_jsonl_path):
        with open(annotations_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    text = data.get("text", "")
                    for ent in data.get("entities", []):
                        s, e = ent.get("start", -1), ent.get("end", -1)
                        if 0 <= s < e <= len(text):
                            known.add(text[s:e].strip().lower())
                        elif "term" in ent:
                            known.add(str(ent["term"]).strip().lower())

    logger.info(f"Loaded {len(known)} known unique terms for seen/unseen discrimination.")
    return known


def suggest_term_status(term: str, known_terms: Set[str]) -> str:
    """Suggests whether a candidate entity term is 'seen' or 'unseen'.
    
    Args:
        term: The extracted entity text.
        known_terms: Set of all lowercase terms from terms.csv and training annotations.
        
    Returns:
        'seen' if term or any direct component exists in known_terms, else 'unseen'.
    """
    clean_term = term.strip().lower()
    if clean_term in known_terms:
        return "seen"

    # Match individual word components for compounds if length > 3
    tokens = [t for t in clean_term.split() if len(t) > 3]
    for k in known_terms:
        if any(t == k for t in tokens):
            return "seen"

    return "unseen"


def split_into_sentences(text: str, nlp: Optional[spacy.language.Language] = None) -> List[str]:
    """Splits arbitrary text into clean sentences using spaCy's rule-based sentencizer."""
    if nlp is None:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        s_text = sent.text.strip()
        # Filter out empty or whitespace-only lines
        if s_text and len(s_text) > 5:
            sentences.append(s_text)
    return sentences


def generate_holdout_draft(
    raw_dir: str = "data/test/raw",
    draft_output_path: str = "data/test/holdout_draft.jsonl",
    terms_csv_path: str = "data/terms.csv",
    annotations_jsonl_path: str = "data/data.jsonl",
    pipeline: Optional[HybridJournalPipeline] = None,
) -> Dict[str, Any]:
    """Processes raw journal text files to produce a draft holdout JSONL for human audit."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(draft_output_path), exist_ok=True)

    raw_files = sorted([
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if f.endswith((".txt", ".md", ".log"))
    ])

    if not raw_files:
        logger.warning(
            f"No raw text files found in '{raw_dir}'. "
            "Please place anonymized real journal text files in data/test/raw/ and rerun."
        )
        return {
            "status": "NO_RAW_FILES",
            "message": f"No raw files found in {raw_dir}",
            "processed_sentences": 0,
            "draft_output_path": draft_output_path,
        }

    if pipeline is None:
        pipeline = HybridJournalPipeline(terms_csv_path=terms_csv_path)

    known_terms = load_known_training_terms(
        terms_csv_path=terms_csv_path,
        annotations_jsonl_path=annotations_jsonl_path,
    )

    sentencizer_nlp = spacy.blank("en")
    sentencizer_nlp.add_pipe("sentencizer")

    total_sentences = 0
    total_entities_suggested = 0
    draft_records: List[Dict[str, Any]] = []

    for file_path in raw_files:
        filename = os.path.basename(file_path)
        logger.info(f"Processing raw holdout source: '{filename}'...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sentences = split_into_sentences(content, nlp=sentencizer_nlp)
        for sent in sentences:
            total_sentences += 1
            pred = pipeline.predict(sent)
            pred_ents = pred.get("entities", [])

            annotated_entities = []
            for ent in pred_ents:
                term = ent["term"]
                status_suggestion = suggest_term_status(term, known_terms)
                annotated_entities.append({
                    "start": ent["start"],
                    "end": ent["end"],
                    "term": term,
                    "label": ent["category"],
                    "term_status": status_suggestion,
                })
                total_entities_suggested += 1

            draft_records.append({
                "text": sent,
                "entities": annotated_entities,
                "source": "real_journal_holdout",
                "notes": f"Suggested from {filename} (Review before promoting to holdout.jsonl)"
            })

    # Write draft records (NEVER directly to holdout.jsonl)
    with open(draft_output_path, "w", encoding="utf-8") as f:
        for r in draft_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        f"Generated {len(draft_records)} draft records ({total_entities_suggested} suggested entities) "
        f"saved to '{draft_output_path}'"
    )

    return {
        "status": "SUCCESS",
        "raw_files_processed": len(raw_files),
        "processed_sentences": total_sentences,
        "suggested_entities": total_entities_suggested,
        "draft_output_path": draft_output_path,
        "sample": draft_records[:3] if draft_records else [],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    res = generate_holdout_draft()
    print("\n" + "=" * 70)
    print("HOLDOUT DRAFT SCAFFOLDING RESULT")
    print("=" * 70)
    print(json.dumps(res, indent=2))
