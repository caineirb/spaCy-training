"""
Build the TRSTR-LLM augmented training dataset.

Combines authentic real training documents (from data/training/train.spacy, 687 docs)
with LLM-direct generated synthetic records (from data/synthetic_llm_generated.jsonl).
Produces:
1. data/training_trstr_llm.jsonl (complete JSONL training pool with augmentation_type labels)
2. data/training/train_trstr_llm.spacy (DocBin file for spaCy transformer fine-tuning)

Diagnostics (negative ratio, class balance) are reported for full transparency.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
import spacy
from spacy.tokens import DocBin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.annotation import report_dataset_diagnostics, convert_records_to_docbin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ojt_pipeline.prepare_trstr_llm")


def build_trstr_llm_dataset(
    train_spacy_path: str = "data/training/train.spacy",
    synthetic_jsonl_path: str = "data/synthetic_llm_generated.jsonl",
    mined_negatives_path: Optional[str] = "data/review/mined_hard_negatives.jsonl",
    dev_spacy_path: str = "data/training/dev.spacy",
    test_spacy_path: str = "data/training/test.spacy",
    terms_csv_path: str = "data/terms.csv",
    output_jsonl_path: str = "data/training_trstr_llm.jsonl",
    output_spacy_path: str = "data/training/train_trstr_llm.spacy",
    filter_evaluation_leaks: bool = True,
    filter_accidental_entities: bool = True,
) -> Dict[str, Any]:
    """Builds the combined TRSTR-LLM training dataset with integrated mined hard negatives."""
    nlp = spacy.blank("en")
    
    # 1. Load real training docs from train.spacy
    if not os.path.exists(train_spacy_path):
        raise FileNotFoundError(f"Real train DocBin not found at {train_spacy_path}")
    
    real_db = DocBin().from_disk(train_spacy_path)
    real_docs = list(real_db.get_docs(nlp.vocab))
    logger.info(f"Loaded {len(real_docs)} real training documents from {train_spacy_path}")
    
    real_records = []
    for doc in real_docs:
        real_records.append({
            "text": doc.text,
            "entities": [
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
                for ent in doc.ents
            ],
            "augmentation_type": "real",
        })
    
    # 2. Load synthetic LLM-generated records
    if not os.path.exists(synthetic_jsonl_path):
        raise FileNotFoundError(f"Synthetic file not found at {synthetic_jsonl_path}")
    
    synthetic_records = []
    with open(synthetic_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                synthetic_records.append(json.loads(line))
    logger.info(f"Loaded {len(synthetic_records)} synthetic records from {synthetic_jsonl_path}")
    
    # 3. Load and curate mined hard negative records (Task 1)
    mined_records = []
    if mined_negatives_path and os.path.exists(mined_negatives_path):
        raw_mined = []
        with open(mined_negatives_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_mined.append(json.loads(line))
        logger.info(f"Loaded {len(raw_mined)} raw mined hard negative entries from {mined_negatives_path}")

        # Guard against dev and test leakage (Zero Data Leakage constraint)
        dev_texts = set()
        test_texts = set()
        if filter_evaluation_leaks:
            if os.path.exists(dev_spacy_path):
                dev_db = DocBin().from_disk(dev_spacy_path)
                dev_texts = {d.text.strip() for d in dev_db.get_docs(nlp.vocab)}
            if os.path.exists(test_spacy_path):
                test_db = DocBin().from_disk(test_spacy_path)
                test_texts = {d.text.strip() for d in test_db.get_docs(nlp.vocab)}

        # Guard against accidental unannotated entity terms from terms.csv
        known_terms = set()
        if filter_accidental_entities and os.path.exists(terms_csv_path):
            import pandas as pd
            terms_df = pd.read_csv(terms_csv_path)
            known_terms = set(terms_df["term"].str.lower().str.strip())

        seen_texts = set()
        duplicates_count = 0
        leaks_dev_count = 0
        leaks_test_count = 0
        accidental_count = 0

        for r in raw_mined:
            text = r.get("text", "").strip()
            if not text:
                continue
            if text in seen_texts:
                duplicates_count += 1
                continue
            seen_texts.add(text)

            if filter_evaluation_leaks:
                if text in test_texts:
                    leaks_test_count += 1
                    continue
                if text in dev_texts:
                    leaks_dev_count += 1
                    continue

            if filter_accidental_entities:
                text_lower = text.lower()
                has_entity = any(
                    f" {t} " in f" {text_lower} " or f" {t}," in f" {text_lower} " or f" {t}." in f" {text_lower} "
                    for t in known_terms
                )
                if has_entity:
                    accidental_count += 1
                    continue

            mined_records.append({
                "text": text,
                "entities": [],
                "augmentation_type": "mined_negative",
            })

        logger.info(
            f"Mined negatives curation audit: {len(raw_mined)} raw -> "
            f"{duplicates_count} intra-file duplicates removed, "
            f"{leaks_test_count} held-out test leaks blocked, "
            f"{leaks_dev_count} dev split leaks blocked, "
            f"{accidental_count} accidental entity sentences filtered -> "
            f"{len(mined_records)} clean hard negatives accepted."
        )

    # 4. Combine pools
    combined_records = real_records + synthetic_records + mined_records
    logger.info(
        f"Combined TRSTR-LLM pool: {len(combined_records)} total records "
        f"({len(real_records)} real + {len(synthetic_records)} synthetic + {len(mined_records)} mined negatives)"
    )
    
    # 5. Save combined JSONL
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for rec in combined_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(combined_records)} records to {output_jsonl_path}")
    
    # 6. Build combined DocBin
    combined_db = DocBin(attrs=real_db.attrs)
    for doc in real_docs:
        combined_db.add(doc)
    
    # Convert synthetic records to docs and add
    syn_converted = 0
    syn_dropped = 0
    for r in synthetic_records:
        text = r.get("text", "")
        doc = nlp.make_doc(text)
        spans = []
        for ent in r.get("entities", []):
            start, end, label = ent["start"], ent["end"], ent["label"]
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
            if span is not None:
                spans.append(span)
            else:
                syn_dropped += 1
        doc.ents = spacy.util.filter_spans(spans)
        combined_db.add(doc)
        syn_converted += 1

    # Convert mined negative records (zero entities) to docs and add
    mined_converted = 0
    for r in mined_records:
        text = r.get("text", "")
        doc = nlp.make_doc(text)
        doc.ents = []
        combined_db.add(doc)
        mined_converted += 1
    
    os.makedirs(os.path.dirname(output_spacy_path), exist_ok=True)
    combined_db.to_disk(output_spacy_path)
    logger.info(
        f"Saved combined DocBin to {output_spacy_path} "
        f"({len(real_docs) + syn_converted + mined_converted} total docs, "
        f"{syn_dropped} synthetic entities dropped)"
    )
    
    # 7. Report diagnostics and check negative ratio target band (25-35%)
    diag_real = report_dataset_diagnostics(real_records, dataset_label="Real Training Split")
    diag_syn = report_dataset_diagnostics(synthetic_records, dataset_label="Synthetic Augmentation")
    diag_mined = report_dataset_diagnostics(mined_records, dataset_label="Mined Hard Negatives") if mined_records else None
    diag_combined = report_dataset_diagnostics(combined_records, dataset_label="TRSTR-LLM Combined Training Pool")
    
    total_negs = sum(1 for r in combined_records if len(r.get("entities", [])) == 0)
    neg_ratio = total_negs / len(combined_records) if combined_records else 0.0
    if neg_ratio < 0.25 or neg_ratio > 0.35:
        logger.warning(
            f"[FLAG] Combined training pool negative ratio is {neg_ratio * 100:.2f}%, "
            f"which pushes outside the established 25–35% target band! "
            f"(Total records: {len(combined_records)}, Negatives: {total_negs})"
        )
    else:
        logger.info(
            f"[PASS] Combined training pool negative ratio is {neg_ratio * 100:.2f}% (within 25–35% target band)."
        )

    return {
        "total_records": len(combined_records),
        "real_count": len(real_records),
        "synthetic_count": len(synthetic_records),
        "mined_count": len(mined_records),
        "total_negatives": total_negs,
        "negative_ratio_pct": round(neg_ratio * 100, 2),
        "negative_ratio_flagged": (neg_ratio < 0.25 or neg_ratio > 0.35),
        "output_jsonl": output_jsonl_path,
        "output_spacy": output_spacy_path,
        "diagnostics": {
            "real": diag_real,
            "synthetic": diag_syn,
            "mined": diag_mined,
            "combined": diag_combined,
        },
    }


if __name__ == "__main__":
    build_trstr_llm_dataset()
