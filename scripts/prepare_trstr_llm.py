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
from typing import List, Dict, Any
import spacy
from spacy.tokens import DocBin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.annotation import report_dataset_diagnostics, convert_records_to_docbin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ojt_pipeline.prepare_trstr_llm")


def build_trstr_llm_dataset(
    train_spacy_path: str = "data/training/train.spacy",
    synthetic_jsonl_path: str = "data/synthetic_llm_generated.jsonl",
    output_jsonl_path: str = "data/training_trstr_llm.jsonl",
    output_spacy_path: str = "data/training/train_trstr_llm.spacy",
) -> Dict[str, Any]:
    """Builds the combined TRSTR-LLM training dataset."""
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
    
    # 3. Combine pools
    combined_records = real_records + synthetic_records
    logger.info(f"Combined TRSTR-LLM pool: {len(combined_records)} total records ({len(real_records)} real + {len(synthetic_records)} synthetic)")
    
    # 4. Save combined JSONL
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for rec in combined_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(combined_records)} records to {output_jsonl_path}")
    
    # 5. Build combined DocBin
    # Combine the pre-existing real docs with newly converted synthetic docs
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
    
    os.makedirs(os.path.dirname(output_spacy_path), exist_ok=True)
    combined_db.to_disk(output_spacy_path)
    logger.info(f"Saved combined DocBin to {output_spacy_path} ({len(real_docs) + syn_converted} total docs, {syn_dropped} synthetic entities dropped)")
    
    # 6. Report diagnostics
    diag_real = report_dataset_diagnostics(real_records, dataset_label="Real Training Split")
    diag_syn = report_dataset_diagnostics(synthetic_records, dataset_label="Synthetic Augmentation")
    diag_combined = report_dataset_diagnostics(combined_records, dataset_label="TRSTR-LLM Combined Training Pool")
    
    return {
        "total_records": len(combined_records),
        "real_count": len(real_records),
        "synthetic_count": len(synthetic_records),
        "output_jsonl": output_jsonl_path,
        "output_spacy": output_spacy_path,
        "diagnostics": {
            "real": diag_real,
            "synthetic": diag_syn,
            "combined": diag_combined,
        },
    }


if __name__ == "__main__":
    build_trstr_llm_dataset()
