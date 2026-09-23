#!/usr/bin/env python3
"""
Active Learning Retraining Workflow.

Takes one or more manually reviewed/verified details.jsonl files
(e.g., data/human_written_details.jsonl and/or data/structured_details.jsonl):
1. Ingests approved entities (skipping any line marked REJECTED, DISCARD, or FALSE_POSITIVE).
2. Automatically updates `data/terms.csv` with newly validated terms.
3. Groups entity spans by sentence into the standard project JSONL schema and appends to `data/data.jsonl`.
4. Re-compiles spaCy binary DocBins (`train.spacy`, `dev.spacy`).
5. Fine-tunes the transformer model on the GPU (RTX 3060).
6. Re-packages the updated hybrid pipeline into `models/hybrid_pipeline`.
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any, Set, Tuple, Union
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
from scripts.annotation import (
    load_terms_dictionary,
    split_and_convert_dataset,
    load_jsonl,
    save_jsonl,
)
from scripts.training import train_ner_trf
from scripts.pipeline import HybridJournalPipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ojt_retrain")


def parse_verified_details(
    details_paths: Union[str, List[str]],
    terms_csv_path: str = "data/terms.csv"
) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, str]], List[Dict[str, Any]]]:
    """Parses one or more reviewed details.jsonl files.
    
    Accepts records where status is 'ACCEPTED' or 'APPROVED' (skips 'REJECTED', 'DISCARD', etc.).
    Groups entity spans by sentence.
    
    Returns:
        (new_training_records, new_terms_to_add, rejected_records)
    """
    if isinstance(details_paths, str):
        details_paths = [details_paths]

    # Load existing terms to avoid redundant additions
    df_terms = pd.read_csv(terms_csv_path)
    existing_terms = {t.lower(): lbl for t, lbl in zip(df_terms["term"], df_terms["label"])}

    text_to_entities: Dict[str, List[Dict[str, Any]]] = {}
    new_terms_to_add: Set[Tuple[str, str]] = set()
    rejected_records: List[Dict[str, Any]] = []

    for file_path in details_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found, skipping: {file_path}")
            continue

        logger.info(f"Parsing verified records from: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                status = str(rec.get("status", "ACCEPTED")).upper()
                term = rec.get("term", "").strip()
                label = rec.get("classification") or rec.get("label", "IT_TERM")
                text = rec.get("context") or rec.get("text", "")
                start = rec.get("start")
                end = rec.get("end")

                # If manually rejected, log and skip
                if status in ["REJECTED", "DISCARD", "FALSE_POSITIVE", "IGNORE", "NO"]:
                    rejected_records.append({"term": term, "context": text, "status": status})
                    continue

                if not text or not term:
                    continue

                if text not in text_to_entities:
                    text_to_entities[text] = []

                # Ensure span character offsets are valid
                if start is not None and end is not None and text[start:end] == term:
                    text_to_entities[text].append({"start": start, "end": end, "label": label, "term": term})
                else:
                    # Re-locate exact substring index in text
                    idx = text.find(term)
                    if idx != -1:
                        text_to_entities[text].append({"start": idx, "end": idx + len(term), "label": label, "term": term})

                # Check if this valid term is novel and should be added to dictionary
                if term.lower() not in existing_terms and len(term) > 1:
                    raw_label = "IT_TASK" if "IT" in label else "CLERICAL"
                    new_terms_to_add.add((term, raw_label))

    # Convert text_to_entities mapping into standard training records
    new_records = []
    for text, ents in text_to_entities.items():
        ents.sort(key=lambda x: x["start"])
        clean_ents = []
        occupied: List[Tuple[int, int]] = []
        for e in ents:
            s, en = e["start"], e["end"]
            if not any(max(s, os) < min(en, oe) for os, oe in occupied):
                clean_ents.append({"start": s, "end": en, "label": e["label"]})
                occupied.append((s, en))

        new_records.append({"text": text, "entities": clean_ents})

    return new_records, new_terms_to_add, rejected_records


def update_terms_csv(
    new_terms: Set[Tuple[str, str]],
    terms_csv_path: str = "data/terms.csv"
) -> int:
    """Appends newly validated terms to data/terms.csv."""
    if not new_terms:
        logger.info("No new unique terms to add to dictionary.")
        return 0

    df = pd.read_csv(terms_csv_path)
    existing_terms = set(df["term"].str.lower())
    rows_to_add = []

    for term, label in new_terms:
        if term.lower() not in existing_terms:
            rows_to_add.append({"term": term, "label": label})
            existing_terms.add(term.lower())

    if rows_to_add:
        df_new = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)
        df_new.to_csv(terms_csv_path, index=False)
        logger.info(f"Appended {len(rows_to_add)} validated terms into {terms_csv_path}.")
        return len(rows_to_add)

    return 0


def retrain_from_verified_data(
    details_paths: Union[str, List[str]],
    terms_csv_path: str = "data/terms.csv",
    annotations_jsonl_path: str = "data/data.jsonl",
    training_data_dir: str = "data/training",
    model_output_dir: str = "models/ner_trf",
    hybrid_pipeline_dir: str = "models/hybrid_pipeline",
    max_steps: int = 200,
    eval_frequency: int = 50,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Complete active learning loop: updates dictionary, dataset, retrains, and saves pipeline."""
    scripts.init_gpu()

    logger.info(f"--- Step 1: Parsing verified records ---")
    new_records, new_terms, rejected = parse_verified_details(details_paths, terms_csv_path)

    logger.info(f"Summary of parsed input:")
    logger.info(f"  - Validated Sentences: {len(new_records)}")
    logger.info(f"  - Novel Dictionary Terms to Add: {len(new_terms)}")
    logger.info(f"  - Rejected / Discarded Spans: {len(rejected)}")

    if dry_run:
        logger.info("[DRY RUN] Previewing changes without modifying files or retraining.")
        return {
            "status": "DRY_RUN",
            "new_terms_to_add": list(new_terms),
            "rejected_items": rejected,
            "new_sentence_count": len(new_records),
        }

    logger.info("--- Step 2: Updating seed dictionary ---")
    added_count = update_terms_csv(new_terms, terms_csv_path)

    logger.info("--- Step 3: Merging training annotations ---")
    existing_records = load_jsonl(annotations_jsonl_path) if os.path.exists(annotations_jsonl_path) else []
    existing_texts = {r["text"] for r in existing_records}

    appended_records = 0
    for rec in new_records:
        if rec["text"] not in existing_texts:
            existing_records.append(rec)
            existing_texts.add(rec["text"])
            appended_records += 1

    save_jsonl(existing_records, annotations_jsonl_path)
    logger.info(f"Total training dataset size: {len(existing_records)} records (+{appended_records} new).")

    logger.info("--- Step 4: Re-compiling spaCy DocBin binary datasets ---")
    spacy_paths = split_and_convert_dataset(existing_records, output_dir=training_data_dir)

    logger.info(f"--- Step 5: Fine-tuning Transformer on GPU ({max_steps} steps) ---")
    train_res = train_ner_trf(
        train_path=spacy_paths["train"],
        dev_path=spacy_paths["dev"],
        output_dir=model_output_dir,
        max_steps=max_steps,
        eval_frequency=eval_frequency,
        use_gpu=0,
    )

    logger.info("--- Step 6: Packaging updated Hybrid Pipeline ---")
    best_model_path = train_res["best_model_path"]
    updated_pipeline = HybridJournalPipeline(
        model_path=best_model_path,
        terms_csv_path=terms_csv_path
    )
    updated_pipeline.save(hybrid_pipeline_dir)
    logger.info(f"Successfully packaged and saved updated pipeline to {hybrid_pipeline_dir}")

    return {
        "status": "SUCCESS",
        "new_terms_added": added_count,
        "new_sentences_added": appended_records,
        "total_dataset_size": len(existing_records),
        "best_model": best_model_path,
        "hybrid_pipeline": hybrid_pipeline_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Retrain OJT Pipeline from manually reviewed details.jsonl file(s)."
    )
    parser.add_argument(
        "-i", "--verified-files",
        nargs="+",
        required=True,
        help="Path(s) to reviewed details.jsonl (e.g. data/human_written_details.jsonl data/structured_details.jsonl)"
    )
    parser.add_argument(
        "--terms-path",
        default="data/terms.csv",
        help="Path to terms dictionary (default: data/terms.csv)"
    )
    parser.add_argument(
        "--annotations-path",
        default="data/data.jsonl",
        help="Path to golden reviewed annotations"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Training steps (default: 200)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect novel terms and sentences to be added without modifying files or retraining"
    )
    args = parser.parse_args()

    results = retrain_from_verified_data(
        details_paths=args.verified_files,
        terms_csv_path=args.terms_path,
        annotations_jsonl_path=args.annotations_path,
        max_steps=args.steps,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    if args.dry_run:
        print("RETRAINING DRY RUN PREVIEW")
        print("=" * 60)
        print(f"Validated sentences parsed  : {results['new_sentence_count']}")
        print(f"Novel terms to be added     : {len(results['new_terms_to_add'])}")
        for t, lbl in sorted(results["new_terms_to_add"]):
            print(f"  + {t:<25} ({lbl})")
        if results["rejected_items"]:
            print(f"\nRejected / Discarded items  : {len(results['rejected_items'])}")
            for r in results["rejected_items"]:
                print(f"  - {r['term']:<25} (status: {r['status']})")
        print("=" * 60)
        print("Run without --dry-run to apply updates and train on GPU.")
    else:
        print("RETRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"New terms added to dictionary : {results['new_terms_added']}")
        print(f"New sentences added to dataset: {results['new_sentences_added']}")
        print(f"Total training dataset size   : {results['total_dataset_size']}")
        print(f"Updated Hybrid Pipeline saved : {results['hybrid_pipeline']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
