"""
Annotation tooling and dataset preparation module.

Provides:
- Seed dictionary loading & normalization (IT_TASK -> IT_TERM, CLERICAL -> CLERICAL_TERM)
- Weak supervision / string-matching candidate annotation
- JSONL I/O for gold annotation storage
- Deduplication and train/dev/test splitting
- Conversion from JSONL into spaCy binary DocBin (.spacy) format
- Dataset diagnostic reporting (negative ratio, class balance)
"""

import os
import sys
import re
import json
import random
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import pandas as pd
import spacy
from spacy.tokens import DocBin, Doc

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.labels import (
    LABEL_MAPPING,
    normalize_to_ner_label,
    normalize_to_dictionary_label,
    CANONICAL_NER_LABELS,
    CANONICAL_DICTIONARY_LABELS,
)

logger = logging.getLogger("ojt_pipeline.annotation")


def load_terms_dictionary(terms_csv_path: str = "data/terms.csv") -> Dict[str, str]:
    """Loads terms dictionary and normalizes labels.
    
    Terms are sorted by length descending so longer phrases match before sub-phrases.
    """
    if not os.path.exists(terms_csv_path):
        raise FileNotFoundError(f"Terms dictionary not found at {terms_csv_path}")

    df = pd.read_csv(terms_csv_path)
    terms_dict = {}
    for _, row in df.iterrows():
        term = str(row["term"]).strip()
        raw_label = str(row["label"]).strip()
        norm_label = LABEL_MAPPING.get(raw_label, raw_label)
        if term:
            terms_dict[term] = norm_label

    # Sort descending by length to handle multi-word terms prior to single tokens
    sorted_terms = dict(sorted(terms_dict.items(), key=lambda item: len(item[0]), reverse=True))
    logger.info(f"Loaded {len(sorted_terms)} unique terms from {terms_csv_path}")
    return sorted_terms


def find_term_spans(text: str, terms_dict: Dict[str, str]) -> List[Dict[str, Any]]:
    """Finds non-overlapping entity spans in text using boundary-sensitive matching."""
    spans: List[Tuple[int, int, str, str]] = []
    occupied_ranges: List[Tuple[int, int]] = []

    for term, label in terms_dict.items():
        escaped_term = re.escape(term)
        # Handle word boundaries safely (including terms with punctuation like C#, Vue.js, .NET)
        pattern = r"(?<!\w)" + escaped_term + r"(?!\w)"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span()
            # Check overlap with existing spans
            overlaps = any(max(start, o_start) < min(end, o_end) for o_start, o_end in occupied_ranges)
            if not overlaps:
                spans.append((start, end, label, text[start:end]))
                occupied_ranges.append((start, end))

    # Sort spans by character offset
    spans.sort(key=lambda s: s[0])
    return [
        {"start": start, "end": end, "label": label, "term": text_span}
        for start, end, label, text_span in spans
    ]


def weak_label_corpus(texts: List[str], terms_dict: Dict[str, str]) -> List[Dict[str, Any]]:
    """Applies weak labeling across a collection of journal text entries."""
    records = []
    for text in texts:
        entities = find_term_spans(text, terms_dict)
        clean_entities = [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in entities]
        records.append({"text": text, "entities": clean_entities})
    return records


def save_jsonl(records: List[Dict[str, Any]], output_jsonl_path: str) -> None:
    """Saves records in the standard project JSONL schema."""
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} JSONL records to {output_jsonl_path}")


def load_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Loads records from a JSONL file."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def convert_records_to_docbin(
    records: List[Dict[str, Any]],
    output_spacy_path: str,
    nlp: Optional[spacy.language.Language] = None
) -> Tuple[int, int]:
    """Converts annotation records into a binary spaCy DocBin (.spacy).
    
    Returns:
        (total_docs, total_entities_stored)
    """
    if nlp is None:
        nlp = spacy.blank("en")

    doc_bin = DocBin()
    total_docs = 0
    total_entities = 0
    dropped_entities = 0

    for rec in records:
        text = rec["text"]
        entities = rec.get("entities", [])
        doc = nlp.make_doc(text)
        spans = []

        for ent in entities:
            start, end, label = ent["start"], ent["end"], ent["label"]
            # Align exact character span with token boundaries
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                # Try expand alignment if contract was too strict
                span = doc.char_span(start, end, label=label, alignment_mode="expand")

            if span is not None:
                spans.append(span)
                total_entities += 1
            else:
                logger.debug(f"Could not align span [{start}:{end}] '{text[start:end]}' with tokens in '{text}'")
                dropped_entities += 1

        # Filter overlapping spans if any
        doc.ents = spacy.util.filter_spans(spans)
        doc_bin.add(doc)
        total_docs += 1

    os.makedirs(os.path.dirname(output_spacy_path), exist_ok=True)
    doc_bin.to_disk(output_spacy_path)
    logger.info(
        f"Saved {total_docs} docs ({total_entities} valid entities, "
        f"{dropped_entities} dropped) to {output_spacy_path}"
    )
    return total_docs, total_entities


def deduplicate_records(
    records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deduplicates annotation records on exact sentence text match before splitting.

    Verifies whether duplicate sentences carry identical or conflicting entity annotations.
    Preserves original record ordering for the first occurrence.

    Returns:
        (deduplicated_records, audit_stats)
    """
    seen_texts: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}
    deduped: List[Dict[str, Any]] = []
    inconsistent: List[Dict[str, Any]] = []
    duplicates_removed = 0

    for idx, r in enumerate(records):
        text = r.get("text", "").strip()
        if not text:
            continue

        ents = sorted(
            r.get("entities", []),
            key=lambda e: (e.get("start", -1), e.get("end", -1), str(e.get("label", "")))
        )

        if text in seen_texts:
            duplicates_removed += 1
            prev_idx, prev_ents = seen_texts[text]
            if ents != prev_ents:
                inconsistent.append({
                    "text": text,
                    "first_seen_index": prev_idx,
                    "first_seen_entities": prev_ents,
                    "duplicate_index": idx,
                    "duplicate_entities": ents,
                })
        else:
            seen_texts[text] = (idx, ents)
            deduped.append(r)

    if inconsistent:
        logger.warning(
            f"Found {len(inconsistent)} duplicate sentence texts with conflicting annotations!"
        )
        for inc in inconsistent[:5]:
            logger.warning(f"  Conflict on: '{inc['text']}'")
    else:
        logger.info(
            f"Deduplication complete: {len(records)} -> {len(deduped)} records "
            f"({duplicates_removed} duplicates removed, 0 annotation conflicts)."
        )

    stats = {
        "original_count": len(records),
        "deduped_count": len(deduped),
        "duplicates_removed": duplicates_removed,
        "inconsistent_count": len(inconsistent),
        "inconsistent_details": inconsistent,
    }
    return deduped, stats


def report_dataset_diagnostics(
    records: List[Dict[str, Any]],
    dataset_label: str = "Full Dataset"
) -> Dict[str, Any]:
    """Reports negative ratio and class balance diagnostics for annotation records.
    
    Checks against the project's target ranges:
    - Negative ratio: 25-35%
    - Class balance: roughly equal IT_TERM / CLERICAL_TERM entity counts
    
    This is a diagnostic report only — it does not fabricate or discard examples.
    """
    total = len(records)
    if total == 0:
        logger.warning(f"[{dataset_label}] No records to analyze.")
        return {"label": dataset_label, "total_records": 0}

    negatives = sum(1 for r in records if not r.get("entities"))
    positives = total - negatives

    label_counts: Dict[str, int] = {}
    for r in records:
        for ent in r.get("entities", []):
            lbl = ent.get("label", "UNKNOWN")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    neg_ratio = round(negatives / total * 100, 1)

    logger.info(f"--- {dataset_label} Diagnostics ---")
    logger.info(f"  Total records : {total}")
    logger.info(f"  Positives     : {positives}")
    logger.info(f"  Negatives     : {negatives} ({neg_ratio}%)")
    logger.info(f"  Entity labels : {label_counts}")

    if neg_ratio < 25:
        logger.warning(
            f"  Negative ratio ({neg_ratio}%) is below target range (25-35%). "
            "Consider adding more negative examples."
        )
    elif neg_ratio > 35:
        logger.warning(
            f"  Negative ratio ({neg_ratio}%) is above target range (25-35%). "
            "Consider adding more positive examples."
        )
    else:
        logger.info(f"  Negative ratio is within target range (25-35%).")

    it_count = label_counts.get("IT_TERM", 0)
    clerical_count = label_counts.get("CLERICAL_TERM", 0)
    if it_count > 0 and clerical_count > 0:
        ratio = round(max(it_count, clerical_count) / min(it_count, clerical_count), 2)
        if ratio > 2.0:
            logger.warning(
                f"  Class imbalance: IT_TERM={it_count}, CLERICAL_TERM={clerical_count} "
                f"(ratio {ratio}:1)"
            )
        else:
            logger.info(
                f"  Class balance acceptable: IT_TERM={it_count}, CLERICAL_TERM={clerical_count} "
                f"(ratio {ratio}:1)"
            )

    return {
        "label": dataset_label,
        "total_records": total,
        "positive_records": positives,
        "negative_records": negatives,
        "negative_ratio_pct": neg_ratio,
        "entity_label_counts": label_counts,
    }


def split_and_convert_dataset(
    records: List[Dict[str, Any]],
    output_dir: str = "data/training",
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, str]:
    """Deduplicates records, splits into train/dev/test, and exports spaCy binary files.
    
    Guarantees zero sentence-level data leakage across splits by grouping
    multi-sentence records that share sentences into the same split.
    Reports per-split diagnostics (negative ratio, class balance) at split time.
    """
    deduped_records, stats = deduplicate_records(records)

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    # Map each sentence to the record indices that contain it
    sent_to_idxs: Dict[str, List[int]] = {}
    for idx, r in enumerate(deduped_records):
        doc = nlp(r.get("text", ""))
        for s in doc.sents:
            st = s.text.strip()
            if st:
                sent_to_idxs.setdefault(st, []).append(idx)

    # Build connected components of records that share sentences
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(deduped_records))}
    for st, idxs in sent_to_idxs.items():
        if len(idxs) > 1:
            for i in idxs:
                for j in idxs:
                    if i != j:
                        adj[i].add(j)

    visited: Set[int] = set()
    components: List[List[int]] = []
    for i in range(len(deduped_records)):
        if i not in visited:
            comp: List[int] = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    random.seed(seed)
    random.shuffle(components)

    train_recs: List[Dict[str, Any]] = []
    dev_recs: List[Dict[str, Any]] = []
    test_recs: List[Dict[str, Any]] = []

    target_train = int(len(deduped_records) * train_ratio)
    target_dev = int(len(deduped_records) * dev_ratio)

    for comp in components:
        comp_recs = [deduped_records[i] for i in comp]
        if len(train_recs) + len(comp_recs) <= target_train or (len(dev_recs) >= target_dev and len(train_recs) < target_train):
            train_recs.extend(comp_recs)
        elif len(dev_recs) + len(comp_recs) <= target_dev:
            dev_recs.extend(comp_recs)
        else:
            test_recs.extend(comp_recs)

    paths = {
        "train": os.path.join(output_dir, "train.spacy"),
        "dev": os.path.join(output_dir, "dev.spacy"),
        "test": os.path.join(output_dir, "test.spacy"),
    }

    logger.info(
        f"Dataset split ({len(deduped_records)} unique records, {len(components)} sentence components): "
        f"Train={len(train_recs)}, Dev={len(dev_recs)}, Test={len(test_recs)}"
    )

    # Report per-split diagnostics
    for split_name, split_recs in [("Train", train_recs), ("Dev", dev_recs), ("Test", test_recs)]:
        report_dataset_diagnostics(split_recs, dataset_label=split_name)

    convert_records_to_docbin(train_recs, paths["train"], nlp=nlp)
    convert_records_to_docbin(dev_recs, paths["dev"], nlp=nlp)
    convert_records_to_docbin(test_recs, paths["test"], nlp=nlp)

    return paths


def prepare_real_data_pipeline(
    data_jsonl_path: str = "data/data.jsonl",
    output_dir: str = "data/training",
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Loads real annotated data, reports diagnostics, deduplicates, and splits into train/dev/test.
    
    This is the primary entry point for the real-data training pipeline.
    Does not fabricate data — diagnostics are reported for the user to assess
    whether the dataset needs more examples of any category before training.
    
    Args:
        data_jsonl_path: Path to the annotated data JSONL file.
        output_dir: Directory for train.spacy / dev.spacy / test.spacy output.
        train_ratio: Proportion of data for training.
        dev_ratio: Proportion of data for development/validation.
        seed: Random seed for reproducible splitting.
    
    Returns:
        Dict containing diagnostics, split paths, and data source path.
    """
    if not os.path.exists(data_jsonl_path):
        raise FileNotFoundError(
            f"Real data file not found at '{data_jsonl_path}'. "
            "Please supply the cleaned annotated data.jsonl before running the pipeline."
        )

    records = load_jsonl(data_jsonl_path)
    logger.info(f"Loaded {len(records)} records from {data_jsonl_path}")

    diagnostics = report_dataset_diagnostics(records, dataset_label="Full Dataset")

    spacy_paths = split_and_convert_dataset(
        records,
        output_dir=output_dir,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
    )

    return {
        "diagnostics": diagnostics,
        "spacy_paths": spacy_paths,
        "data_jsonl_path": data_jsonl_path,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = prepare_real_data_pipeline()
    print("Real-data pipeline preparation complete:")
    import pprint
    pprint.pprint(res)
