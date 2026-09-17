#!/usr/bin/env python3
"""
Production Deployment Script for OJT Journal Task Tagging.

Features:
- Streaming line-by-line processing with constant memory overhead (no file size limit)
- Hybrid extraction: Deterministic EntityRuler + Fine-tuned Transformer NER
- Outputs identified terms, count per term, classification, source, and confidence
- Supports CSV, JSON, and rich tabular terminal output
- Optional streaming detailed line-by-line output
"""

import os
import sys
import json
import time
import argparse
import logging
from collections import defaultdict
from typing import Generator, List, Tuple, Dict, Any, Optional
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
from scripts.pipeline import HybridJournalPipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ojt_deploy")


def stream_file_batches(
    filepath: str,
    batch_size: int = 64
) -> Generator[List[Tuple[int, str]], None, None]:
    """Streams lines from a file in fixed-size batches without loading the whole file into RAM.
    
    Yields:
        List of (line_number, line_text)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    batch = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            # Ignore purely empty lines and header comments starting with #
            if line and not line.startswith("#"):
                batch.append((line_no, line))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

    if batch:
        yield batch


def process_text_file(
    input_filepath: str,
    pipeline: HybridJournalPipeline,
    batch_size: int = 64,
    confidence_threshold: float = 0.80,
    detailed_output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Processes an arbitrary-sized text file using streaming batch inference.
    
    Args:
        input_filepath: Path to the input .txt file.
        pipeline: HybridJournalPipeline instance.
        batch_size: Streaming batch size.
        confidence_threshold: Threshold to accept ML extractions.
        detailed_output_path: Optional file to append line-by-line JSONL annotations.
        
    Returns:
        pd.DataFrame containing aggregated term statistics.
    """
    scripts.init_gpu()

    # Streaming accumulators (constant memory)
    term_counts: Dict[str, int] = defaultdict(int)
    term_labels: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    term_sources: Dict[str, str] = {}
    term_confs: Dict[str, List[float]] = defaultdict(list)
    term_sample_contexts: Dict[str, str] = {}

    detailed_file = None
    if detailed_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(detailed_output_path)), exist_ok=True)
        detailed_file = open(detailed_output_path, "w", encoding="utf-8")

    total_lines = 0
    total_entities_found = 0
    start_time = time.time()

    logger.info(f"Beginning streaming inference on '{input_filepath}' (batch_size={batch_size})...")

    try:
        for batch in stream_file_batches(input_filepath, batch_size=batch_size):
            line_numbers = [item[0] for item in batch]
            texts = [item[1] for item in batch]
            total_lines += len(texts)

            # High-throughput batch inference
            docs = list(pipeline.nlp.pipe(texts, batch_size=len(texts)))

            for line_no, text, doc in zip(line_numbers, texts, docs):
                line_entities = []

                for ent in doc.ents:
                    term = ent.text.strip()
                    if not term:
                        continue

                    label = ent.label_
                    is_dict = term.lower() in pipeline._lower_terms_set

                    if is_dict:
                        source = "dictionary"
                        conf = 1.00
                        status = "ACCEPTED"
                    else:
                        source = "ML"
                        conf = pipeline._calculate_ml_confidence(doc, ent)
                        status = "ACCEPTED" if conf >= confidence_threshold else "NEEDS_REVIEW"

                    # Update aggregated term metrics
                    term_counts[term] += 1
                    term_labels[term][label] += 1
                    term_confs[term].append(conf)
                    if term not in term_sources or source == "dictionary":
                        term_sources[term] = source
                    if term not in term_sample_contexts:
                        term_sample_contexts[term] = text

                    total_entities_found += 1
                    line_entities.append({
                        "line_number": line_no,
                        "term": term,
                        "classification": label,
                        "source": source,
                        "confidence": round(conf, 4),
                        "status": status,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    })

                # Stream detailed log immediately to disk
                if detailed_file and line_entities:
                    for item in line_entities:
                        item["context"] = text
                        detailed_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    detailed_file.flush()

    finally:
        if detailed_file:
            detailed_file.close()

    elapsed = time.time() - start_time
    throughput = (total_lines / elapsed) if elapsed > 0 else 0.0

    # Build final aggregated DataFrame
    rows = []
    for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
        # Determine dominant classification
        label_counter = term_labels[term]
        dominant_label = max(label_counter.items(), key=lambda x: x[1])[0]
        confs = term_confs[term]
        mean_conf = round(sum(confs) / len(confs), 4) if confs else 1.00

        rows.append({
            "term": term,
            "count": count,
            "classification": dominant_label,
            "source": term_sources.get(term, "ML"),
            "confidence": mean_conf,
            "sample_context": term_sample_contexts.get(term, ""),
        })

    df = pd.DataFrame(rows)

    logger.info(
        f"Processing complete: {total_lines} lines in {elapsed:.2f}s "
        f"({throughput:.1f} lines/sec). "
        f"Identified {total_entities_found} total entity occurrences "
        f"({len(df)} unique terms)."
    )
    return df


def export_results(
    df: pd.DataFrame,
    output_path: str,
    output_format: str = "csv"
) -> None:
    """Exports aggregated results to CSV, JSON, or text table."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if output_format == "csv" or output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    elif output_format == "json" or output_path.endswith(".json"):
        records = df.to_dict(orient="records")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    elif output_format == "jsonl" or output_path.endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))

    logger.info(f"Exported aggregated terms to '{output_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming OJT Journal Entity Tagger (Deployment Pipeline)"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input text file (no size limit, processed via streaming)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to save aggregated results (e.g. output.csv or output.json)"
    )
    parser.add_argument(
        "-d", "--detailed-output",
        default=None,
        help="Optional path to save streaming line-by-line JSONL extraction logs"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["csv", "json", "jsonl", "text"],
        default="csv",
        help="Output serialization format (default: csv)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=64,
        help="Streaming batch size for GPU inference (default: 64)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.80,
        help="Confidence threshold for ML acceptance (default: 0.80)"
    )
    parser.add_argument(
        "-m", "--model-path",
        default="models/hybrid_pipeline",
        help="Path to hybrid pipeline model directory"
    )
    parser.add_argument(
        "--terms-path",
        default="data/terms.csv",
        help="Path to terms dictionary CSV"
    )

    args = parser.parse_args()

    # Determine default output file if none provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"data/{base_name}_tagged.{args.format}"

    # Load hybrid pipeline
    pipeline = HybridJournalPipeline(
        model_path=args.model_path,
        terms_csv_path=args.terms_path,
        confidence_threshold=args.threshold
    )

    # Execute streaming processing
    df_results = process_text_file(
        input_filepath=args.input,
        pipeline=pipeline,
        batch_size=args.batch_size,
        confidence_threshold=args.threshold,
        detailed_output_path=args.detailed_output
    )

    # Save output
    export_results(df_results, args.output, output_format=args.format)

    # Print clean formatted summary to console
    print("\n" + "=" * 80)
    print(f"DEPLOYMENT EXTRACTION SUMMARY: {args.input}")
    print("=" * 80)
    if len(df_results) == 0:
        print("No task entities identified in the provided file.")
    else:
        # Display top identified terms
        preview_df = df_results[["term", "count", "classification", "source", "confidence"]]
        print(preview_df.to_string(index=False))
    print("=" * 80)
    print(f"Results saved to: {args.output}")
    if args.detailed_output:
        print(f"Detailed line-by-line logs saved to: {args.detailed_output}")


if __name__ == "__main__":
    main()
