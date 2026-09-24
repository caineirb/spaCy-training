"""
Standalone Entity-Term Inspector.

Reads a JSONL file containing NER annotation records, extracts the actual
entity surface text from text[start:end] for each entity, and adds/overwrites
a "term" field inside each entity object for human readability.

This is a utility for manual inspection — it is NOT part of the annotation,
training, or evaluation pipeline.

Usage:
    python scripts/inspect_entity_terms.py data/synthetic_paraphrases.jsonl

Output:
    data/review/synthetic_paraphrases_with_terms.jsonl
"""

import sys
import os
import json


def add_terms_to_entities(input_path: str) -> str:
    """Reads a JSONL file, adds 'term' fields to entities, writes to a new file.
    
    Args:
        input_path: Path to the input JSONL file.
        
    Returns:
        Path to the output file.
    """
    review_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "review")
    os.makedirs(review_dir, exist_ok=True)
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(review_dir, f"{name}_with_terms{ext}")

    records_processed = 0
    entities_annotated = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            text = record.get("text", "")

            for ent in record.get("entities", []):
                start = ent.get("start", 0)
                end = ent.get("end", 0)
                ent["term"] = text[start:end]
                entities_annotated += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_processed += 1

    return output_path, records_processed, entities_annotated


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    output_path, records, entities = add_terms_to_entities(input_path)
    print(f"Processed {records} records, annotated {entities} entities.")
    print(f"Output: {output_path}")
