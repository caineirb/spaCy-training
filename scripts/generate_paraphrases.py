"""
Standalone Paraphrase Generation Script for OJT NER Data Augmentation.

Generates T5 seq2seq paraphrases from the real training split, with full source
provenance tracking (source_text, source_id) to enable leak-free cross-validation
filtering.

Usage:
    python scripts/generate_paraphrases.py [--seed 42] [--output data/synthetic_paraphrases.jsonl]

Extracted from main.ipynb cells 16-19 to prevent notebook-only generation drift.
"""

import os
import sys
import json
import random
import hashlib
import argparse
from typing import List, Dict, Any, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
scripts.init_gpu()

import torch
import spacy
from spacy.tokens import DocBin
from transformers import T5ForConditionalGeneration, T5Tokenizer


# ---------------------------------------------------------------------------
# Constants: Generic noun blocklists (canonical, shared with generate_llm_synthetic.py)
# ---------------------------------------------------------------------------

GENERIC_NOUN_BLOCKLIST = {
    "program", "encode", "encoded", "office documents",
    "coordination", "orientation", "deployment", "front page",
    "copies", "issued", "stalls", "proctoring",
    "reference numbers", "field trials", "data quality",
    "system workflows", "data requirements", "system exploration",
    # Rejected generic administrative artifacts & paperwork (Group 3 policy decision)
    "visitor ids", "visitor id", "graduation credentials", "graduation credential",
    "faculty academic records", "faculty academic record", "office supply lists", "office supply list",
    "court summons", "employee leave records", "employee leave record", "leave records",
    "fiscal department line items", "fiscal department line item", "payroll discrepancies", "payroll discrepancy",
    "deed logs", "deed log", "public oaths", "public oath", "financial aid allowances", "financial aid allowance",
    "employee tax forms", "employee tax form", "tax forms", "warehouse receipt ledgers", "warehouse receipt ledger",
    # Rejected non-task environmental & conversational artifacts (Group 4 policy decision)
    "blockers", "blocker", "attendance machine", "intern time log sheet", "time log sheet",
    "visitor logbook", "guest identification badges", "guest identification badge",
    "notebook notes", "notebook note", "projector monitors", "projector monitor",
}

BARE_GENERICS = {
    "system", "systems", "database", "databases", "software", "softwares",
    "code", "codes", "computer", "computers", "website", "websites",
    "application", "applications", "server", "servers", "backend", "backends",
    "frontend", "frontends", "data", "table", "tables", "document", "documents",
    "file", "files", "record", "records", "form", "forms", "paper", "papers",
    "program", "programs", "network", "networks", "printer", "printers",
    "scanner", "scanners", "monitor", "monitors", "keyboard", "keyboards",
    "desk", "desks", "office", "offices", "university", "universities",
    "folder", "folders", "department", "departments",
}


# ---------------------------------------------------------------------------
# Utility: source_id computation
# ---------------------------------------------------------------------------

def compute_source_id(text: str) -> str:
    """Compute a stable source_id from normalized text (lowercase, whitespace-collapsed).

    Returns the first 16 hex characters of a SHA-256 hash, providing a 64-bit
    collision-resistant identifier suitable for cross-referencing paraphrases
    back to their source records.
    """
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def generate_paraphrases(
    text: str,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    device: torch.device,
    num_return_sequences: int = 5,
    max_length: int = 128,
) -> List[str]:
    """Generate paraphrase candidates using the T5 model."""
    input_text = f"paraphrase: {text} </s>"
    encoding = tokenizer.encode_plus(
        input_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=10,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

    candidates = []
    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        decoded = decoded.strip()
        if decoded and decoded.lower() != text.lower():
            candidates.append(decoded)
    return candidates


def relocate_entities(
    original_text: str,
    paraphrase_text: str,
    original_entities: List[Dict[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Relocate entity spans from original to paraphrase via exact substring match.

    Returns (success, relocated_entities) where success=False if any entity
    text cannot be found in the paraphrase with valid word boundaries.
    """
    relocated = []
    used_ranges = []

    for ent in original_entities:
        entity_text = original_text[ent["start"]:ent["end"]]
        label = ent["label"]

        if not entity_text.strip():
            return False, []

        # Search for exact substring match with word-boundary validation
        search_start = 0
        found = False
        while search_start < len(paraphrase_text):
            # Case-sensitive first, then case-insensitive fallback
            pos = paraphrase_text.find(entity_text, search_start)
            if pos == -1:
                # Try case-insensitive
                pos = paraphrase_text.lower().find(entity_text.lower(), search_start)
                if pos == -1:
                    break

            start_idx = pos
            end_idx = pos + len(entity_text)

            # Word-boundary check
            before = paraphrase_text[start_idx - 1] if start_idx > 0 else " "
            after = paraphrase_text[end_idx] if end_idx < len(paraphrase_text) else " "

            if before.isalnum() or after.isalnum():
                search_start = pos + 1
                continue

            # Check overlap
            overlap = any(max(start_idx, us) < min(end_idx, ue) for us, ue in used_ranges)
            if overlap:
                search_start = pos + 1
                continue

            # Use the actual text from the paraphrase (preserving case)
            relocated.append({"start": start_idx, "end": end_idx, "label": label})
            used_ranges.append((start_idx, end_idx))
            found = True
            break

        if not found:
            return False, []

    # Sort by start offset and verify no overlaps
    relocated.sort(key=lambda e: e["start"])
    for i in range(len(relocated) - 1):
        if relocated[i]["end"] > relocated[i + 1]["start"]:
            return False, []

    return True, relocated


def passes_paraphrase_quality_check(
    text: str,
    entities: List[Dict[str, Any]],
    seen_texts: Set[str],
    eval_sentences: Set[str],
) -> Tuple[bool, str]:
    """Quality check for a paraphrase candidate."""
    if not text or len(text.strip()) < 10 or len(text) > 500:
        return False, "Length"

    text_lower = text.strip().lower()
    if text_lower in seen_texts:
        return False, "Duplicate"
    if text_lower in eval_sentences:
        return False, "Eval leak"

    for ent in entities:
        term = text[ent["start"]:ent["end"]].strip().lower()
        if term in GENERIC_NOUN_BLOCKLIST or term in BARE_GENERICS:
            return False, f"Blocked term: {term}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def run_paraphrase_generation(
    train_spacy_path: str = "data/training/train.spacy",
    output_path: str = "data/synthetic_paraphrases.jsonl",
    seed: int = 42,
    num_candidates: int = 5,
    max_length: int = 128,
) -> List[Dict[str, Any]]:
    """Generate paraphrases for all records in the training split.

    Each paraphrase record includes source provenance:
      - source_text: the original text that was paraphrased
      - source_id: SHA-256 hash of the normalized source text

    Returns the list of accepted paraphrase records.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load training records from DocBin
    print(f"Loading training records from {train_spacy_path}...")
    nlp_blank = spacy.blank("en")
    train_db = DocBin().from_disk(train_spacy_path)
    train_docs = list(train_db.get_docs(nlp_blank.vocab))

    train_records = []
    for doc in train_docs:
        train_records.append({
            "text": doc.text,
            "entities": [
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
                for ent in doc.ents
            ],
        })

    print(f"Loaded {len(train_records)} real training records for paraphrasing.")
    print(f"  Positive: {sum(1 for r in train_records if r['entities'])}")
    print(f"  Negative: {sum(1 for r in train_records if not r['entities'])}")

    # Build evaluation sentence set for dedup guard
    eval_sentences: Set[str] = set()
    for split in ["dev", "test"]:
        path = f"data/training/{split}.spacy"
        if os.path.exists(path):
            db = DocBin().from_disk(path)
            for doc in db.get_docs(nlp_blank.vocab):
                eval_sentences.add(doc.text.strip().lower())

    for eval_path in ["data/test/unseen_benchmark.jsonl", "data/test/holdout.jsonl"]:
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            eval_sentences.add(rec["text"].strip().lower())
                        except Exception:
                            pass

    # Include all real training sentences too
    real_sentences = {r["text"].strip().lower() for r in train_records}
    eval_sentences.update(real_sentences)
    print(f"Dedup guard: {len(eval_sentences)} protected sentences.")

    # Load T5 paraphraser
    PARAPHRASE_MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
    print(f"\nLoading paraphraser: {PARAPHRASE_MODEL_NAME}...")
    para_tokenizer = T5Tokenizer.from_pretrained(PARAPHRASE_MODEL_NAME)
    para_model = T5ForConditionalGeneration.from_pretrained(PARAPHRASE_MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    para_model = para_model.to(device)
    para_model.eval()
    print(f"Paraphraser loaded on {device}.")

    # Generate paraphrases
    paraphrase_records: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()
    total_attempted = 0
    total_span_failures = 0
    total_quality_failures = 0

    print(f"\nGenerating paraphrases for {len(train_records)} training records...")
    print(f"Target: ~1 accepted paraphrase per record (best-of-{num_candidates} candidates)")
    print()

    for idx, rec in enumerate(train_records):
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(train_records)} records processed, "
                  f"{len(paraphrase_records)} paraphrases accepted")

        text = rec["text"]
        entities = rec["entities"]

        # Generate candidates
        try:
            candidates = generate_paraphrases(
                text, para_model, para_tokenizer, device,
                num_return_sequences=num_candidates, max_length=max_length,
            )
        except Exception:
            continue

        total_attempted += len(candidates)

        # Try to accept the best valid paraphrase
        for cand in candidates:
            if entities:
                # Positive record: relocate entities
                success, relocated = relocate_entities(text, cand, entities)
                if not success:
                    total_span_failures += 1
                    continue
            else:
                # Negative record: just accept paraphrase with empty entities
                relocated = []

            # Quality check
            passes, reason = passes_paraphrase_quality_check(
                cand, relocated, seen_texts, eval_sentences
            )
            if not passes:
                total_quality_failures += 1
                continue

            # Accept this paraphrase WITH SOURCE PROVENANCE
            para_rec = {
                "text": cand,
                "entities": relocated,
                "augmentation_type": "paraphrase",
                "source_text": text,
                "source_id": compute_source_id(text),
            }
            paraphrase_records.append(para_rec)
            seen_texts.add(cand.strip().lower())
            break  # Only accept 1 paraphrase per original record

    # Unload T5 model
    del para_model, para_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Report statistics
    pos_para = sum(1 for r in paraphrase_records if r["entities"])
    neg_para = sum(1 for r in paraphrase_records if not r["entities"])

    print(f"\n{'=' * 60}")
    print(f"PARAPHRASE GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total accepted paraphrases : {len(paraphrase_records)}")
    print(f"  Positive paraphrases     : {pos_para}")
    print(f"  Negative paraphrases     : {neg_para} ({neg_para / max(len(paraphrase_records), 1) * 100:.1f}%)")
    print(f"  Span relocation failures : {total_span_failures}")
    print(f"  Quality check failures   : {total_quality_failures}")
    print(f"  Total candidates tried   : {total_attempted}")

    # Save paraphrases
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in paraphrase_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(paraphrase_records)} paraphrases to {output_path}")

    # Verify source provenance
    source_ids = {r["source_id"] for r in paraphrase_records}
    print(f"Unique source IDs: {len(source_ids)} (from {len(train_records)} source records)")

    return paraphrase_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate T5 paraphrases with source provenance for OJT NER"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output", type=str, default="data/synthetic_paraphrases.jsonl",
        help="Output path for paraphrase JSONL"
    )
    parser.add_argument(
        "--train-path", type=str, default="data/training/train.spacy",
        help="Path to the training DocBin"
    )
    args = parser.parse_args()

    run_paraphrase_generation(
        train_spacy_path=args.train_path,
        output_path=args.output,
        seed=args.seed,
    )
