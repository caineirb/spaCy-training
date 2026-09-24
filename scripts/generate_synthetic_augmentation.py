"""
Synthetic Data Augmentation Pipeline for OJT Journal NER.

Generates:
1. Paraphrase-Based Augmentation: Syntactic and grammatical variants of real training
   sentences using T5-base with strict entity span preservation and task-stability guardrails.
2. Template-Based Supplement: Syntactic frame variations targeting under-represented
   terms from data/terms.csv and observed real vocabulary.

Guarantees:
- Strict zero leakage: No synthetic sentence is an exact or near-duplicate (similarity >= 0.70)
  of any sentence in dev.spacy, test.spacy, unseen_benchmark.jsonl, or holdout.jsonl.
- Exact literal entity surface preservation.
- Output isolation: Paraphrase set, template set, and combined TRSTR training pool
  are stored separately from the pure-real data.
"""

import os
import sys
import json
import re
import random
import difflib
from typing import List, Dict, Any, Tuple, Set
import spacy
from spacy.tokens import DocBin, Doc
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.annotation import load_terms_dictionary, report_dataset_diagnostics
from scripts.eval import load_unseen_benchmark

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Generic nouns that exist in data.jsonl but should NOT propagate into synthetic data.
# These are borderline annotations (activities/concepts rather than specific tools/tasks)
# that we keep in real training data but filter out during augmentation to avoid
# amplifying annotation noise.
GENERIC_NOUN_BLOCKLIST = {
    "coding", "system design", "system development", "encode",
    "accounts", "formatting", "notices", "requirements", "network",
    "copies", "coordination", "policies", "orientation", "survey",
    "stalls", "proctoring", "deployment", "front page", "drivers",
    "office systems", "data organization", "field trials",
    "reference numbers", "data quality", "system workflows",
    "data requirements", "system exploration",
    # Added after Task 3 quality review:
    "office documents", "program", "issued", "encoded", "technical",
    "layouts", "debugging",
}

# Entities that should be relabeled (not removed) during synthetic generation.
# Key: lowercased entity surface text. Value: corrected label.
ENTITY_RELABEL_MAP = {
    "printing": "CLERICAL_TERM",
}


def load_eval_sentences_and_terms() -> Tuple[Set[str], Set[str]]:
    """Loads all sentences from evaluation sets and all unseen benchmark terms."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    eval_sentences = set()

    for split in ["dev", "test"]:
        path = f"data/training/{split}.spacy"
        if os.path.exists(path):
            db = DocBin().from_disk(path)
            for doc in db.get_docs(nlp.vocab):
                sent_doc = nlp(doc.text)
                for s in sent_doc.sents:
                    clean = s.text.strip().lower()
                    if clean:
                        eval_sentences.add(clean)

    # Benchmark samples
    bench_samples = load_unseen_benchmark("data/test/unseen_benchmark.jsonl")
    bench_terms = set()
    for b in bench_samples:
        clean = b["text"].strip().lower()
        if clean:
            eval_sentences.add(clean)
        for e in b.get("entities", []):
            term = b["text"][e["start"]:e["end"]].strip().lower()
            if term:
                bench_terms.add(term)

    # Holdout
    holdout_path = "data/test/holdout.jsonl"
    if os.path.exists(holdout_path):
        with open(holdout_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    clean = rec["text"].strip().lower()
                    if clean:
                        eval_sentences.add(clean)

    print(f"Loaded {len(eval_sentences)} evaluation sentences to guard against leakage.")
    print(f"Loaded {len(bench_terms)} unseen benchmark terms to guard against contamination.")
    return eval_sentences, bench_terms


def is_near_duplicate(text: str, eval_sentences: Set[str], threshold: float = 0.70) -> bool:
    """Checks if text or any constituent sentence has >= threshold similarity to any evaluation sentence."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    candidate_sents = [s.text.strip().lower() for s in doc.sents if s.text.strip()]
    if not candidate_sents:
        candidate_sents = [text.strip().lower()]

    for s_clean in candidate_sents:
        if s_clean in eval_sentences:
            return True

        s_words = set(re.findall(r"\w+", s_clean))
        if not s_words:
            continue

        for ev in eval_sentences:
            ev_words = set(re.findall(r"\w+", ev))
            if not ev_words:
                continue
            jaccard = len(s_words & ev_words) / len(s_words | ev_words)
            if jaccard >= 0.35:
                sim = difflib.SequenceMatcher(None, s_clean, ev).ratio()
                if sim >= threshold:
                    return True
    return False


def relocate_entities(orig_text: str, para_text: str, orig_entities: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Verifies that all entity surface texts survive intact and non-overlapping."""
    new_entities = []
    sorted_ents = sorted(orig_entities, key=lambda x: x["start"])

    for ent in sorted_ents:
        surface = orig_text[ent["start"]:ent["end"]]
        pos = para_text.find(surface)
        if pos == -1:
            return False, []

        start_idx = pos
        end_idx = pos + len(surface)

        # Boundary check
        before = para_text[start_idx-1] if start_idx > 0 else " "
        after = para_text[end_idx] if end_idx < len(para_text) else " "
        if before.isalnum() or after.isalnum():
            return False, []

        new_entities.append({
            "start": start_idx,
            "end": end_idx,
            "label": ent["label"],
            "text": surface
        })

    # Sort and check overlaps
    new_entities.sort(key=lambda x: x["start"])
    for i in range(len(new_entities) - 1):
        if new_entities[i]["end"] > new_entities[i+1]["start"]:
            return False, []

    return True, new_entities


def check_task_stability(orig_text: str, para_text: str, labels: List[str]) -> bool:
    """Checks for cross-domain contamination and length divergence."""
    orig_lower = orig_text.lower()
    para_lower = para_text.lower()

    len_ratio = len(para_text) / max(len(orig_text), 1)
    if len_ratio < 0.5 or len_ratio > 2.0:
        return False

    it_keywords = ["coding", "scripting", "python", "javascript", "sql query", "programming", "compiled", "vba macro", "debugging", "git commit"]
    clerical_keywords = ["photocopying", "paper filing", "filing documents", "hard copy", "shredding", "stapling"]

    has_it = "IT_TERM" in labels
    has_clerical = "CLERICAL_TERM" in labels

    if has_clerical and not has_it:
        for kw in it_keywords:
            if kw in para_lower and kw not in orig_lower:
                return False

    if has_it and not has_clerical:
        for kw in clerical_keywords:
            if kw in para_lower and kw not in orig_lower:
                return False

    return True


def extract_real_training_records() -> List[Dict[str, Any]]:
    """Loads training split docs from data/training/train.spacy into JSON-serializable records."""
    nlp = spacy.blank("en")
    db = DocBin().from_disk("data/training/train.spacy")
    records = []
    for doc in db.get_docs(nlp.vocab):
        ents = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
        records.append({"text": doc.text, "entities": ents})
    return records


def generate_paraphrase_pool(
    train_records: List[Dict[str, Any]],
    eval_sentences: Set[str],
    bench_terms: Set[str],
    max_variants_per_record: int = 2
) -> List[Dict[str, Any]]:
    """Generates syntactic variants of real training records using T5."""
    print("Loading T5 paraphrasing model...")
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    paraphrase_records = []
    seen_texts = set(r["text"].strip().lower() for r in train_records)

    pos_records = [r for r in train_records if len(r.get("entities", [])) > 0]
    neg_records = [r for r in train_records if len(r.get("entities", [])) == 0]

    print(f"Generating paraphrases from {len(pos_records)} positive and {len(neg_records)} negative training records...")

    # Process positive records
    for idx, r in enumerate(pos_records):
        orig_text = r["text"].strip()
        # Filter out generic-noun entities and apply relabeling before paraphrasing
        entities = []
        for e in r["entities"]:
            surface = orig_text[e["start"]:e["end"]].strip().lower()
            if surface in GENERIC_NOUN_BLOCKLIST:
                continue
            if surface in ENTITY_RELABEL_MAP:
                e = {**e, "label": ENTITY_RELABEL_MAP[surface]}
            entities.append(e)
        if not entities:
            # All entities were generic — skip this record entirely
            continue
        labels = [e["label"] for e in entities]
        ent_texts = [orig_text[e["start"]:e["end"]].lower() for e in entities]

        # Guard: check if orig text had benchmark term (should be 0)
        if any(bt in orig_text.lower() for bt in bench_terms):
            continue

        prompt = f"paraphrase: {orig_text}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=96,
                num_beams=8,
                num_return_sequences=5,
                no_repeat_ngram_size=2,
                temperature=1.0
            )

        added_for_this = 0
        for out in outputs:
            if added_for_this >= max_variants_per_record:
                break
            cand = tokenizer.decode(out, skip_special_tokens=True).strip()
            cand_lower = cand.lower()

            # Rule: non-trivial variation
            if cand_lower == orig_text.lower() or cand_lower in seen_texts:
                continue

            # Rule: zero unseen benchmark terms
            if any(bt in cand_lower for bt in bench_terms):
                continue

            # Rule: zero near-duplicate to evaluation sets
            if is_near_duplicate(cand, eval_sentences):
                continue

            # Rule: exact entity preservation
            ok_ent, new_ents = relocate_entities(orig_text, cand, entities)
            if not ok_ent:
                continue

            # Rule: task stability
            if not check_task_stability(orig_text, cand, labels):
                continue

            clean_ents = [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in new_ents]
            paraphrase_records.append({
                "text": cand,
                "entities": clean_ents,
                "augmentation_type": "paraphrase",
                "source_text": orig_text
            })
            seen_texts.add(cand_lower)
            added_for_this += 1

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(pos_records)} positive records -> {len(paraphrase_records)} valid paraphrases")

    # Process negative records (target ~25-35% negative ratio in the synthetic pool)
    target_negs = int(len(paraphrase_records) * 0.35)
    added_negs = 0
    print(f"Generating ~{target_negs} negative paraphrases...")

    for r in neg_records:
        if added_negs >= target_negs:
            break
        orig_text = r["text"].strip()
        prompt = f"paraphrase: {orig_text}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=96,
                num_beams=6,
                num_return_sequences=3,
                no_repeat_ngram_size=2
            )

        for out in outputs:
            cand = tokenizer.decode(out, skip_special_tokens=True).strip()
            cand_lower = cand.lower()

            if cand_lower == orig_text.lower() or cand_lower in seen_texts:
                continue
            if any(bt in cand_lower for bt in bench_terms):
                continue
            if is_near_duplicate(cand, eval_sentences):
                continue

            paraphrase_records.append({
                "text": cand,
                "entities": [],
                "augmentation_type": "paraphrase",
                "source_text": orig_text
            })
            seen_texts.add(cand_lower)
            added_negs += 1
            break

    print(f"Total paraphrase pool: {len(paraphrase_records)} records ({added_negs} negatives).")
    return paraphrase_records


def generate_template_pool(
    terms_dict: Dict[str, str],
    eval_sentences: Set[str],
    bench_terms: Set[str],
    count: int = 180
) -> List[Dict[str, Any]]:
    """Generates synthetic sentences using syntactic template frames with concrete vocabulary terms.
    
    Frames follow diverse syntactic patterns:
    - Active past: 'I configured and tested [TERM] for office operations.'
    - Passive: '[TERM] was deployed across the local network.'
    - Gerund fronted: 'Assisting in [TERM], our team completed the task.'
    - Inverted/Prepositional: 'For daily workflows, we relied on [TERM].'
    """
    it_terms = [t for t, l in terms_dict.items() if l == "IT_TERM" and t.lower() not in bench_terms and t.lower() not in GENERIC_NOUN_BLOCKLIST]
    clerical_terms = [t for t, l in terms_dict.items() if l == "CLERICAL_TERM" and t.lower() not in bench_terms and t.lower() not in GENERIC_NOUN_BLOCKLIST]

    # Syntactic frames: {slot} will be replaced with term
    it_frames = [
        ("I installed and configured {term} on the office workstation.", "IT_TERM"),
        ("Resolved network issues by running diagnostics using {term}.", "IT_TERM"),
        ("During the internship, our team performed maintenance on {term}.", "IT_TERM"),
        ("Assisted the IT supervisor in managing {term} for company systems.", "IT_TERM"),
        ("{term} was updated to the latest stable release to ensure security.", "IT_TERM"),
        ("Conducted hardware troubleshooting and routine checkups on {term}.", "IT_TERM"),
        ("Implemented automated scripts to monitor {term} performance.", "IT_TERM"),
        ("For the daily infrastructure tasks, we utilized {term} directly.", "IT_TERM"),
        ("Configured access permissions and user accounts within {term}.", "IT_TERM"),
        ("Verified database connectivity and backup integrity for {term}.", "IT_TERM"),
    ]

    clerical_frames = [
        ("I encoded and organized {term} for the department supervisor.", "CLERICAL_TERM"),
        ("Arranged, filed, and audited {term} in accordance with office policy.", "CLERICAL_TERM"),
        ("During the morning shift, we processed and stamped {term}.", "CLERICAL_TERM"),
        ("Assisted in verifying client information recorded on {term}.", "CLERICAL_TERM"),
        ("{term} was systematically categorized and filed into official binders.", "CLERICAL_TERM"),
        ("Prepared daily transaction summaries and cross-checked {term}.", "CLERICAL_TERM"),
        ("Reviewed official documentation to ensure {term} had complete signatures.", "CLERICAL_TERM"),
        ("For administrative records, we digitized physical copies of {term}.", "CLERICAL_TERM"),
        ("Assisted walk-in clients with submitting and releasing their {term}.", "CLERICAL_TERM"),
        ("Logged outgoing correspondence and verified tracking numbers on {term}.", "CLERICAL_TERM"),
    ]

    # Negative frames (clerical/IT activities with NO domain terms)
    negative_templates = [
        "Attended the morning flag ceremony and department briefing.",
        "Assisted in rearranging office chairs and tables for the afternoon conference.",
        "Escorted visitors to the appropriate administrative office desk.",
        "Participated in the general office cleaning and 5S organization activity.",
        "Helped pack promotional materials and supplies for the community outreach program.",
        "Observed daily morning workflow and noted supervisory feedback.",
        "Organized office desk supplies and stationery materials in the supply cabinet.",
        "Guided guests to the registration booth during the anniversary celebration."
    ]

    records = []
    seen = set()

    # Generate IT templates (~40%)
    target_it = int(count * 0.35)
    it_added = 0
    while it_added < target_it and it_terms:
        term = random.choice(it_terms)
        frame, label = random.choice(it_frames)
        sentence = frame.format(term=term)
        s_clean = sentence.strip().lower()
        if s_clean in seen or is_near_duplicate(sentence, eval_sentences) or any(bt in s_clean for bt in bench_terms):
            continue
        start = sentence.find(term)
        end = start + len(term)
        records.append({
            "text": sentence,
            "entities": [{"start": start, "end": end, "label": label}],
            "augmentation_type": "template",
            "source_text": None
        })
        seen.add(s_clean)
        it_added += 1

    # Generate Clerical templates (~35%)
    target_cl = int(count * 0.35)
    cl_added = 0
    while cl_added < target_cl and clerical_terms:
        term = random.choice(clerical_terms)
        frame, label = random.choice(clerical_frames)
        sentence = frame.format(term=term)
        s_clean = sentence.strip().lower()
        if s_clean in seen or is_near_duplicate(sentence, eval_sentences) or any(bt in s_clean for bt in bench_terms):
            continue
        start = sentence.find(term)
        end = start + len(term)
        records.append({
            "text": sentence,
            "entities": [{"start": start, "end": end, "label": label}],
            "augmentation_type": "template",
            "source_text": None
        })
        seen.add(s_clean)
        cl_added += 1

    # Generate Negative templates (~30%)
    target_neg = count - it_added - cl_added
    neg_added = 0
    for t in negative_templates * 5:
        if neg_added >= target_neg:
            break
        s_clean = t.strip().lower()
        if s_clean in seen or is_near_duplicate(t, eval_sentences):
            continue
        records.append({
            "text": t,
            "entities": [],
            "augmentation_type": "template",
            "source_text": None
        })
        seen.add(s_clean)
        neg_added += 1

    print(f"Total template pool: {len(records)} records (IT: {it_added}, Clerical: {cl_added}, Neg: {neg_added}).")
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    """Saves records to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def convert_records_to_docbin(records: List[Dict[str, Any]], output_path: str) -> None:
    """Converts annotation records into a spaCy DocBin file."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc_bin = DocBin()

    valid_count = 0
    for r in records:
        text = r["text"]
        doc = nlp.make_doc(text)
        spans = []
        for ent in r.get("entities", []):
            span = doc.char_span(ent["start"], ent["end"], label=ent["label"], alignment_mode="strict")
            if span is not None:
                spans.append(span)
            else:
                # Try contract alignment mode if strict failed on token boundary
                span = doc.char_span(ent["start"], ent["end"], label=ent["label"], alignment_mode="contract")
                if span is not None:
                    spans.append(span)

        doc.ents = spans
        doc_bin.add(doc)
        valid_count += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc_bin.to_disk(output_path)
    print(f"Exported {valid_count} docs to DocBin: {output_path}")


def run_synthetic_pipeline():
    print("=" * 70)
    print("STARTING SYNTHETIC AUGMENTATION PIPELINE (TASK 1 & TASK 3)")
    print("=" * 70)

    # 1. Load evaluation guard sets
    eval_sentences, bench_terms = load_eval_sentences_and_terms()

    # 2. Load real training records (isolated from dev/test)
    real_train_records = extract_real_training_records()
    print(f"Loaded {len(real_train_records)} real training records from train.spacy.")

    # 3. Generate paraphrase pool
    paraphrase_pool = generate_paraphrase_pool(real_train_records, eval_sentences, bench_terms, max_variants_per_record=3)
    save_jsonl(paraphrase_pool, "data/synthetic_paraphrases.jsonl")

    # 4. Generate template pool
    terms_dict = load_terms_dictionary("data/terms.csv")
    template_pool = generate_template_pool(terms_dict, eval_sentences, bench_terms, count=180)
    save_jsonl(template_pool, "data/synthetic_templates.jsonl")

    # 5. Combine into TRSTR training pool
    combined_trstr = []
    # Add real training records
    for r in real_train_records:
        combined_trstr.append({
            "text": r["text"],
            "entities": r["entities"],
            "augmentation_type": "real",
            "source_text": None
        })

    # Add synthetic pools
    combined_trstr.extend(paraphrase_pool)
    combined_trstr.extend(template_pool)

    # Shuffle for training
    random.shuffle(combined_trstr)

    # Save TRSTR pool
    save_jsonl(combined_trstr, "data/training_trstr.jsonl")
    convert_records_to_docbin(combined_trstr, "data/training/train_trstr.spacy")

    # Report diagnostics
    print("\n--- COMPOSITION REPORT ---")
    print(f"Pure Real Training Pool   : {len(real_train_records)} records")
    print(f"Paraphrase Augmentation   : {len(paraphrase_pool)} records")
    print(f"Template Supplement       : {len(template_pool)} records")
    print(f"Total TRSTR Training Pool : {len(combined_trstr)} records (Real: {len(real_train_records)}, Synthetic: {len(paraphrase_pool) + len(template_pool)})")
    
    report_dataset_diagnostics(combined_trstr, "TRSTR Combined Training Pool")
    print("=" * 70)


if __name__ == "__main__":
    run_synthetic_pipeline()
