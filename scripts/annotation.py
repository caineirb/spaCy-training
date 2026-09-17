"""
Annotation tooling and dataset preparation module.

Provides:
- Seed dictionary loading & normalization (IT_TASK -> IT_TERM, CLERICAL -> CLERICAL_TERM)
- Weak supervision / string-matching candidate annotation
- Diverse OJT journal sentence generation with realistic contexts & negative examples
- Review export (CSV) and golden annotation storage (JSONL)
- Conversion from JSONL into spaCy binary DocBin (.spacy) format
"""

import os
import re
import json
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import spacy
from spacy.tokens import DocBin, Doc

logger = logging.getLogger("ojt_pipeline.annotation")

LABEL_MAPPING = {
    "IT_TASK": "IT_TERM",
    "CLERICAL": "CLERICAL_TERM",
    "IT_TERM": "IT_TERM",
    "CLERICAL_TERM": "CLERICAL_TERM",
}


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


def export_for_review_csv(records: List[Dict[str, Any]], output_csv_path: str) -> None:
    """Exports weakly-labeled records to a flat tabular CSV format for human inspection."""
    rows = []
    for idx, rec in enumerate(records):
        text = rec["text"]
        entities = rec.get("entities", [])
        if not entities:
            rows.append({
                "record_id": idx,
                "text": text,
                "term": "[NO_ENTITY]",
                "start": -1,
                "end": -1,
                "label": "NONE",
                "status": "AUTO_NEGATIVE"
            })
        else:
            for ent in entities:
                s, e = ent["start"], ent["end"]
                term = text[s:e]
                rows.append({
                    "record_id": idx,
                    "text": text,
                    "term": term,
                    "start": s,
                    "end": e,
                    "label": ent["label"],
                    "status": "AUTO_MATCH"
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Exported {len(df)} review entries to {output_csv_path}")


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


def generate_synthetic_ojt_dataset(
    terms_dict: Dict[str, str],
    target_samples_per_label: int = 400,
    target_negatives: int = 200,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Generates a linguistically diverse OJT weekly journal dataset.
    
    Produces distinct sentence structures (development, debugging, data entry, filing,
    system maintenance, reporting) and pure negative sentences (meetings, break, general talk).
    """
    random.seed(seed)

    it_terms = [t for t, l in terms_dict.items() if l == "IT_TERM"]
    clerical_terms = [t for t, l in terms_dict.items() if l == "CLERICAL_TERM"]

    # Contextual sentence templates with syntactic diversity
    it_templates = [
        "I developed a web application feature using {term}.",
        "Configured the development environment and set up {term}.",
        "Worked on backend API services and integrated {term} for persistent storage.",
        "Refactored legacy code modules and migrated the system to {term}.",
        "Resolved an unexpected crash in production by debugging our {term} service.",
        "Created comprehensive documentation and guidelines for {term} usage.",
        "Assisted our senior engineer in deploying our application using {term}.",
        "Conducted automated unit testing and performance benchmarks on {term}.",
        "Researched modern architectural best practices involving {term}.",
        "Implemented secure authentication and access controls in {term}.",
        "Monitored server resource utilization and optimized {term} queries.",
        "Collaborated with the infrastructure team to configure {term}.",
        "Wrote clean, maintainable microservice code in {term} under supervision.",
        "Installed necessary security patches and updated the {term} dependency.",
        "Designed the database schema and executed migrations using {term}.",
        "Attended a technical workshop on best practices for {term}.",
        "Troubleshot network connectivity and configuration errors in {term}.",
        "Automated our daily build and deployment pipeline using {term}.",
        "Created an interactive user interface component with {term}.",
        "Implemented background asynchronous task workers using {term}."
    ]

    clerical_templates = [
        "Assisted the administrative supervisor with daily {term}.",
        "Handled confidential student records during {term}.",
        "Completed the assigned weekly tasks regarding {term} on schedule.",
        "Printed, stamped, and organized physical documents for {term}.",
        "Cross-checked incoming department requests as part of {term}.",
        "Maintained office order and coordinated department logistics through {term}.",
        "Prepared the required paperwork and verified attachments for {term}.",
        "Encoded information accurately into the office registry during {term}.",
        "Submitted the finalized documentation package required for {term}.",
        "Assisted staff members and visitors with inquiries regarding {term}.",
        "Reviewed transaction logs and audited files as part of {term}.",
        "Organized file cabinets, categorized binders, and conducted {term}.",
        "Drafted official correspondence and summarized notes for {term}.",
        "Verified signatures and filed department clearance forms during {term}.",
        "Distributed printed handouts and supported the team during {term}.",
        "Facilitated document routing and tracked outgoing mail for {term}.",
        "Compiled weekly department statistics and created charts for {term}.",
        "Managed physical inventory counts and logged discrepancies during {term}."
    ]

    # Negative examples: Realistic OJT sentences containing NO IT/Clerical entities
    negative_sentences = [
        "Attended the morning standup meeting with the supervisor to discuss daily goals.",
        "Joined the weekly team retrospective to share progress updates and blockers.",
        "Took a short lunch break with fellow student interns at the company cafeteria.",
        "Participated in the company-wide orientation regarding workplace ethics and rules.",
        "Waited for feedback from the team lead before proceeding with the next assignment.",
        "Greeted visitors at the entrance and escorted them to the conference room.",
        "Cleaned the intern workstation and organized desk accessories before logging out.",
        "Listened to a presentation given by the department head on strategic objectives.",
        "Took detailed personal notes during the department town hall assembly.",
        "Arrived at the office on time and logged in to the biometric attendance machine.",
        "Discussed project deadlines and milestone expectations during the afternoon huddle.",
        "Helped a colleague locate the designated conference room on the third floor.",
        "Reviewed general company guidelines regarding data privacy and acceptable use.",
        "Attended the farewell gathering for the departing senior supervisor.",
        "Submitted the bi-weekly intern time log sheet to the human resource assistant.",
        "Participated in an open discussion regarding team workflow enhancements.",
        "Signed the visitor logbook and collected the guest identification badges.",
        "Checked my work email inbox and responded to routine greeting messages.",
        "Organized notebook notes and planned personal task priorities for tomorrow morning.",
        "Had a quick alignment call with the project mentor to review weekly objectives."
    ]

    records: List[Dict[str, Any]] = []

    # Generate IT instances
    for i in range(target_samples_per_label):
        term = it_terms[i % len(it_terms)]
        template = random.choice(it_templates)
        # Randomize minor linguistic variation
        prefix = random.choice(["Today, ", "In the morning, ", "This week, ", "For my primary task, ", ""])
        sentence = prefix + template.format(term=term)
        spans = find_term_spans(sentence, {term: "IT_TERM"})
        records.append({
            "text": sentence,
            "entities": [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans]
        })

    # Generate Clerical instances
    for i in range(target_samples_per_label):
        term = clerical_terms[i % len(clerical_terms)]
        template = random.choice(clerical_templates)
        prefix = random.choice(["Today, ", "During my shift, ", "In the afternoon, ", "As assigned, ", ""])
        sentence = prefix + template.format(term=term)
        spans = find_term_spans(sentence, {term: "CLERICAL_TERM"})
        records.append({
            "text": sentence,
            "entities": [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans]
        })

    # Generate Negative instances
    while len([r for r in records if len(r["entities"]) == 0]) < target_negatives:
        neg_sent = random.choice(negative_sentences)
        prefix = random.choice(["First, ", "Afterwards, ", "On Thursday, ", "Early in the day, ", ""])
        combined_neg = prefix + neg_sent
        records.append({"text": combined_neg, "entities": []})

    # Shuffle records
    random.shuffle(records)
    logger.info(f"Generated synthetic OJT dataset: {len(records)} total records")
    return records, negative_sentences


def split_and_convert_dataset(
    records: List[Dict[str, Any]],
    output_dir: str = "data/training",
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, str]:
    """Splits records into train/dev/test and exports spaCy binary files."""
    random.seed(seed)
    shuffled = list(records)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)

    train_recs = shuffled[:n_train]
    dev_recs = shuffled[n_train:n_train + n_dev]
    test_recs = shuffled[n_train + n_dev:]

    paths = {
        "train": os.path.join(output_dir, "train.spacy"),
        "dev": os.path.join(output_dir, "dev.spacy"),
        "test": os.path.join(output_dir, "test.spacy"),
    }

    nlp = spacy.blank("en")
    logger.info(f"Dataset split: Train={len(train_recs)}, Dev={len(dev_recs)}, Test={len(test_recs)}")

    convert_records_to_docbin(train_recs, paths["train"], nlp=nlp)
    convert_records_to_docbin(dev_recs, paths["dev"], nlp=nlp)
    convert_records_to_docbin(test_recs, paths["test"], nlp=nlp)

    return paths


def build_full_dataset_pipeline(
    terms_csv_path: str = "data/terms.csv",
    output_raw_txt: str = "data/raw/sample_journals.txt",
    output_jsonl: str = "data/reviewed/annotations.jsonl",
    output_review_csv: str = "data/reviewed/annotations_review.csv",
    output_training_dir: str = "data/training",
    samples_per_label: int = 400,
    negatives: int = 200,
) -> Dict[str, Any]:
    """End-to-end dataset builder: loads terms, generates corpus, exports JSONL, CSV and DocBin."""
    terms_dict = load_terms_dictionary(terms_csv_path)
    records, _ = generate_synthetic_ojt_dataset(
        terms_dict,
        target_samples_per_label=samples_per_label,
        target_negatives=negatives
    )

    # Save raw sentences
    os.makedirs(os.path.dirname(output_raw_txt), exist_ok=True)
    with open(output_raw_txt, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r["text"] + "\n")

    # Save JSONL and review CSV
    save_jsonl(records, output_jsonl)
    export_for_review_csv(records, output_review_csv)

    # Split into train/dev/test spaCy binaries
    spacy_paths = split_and_convert_dataset(records, output_dir=output_training_dir)

    return {
        "total_records": len(records),
        "jsonl_path": output_jsonl,
        "review_csv_path": output_review_csv,
        "spacy_paths": spacy_paths,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = build_full_dataset_pipeline()
    print("Dataset generation complete:", res)
