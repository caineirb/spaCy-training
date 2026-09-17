"""
Project Label Taxonomy & Normalization Layer.

Explicitly formalizes the two-tier label architecture used across this project:

Tier 1 — Dictionary Vocabulary Labels (data/terms.csv):
    - `IT_TASK`: Seed technology, programming language, software tool, or IT task phrase.
    - `CLERICAL`: Seed clerical activity, office suite software, or administrative workflow phrase.

Tier 2 — NER Span Entity Labels (annotations.jsonl, DocBins, NER model checkpoints):
    - `IT_TERM`: Token span extracted by spaCy NER identifying an IT task, tool, or technology.
    - `CLERICAL_TERM`: Token span extracted by spaCy NER identifying a clerical/administrative activity or office tool.

This explicit mapping layer guarantees traceability between vocabulary definitions and NER spans.
"""

from typing import Dict, Set

# Canonical label sets
CANONICAL_DICTIONARY_LABELS: Set[str] = {"IT_TASK", "CLERICAL"}
CANONICAL_NER_LABELS: Set[str] = {"IT_TERM", "CLERICAL_TERM"}

# Explicit mapping from dictionary vocabulary categories to NER span entity labels
DICTIONARY_TO_NER_MAP: Dict[str, str] = {
    "IT_TASK": "IT_TERM",
    "CLERICAL": "CLERICAL_TERM",
    "IT_TERM": "IT_TERM",               # Identity for already normalized labels
    "CLERICAL_TERM": "CLERICAL_TERM",   # Identity for already normalized labels
}

# Reverse mapping from NER span entity labels back to dictionary categories
NER_TO_DICTIONARY_MAP: Dict[str, str] = {
    "IT_TERM": "IT_TASK",
    "CLERICAL_TERM": "CLERICAL",
    "IT_TASK": "IT_TASK",               # Identity for already dictionary labels
    "CLERICAL": "CLERICAL",             # Identity for already dictionary labels
}

# Comprehensive lookup dictionary for backward-compatible imports
LABEL_MAPPING: Dict[str, str] = dict(DICTIONARY_TO_NER_MAP)


def normalize_to_ner_label(raw_label: str) -> str:
    """Normalizes any recognized raw label to its canonical NER span label (IT_TERM or CLERICAL_TERM).
    
    Args:
        raw_label: Input label string (e.g. 'IT_TASK', 'CLERICAL', 'IT_TERM', 'CLERICAL_TERM').
        
    Returns:
        Canonical NER span label ('IT_TERM' or 'CLERICAL_TERM').
        
    Raises:
        ValueError: If raw_label is not recognized in the project taxonomy.
    """
    clean = str(raw_label).strip().upper()
    if clean in DICTIONARY_TO_NER_MAP:
        return DICTIONARY_TO_NER_MAP[clean]
    raise ValueError(
        f"Unrecognized label '{raw_label}'. "
        f"Valid dictionary labels are {sorted(list(CANONICAL_DICTIONARY_LABELS))}; "
        f"valid NER entity labels are {sorted(list(CANONICAL_NER_LABELS))}."
    )


def normalize_to_dictionary_label(raw_label: str) -> str:
    """Normalizes any recognized raw label to its canonical dictionary category (IT_TASK or CLERICAL).
    
    Args:
        raw_label: Input label string (e.g. 'IT_TASK', 'CLERICAL', 'IT_TERM', 'CLERICAL_TERM').
        
    Returns:
        Canonical dictionary category ('IT_TASK' or 'CLERICAL').
        
    Raises:
        ValueError: If raw_label is not recognized in the project taxonomy.
    """
    clean = str(raw_label).strip().upper()
    if clean in NER_TO_DICTIONARY_MAP:
        return NER_TO_DICTIONARY_MAP[clean]
    raise ValueError(
        f"Unrecognized label '{raw_label}'. "
        f"Valid dictionary labels are {sorted(list(CANONICAL_DICTIONARY_LABELS))}; "
        f"valid NER entity labels are {sorted(list(CANONICAL_NER_LABELS))}."
    )


def is_valid_label(label: str) -> bool:
    """Returns True if the label belongs to either the dictionary or NER label sets."""
    clean = str(label).strip().upper()
    return clean in DICTIONARY_TO_NER_MAP
