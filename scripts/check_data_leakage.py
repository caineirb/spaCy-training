"""
Data Leakage and Benchmark Isolation Audit Script.

Verifies and prints PASS/FAIL for each of 6 strict leakage checks:
1. No duplicate documents across train/dev/test splits.
2. No duplicate sentences across train/dev/test splits.
3. No unseen-benchmark terms appear in data/terms.csv.
4. No unseen-benchmark terms appear in training annotations (data/data.jsonl).
5. No unseen-benchmark terms are matchable by the EntityRuler as currently configured.
6. No sentences in data/test/holdout.jsonl or data/test/raw/ also appear in
   train.spacy / dev.spacy / test.spacy or the synthetic annotation pipeline's source data.

Exits with code 1 if ANY check fails. Exits with code 0 only if ALL checks PASS.
"""

import os
import sys
import json
import glob
from typing import Dict, List, Set, Tuple, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
import spacy
from spacy.tokens import DocBin
from scripts.annotation import load_terms_dictionary
from scripts.pipeline import HybridJournalPipeline
from scripts.eval import load_unseen_benchmark


def print_section(title: str) -> None:
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def check_document_duplicates(
    train_docs: List[str],
    dev_docs: List[str],
    test_docs: List[str],
) -> Tuple[bool, Dict[str, Any]]:
    """Check 1: Verify zero duplicate documents across train/dev/test splits."""
    train_set = set(t.strip() for t in train_docs if t.strip())
    dev_set = set(t.strip() for t in dev_docs if t.strip())
    test_set = set(t.strip() for t in test_docs if t.strip())

    train_dev = train_set & dev_set
    train_test = train_set & test_set
    dev_test = dev_set & test_set

    # Intra-split duplicate check for informational logging
    train_intra = len(train_docs) - len(train_set)
    dev_intra = len(dev_docs) - len(dev_set)
    test_intra = len(test_docs) - len(test_set)

    passed = (len(train_dev) == 0 and len(train_test) == 0 and len(dev_test) == 0)

    details = {
        "train_count": len(train_docs),
        "dev_count": len(dev_docs),
        "test_count": len(test_docs),
        "train_intra_duplicates": train_intra,
        "dev_intra_duplicates": dev_intra,
        "test_intra_duplicates": test_intra,
        "train_dev_overlap_count": len(train_dev),
        "train_test_overlap_count": len(train_test),
        "dev_test_overlap_count": len(dev_test),
        "train_dev_overlap_samples": list(train_dev)[:5],
        "train_test_overlap_samples": list(train_test)[:5],
        "dev_test_overlap_samples": list(dev_test)[:5],
    }
    return passed, details


def check_sentence_duplicates(
    nlp: spacy.language.Language,
    train_docs: List[str],
    dev_docs: List[str],
    test_docs: List[str],
) -> Tuple[bool, Dict[str, Any]]:
    """Check 2: Verify zero duplicate sentences across train/dev/test splits."""
    def extract_sentences(docs: List[str]) -> List[str]:
        sents = []
        for text in docs:
            if not text.strip():
                continue
            doc = nlp(text)
            for s in doc.sents:
                clean_s = s.text.strip()
                if clean_s:
                    sents.append(clean_s)
        return sents

    train_sents = extract_sentences(train_docs)
    dev_sents = extract_sentences(dev_docs)
    test_sents = extract_sentences(test_docs)

    train_sent_set = set(train_sents)
    dev_sent_set = set(dev_sents)
    test_sent_set = set(test_sents)

    train_dev = train_sent_set & dev_sent_set
    train_test = train_sent_set & test_sent_set
    dev_test = dev_sent_set & test_sent_set

    passed = (len(train_dev) == 0 and len(train_test) == 0 and len(dev_test) == 0)

    details = {
        "train_sentence_count": len(train_sents),
        "dev_sentence_count": len(dev_sents),
        "test_sentence_count": len(test_sents),
        "train_dev_sentence_overlap_count": len(train_dev),
        "train_test_sentence_overlap_count": len(train_test),
        "dev_test_sentence_overlap_count": len(dev_test),
        "train_dev_samples": list(train_dev)[:5],
        "train_test_samples": list(train_test)[:5],
        "dev_test_samples": list(dev_test)[:5],
    }
    return passed, details


def check_unseen_in_terms_csv(
    unseen_gold_terms: Set[str],
    terms_csv_path: str = "data/terms.csv",
) -> Tuple[bool, Dict[str, Any]]:
    """Check 3: Verify zero unseen benchmark terms appear in data/terms.csv."""
    terms_dict = load_terms_dictionary(terms_csv_path)
    terms_lower = {t.strip().lower() for t in terms_dict.keys()}

    leaked = set()
    for term in unseen_gold_terms:
        if term.lower() in terms_lower:
            leaked.add(term)

    passed = (len(leaked) == 0)
    details = {
        "total_unseen_terms_checked": len(unseen_gold_terms),
        "total_dictionary_terms": len(terms_dict),
        "leaked_count": len(leaked),
        "leaked_terms": sorted(list(leaked)),
    }
    return passed, details


def check_unseen_in_training_annotations(
    unseen_gold_terms: Set[str],
    annotations_path: str = "data/data.jsonl",
) -> Tuple[bool, Dict[str, Any]]:
    """Check 4: Verify zero unseen benchmark terms appear in data/data.jsonl."""
    annotated_entities: Set[str] = set()

    if os.path.exists(annotations_path):
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    for ent in record.get("entities", []):
                        if isinstance(ent, list) and len(ent) == 3:
                            s, e, _ = ent
                            annotated_entities.add(text[s:e].strip().lower())
                        elif isinstance(ent, dict):
                            t = ent.get("term", text[ent["start"]:ent["end"]] if "start" in ent else "")
                            if t:
                                annotated_entities.add(t.strip().lower())
                except Exception:
                    continue

    leaked = set()
    for term in unseen_gold_terms:
        if term.lower() in annotated_entities:
            leaked.add(term)

    passed = (len(leaked) == 0)
    details = {
        "total_unseen_terms_checked": len(unseen_gold_terms),
        "total_annotated_entities_pool": len(annotated_entities),
        "leaked_count": len(leaked),
        "leaked_terms": sorted(list(leaked)),
    }
    return passed, details


def check_unseen_matchable_by_entity_ruler(
    pipeline: HybridJournalPipeline,
    benchmark_samples: List[Dict[str, Any]],
    unseen_gold_terms: Set[str],
) -> Tuple[bool, Dict[str, Any]]:
    """Check 5: Verify zero unseen benchmark terms are matchable by the EntityRuler."""
    # Test 1: Direct term string match by EntityRuler
    exact_term_matches: List[Tuple[str, str, str]] = []
    subspan_matches: List[Tuple[str, str, str]] = []

    with pipeline.nlp.select_pipes(disable=[p for p in ["transformer", "ner"] if p in pipeline.nlp.pipe_names]):
        for term in unseen_gold_terms:
            doc = pipeline.nlp(term)
            for ent in doc.ents:
                if ent.text.strip().lower() == term.lower():
                    exact_term_matches.append((term, ent.text, ent.label_))
                else:
                    subspan_matches.append((term, ent.text, ent.label_))

        # Test 2: In-context match on the exact target unseen entity span in benchmark sentences
        context_exact_span_matches: List[Dict[str, Any]] = []
        for sample in benchmark_samples:
            text = sample["text"]
            gold_ents = sample.get("entities", [])
            doc = pipeline.nlp(text)
            pred_spans = {(e.start_char, e.end_char, e.label_, e.text) for e in doc.ents}

            for g in gold_ents:
                g_start = g.get("start")
                g_end = g.get("end")
                g_label = g.get("label")
                g_term = g.get("term")

                if (g_start, g_end, g_label, g_term) in pred_spans:
                    context_exact_span_matches.append({
                        "text": text,
                        "gold_term": g_term,
                        "span": [g_start, g_end],
                        "label": g_label,
                    })

    # Exact matches of the unseen term by EntityRuler indicate pure dictionary leakage
    passed = (len(exact_term_matches) == 0 and len(context_exact_span_matches) == 0)

    details = {
        "exact_term_matches_count": len(exact_term_matches),
        "exact_term_matches": exact_term_matches,
        "context_exact_span_matches_count": len(context_exact_span_matches),
        "context_exact_span_matches": context_exact_span_matches,
        "subspan_partial_matches_count": len(subspan_matches),
        "subspan_partial_matches": subspan_matches,
    }
    return passed, details


def check_real_holdout_isolation(
    nlp: spacy.language.Language,
    train_docs: List[str],
    dev_docs: List[str],
    test_docs: List[str],
    holdout_jsonl_path: str = "data/test/holdout.jsonl",
    raw_holdout_dir: str = "data/test/raw",
    annotations_path: str = "data/data.jsonl",
) -> Tuple[bool, Dict[str, Any]]:
    """Check 6: Verify zero sentences in holdout.jsonl or data/test/raw/ appear in training/dev/test/synthetic data."""
    def extract_sentences_from_text(text: str) -> List[str]:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    # Collect all existing training & synthetic sentences
    known_pool: Set[str] = set()
    for d in train_docs + dev_docs + test_docs:
        for s in extract_sentences_from_text(d):
            known_pool.add(s)

    if os.path.exists(annotations_path):
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    try:
                        rec = json.loads(line)
                        for s in extract_sentences_from_text(rec.get("text", "")):
                            known_pool.add(s)
                    except Exception:
                        pass

    # Collect holdout sentences
    holdout_sentences: List[str] = []

    # From holdout.jsonl
    if os.path.exists(holdout_jsonl_path) and os.path.getsize(holdout_jsonl_path) > 0:
        with open(holdout_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        rec = json.loads(line)
                        holdout_sentences.extend(extract_sentences_from_text(rec.get("text", "")))
                    except Exception:
                        pass

    # From data/test/raw/*
    if os.path.exists(raw_holdout_dir):
        raw_files = glob.glob(os.path.join(raw_holdout_dir, "**", "*"), recursive=True)
        for rf in raw_files:
            if os.path.isfile(rf) and os.path.getsize(rf) > 0:
                try:
                    with open(rf, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        holdout_sentences.extend(extract_sentences_from_text(content))
                except Exception:
                    pass

    holdout_set = set(holdout_sentences)
    overlap = holdout_set & known_pool

    passed = (len(overlap) == 0)
    details = {
        "holdout_sentences_found": len(holdout_sentences),
        "holdout_unique_sentences": len(holdout_set),
        "training_pool_sentences": len(known_pool),
        "overlap_count": len(overlap),
        "overlapping_sentences": list(overlap)[:5],
        "is_holdout_empty": (len(holdout_sentences) == 0),
    }
    return passed, details


def run_all_leakage_checks() -> bool:
    """Executes all 6 leakage checks, prints structured output, and returns overall success."""
    print("\n" + "#" * 75)
    print("  DATA LEAKAGE & BENCHMARK ISOLATION AUDIT")
    print("#" * 75)

    # 1. Load splits from DocBins
    print("\n[INFO] Loading dataset splits from data/training/*.spacy...")
    sentencizer_nlp = spacy.blank("en")
    sentencizer_nlp.add_pipe("sentencizer")

    def load_docbin_texts(path: str) -> List[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing DocBin file: {path}")
        db = DocBin().from_disk(path)
        return [doc.text for doc in db.get_docs(sentencizer_nlp.vocab)]

    train_docs = load_docbin_texts("data/training/train.spacy")
    dev_docs = load_docbin_texts("data/training/dev.spacy")
    test_docs = load_docbin_texts("data/training/test.spacy")

    # 2. Extract unseen benchmark terms
    benchmark_samples = load_unseen_benchmark()
    unseen_gold_terms: Set[str] = set()
    for sample in benchmark_samples:
        for ent in sample.get("entities", []):
            term = ent.get("term", "").strip()
            if term:
                unseen_gold_terms.add(term)

    # 3. Load hybrid pipeline for EntityRuler checks
    print("[INFO] Initializing pipeline for EntityRuler matchability audit...")
    pipeline = HybridJournalPipeline()

    results: Dict[str, Tuple[bool, str]] = {}

    # -------------------------------------------------------------
    # Check 1: Duplicate documents across train/dev/test splits
    # -------------------------------------------------------------
    print_section("Check 1: Cross-Split Document Duplication")
    c1_pass, c1_details = check_document_duplicates(train_docs, dev_docs, test_docs)
    if c1_pass:
        print("[PASS] Zero duplicate documents across train/dev/test splits.")
        print(f"       Docs: train={c1_details['train_count']}, dev={c1_details['dev_count']}, test={c1_details['test_count']}")
        results["Check 1 (Document Duplication)"] = (True, "PASS")
    else:
        print("[FAIL] Duplicate documents detected across dataset splits!")
        print(f"       train <-> dev overlap : {c1_details['train_dev_overlap_count']} duplicates")
        print(f"       train <-> test overlap: {c1_details['train_test_overlap_count']} duplicates")
        print(f"       dev   <-> test overlap: {c1_details['dev_test_overlap_count']} duplicates")
        if c1_details["train_test_overlap_samples"]:
            print("       Sample train <-> test overlap:")
            for s in c1_details["train_test_overlap_samples"][:3]:
                print(f"         - {repr(s)}")
        results["Check 1 (Document Duplication)"] = (
            False,
            f"FAIL (train-dev: {c1_details['train_dev_overlap_count']}, train-test: {c1_details['train_test_overlap_count']}, dev-test: {c1_details['dev_test_overlap_count']})"
        )

    # -------------------------------------------------------------
    # Check 2: Duplicate sentences across train/dev/test splits
    # -------------------------------------------------------------
    print_section("Check 2: Cross-Split Sentence Duplication")
    c2_pass, c2_details = check_sentence_duplicates(sentencizer_nlp, train_docs, dev_docs, test_docs)
    if c2_pass:
        print("[PASS] Zero duplicate sentences across train/dev/test splits.")
        print(f"       Sentences: train={c2_details['train_sentence_count']}, dev={c2_details['dev_sentence_count']}, test={c2_details['test_sentence_count']}")
        results["Check 2 (Sentence Duplication)"] = (True, "PASS")
    else:
        print("[FAIL] Duplicate sentences detected across dataset splits!")
        print(f"       train <-> dev sentence overlap : {c2_details['train_dev_sentence_overlap_count']} duplicates")
        print(f"       train <-> test sentence overlap: {c2_details['train_test_sentence_overlap_count']} duplicates")
        print(f"       dev   <-> test sentence overlap: {c2_details['dev_test_sentence_overlap_count']} duplicates")
        if c2_details.get("train_dev_samples"):
            print("       Sample train <-> dev sentence overlap:")
            for s in c2_details["train_dev_samples"][:3]:
                print(f"         - {repr(s)}")
        if c2_details.get("train_test_samples"):
            print("       Sample train <-> test sentence overlap:")
            for s in c2_details["train_test_samples"][:3]:
                print(f"         - {repr(s)}")
        if c2_details.get("dev_test_samples"):
            print("       Sample dev <-> test sentence overlap:")
            for s in c2_details["dev_test_samples"][:3]:
                print(f"         - {repr(s)}")
        results["Check 2 (Sentence Duplication)"] = (
            False,
            f"FAIL (train-dev: {c2_details['train_dev_sentence_overlap_count']}, train-test: {c2_details['train_test_sentence_overlap_count']}, dev-test: {c2_details['dev_test_sentence_overlap_count']})"
        )

    # -------------------------------------------------------------
    # Check 3: Unseen benchmark terms in data/terms.csv
    # -------------------------------------------------------------
    print_section("Check 3: Unseen Benchmark Terms in data/terms.csv")
    c3_pass, c3_details = check_unseen_in_terms_csv(unseen_gold_terms)
    if c3_pass:
        print(f"[PASS] Zero of {c3_details['total_unseen_terms_checked']} unseen benchmark terms appear in data/terms.csv.")
        print(f"       Dictionary size checked: {c3_details['total_dictionary_terms']} terms.")
        results["Check 3 (Terms CSV Leakage)"] = (True, "PASS")
    else:
        print(f"[FAIL] {c3_details['leaked_count']} unseen benchmark terms found in data/terms.csv!")
        print(f"       Leaked terms: {c3_details['leaked_terms']}")
        results["Check 3 (Terms CSV Leakage)"] = (False, f"FAIL ({c3_details['leaked_count']} terms leaked)")

    # -------------------------------------------------------------
    # Check 4: Unseen benchmark terms in training annotations
    # -------------------------------------------------------------
    print_section("Check 4: Unseen Benchmark Terms in Training Annotations")
    c4_pass, c4_details = check_unseen_in_training_annotations(unseen_gold_terms)
    if c4_pass:
        print(f"[PASS] Zero of {c4_details['total_unseen_terms_checked']} unseen benchmark terms appear in data/data.jsonl.")
        print(f"       Annotated entity pool: {c4_details['total_annotated_entities_pool']} unique entity strings.")
        results["Check 4 (Training Annotation Leakage)"] = (True, "PASS")
    else:
        print(f"[FAIL] {c4_details['leaked_count']} unseen benchmark terms found in training annotations!")
        print(f"       Leaked terms: {c4_details['leaked_terms']}")
        results["Check 4 (Training Annotation Leakage)"] = (False, f"FAIL ({c4_details['leaked_count']} terms leaked)")

    # -------------------------------------------------------------
    # Check 5: Unseen benchmark terms matchable by EntityRuler
    # -------------------------------------------------------------
    print_section("Check 5: EntityRuler Matchability on Unseen Benchmark")
    c5_pass, c5_details = check_unseen_matchable_by_entity_ruler(pipeline, benchmark_samples, unseen_gold_terms)
    if c5_pass:
        print("[PASS] Zero unseen benchmark terms are matchable as exact entities by the EntityRuler.")
        if c5_details["subspan_partial_matches_count"] > 0:
            print(f"       [NOTE] Sub-span token overlap detected for {c5_details['subspan_partial_matches_count']} terms:")
            for t, matched_sub, lbl in c5_details["subspan_partial_matches"]:
                print(f"         - '{t}' contains dictionary token '{matched_sub}' ({lbl})")
        results["Check 5 (EntityRuler Matchability)"] = (True, "PASS")
    else:
        print(f"[FAIL] EntityRuler matched {c5_details['exact_term_matches_count']} unseen benchmark terms directly!")
        if c5_details["exact_term_matches"]:
            print(f"       Exact term matches: {c5_details['exact_term_matches']}")
        if c5_details["context_exact_span_matches"]:
            print(f"       Context span matches: {c5_details['context_exact_span_matches']}")
        results["Check 5 (EntityRuler Matchability)"] = (False, f"FAIL ({c5_details['exact_term_matches_count']} exact matches)")

    # -------------------------------------------------------------
    # Check 6: Real holdout data isolation
    # -------------------------------------------------------------
    print_section("Check 6: Real Holdout Scaffolding & Data Isolation")
    c6_pass, c6_details = check_real_holdout_isolation(sentencizer_nlp, train_docs, dev_docs, test_docs)
    if c6_pass:
        if c6_details["is_holdout_empty"]:
            print("[PASS] Real holdout dataset is currently empty (0 sentences; awaiting manual gold annotations).")
            print(f"       Scanned holdout.jsonl and data/test/raw/ against {c6_details['training_pool_sentences']} training sentences.")
            print("       Isolation verified: zero leakage possible from 0 holdout records.")
        else:
            print(f"[PASS] Verified isolation: zero of {c6_details['holdout_unique_sentences']} holdout sentences appear in training/dev/test.")
        results["Check 6 (Holdout Isolation)"] = (True, "PASS")
    else:
        print(f"[FAIL] {c6_details['overlap_count']} holdout sentences overlap with training/synthetic data!")
        print(f"       Sample overlapping sentences: {c6_details['overlapping_sentences']}")
        results["Check 6 (Holdout Isolation)"] = (False, f"FAIL ({c6_details['overlap_count']} sentences overlap)")

    # -------------------------------------------------------------
    # Overall Audit Summary
    # -------------------------------------------------------------
    print_section("AUDIT SUMMARY")
    all_passed = True
    for check_name, (passed, msg) in results.items():
        status_str = "PASS" if passed else "FAIL"
        print(f"  {check_name:<40}: [{status_str}] - {msg}")
        if not passed:
            all_passed = False

    print("=" * 75)
    if all_passed:
        print(">>> RESULT: ALL 6 CHECKS PASSED. ZERO DATA LEAKAGE DETECTED.")
        print("=" * 75)
        return True
    else:
        print(">>> RESULT: LEAKAGE DETECTED. ONE OR MORE CHECKS FAILED.")
        print("=" * 75)
        return False


if __name__ == "__main__":
    success = run_all_leakage_checks()
    if not success:
        sys.exit(1)
    sys.exit(0)
