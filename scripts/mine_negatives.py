"""
Hard Negative Mining and Stylistic Overlap Analysis Module.

Guarantees:
1. Zero Data Leakage: Mined strictly from authentic training data.
   Held-out test split, dev split, and unseen benchmark are strictly excluded.
2. High-Confidence Negative Mining: Identifies negative sentences where the ML model
   falsely predicts an entity with high confidence (>=0.85).
3. Stylistic Overlap Analysis: Computes n-gram overlap and vocabulary Jaccard similarity
   between synthetic templates and the unseen benchmark.
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import spacy
from spacy.tokens import DocBin

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
from scripts.pipeline import HybridJournalPipeline

logger = logging.getLogger("ojt_pipeline.mine_negatives")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def resolve_default_model_path() -> str:
    """Finds the best available model checkpoint for mining."""
    candidates = [
        "models/ner_trf/model-best",
        "models/ner_trf_trtr/model-best",
        "models/ner_trf_trstr_llm/model-best",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "models/ner_trf_trtr/model-best"


def mine_training_hard_negatives(
    data_jsonl_path: str = "data/data.jsonl",
    test_spacy_path: str = "data/training/test.spacy",
    dev_spacy_path: str = "data/training/dev.spacy",
    unseen_benchmark_path: str = "data/test/unseen_benchmark.jsonl",
    model_path: str = None,
    output_path: str = "data/review/mined_hard_negatives.jsonl",
    confidence_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Mines hard negative candidates from authentic negative sentences without test leakage.
    
    Identifies non-test real sentences where gold annotates 0 entities, but ML produces
    a high-confidence false positive (>= confidence_threshold).
    """
    scripts.init_gpu()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if model_path is None:
        model_path = resolve_default_model_path()

    if not os.path.exists(data_jsonl_path):
        raise FileNotFoundError(f"Data file not found at: {data_jsonl_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    nlp_blank = spacy.blank("en")

    # 1. Build strict set of excluded evaluation texts to guarantee ZERO leakage
    excluded_texts: Set[str] = set()

    if os.path.exists(test_spacy_path):
        test_db = DocBin().from_disk(test_spacy_path)
        for doc in test_db.get_docs(nlp_blank.vocab):
            excluded_texts.add(doc.text.strip().lower())
        logger.info(f"Loaded {len(test_db)} held-out test texts to exclude.")

    if os.path.exists(dev_spacy_path):
        dev_db = DocBin().from_disk(dev_spacy_path)
        for doc in dev_db.get_docs(nlp_blank.vocab):
            excluded_texts.add(doc.text.strip().lower())
        logger.info(f"Loaded {len(dev_db)} dev texts to exclude.")

    if os.path.exists(unseen_benchmark_path):
        with open(unseen_benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        excluded_texts.add(rec["text"].strip().lower())
                    except Exception:
                        pass
        logger.info(f"Excluded unseen benchmark texts. Total excluded: {len(excluded_texts)}")

    # 2. Load authentic records from data.jsonl
    records = []
    with open(data_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 3. Filter authentic negative records (0 gold entities and strictly non-evaluation)
    negatives = [
        r for r in records
        if len(r.get("entities", [])) == 0 and r.get("text", "").strip().lower() not in excluded_texts
    ]

    logger.info(f"Using model checkpoint: {model_path}")
    pipeline = HybridJournalPipeline(model_path=model_path)
    hard_negatives = []
    fp_term_counts = Counter()

    logger.info(f"Scanning {len(negatives)} authentic training negative sentences (threshold={confidence_threshold})...")

    seen_fp_keys = set()
    for r in negatives:
        text = r["text"].strip()
        pred_res = pipeline.predict(text, mode="transformer_only")
        pred_ents = pred_res.get("entities", [])

        # Check for predictions on negative sentences
        for p in pred_ents:
            conf = p["confidence"]
            term = p["term"].strip()
            cat = p["category"]

            if conf >= confidence_threshold and term:
                fp_key = (text, term.lower())
                if fp_key in seen_fp_keys:
                    continue
                seen_fp_keys.add(fp_key)

                fp_term_counts[term.lower()] += 1
                hard_negatives.append({
                    "text": text,
                    "false_positive_term": term,
                    "false_positive_category": cat,
                    "confidence": round(conf, 4),
                    "gold_entity_count": 0,
                })

    # Save mined candidates
    with open(output_path, "w", encoding="utf-8") as f:
        for hn in hard_negatives:
            f.write(json.dumps(hn, ensure_ascii=False) + "\n")

    logger.info(
        f"Mined {len(hard_negatives)} hard negative instances across training corpus. "
        f"Saved to: {output_path}"
    )

    top_fp_terms = fp_term_counts.most_common(20)
    logger.info(f"Top 10 hard negative terms: {top_fp_terms[:10]}")

    return {
        "total_instances_mined": len(hard_negatives),
        "unique_fp_terms": len(fp_term_counts),
        "top_fp_terms": top_fp_terms,
        "output_path": output_path,
        "model_used": model_path,
    }


def get_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Extracts unique n-grams from a list of tokens."""
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_stylistic_overlap(
    synthetic_path: str = "data/synthetic_llm_generated.jsonl",
    benchmark_path: str = "data/test/unseen_benchmark.jsonl",
) -> Dict[str, Any]:
    """Computes stylistic and n-gram overlap between synthetic templates and unseen benchmark."""
    if not os.path.exists(synthetic_path):
        logger.warning(f"Synthetic file not found at: {synthetic_path}. Skipping stylistic overlap.")
        return {}
    if not os.path.exists(benchmark_path):
        logger.warning(f"Benchmark file not found at: {benchmark_path}. Skipping stylistic overlap.")
        return {}

    nlp = spacy.blank("en")

    def load_corpus_tokens(path: str) -> Tuple[List[str], List[List[str]]]:
        all_tokens = []
        doc_tokens = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = rec.get("text", "")
                tokens = [t.text.lower() for t in nlp(text) if not t.is_space]
                all_tokens.extend(tokens)
                doc_tokens.append(tokens)
        return all_tokens, doc_tokens

    syn_tokens, syn_docs = load_corpus_tokens(synthetic_path)
    bench_tokens, bench_docs = load_corpus_tokens(benchmark_path)

    # Vocabulary Jaccard
    syn_vocab = set(syn_tokens)
    bench_vocab = set(bench_tokens)
    vocab_jaccard = len(syn_vocab & bench_vocab) / len(syn_vocab | bench_vocab) if (syn_vocab | bench_vocab) else 0.0

    # Bigram & Trigram Jaccard
    syn_bigrams = set().union(*[get_ngrams(d, 2) for d in syn_docs])
    bench_bigrams = set().union(*[get_ngrams(d, 2) for d in bench_docs])
    bigram_jaccard = len(syn_bigrams & bench_bigrams) / len(syn_bigrams | bench_bigrams) if (syn_bigrams | bench_bigrams) else 0.0

    syn_trigrams = set().union(*[get_ngrams(d, 3) for d in syn_docs])
    bench_trigrams = set().union(*[get_ngrams(d, 3) for d in bench_docs])
    trigram_jaccard = len(syn_trigrams & bench_trigrams) / len(syn_trigrams | bench_trigrams) if (syn_trigrams | bench_trigrams) else 0.0

    # Length statistics
    syn_lengths = [len(d) for d in syn_docs]
    bench_lengths = [len(d) for d in bench_docs]

    overlap_report = {
        "synthetic_records": len(syn_docs),
        "benchmark_records": len(bench_docs),
        "vocab_overlap_pct": round(len(syn_vocab & bench_vocab) / len(bench_vocab) * 100, 2) if bench_vocab else 0.0,
        "vocab_jaccard_similarity": round(vocab_jaccard, 4),
        "bigram_jaccard_similarity": round(bigram_jaccard, 4),
        "trigram_jaccard_similarity": round(trigram_jaccard, 4),
        "synthetic_mean_token_length": round(sum(syn_lengths) / len(syn_lengths), 1) if syn_lengths else 0.0,
        "benchmark_mean_token_length": round(sum(bench_lengths) / len(bench_lengths), 1) if bench_lengths else 0.0,
    }

    logger.info("Stylistic Overlap Report:")
    for k, v in overlap_report.items():
        logger.info(f"  {k}: {v}")

    return overlap_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine hard negatives from authentic training data without test leakage.")
    parser.add_argument("--data-path", type=str, default="data/data.jsonl", help="Path to data.jsonl")
    parser.add_argument("--test-spacy", type=str, default="data/training/test.spacy", help="Path to test.spacy")
    parser.add_argument("--dev-spacy", type=str, default="data/training/dev.spacy", help="Path to dev.spacy")
    parser.add_argument("--unseen-benchmark", type=str, default="data/test/unseen_benchmark.jsonl", help="Path to unseen_benchmark.jsonl")
    parser.add_argument("--model-path", type=str, default=None, help="Model checkpoint path to evaluate for false positives")
    parser.add_argument("--output-path", type=str, default="data/review/mined_hard_negatives.jsonl", help="Output JSONL path")
    parser.add_argument("--confidence-threshold", type=float, default=0.85, help="Confidence threshold for false positive extraction")
    args = parser.parse_args()

    logger.info("Starting hard negative mining and stylistic overlap evaluation...")
    mining_res = mine_training_hard_negatives(
        data_jsonl_path=args.data_path,
        test_spacy_path=args.test_spacy,
        dev_spacy_path=args.dev_spacy,
        unseen_benchmark_path=args.unseen_benchmark,
        model_path=args.model_path,
        output_path=args.output_path,
        confidence_threshold=args.confidence_threshold,
    )
    overlap_res = compute_stylistic_overlap()
    print("\n" + "=" * 80)
    print("MINING & STYLISTIC OVERLAP COMPLETE")
    print("=" * 80)
    print("Mining summary:", json.dumps(mining_res, indent=2))
    print("Stylistic overlap:", json.dumps(overlap_res, indent=2))
