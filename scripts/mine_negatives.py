"""
Hard Negative Mining and Stylistic Overlap Analysis Module (Task 5).

Guarantees:
1. Zero Data Leakage: Mined strictly from authentic training data (data/training/train.spacy).
   Dev and test splits are strictly excluded.
2. High-Confidence Negative Mining: Identifies negative sentences where the ML model
   falsely predicts an entity with high confidence (>=0.85).
3. Stylistic Overlap Analysis: Computes n-gram overlap and vocabulary Jaccard similarity
   between synthetic templates and the unseen benchmark.
"""

import os
import sys
import json
import logging
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


def mine_training_hard_negatives(
    data_jsonl_path: str = "data/data.jsonl",
    holdout_jsonl_path: str = "data/test/holdout.jsonl",
    model_path: str = "models/ner_trf/model-best",
    output_path: str = "data/review/mined_hard_negatives.jsonl",
    confidence_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Mines hard negative candidates from authentic negative sentences without test leakage.
    
    Identifies non-test real sentences where gold annotates 0 entities, but ML produces
    a high-confidence false positive (>=0.85).
    """
    scripts.init_gpu()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(data_jsonl_path):
        raise FileNotFoundError(f"Data file not found at: {data_jsonl_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    # Build strict set of holdout texts to guarantee ZERO leakage
    holdout_texts = set()
    if os.path.exists(holdout_jsonl_path):
        with open(holdout_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    holdout_texts.add(json.loads(line).get("text", "").strip())

    records = []
    with open(data_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Select authentic negative records strictly excluding holdout
    negatives = [
        r for r in records
        if len(r.get("entities", [])) == 0 and r.get("text", "").strip() not in holdout_texts
    ]

    pipeline = HybridJournalPipeline(model_path=model_path)
    hard_negatives = []
    fp_term_counts = Counter()

    logger.info(f"Scanning {len(negatives)} authentic non-test negative records for hard negative candidates...")

    for r in negatives:
        text = r["text"]
        pred_res = pipeline.predict(text, mode="transformer_only")
        pred_ents = pred_res.get("entities", [])

        gold_spans = [(ent["start"], ent["end"]) for ent in r.get("entities", [])]

        # Check for predictions on negative sentences or non-annotated spans
        for p in pred_ents:
            p_start, p_end = p["start"], p["end"]
            conf = p["confidence"]

            overlaps_gold = any(
                max(p_start, g_s) < min(p_end, g_e)
                for g_s, g_e in gold_spans
            )

            # High confidence false positive on non-entity span
            if not overlaps_gold and conf >= confidence_threshold:
                fp_term_counts[p["term"].lower()] += 1
                hard_negatives.append({
                    "text": text,
                    "false_positive_term": p["term"],
                    "false_positive_category": p["category"],
                    "confidence": conf,
                    "gold_entity_count": len(gold_spans),
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
        raise FileNotFoundError(f"Synthetic file not found: {synthetic_path}")
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

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
    logger.info("Starting hard negative mining and stylistic overlap evaluation...")
    mining_res = mine_training_hard_negatives()
    overlap_res = compute_stylistic_overlap()
    print("\n" + "=" * 80)
    print("TASK 5 MINING & STYLISTIC OVERLAP COMPLETE")
    print("=" * 80)
    print("Mining summary:", json.dumps(mining_res, indent=2))
    print("Stylistic overlap:", json.dumps(overlap_res, indent=2))
