"""
Grouped K-Fold Cross-Validation for OJT NER: TRTR vs. TRSTR-Paraphrase vs. TRSTR-LLM.

Protocol:
1. Pure Real Evaluation: K document-level folds partitioned over all authentic student journal
   records in `data/data.jsonl` (1,241 records). Each validation fold contains authentic records only.
2. Zero Evaluation Contamination: Synthetic paraphrases, LLM-generated records, and mined negatives
   are filtered and added ONLY to training folds, never to validation folds.
3. Three Canonical Conditions:
   (i)   trtr: Baseline trained strictly on authentic training records for that fold.
   (ii)  trstr_paraphrase: Authentic training fold + T5-generated paraphrases (data/synthetic_paraphrases.jsonl).
   (iii) trstr_llm: Authentic training fold + Gemini LLM synthetic data (data/synthetic_llm_generated.jsonl)
         + curated mined hard negatives (data/review/mined_hard_negatives.jsonl).
4. Evaluation:
   - Evaluated on the identical pure-real validation fold for each split.
   - Secondary generalization probe on the isolated unseen-term benchmark (data/test/unseen_benchmark.jsonl).
   - Reports Mean ± Standard Deviation for Strict Precision, Recall, and F1 (Overall, IT_TERM, CLERICAL_TERM).
"""

import os
import sys
import gc
import json
import logging
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
scripts.init_gpu()

import spacy
from spacy.tokens import DocBin
from spacy.cli.train import train as spacy_train
from sklearn.model_selection import StratifiedKFold
from scripts.pipeline import HybridJournalPipeline
from scripts.eval import load_unseen_benchmark, evaluate_unseen_mode

logger = logging.getLogger("ojt_cv")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

CV_DATA_DIR = "data/cv"
CV_RESULTS_DIR = "data/cv/results"
CV_MODELS_DIR = "models/cv"


def extract_all_real_records(data_path: str = "data/data.jsonl") -> List[Dict[str, Any]]:
    """Loads all authentic real records from data/data.jsonl."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source data file not found: {data_path}")
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} authentic documents from {data_path}.")
    return records


def get_doc_stratification_label(record: Dict[str, Any]) -> str:
    """Stratifies records by presence and balance of entity types."""
    labels = [e["label"] for e in record.get("entities", [])]
    if not labels:
        return "negative"
    has_it = "IT_TERM" in labels
    has_cl = "CLERICAL_TERM" in labels
    if has_it and has_cl:
        return "both"
    elif has_it:
        return "it_only"
    else:
        return "cl_only"


from scripts.annotation import convert_records_to_docbin

def records_to_docbin(records: List[Dict[str, Any]], output_path: str) -> None:
    """Exports a list of records to spaCy DocBin format using safe span filtering."""
    convert_records_to_docbin(records, output_path)


def prepare_3way_cv_folds(
    data_path: str = "data/data.jsonl",
    synthetic_paraphrase_path: str = "data/synthetic_paraphrases.jsonl",
    synthetic_llm_path: str = "data/synthetic_llm_generated.jsonl",
    mined_negatives_path: str = "data/review/mined_hard_negatives.jsonl",
    terms_csv_path: str = "data/terms.csv",
    n_splits: int = 5,
    random_seed: int = 42,
    force: bool = False,
) -> Dict[str, Any]:
    """Prepares stratified K-fold partitions across real data and builds 3 condition pools per fold."""
    os.makedirs(CV_DATA_DIR, exist_ok=True)
    meta_path = os.path.join(CV_DATA_DIR, f"cv_metadata_{n_splits}fold.json")
    if os.path.exists(meta_path) and not force:
        logger.info(f"CV folds already exist at {CV_DATA_DIR}. Loading metadata.")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    real_records = extract_all_real_records(data_path)

    # Load synthetic paraphrases
    paraphrase_records = []
    if os.path.exists(synthetic_paraphrase_path):
        with open(synthetic_paraphrase_path, "r", encoding="utf-8") as f:
            paraphrase_records = [json.loads(l) for l in f if l.strip()]
        logger.info(f"Loaded {len(paraphrase_records)} synthetic paraphrases.")

    # Load LLM synthetic records
    synthetic_llm_records = []
    if os.path.exists(synthetic_llm_path):
        with open(synthetic_llm_path, "r", encoding="utf-8") as f:
            synthetic_llm_records = [json.loads(l) for l in f if l.strip()]
        logger.info(f"Loaded {len(synthetic_llm_records)} LLM synthetic records.")

    # Load mined hard negatives
    mined_negatives = []
    if os.path.exists(mined_negatives_path):
        with open(mined_negatives_path, "r", encoding="utf-8") as f:
            mined_negatives = [json.loads(l) for l in f if l.strip()]
        logger.info(f"Loaded {len(mined_negatives)} mined hard negatives.")

    # Load known dictionary terms to filter accidental entities from mined negatives
    known_terms = set()
    if os.path.exists(terms_csv_path):
        import pandas as pd
        df = pd.read_csv(terms_csv_path)
        known_terms = set(df["term"].dropna().str.lower().str.strip())

    strat_labels = [get_doc_stratification_label(r) for r in real_records]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    fold_assignments = [0] * len(real_records)
    for fold_idx, (_, val_indices) in enumerate(skf.split(real_records, strat_labels)):
        for idx in val_indices:
            fold_assignments[idx] = fold_idx

    metadata: Dict[str, Any] = {"n_splits": n_splits, "folds": []}

    for fold_idx in range(n_splits):
        fold_dir = os.path.join(CV_DATA_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        val_real = [r for i, r in enumerate(real_records) if fold_assignments[i] == fold_idx]
        train_real = [r for i, r in enumerate(real_records) if fold_assignments[i] != fold_idx]
        val_texts = {r["text"].strip().lower() for r in val_real}

        # 1. Condition TRTR: Authentic training records only
        pool_trtr = list(train_real)
        random.seed(random_seed + fold_idx)
        random.shuffle(pool_trtr)

        # 2. Condition TRSTR-Paraphrase: train_real + leak-free paraphrases
        clean_paraphrases = [
            r for r in paraphrase_records
            if r.get("text", "").strip().lower() not in val_texts
        ]
        pool_trstr_para = list(train_real) + clean_paraphrases
        random.shuffle(pool_trstr_para)

        # 3. Condition TRSTR-LLM: train_real + leak-free LLM synthetic + curated negatives
        clean_llm = [
            r for r in synthetic_llm_records
            if r.get("text", "").strip().lower() not in val_texts
        ]
        clean_mined = []
        for r in mined_negatives:
            t = r.get("text", "").strip()
            t_lower = t.lower()
            if not t or t_lower in val_texts:
                continue
            has_term = any(
                f" {kt} " in f" {t_lower} " or f" {kt}," in f" {t_lower} " or f" {kt}." in f" {t_lower} "
                for kt in known_terms
            )
            if not has_term:
                clean_mined.append({"text": t, "entities": [], "augmentation_type": "mined_negative"})

        pool_trstr_llm = list(train_real) + clean_llm + clean_mined
        random.shuffle(pool_trstr_llm)

        # Export DocBins
        val_path = os.path.join(fold_dir, "val_real.spacy")
        trtr_path = os.path.join(fold_dir, "train_trtr.spacy")
        para_path = os.path.join(fold_dir, "train_trstr_paraphrase.spacy")
        llm_path = os.path.join(fold_dir, "train_trstr_llm.spacy")

        records_to_docbin(val_real, val_path)
        records_to_docbin(pool_trtr, trtr_path)
        records_to_docbin(pool_trstr_para, para_path)
        records_to_docbin(pool_trstr_llm, llm_path)

        metadata["folds"].append({
            "fold": fold_idx,
            "val_real_count": len(val_real),
            "train_real_count": len(train_real),
            "trstr_para_count": len(pool_trstr_para),
            "trstr_llm_count": len(pool_trstr_llm),
            "val_path": val_path,
            "trtr_path": trtr_path,
            "trstr_para_path": para_path,
            "trstr_llm_path": llm_path,
        })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Generated {n_splits} cross-validation folds. Metadata saved to {meta_path}.")
    return metadata


def train_cv_model(
    condition: str,
    fold_idx: int,
    train_path: str,
    dev_path: str,
    config_path: str = "config_trf.cfg",
    max_steps: int = 1500,
    eval_frequency: int = 50,
    patience: int = 250,
    seed: int = 42,
) -> str:
    """Trains a transformer model for a specific fold and condition with GPU memory safety."""
    output_dir = os.path.join(CV_MODELS_DIR, f"{condition}_f{fold_idx}")
    best_model_path = os.path.join(output_dir, "model-best")

    if os.path.exists(os.path.join(best_model_path, "meta.json")):
        logger.info(f"Found existing trained checkpoint at {best_model_path}. Skipping training.")
        return best_model_path

    # Clean GPU memory before training
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    overrides = {
        "paths.train": train_path,
        "paths.dev": dev_path,
        "system.seed": seed,
        "training.seed": seed,
        "training.max_steps": max_steps,
        "training.eval_frequency": eval_frequency,
        "training.patience": patience,
        "training.optimizer.learn_rate.total_steps": max_steps,
        "components.transformer.model.mixed_precision": True,
        "components.transformer.max_batch_items": 2048,
    }

    logger.info(f"Training {condition.upper()} (Fold {fold_idx}) for up to {max_steps} steps...")
    spacy_train(config_path, output_dir, use_gpu=0, overrides=overrides)
    return best_model_path


def match_entities_strict(
    gold_ents: List[Dict[str, Any]],
    pred_ents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes exact-boundary strict TP, FP, FN with label breakdowns."""
    matched_g = set()
    matched_p = set()

    for p_idx, p in enumerate(pred_ents):
        for g_idx, g in enumerate(gold_ents):
            if g_idx in matched_g:
                continue
            if p["start"] == g["start"] and p["end"] == g["end"] and p["category"] == g["label"]:
                matched_p.add(p_idx)
                matched_g.add(g_idx)
                break

    strict_tp = len(matched_g)
    strict_fp = len(pred_ents) - len(matched_p)
    strict_fn = len(gold_ents) - len(matched_g)

    labels = ["CLERICAL_TERM", "IT_TERM"]
    per_label = {}
    for lbl in labels:
        g_lbl = {i for i, g in enumerate(gold_ents) if g["label"] == lbl}
        p_lbl = {i for i, p in enumerate(pred_ents) if p["category"] == lbl}
        tp = len(matched_g & g_lbl)
        fp = len(p_lbl - matched_p)
        fn = len(g_lbl - matched_g)
        per_label[lbl] = {"tp": tp, "fp": fp, "fn": fn}

    return {
        "overall": {"tp": strict_tp, "fp": strict_fp, "fn": strict_fn},
        "per_label": per_label,
    }


def evaluate_cv_fold(
    pipeline: HybridJournalPipeline,
    val_spacy_path: str,
) -> Dict[str, Any]:
    """Evaluates the pipeline on a pure real validation fold."""
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(val_spacy_path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    strict_tp, strict_fp, strict_fn = 0, 0, 0
    lbl_counts = {"CLERICAL_TERM": {"tp": 0, "fp": 0, "fn": 0}, "IT_TERM": {"tp": 0, "fp": 0, "fn": 0}}

    for doc in docs:
        text = doc.text
        gold_ents = [
            {"start": ent.start_char, "end": ent.end_char, "label": ent.label_, "term": ent.text}
            for ent in doc.ents
        ]
        pred_res = pipeline.predict(text)
        pred_ents = pred_res.get("entities", [])

        m = match_entities_strict(gold_ents, pred_ents)
        strict_tp += m["overall"]["tp"]
        strict_fp += m["overall"]["fp"]
        strict_fn += m["overall"]["fn"]

        for lbl in ["CLERICAL_TERM", "IT_TERM"]:
            lbl_counts[lbl]["tp"] += m["per_label"][lbl]["tp"]
            lbl_counts[lbl]["fp"] += m["per_label"][lbl]["fp"]
            lbl_counts[lbl]["fn"] += m["per_label"][lbl]["fn"]

    def calc_prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)

    ov_p, ov_r, ov_f1 = calc_prf(strict_tp, strict_fp, strict_fn)
    labels_report = {}
    for lbl in ["CLERICAL_TERM", "IT_TERM"]:
        c = lbl_counts[lbl]
        lp, lr, lf = calc_prf(c["tp"], c["fp"], c["fn"])
        labels_report[lbl] = {"precision": lp, "recall": lr, "f1": lf}

    return {
        "overall": {"precision": ov_p, "recall": ov_r, "f1": ov_f1},
        "labels": labels_report,
        "counts": {"tp": strict_tp, "fp": strict_fp, "fn": strict_fn},
    }


def compute_metric_stats(values: List[float]) -> Dict[str, float]:
    """Computes mean and standard deviation for a list of metric percentages."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": round(float(np.mean(values)), 2),
        "std": round(float(np.std(values)), 2),
    }


def run_3way_cross_validation(
    data_path: str = "data/data.jsonl",
    terms_csv_path: str = "data/terms.csv",
    unseen_benchmark_path: str = "data/test/unseen_benchmark.jsonl",
    n_splits: int = 5,
    conditions: Optional[List[str]] = None,
    folds_to_run: Optional[List[int]] = None,
    max_steps: int = 1500,
    eval_frequency: int = 50,
    patience: int = 250,
    force_train: bool = False,
) -> Dict[str, Any]:
    """Executes 3-way K-Fold Cross-Validation: TRTR vs. TRSTR-Paraphrase vs. TRSTR-LLM."""
    if conditions is None:
        conditions = ["trtr", "trstr_paraphrase", "trstr_llm"]

    os.makedirs(CV_RESULTS_DIR, exist_ok=True)
    metadata = prepare_3way_cv_folds(data_path=data_path, n_splits=n_splits)

    if folds_to_run is None:
        folds_to_run = list(range(n_splits))

    unseen_benchmark = None
    if os.path.exists(unseen_benchmark_path):
        unseen_benchmark = load_unseen_benchmark(unseen_benchmark_path)

    path_key_map = {
        "trtr": "trtr_path",
        "trstr_paraphrase": "trstr_para_path",
        "trstr_llm": "trstr_llm_path",
    }

    all_condition_runs: Dict[str, List[Dict[str, Any]]] = {cond: [] for cond in conditions}

    for cond in conditions:
        logger.info(f"\n{'='*75}\nSTARTING CV EVALUATION: {cond.upper()}\n{'='*75}")

        for fold_idx in folds_to_run:
            fold_meta = metadata["folds"][fold_idx]
            res_path = os.path.join(CV_RESULTS_DIR, f"{cond}_f{fold_idx}.json")

            if os.path.exists(res_path) and not force_train:
                logger.info(f"Loaded cached run results from {res_path}")
                with open(res_path, "r", encoding="utf-8") as f:
                    run_metrics = json.load(f)
            else:
                train_path = fold_meta[path_key_map[cond]]
                val_path = fold_meta["val_path"]

                model_path = train_cv_model(
                    condition=cond,
                    fold_idx=fold_idx,
                    train_path=train_path,
                    dev_path=val_path,
                    max_steps=max_steps,
                    eval_frequency=eval_frequency,
                    patience=patience,
                )

                pipeline = HybridJournalPipeline(
                    model_path=model_path,
                    terms_csv_path=terms_csv_path,
                    use_gpu=True,
                )
                run_metrics = evaluate_cv_fold(pipeline, val_path)

                # Secondary probe on isolated unseen benchmark
                if unseen_benchmark:
                    unseen_trf = evaluate_unseen_mode(pipeline, unseen_benchmark, mode="transformer_only")
                    unseen_hyb = evaluate_unseen_mode(pipeline, unseen_benchmark, mode="hybrid")
                    run_metrics["unseen_benchmark"] = {
                        "transformer_recall": unseen_trf["recall_pct"],
                        "transformer_f1": unseen_trf["f1_pct"],
                        "hybrid_recall": unseen_hyb["recall_pct"],
                        "hybrid_f1": unseen_hyb["f1_pct"],
                    }

                run_metrics["condition"] = cond
                run_metrics["fold"] = fold_idx

                with open(res_path, "w", encoding="utf-8") as f:
                    json.dump(run_metrics, f, indent=2)

                # Unload pipeline and clear GPU cache
                del pipeline
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            all_condition_runs[cond].append(run_metrics)

    # Consolidated Statistics
    summary_report: Dict[str, Any] = {}
    for cond in conditions:
        runs = all_condition_runs[cond]
        f1_list = [r["overall"]["f1"] for r in runs]
        p_list = [r["overall"]["precision"] for r in runs]
        r_list = [r["overall"]["recall"] for r in runs]

        it_f1_list = [r["labels"]["IT_TERM"]["f1"] for r in runs]
        cl_f1_list = [r["labels"]["CLERICAL_TERM"]["f1"] for r in runs]

        unseen_trf_r_list = [r.get("unseen_benchmark", {}).get("transformer_recall", 0.0) for r in runs]
        unseen_trf_f1_list = [r.get("unseen_benchmark", {}).get("transformer_f1", 0.0) for r in runs]

        summary_report[cond] = {
            "n_folds": len(runs),
            "overall_f1": compute_metric_stats(f1_list),
            "overall_precision": compute_metric_stats(p_list),
            "overall_recall": compute_metric_stats(r_list),
            "it_term_f1": compute_metric_stats(it_f1_list),
            "clerical_term_f1": compute_metric_stats(cl_f1_list),
            "unseen_trf_recall": compute_metric_stats(unseen_trf_r_list),
            "unseen_trf_f1": compute_metric_stats(unseen_trf_f1_list),
            "runs": runs,
        }

    # Save consolidated summary
    summary_path = os.path.join(CV_DATA_DIR, "cv_3way_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    logger.info(f"Consolidated CV summary saved to: {summary_path}")

    # Display comparison table
    print_cv_comparison_table(summary_report)
    return summary_report


def print_cv_comparison_table(summary: Dict[str, Any]) -> None:
    """Prints a formatted comparative table of cross-validation results across conditions."""
    cond_names = {
        "trtr": "TRTR (Real-Only)",
        "trstr_paraphrase": "TRSTR-Paraphrase",
        "trstr_llm": "TRSTR-LLM",
    }
    active_conds = [c for c in ["trtr", "trstr_paraphrase", "trstr_llm"] if c in summary]

    print("\n" + "=" * 90)
    print("      K-FOLD CROSS-VALIDATION SUMMARY (MEAN ± STD DEV)")
    print("=" * 90)
    header = f"{'METRIC':<36}"
    for c in active_conds:
        header += f" | {cond_names.get(c, c):>16}"
    print(header)
    print("-" * 90)

    rows = [
        ("Held-Out Validation Overall F1", "overall_f1"),
        ("Held-Out Validation Overall Precision", "overall_precision"),
        ("Held-Out Validation Overall Recall", "overall_recall"),
        ("  IT_TERM F1", "it_term_f1"),
        ("  CLERICAL_TERM F1", "clerical_term_f1"),
        ("Unseen Benchmark TRF Recall", "unseen_trf_recall"),
        ("Unseen Benchmark TRF F1", "unseen_trf_f1"),
    ]

    for label, key in rows:
        row_str = f"{label:<36}"
        for c in active_conds:
            stat = summary[c].get(key, {"mean": 0.0, "std": 0.0})
            row_str += f" | {stat['mean']:>6.2f} ± {stat['std']:<4.2f}%"
        print(row_str)

    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-Way Cross-Validation: TRTR vs TRSTR-Para vs TRSTR-LLM")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--conditions", type=str, default="trtr,trstr_paraphrase,trstr_llm", help="Comma-separated conditions")
    parser.add_argument("--folds-to-run", type=str, default=None, help="Comma-separated fold indices to run (e.g. '0,1')")
    parser.add_argument("--max-steps", type=int, default=1500, help="Max fine-tuning steps per fold")
    parser.add_argument("--force", action="store_true", help="Force retrain even if cached result exists")
    args = parser.parse_args()

    cond_list = [c.strip() for c in args.conditions.split(",")]
    folds_idx = [int(f.strip()) for f in args.folds_to_run.split(",")] if args.folds_to_run else None

    run_3way_cross_validation(
        n_splits=args.folds,
        conditions=cond_list,
        folds_to_run=folds_idx,
        max_steps=args.max_steps,
        force_train=args.force,
    )
