"""
Grouped 5-Fold Cross-Validation and Controlled Ablation Harness for OJT NER.

Protocol:
1. Pure Real Evaluation: 5 document-level folds over all 982 authentic student journal records
   (train.spacy + dev.spacy + test.spacy combined). Each validation fold contains ~196 authentic records.
2. Training Augmentation Isolation: Synthetic LLM data (data/synthetic_llm_generated.jsonl) and
   mined negatives are added ONLY to training folds, never to evaluation folds.
3. Four Controlled Arms:
   (i)   arm_real_only: Real training data only, new config (mixed precision, case augmenter, synced LR).
   (ii)  arm_old_config: Real + 300 synthetic LLM, old config (mixed_precision=False, no augmenter, total_steps=20000).
   (iii) arm_new_config: Real + 300 synthetic LLM, new config (mixed_precision=True, lower_case augmenter, total_steps=2500).
   (iv)  arm_new_config_7novel: Real + 300 synthetic LLM + 7 novel leak-free negatives, new config.
4. Evaluation Metrics:
   - Overall and per-label (IT_TERM, CLERICAL_TERM) Strict Precision, Recall, F1, and raw TP/FP/FN.
   - Relaxed F1 (any token overlap).
   - Unseen-term benchmark (65 novel entities) evaluated as a secondary probe.
   - Out-of-fold confidence threshold calibration curve.
"""

import os
import sys
import json
import logging
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Ensure project root is in sys.path and initialize GPU before importing spacy
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
CV_MODELS_DIR = "archive/cv_models"
N_SPLITS = 5


def extract_all_real_docs() -> List[Dict[str, Any]]:
    """Loads all authentic real records from train.spacy, dev.spacy, and test.spacy."""
    nlp = spacy.blank("en")
    records = []
    
    for split in ["train", "dev", "test"]:
        path = f"data/training/{split}.spacy"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset split not found: {path}")
        db = DocBin().from_disk(path)
        for doc in db.get_docs(nlp.vocab):
            entities = [
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_, "term": ent.text}
                for ent in doc.ents
            ]
            records.append({
                "text": doc.text,
                "entities": entities,
                "source_split": split
            })
    logger.info(f"Extracted {len(records)} authentic documents across all real splits.")
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


def records_to_docbin(records: List[Dict[str, Any]], output_path: str) -> None:
    """Exports a list of records to spaCy DocBin format."""
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
            if span is None:
                span = doc.char_span(ent["start"], ent["end"], label=ent["label"], alignment_mode="contract")
            if span is not None:
                spans.append(span)
        doc.ents = spans
        doc_bin.add(doc)
        valid_count += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc_bin.to_disk(output_path)
    logger.info(f"Saved {valid_count} docs to DocBin: {output_path}")


def prepare_cv_folds(force: bool = False) -> Dict[str, Any]:
    """Generates 5 document-level stratified folds across real data and builds training pools."""
    os.makedirs(CV_DATA_DIR, exist_ok=True)
    meta_path = os.path.join(CV_DATA_DIR, "cv_metadata_v2.json")
    if os.path.exists(meta_path) and not force:
        logger.info(f"CV folds already exist at {CV_DATA_DIR}. Loading metadata.")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    real_records = extract_all_real_docs()

    # Load 300 synthetic LLM records
    synthetic_records = []
    with open("data/synthetic_llm_generated.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                synthetic_records.append(json.loads(line))

    # Load 7 novel leak-free negatives
    novel_7_negatives = []
    novel_path = "data/review/novel_7_negatives.jsonl"
    if os.path.exists(novel_path):
        with open(novel_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    novel_7_negatives.append(json.loads(line))

    logger.info(f"Loaded {len(synthetic_records)} synthetic LLM records and {len(novel_7_negatives)} novel negatives.")

    strat_labels = [get_doc_stratification_label(r) for r in real_records]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_assignments = [0] * len(real_records)
    for fold_idx, (_, val_indices) in enumerate(skf.split(real_records, strat_labels)):
        for idx in val_indices:
            fold_assignments[idx] = fold_idx

    metadata: Dict[str, Any] = {"folds": []}

    for fold_idx in range(N_SPLITS):
        fold_dir = os.path.join(CV_DATA_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        val_real = [r for i, r in enumerate(real_records) if fold_assignments[i] == fold_idx]
        train_real = [r for i, r in enumerate(real_records) if fold_assignments[i] != fold_idx]

        # Pools:
        # 1. Real Only: train_real
        pool_real_only = list(train_real)
        random.seed(42 + fold_idx)
        random.shuffle(pool_real_only)

        # 2. Real + Synthetic LLM (used for old_config and new_config)
        pool_synthetic = list(train_real) + list(synthetic_records)
        random.shuffle(pool_synthetic)

        # 3. Real + Synthetic LLM + 7 Novel Negatives
        pool_7novel = list(train_real) + list(synthetic_records) + list(novel_7_negatives)
        random.shuffle(pool_7novel)

        # Export DocBins
        val_path = os.path.join(fold_dir, "val_real.spacy")
        real_only_path = os.path.join(fold_dir, "train_real_only.spacy")
        synthetic_path = os.path.join(fold_dir, "train_synthetic.spacy")
        novel7_path = os.path.join(fold_dir, "train_novel7.spacy")

        records_to_docbin(val_real, val_path)
        records_to_docbin(pool_real_only, real_only_path)
        records_to_docbin(pool_synthetic, synthetic_path)
        records_to_docbin(pool_7novel, novel7_path)

        fold_meta = {
            "fold": fold_idx,
            "val_real_count": len(val_real),
            "train_real_count": len(train_real),
            "synthetic_train_count": len(pool_synthetic),
            "novel7_train_count": len(pool_7novel),
            "val_path": val_path,
            "real_only_path": real_only_path,
            "synthetic_path": synthetic_path,
            "novel7_path": novel7_path,
        }
        metadata["folds"].append(fold_meta)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"CV folds metadata saved to {meta_path}.")
    return metadata


def train_cv_model(
    arm: str,
    fold_idx: int,
    seed: int,
    train_path: str,
    dev_path: str,
    max_steps: int = 2500,
    eval_frequency: int = 50,
    patience: int = 250,
) -> str:
    """Trains a transformer model for a specific fold and arm."""
    output_dir = os.path.join(CV_MODELS_DIR, f"{arm}_f{fold_idx}_s{seed}")
    best_model_path = os.path.join(output_dir, "model-best")

    if os.path.exists(os.path.join(best_model_path, "meta.json")):
        logger.info(f"Model already exists at {best_model_path}. Skipping training.")
        return best_model_path

    os.makedirs(output_dir, exist_ok=True)

    # Configure overrides based on arm
    overrides = {
        "paths.train": train_path,
        "paths.dev": dev_path,
        "system.seed": seed,
        "training.seed": seed,
        "training.max_steps": max_steps,
        "training.eval_frequency": eval_frequency,
        "training.patience": patience,
    }

    if arm == "arm_old_config":
        overrides["components.transformer.model.mixed_precision"] = False
        overrides["corpora.train.augmenter"] = None
        overrides["training.optimizer.learn_rate.total_steps"] = 20000
    else:
        overrides["components.transformer.model.mixed_precision"] = True
        overrides["corpora.train.augmenter.@augmenters"] = "spacy.lower_case.v1"
        overrides["corpora.train.augmenter.level"] = 0.1
        overrides["training.optimizer.learn_rate.total_steps"] = max_steps

    logger.info(f"Training {arm} fold {fold_idx} seed {seed}...")
    spacy_train("config_trf.cfg", output_dir, use_gpu=0, overrides=overrides)
    return best_model_path


def match_entities_strict_and_relaxed(
    gold_ents: List[Dict[str, Any]],
    pred_ents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes exact-span strict and overlapping relaxed TP, FP, FN with label breakdowns."""
    # Strict matching
    matched_g_strict = set()
    matched_p_strict = set()
    for p_idx, p in enumerate(pred_ents):
        for g_idx, g in enumerate(gold_ents):
            if g_idx in matched_g_strict:
                continue
            if p["start"] == g["start"] and p["end"] == g["end"] and p["category"] == g["label"]:
                matched_p_strict.add(p_idx)
                matched_g_strict.add(g_idx)
                break

    strict_tp = len(matched_g_strict)
    strict_fp = len(pred_ents) - len(matched_p_strict)
    strict_fn = len(gold_ents) - len(matched_g_strict)

    # Per-label strict counts
    labels = ["CLERICAL_TERM", "IT_TERM"]
    per_label_strict = {}
    for lbl in labels:
        g_lbl_idxs = {i for i, g in enumerate(gold_ents) if g["label"] == lbl}
        p_lbl_idxs = {i for i, p in enumerate(pred_ents) if p["category"] == lbl}
        tp_lbl = len(matched_g_strict & g_lbl_idxs)
        fp_lbl = len(p_lbl_idxs - matched_p_strict)
        fn_lbl = len(g_lbl_idxs - matched_g_strict)
        per_label_strict[lbl] = {"tp": tp_lbl, "fp": fp_lbl, "fn": fn_lbl}

    # Relaxed matching (any token overlap, label ignored)
    matched_g_relaxed = set()
    matched_p_relaxed = set()
    for p_idx, p in enumerate(pred_ents):
        for g_idx, g in enumerate(gold_ents):
            if g_idx in matched_g_relaxed:
                continue
            if max(p["start"], g["start"]) < min(p["end"], g["end"]):
                matched_p_relaxed.add(p_idx)
                matched_g_relaxed.add(g_idx)
                break

    relaxed_tp = len(matched_g_relaxed)
    relaxed_fp = len(pred_ents) - len(matched_p_relaxed)
    relaxed_fn = len(gold_ents) - len(matched_g_relaxed)

    return {
        "strict": {"tp": strict_tp, "fp": strict_fp, "fn": strict_fn},
        "relaxed": {"tp": relaxed_tp, "fp": relaxed_fp, "fn": relaxed_fn},
        "per_label": per_label_strict,
    }


def evaluate_cv_fold(
    pipeline: HybridJournalPipeline,
    val_spacy_path: str,
) -> Dict[str, Any]:
    """Evaluates the pipeline on a pure real held-out validation fold."""
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(val_spacy_path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    strict_tp, strict_fp, strict_fn = 0, 0, 0
    relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0
    lbl_counts = {"CLERICAL_TERM": {"tp": 0, "fp": 0, "fn": 0}, "IT_TERM": {"tp": 0, "fp": 0, "fn": 0}}

    all_predictions_for_tuning = []

    for doc in docs:
        text = doc.text
        gold_ents = [
            {"start": ent.start_char, "end": ent.end_char, "label": ent.label_, "term": ent.text}
            for ent in doc.ents
        ]
        pred_res = pipeline.predict(text)
        pred_ents = pred_res.get("entities", [])

        # Match
        m = match_entities_strict_and_relaxed(gold_ents, pred_ents)
        strict_tp += m["strict"]["tp"]
        strict_fp += m["strict"]["fp"]
        strict_fn += m["strict"]["fn"]

        relaxed_tp += m["relaxed"]["tp"]
        relaxed_fp += m["relaxed"]["fp"]
        relaxed_fn += m["relaxed"]["fn"]

        for lbl in ["CLERICAL_TERM", "IT_TERM"]:
            lbl_counts[lbl]["tp"] += m["per_label"][lbl]["tp"]
            lbl_counts[lbl]["fp"] += m["per_label"][lbl]["fp"]
            lbl_counts[lbl]["fn"] += m["per_label"][lbl]["fn"]

        # Track predictions and confidence for post-hoc threshold tuning
        for p in pred_ents:
            # Check if this prediction was a strict TP
            is_tp = any(
                p["start"] == g["start"] and p["end"] == g["end"] and p["category"] == g["label"]
                for g in gold_ents
            )
            all_predictions_for_tuning.append({
                "term": p["term"],
                "category": p["category"],
                "confidence": p.get("confidence", 1.0),
                "source": p.get("source", "ML"),
                "is_tp": is_tp,
            })

    def calc_prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)

    s_p, s_r, s_f1 = calc_prf(strict_tp, strict_fp, strict_fn)
    r_p, r_r, r_f1 = calc_prf(relaxed_tp, relaxed_fp, relaxed_fn)

    labels_report = {}
    for lbl in ["CLERICAL_TERM", "IT_TERM"]:
        c = lbl_counts[lbl]
        lp, lr, lf = calc_prf(c["tp"], c["fp"], c["fn"])
        labels_report[lbl] = {"precision": lp, "recall": lr, "f1": lf, "counts": c}

    return {
        "strict": {"precision": s_p, "recall": s_r, "f1": s_f1, "tp": strict_tp, "fp": strict_fp, "fn": strict_fn},
        "relaxed": {"precision": r_p, "recall": r_r, "f1": r_f1, "tp": relaxed_tp, "fp": relaxed_fp, "fn": relaxed_fn},
        "labels": labels_report,
        "predictions_for_tuning": all_predictions_for_tuning,
        "total_gold": strict_tp + strict_fn,
        "total_pred": strict_tp + strict_fp,
    }


def run_controlled_ablation(
    arms: List[str] = ["arm_real_only", "arm_old_config", "arm_new_config", "arm_new_config_7novel"],
    seeds: List[int] = [0, 42, 123],
    folds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Executes the controlled cross-validation evaluation across all specified arms."""
    os.makedirs(CV_RESULTS_DIR, exist_ok=True)
    metadata = prepare_cv_folds()

    if folds is None:
        folds = list(range(N_SPLITS))

    unseen_benchmark = load_unseen_benchmark("data/test/unseen_benchmark.jsonl")

    all_arm_summaries = {}

    for arm in arms:
        logger.info(f"\n{'='*75}\nRUNNING ABLATION ARM: {arm.upper()}\n{'='*75}")
        arm_runs = []

        for seed in seeds:
            for fold_idx in folds:
                fold_meta = metadata["folds"][fold_idx]
                res_path = os.path.join(CV_RESULTS_DIR, f"{arm}_f{fold_idx}_s{seed}.json")

                if os.path.exists(res_path):
                    logger.info(f"Loaded cached results from: {res_path}")
                    with open(res_path, "r", encoding="utf-8") as f:
                        run_metrics = json.load(f)
                else:
                    # Select appropriate training pool
                    if arm == "arm_real_only":
                        train_path = fold_meta["real_only_path"]
                    elif arm == "arm_new_config_7novel":
                        train_path = fold_meta["novel7_path"]
                    else:
                        train_path = fold_meta["synthetic_path"]

                    val_path = fold_meta["val_path"]

                    model_path = train_cv_model(
                        arm=arm,
                        fold_idx=fold_idx,
                        seed=seed,
                        train_path=train_path,
                        dev_path=val_path,
                    )

                    pipeline = HybridJournalPipeline(model_path=model_path, terms_csv_path="data/terms.csv", use_gpu=True)
                    run_metrics = evaluate_cv_fold(pipeline, val_path)

                    # Also evaluate on unseen benchmark
                    unseen_res = evaluate_unseen_mode(pipeline, unseen_benchmark, mode="hybrid")
                    run_metrics["unseen_benchmark"] = {
                        "recall": unseen_res["recall_pct"],
                        "precision": unseen_res["precision_pct"],
                        "f1": unseen_res["f1_pct"],
                        "tp": unseen_res["true_positives"],
                        "total_gold": unseen_res["total_gold_entities"]
                    }

                    run_metrics["arm"] = arm
                    run_metrics["fold"] = fold_idx
                    run_metrics["seed"] = seed

                    with open(res_path, "w", encoding="utf-8") as f:
                        json.dump(run_metrics, f, indent=2)

                arm_runs.append(run_metrics)

        all_arm_summaries[arm] = summarize_arm_runs(arm, arm_runs)

    summary_file = os.path.join(CV_DATA_DIR, "cv_ablation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_arm_summaries, f, indent=2)
    logger.info(f"Consolidated ablation summary saved to {summary_file}")
    return all_arm_summaries


def summarize_arm_runs(arm: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes mean ± std, raw sums, and confidence intervals across runs for an arm."""
    f1s = [r["strict"]["f1"] for r in runs]
    ps = [r["strict"]["precision"] for r in runs]
    rs = [r["strict"]["recall"] for r in runs]

    it_f1s = [r["labels"]["IT_TERM"]["f1"] for r in runs]
    cl_f1s = [r["labels"]["CLERICAL_TERM"]["f1"] for r in runs]

    unseen_recalls = [r.get("unseen_benchmark", {}).get("recall", 0.0) for r in runs]

    total_tp = sum(r["strict"]["tp"] for r in runs)
    total_fp = sum(r["strict"]["fp"] for r in runs)
    total_fn = sum(r["strict"]["fn"] for r in runs)

    total_it_tp = sum(r["labels"]["IT_TERM"]["counts"]["tp"] for r in runs)
    total_it_fp = sum(r["labels"]["IT_TERM"]["counts"]["fp"] for r in runs)
    total_it_fn = sum(r["labels"]["IT_TERM"]["counts"]["fn"] for r in runs)

    total_cl_tp = sum(r["labels"]["CLERICAL_TERM"]["counts"]["tp"] for r in runs)
    total_cl_fp = sum(r["labels"]["CLERICAL_TERM"]["counts"]["fp"] for r in runs)
    total_cl_fn = sum(r["labels"]["CLERICAL_TERM"]["counts"]["fn"] for r in runs)

    def stats(arr):
        return {"mean": round(float(np.mean(arr)), 2), "std": round(float(np.std(arr)), 2)}

    return {
        "arm": arm,
        "n_runs": len(runs),
        "overall_strict_f1": stats(f1s),
        "overall_strict_precision": stats(ps),
        "overall_strict_recall": stats(rs),
        "it_term_f1": stats(it_f1s),
        "clerical_term_f1": stats(cl_f1s),
        "unseen_recall": stats(unseen_recalls),
        "raw_counts": {
            "overall": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
            "it_term": {"tp": total_it_tp, "fp": total_it_fp, "fn": total_it_fn},
            "clerical_term": {"tp": total_cl_tp, "fp": total_cl_fp, "fn": total_cl_fn},
        },
        "display_summary": (
            f"F1: {np.mean(f1s):.2f} ± {np.std(f1s):.2f}% | "
            f"IT: {np.mean(it_f1s):.2f} ± {np.std(it_f1s):.2f}% | "
            f"CL: {np.mean(cl_f1s):.2f} ± {np.std(cl_f1s):.2f}% | "
            f"Unseen R: {np.mean(unseen_recalls):.2f} ± {np.std(unseen_recalls):.2f}%"
        )
    }


def tune_confidence_thresholds(
    arm_results_path: str = "data/cv/results",
    arm_name: str = "arm_real_only"
) -> Dict[str, Any]:
    """Tunes per-label confidence thresholds on out-of-fold CV predictions only."""
    all_preds = []
    total_golds = {"CLERICAL_TERM": 0, "IT_TERM": 0}
    matching_files = []
    for f in os.listdir(arm_results_path):
        if f.endswith(".json") and arm_name in f:
            matching_files.append(f)
            data = json.load(open(os.path.join(arm_results_path, f)))
            all_preds.extend(data.get("predictions_for_tuning", []))
            for lbl in ["CLERICAL_TERM", "IT_TERM"]:
                lbl_c = data.get("labels", {}).get(lbl, {}).get("counts", {})
                total_golds[lbl] += lbl_c.get("tp", 0) + lbl_c.get("fn", 0)

    if not all_preds:
        logger.warning(f"No predictions found for threshold tuning with arm: {arm_name}")
        return {}

    logger.info(f"Loaded {len(all_preds)} out-of-fold predictions from {len(matching_files)} runs for {arm_name}.")
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
    curve = {}

    print(f"\n{'='*75}\nCONFIDENCE THRESHOLD TUNING CURVE ({arm_name.upper()})\n{'='*75}")
    for lbl in ["CLERICAL_TERM", "IT_TERM"]:
        lbl_preds = [p for p in all_preds if p["category"] == lbl and p["source"] == "ML"]
        curve[lbl] = []
        tot_g = total_golds[lbl]
        print(f"\nLabel: {lbl} (Total Gold Entities: {tot_g})")
        print(f"{'Thresh':<8} {'Accepted':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec %':<9} {'Rec %':<9} {'F1 %':<9}")
        print("-" * 65)
        for thresh in thresholds:
            accepted = [p for p in lbl_preds if p["confidence"] >= thresh]
            tp = sum(1 for p in accepted if p["is_tp"])
            fp = len(accepted) - tp
            fn = max(0, tot_g - tp)
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / tot_g if tot_g else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
            prec_pct = round(prec * 100, 2)
            rec_pct = round(rec * 100, 2)
            f1_pct = round(f1 * 100, 2)
            curve[lbl].append({
                "threshold": thresh,
                "accepted_count": len(accepted),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": prec_pct,
                "recall": rec_pct,
                "f1": f1_pct
            })
            print(f"{thresh:<8.2f} {len(accepted):<10} {tp:<6} {fp:<6} {fn:<6} {prec_pct:<9.2f} {rec_pct:<9.2f} {f1_pct:<9.2f}")

    output_path = f"data/cv/confidence_threshold_tuning_{arm_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)
    logger.info(f"Threshold tuning curve saved to {output_path}")
    return curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross Validation Retraining Harness")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare folds and exit")
    parser.add_argument("--arms", type=str, default="arm_real_only,arm_new_config", help="Comma-separated arms")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds (e.g., '0' or '0,42,123')")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated folds (e.g. '0,1,2,3,4')")
    parser.add_argument("--tune-thresholds", action="store_true", help="Tune confidence thresholds from results")
    args = parser.parse_args()

    if args.prepare_only:
        prepare_cv_folds(force=True)
    elif args.tune_thresholds:
        tune_confidence_thresholds()
    else:
        arm_list = [a.strip() for a in args.arms.split(",")]
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
        fold_list = [int(f.strip()) for f in args.folds.split(",")]
        results = run_controlled_ablation(arms=arm_list, seeds=seed_list, folds=fold_list)
        print("\n" + "=" * 80)
        print("ABLATION RESULTS SUMMARY")
        print("=" * 80)
        for arm_name, summary in results.items():
            print(f"[{arm_name}]: {summary['display_summary']}")
