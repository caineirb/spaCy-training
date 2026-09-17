"""
Evaluation and thesis generalization analysis module.

Computes:
1. Precision, Recall, and F1 per label on the held-out test set (data/training/test.spacy)
2. Statistically robust Unseen-Term Generalization Benchmark (65 unseen entities across IT and Clerical)
3. Quantitative comparison: Pure Dictionary vs. Pure ML vs. Hybrid Pipeline
4. Confidence score distribution and correlation with prediction correctness (Issue 2 calibration metric)
"""

import os
import sys
import json
import math
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from spacy.training import Example

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts
from scripts.annotation import load_terms_dictionary
from scripts.pipeline import HybridJournalPipeline

logger = logging.getLogger("ojt_pipeline.eval")

# Expanded Unseen-Term Benchmark (65 unique entities strictly NOT in data/terms.csv)
UNSEEN_BENCHMARK_SAMPLES = [
    # --- Modern IT Frameworks & Tools (40 unseen entities) ---
    {"text": "I developed a scalable microservice using FastAPI and connected it to our database.", "entities": [{"start": 42, "end": 49, "label": "IT_TERM", "term": "FastAPI"}]},
    {"text": "Migrated our legacy serverless functions to Bun for faster runtime execution.", "entities": [{"start": 44, "end": 47, "label": "IT_TERM", "term": "Bun"}]},
    {"text": "Built an interactive reactive dashboard component using Svelte.", "entities": [{"start": 56, "end": 62, "label": "IT_TERM", "term": "Svelte"}]},
    {"text": "Configured modern responsive web styling using Tailwind CSS.", "entities": [{"start": 47, "end": 59, "label": "IT_TERM", "term": "Tailwind CSS"}]},
    {"text": "Set up authentication and cloud backend storage using Supabase.", "entities": [{"start": 54, "end": 62, "label": "IT_TERM", "term": "Supabase"}]},
    {"text": "Defined database schema models and executed migrations with Prisma.", "entities": [{"start": 60, "end": 66, "label": "IT_TERM", "term": "Prisma"}]},
    {"text": "Implemented caching layers and message queuing services using Redis.", "entities": [{"start": 62, "end": 67, "label": "IT_TERM", "term": "Redis"}]},
    {"text": "Constructed our enterprise backend architecture using NestJS.", "entities": [{"start": 54, "end": 60, "label": "IT_TERM", "term": "NestJS"}]},
    {"text": "Automated cloud infrastructure provisioning using Terraform.", "entities": [{"start": 50, "end": 59, "label": "IT_TERM", "term": "Terraform"}]},
    {"text": "Developed a modern server-rendered portal using Next.js for our campus users.", "entities": [{"start": 48, "end": 55, "label": "IT_TERM", "term": "Next.js"}]},
    {"text": "Implemented universal state management in Nuxt for high SEO performance.", "entities": [{"start": 43, "end": 47, "label": "IT_TERM", "term": "Nuxt"}]},
    {"text": "Refactored the documentation landing page using Astro for static generation.", "entities": [{"start": 48, "end": 53, "label": "IT_TERM", "term": "Astro"}]},
    {"text": "Configured the rapid module bundler using Vite to optimize local build times.", "entities": [{"start": 42, "end": 46, "label": "IT_TERM", "term": "Vite"}]},
    {"text": "Accelerated monorepo build pipelines using Turbopack during continuous integration.", "entities": [{"start": 43, "end": 52, "label": "IT_TERM", "term": "Turbopack"}]},
    {"text": "Streamed analytics and analytical aggregations into ClickHouse for reporting.", "entities": [{"start": 52, "end": 62, "label": "IT_TERM", "term": "ClickHouse"}]},
    {"text": "Configured high-availability distributed data replication in Cassandra.", "entities": [{"start": 61, "end": 70, "label": "IT_TERM", "term": "Cassandra"}]},
    {"text": "Implemented distributed event streaming pipelines with Kafka across microservices.", "entities": [{"start": 55, "end": 60, "label": "IT_TERM", "term": "Kafka"}]},
    {"text": "Handled asynchronous email queues and task workers using RabbitMQ.", "entities": [{"start": 57, "end": 65, "label": "IT_TERM", "term": "RabbitMQ"}]},
    {"text": "Scheduled recurring background computation tasks in Celery with redis brokers.", "entities": [{"start": 52, "end": 58, "label": "IT_TERM", "term": "Celery"}]},
    {"text": "Orchestrated complex data engineering pipelines using Airflow DAGs.", "entities": [{"start": 54, "end": 61, "label": "IT_TERM", "term": "Airflow"}]},
    {"text": "Integrated distributed tracing and performance metrics via OpenTelemetry.", "entities": [{"start": 59, "end": 72, "label": "IT_TERM", "term": "OpenTelemetry"}]},
    {"text": "Designed operational system health dashboards in Grafana for server monitoring.", "entities": [{"start": 49, "end": 56, "label": "IT_TERM", "term": "Grafana"}]},
    {"text": "Configured metric alerting rules and query scraping using Prometheus.", "entities": [{"start": 58, "end": 68, "label": "IT_TERM", "term": "Prometheus"}]},
    {"text": "Packaged Kubernetes deployment charts and configuration values using Helm.", "entities": [{"start": 69, "end": 73, "label": "IT_TERM", "term": "Helm"}]},
    {"text": "Automated GitOps continuous deployment synchronization through ArgoCD.", "entities": [{"start": 63, "end": 69, "label": "IT_TERM", "term": "ArgoCD"}]},
    {"text": "Wrote smart contract verification scripts using Solidity on an Ethereum testnet.", "entities": [{"start": 48, "end": 56, "label": "IT_TERM", "term": "Solidity"}]},
    {"text": "Queried decentralized wallet balances using Web3.js client libraries.", "entities": [{"start": 44, "end": 51, "label": "IT_TERM", "term": "Web3.js"}]},
    {"text": "Enforced strict API runtime request validation using Zod schemas.", "entities": [{"start": 53, "end": 56, "label": "IT_TERM", "term": "Zod"}]},
    {"text": "Implemented end-to-end type-safe client procedures with tRPC.", "entities": [{"start": 56, "end": 60, "label": "IT_TERM", "term": "tRPC"}]},
    {"text": "Automated cross-browser UI regression test suites using Playwright.", "entities": [{"start": 56, "end": 66, "label": "IT_TERM", "term": "Playwright"}]},
    {"text": "Authored automated frontend integration assertions in Cypress.", "entities": [{"start": 54, "end": 61, "label": "IT_TERM", "term": "Cypress"}]},
    {"text": "Constructed an isolated component library showcase with Storybook.", "entities": [{"start": 56, "end": 65, "label": "IT_TERM", "term": "Storybook"}]},
    {"text": "Deployed headless content management instances via Directus.", "entities": [{"start": 51, "end": 59, "label": "IT_TERM", "term": "Directus"}]},
    {"text": "Configured customizable headless REST collections using Strapi.", "entities": [{"start": 56, "end": 62, "label": "IT_TERM", "term": "Strapi"}]},
    {"text": "Implemented instant full-text search indexing with Meilisearch.", "entities": [{"start": 51, "end": 62, "label": "IT_TERM", "term": "Meilisearch"}]},
    {"text": "Engineered real-time document graph synchronization with SurrealDB.", "entities": [{"start": 57, "end": 66, "label": "IT_TERM", "term": "SurrealDB"}]},
    {"text": "Built a self-hosted single-binary backend prototype in PocketBase.", "entities": [{"start": 55, "end": 65, "label": "IT_TERM", "term": "PocketBase"}]},
    {"text": "Executed high-speed TypeScript scripts on the server using Deno.", "entities": [{"start": 59, "end": 63, "label": "IT_TERM", "term": "Deno"}]},
    {"text": "Enforced rigorous data structure parsing models using Pydantic.", "entities": [{"start": 54, "end": 62, "label": "IT_TERM", "term": "Pydantic"}]},
    {"text": "Processed multi-million row tabular datasets rapidly with Polars dataframes.", "entities": [{"start": 58, "end": 64, "label": "IT_TERM", "term": "Polars"}]},

    # --- Institutional & Clerical Workflows (25 unseen entities) ---
    {"text": "Assisted the department head with Curriculum Verification during the semester audit.", "entities": [{"start": 34, "end": 57, "label": "CLERICAL_TERM", "term": "Curriculum Verification"}]},
    {"text": "Organized faculty schedules and facilitated Thesis Defense Scheduling.", "entities": [{"start": 44, "end": 69, "label": "CLERICAL_TERM", "term": "Thesis Defense Scheduling"}]},
    {"text": "Verified institutional compliance folders for Accreditation Auditing.", "entities": [{"start": 46, "end": 68, "label": "CLERICAL_TERM", "term": "Accreditation Auditing"}]},
    {"text": "Logged incoming visitor IDs and managed Visitor Escorting duties.", "entities": [{"start": 40, "end": 57, "label": "CLERICAL_TERM", "term": "Visitor Escorting"}]},
    {"text": "Assisted staff members with daily Biometric Clearance procedures.", "entities": [{"start": 34, "end": 53, "label": "CLERICAL_TERM", "term": "Biometric Clearance"}]},
    {"text": "Sorted physical graduation credentials for Diploma Archiving in the vault.", "entities": [{"start": 43, "end": 60, "label": "CLERICAL_TERM", "term": "Diploma Archiving"}]},
    {"text": "Checked senior requirements and audited files for Graduation Clearance.", "entities": [{"start": 50, "end": 70, "label": "CLERICAL_TERM", "term": "Graduation Clearance"}]},
    {"text": "Compiled faculty academic records as part of Faculty Docketing.", "entities": [{"start": 45, "end": 62, "label": "CLERICAL_TERM", "term": "Faculty Docketing"}]},
    {"text": "Verified official registrar seals for Transcript Notarization packages.", "entities": [{"start": 38, "end": 61, "label": "CLERICAL_TERM", "term": "Transcript Notarization"}]},
    {"text": "Encoded departmental office expenses on the Petty Cash Voucher ledger.", "entities": [{"start": 44, "end": 62, "label": "CLERICAL_TERM", "term": "Petty Cash Voucher"}]},
    {"text": "Prepared itemized office supply lists for Procurement Requisition approval.", "entities": [{"start": 42, "end": 65, "label": "CLERICAL_TERM", "term": "Procurement Requisition"}]},
    {"text": "Assisted legal counsel with notarized Affidavit Processing records.", "entities": [{"start": 38, "end": 58, "label": "CLERICAL_TERM", "term": "Affidavit Processing"}]},
    {"text": "Drafted student order of ceremonies during Commencement Program Drafting.", "entities": [{"start": 43, "end": 72, "label": "CLERICAL_TERM", "term": "Commencement Program Drafting"}]},
    {"text": "Facilitated local community requests for Barangay Certification releases.", "entities": [{"start": 41, "end": 63, "label": "CLERICAL_TERM", "term": "Barangay Certification"}]},
    {"text": "Organized official court summons folders for Subpoena Tracking.", "entities": [{"start": 45, "end": 62, "label": "CLERICAL_TERM", "term": "Subpoena Tracking"}]},
    {"text": "Processed employee leave records during Leave Application Encoding.", "entities": [{"start": 40, "end": 66, "label": "CLERICAL_TERM", "term": "Leave Application Encoding"}]},
    {"text": "Calculated fiscal department line items during Budget Allocation Tabulation.", "entities": [{"start": 47, "end": 75, "label": "CLERICAL_TERM", "term": "Budget Allocation Tabulation"}]},
    {"text": "Audited payroll discrepancies during Salary Differential Auditing.", "entities": [{"start": 37, "end": 65, "label": "CLERICAL_TERM", "term": "Salary Differential Auditing"}]},
    {"text": "Cataloged deed logs and public oaths in Notarial Registry Encoding.", "entities": [{"start": 40, "end": 66, "label": "CLERICAL_TERM", "term": "Notarial Registry Encoding"}]},
    {"text": "Assisted professors with incoming Grade Completion Verification forms.", "entities": [{"start": 34, "end": 63, "label": "CLERICAL_TERM", "term": "Grade Completion Verification"}]},
    {"text": "Distributed financial aid allowances for Scholarship Stipend Distribution.", "entities": [{"start": 42, "end": 74, "label": "CLERICAL_TERM", "term": "Scholarship Stipend Distribution"}]},
    {"text": "Archived retired personnel personnel files for Service Record Archiving.", "entities": [{"start": 47, "end": 71, "label": "CLERICAL_TERM", "term": "Service Record Archiving"}]},
    {"text": "Cross-checked employee tax forms during Tax Withholding Verification.", "entities": [{"start": 40, "end": 68, "label": "CLERICAL_TERM", "term": "Tax Withholding Verification"}]},
    {"text": "Audited warehouse receipt ledgers for Supplies Inventory Balancing.", "entities": [{"start": 38, "end": 66, "label": "CLERICAL_TERM", "term": "Supplies Inventory Balancing"}]},
    {"text": "Reconciled adjunct lecturer timesheets in Honorarium Reconciliation.", "entities": [{"start": 42, "end": 67, "label": "CLERICAL_TERM", "term": "Honorarium Reconciliation"}]},

    # --- Negative Controls (20 sentences with NO entities) ---
    {"text": "Attended the morning standup meeting with the supervisor to discuss daily goals.", "entities": []},
    {"text": "Joined the weekly team retrospective to share progress updates and blockers.", "entities": []},
    {"text": "Took a short lunch break with fellow student interns at the company cafeteria.", "entities": []},
    {"text": "Participated in the company-wide orientation regarding workplace ethics and rules.", "entities": []},
    {"text": "Cleaned the intern workstation and organized desk accessories before logging out.", "entities": []},
    {"text": "Waited for feedback from the team lead before proceeding with the next assignment.", "entities": []},
    {"text": "Helped a colleague locate the designated conference room on the third floor.", "entities": []},
    {"text": "Listened to a presentation given by the department head on strategic objectives.", "entities": []},
    {"text": "Took detailed personal notes during the department town hall assembly.", "entities": []},
    {"text": "Arrived at the office on time and logged in to the biometric attendance machine.", "entities": []},
    {"text": "Discussed project deadlines and milestone expectations during the afternoon huddle.", "entities": []},
    {"text": "Reviewed general company guidelines regarding data privacy and acceptable use.", "entities": []},
    {"text": "Attended the farewell gathering for the departing senior supervisor.", "entities": []},
    {"text": "Submitted the bi-weekly intern time log sheet to the human resource assistant.", "entities": []},
    {"text": "Participated in an open discussion regarding team workflow enhancements.", "entities": []},
    {"text": "Signed the visitor logbook and collected the guest identification badges.", "entities": []},
    {"text": "Checked my work email inbox and responded to routine greeting messages.", "entities": []},
    {"text": "Organized notebook notes and planned personal task priorities for tomorrow morning.", "entities": []},
    {"text": "Had a quick alignment call with the project mentor to review weekly objectives.", "entities": []},
    {"text": "Wiped down the conference room whiteboard and turned off projector monitors.", "entities": []},
]


def evaluate_test_docbin(
    pipeline: HybridJournalPipeline,
    test_spacy_path: str = "data/training/test.spacy"
) -> Dict[str, Any]:
    """Evaluates the hybrid pipeline on the held-out test.spacy DocBin dataset."""
    if not os.path.exists(test_spacy_path):
        raise FileNotFoundError(f"Test dataset not found at {test_spacy_path}")

    nlp = pipeline.nlp
    doc_bin = DocBin().from_disk(test_spacy_path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    scorer = Scorer()
    scored_examples = []

    for gold_doc in docs:
        pred_doc = nlp(gold_doc.text)
        scored_examples.append(Example(pred_doc, gold_doc))

    scores = scorer.score(scored_examples)

    ents_p = scores.get("ents_p", 0.0) or 0.0
    ents_r = scores.get("ents_r", 0.0) or 0.0
    ents_f = scores.get("ents_f", 0.0) or 0.0
    ents_per_type = scores.get("ents_per_type", {})

    report = {
        "overall_precision": round(ents_p * 100, 2),
        "overall_recall": round(ents_r * 100, 2),
        "overall_f1": round(ents_f * 100, 2),
        "total_test_documents": len(docs),
        "labels": {},
    }

    for label, metrics in ents_per_type.items():
        report["labels"][label] = {
            "precision": round((metrics.get("p", 0.0) or 0.0) * 100, 2),
            "recall": round((metrics.get("r", 0.0) or 0.0) * 100, 2),
            "f1": round((metrics.get("f", 0.0) or 0.0) * 100, 2),
        }

    return report


def evaluate_unseen_mode(
    pipeline: HybridJournalPipeline,
    benchmark_samples: List[Dict[str, Any]],
    mode: str = "hybrid",
) -> Dict[str, Any]:
    """Evaluates a single execution mode on the unseen-term benchmark."""
    tp = 0
    fp = 0
    fn = 0
    total_gold_entities = 0

    confidences_correct: List[float] = []
    confidences_incorrect: List[float] = []
    all_confidences: List[float] = []
    detailed_results = []

    for sample in benchmark_samples:
        text = sample["text"]
        gold_ents = sample.get("entities", [])
        total_gold_entities += len(gold_ents)

        pred = pipeline.predict(text, mode=mode)
        pred_ents = pred["entities"]

        gold_set = {(e["term"].lower(), e["label"]) for e in gold_ents}
        pred_set = {(e["term"].lower(), e["category"], e["confidence"]) for e in pred_ents}

        matched_preds = set()

        for g_term, g_lbl in gold_set:
            hit = False
            for p_term, p_lbl, p_conf in pred_set:
                if (p_term == g_term or g_term in p_term or p_term in g_term) and p_lbl == g_lbl:
                    hit = True
                    matched_preds.add((p_term, p_lbl, p_conf))
                    confidences_correct.append(p_conf)
                    all_confidences.append(p_conf)
                    break

            if hit:
                tp += 1
            else:
                fn += 1

        for p_term, p_lbl, p_conf in pred_set:
            if (p_term, p_lbl, p_conf) not in matched_preds:
                fp += 1
                confidences_incorrect.append(p_conf)
                all_confidences.append(p_conf)

        detailed_results.append({
            "text": text,
            "gold": [e["term"] for e in gold_ents],
            "pred": [e["term"] for e in pred_ents],
            "conf": [e["confidence"] for e in pred_ents],
        })

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    mean_conf_correct = round(sum(confidences_correct) / len(confidences_correct), 4) if confidences_correct else 0.0
    mean_conf_incorrect = round(sum(confidences_incorrect) / len(confidences_incorrect), 4) if confidences_incorrect else 0.0
    conf_delta = round(mean_conf_correct - mean_conf_incorrect, 4)

    return {
        "mode": mode,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gold_entities": total_gold_entities,
        "precision_pct": round(prec * 100, 2),
        "recall_pct": round(rec * 100, 2),
        "f1_pct": round(f1 * 100, 2),
        "correctly_identified_raw": f"{tp}/{total_gold_entities}",
        "confidence_calibration": {
            "mean_confidence_correct_entities": mean_conf_correct,
            "mean_confidence_spurious_entities": mean_conf_incorrect,
            "confidence_discrimination_delta": conf_delta,
        },
        "sample_preview": detailed_results[:8],
    }


def evaluate_unseen_generalization(
    pipeline: HybridJournalPipeline,
    benchmark_samples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluates pipeline performance on the unseen-term benchmark across 3 explicit modes:
    1. Transformer-only: EntityRuler disabled/bypassed entirely.
    2. EntityRuler-only: Dictionary matching only, no Transformer NER.
    3. Hybrid: Combined EntityRuler + Transformer NER pipeline.
    
    Reports precision, recall, and F1 separately for each mode without merging.
    """
    if benchmark_samples is None:
        benchmark_samples = UNSEEN_BENCHMARK_SAMPLES

    logger.info("Evaluating unseen benchmark: Mode 1/3 (Transformer-only)...")
    trf_metrics = evaluate_unseen_mode(pipeline, benchmark_samples, mode="transformer_only")

    logger.info("Evaluating unseen benchmark: Mode 2/3 (EntityRuler-only)...")
    ruler_metrics = evaluate_unseen_mode(pipeline, benchmark_samples, mode="entity_ruler_only")

    logger.info("Evaluating unseen benchmark: Mode 3/3 (Hybrid)...")
    hybrid_metrics = evaluate_unseen_mode(pipeline, benchmark_samples, mode="hybrid")

    return {
        "benchmark_sample_size": len(benchmark_samples),
        "total_unseen_entities": hybrid_metrics["total_gold_entities"],
        "total_unseen_benchmark_entities": hybrid_metrics["total_gold_entities"],
        "modes": {
            "transformer_only": trf_metrics,
            "entity_ruler_only": ruler_metrics,
            "hybrid": hybrid_metrics,
        },
        # Top-level direct comparisons across the 3 modes:
        "transformer_only_precision_pct": trf_metrics["precision_pct"],
        "transformer_only_recall_pct": trf_metrics["recall_pct"],
        "transformer_only_f1_pct": trf_metrics["f1_pct"],
        "transformer_only_identified_raw": trf_metrics["correctly_identified_raw"],

        "entity_ruler_only_precision_pct": ruler_metrics["precision_pct"],
        "entity_ruler_only_recall_pct": ruler_metrics["recall_pct"],
        "entity_ruler_only_f1_pct": ruler_metrics["f1_pct"],
        "entity_ruler_only_identified_raw": ruler_metrics["correctly_identified_raw"],

        "hybrid_precision_pct": hybrid_metrics["precision_pct"],
        "hybrid_recall_pct": hybrid_metrics["recall_pct"],
        "hybrid_f1_pct": hybrid_metrics["f1_pct"],
        "hybrid_identified_raw": hybrid_metrics["correctly_identified_raw"],

        # Backward-compatible fields
        "correctly_identified_raw": hybrid_metrics["correctly_identified_raw"],
        "dictionary_recall": f"{ruler_metrics['true_positives']}/{ruler_metrics['total_gold_entities']} ({ruler_metrics['recall_pct']}%)",
        "dictionary_recall_pct": ruler_metrics["recall_pct"],
        "generalization_lift": f"+{round(hybrid_metrics['recall_pct'] - ruler_metrics['recall_pct'], 2)}%",
        "generalization_lift_recall": f"+{round(hybrid_metrics['recall_pct'] - ruler_metrics['recall_pct'], 2)}%",
        "confidence_calibration": hybrid_metrics["confidence_calibration"],
        "sample_preview": hybrid_metrics["sample_preview"],
    }


def evaluate_real_holdout(
    pipeline: HybridJournalPipeline,
    holdout_jsonl_path: str = "data/test/holdout.jsonl",
    terms_csv_path: str = "data/terms.csv",
    annotations_jsonl_path: str = "data/reviewed/annotations.jsonl",
) -> Dict[str, Any]:
    """Evaluates the hybrid pipeline against the permanent real-world holdout dataset.
    
    Skips gracefully if the file does not exist or is empty.
    Computes precision, recall, and F1 partitioned by:
    1. 'seen' terms (terms known from terms.csv / training annotations) in real context
    2. 'unseen' terms (novel terms) in real context
    3. Combined overall holdout metrics
    """
    if not os.path.exists(holdout_jsonl_path) or os.path.getsize(holdout_jsonl_path) == 0:
        logger.info(f"Real-world holdout dataset at '{holdout_jsonl_path}' is empty or not yet populated. Skipping.")
        return {
            "status": "SKIPPED",
            "message": f"Holdout dataset '{holdout_jsonl_path}' is empty or not yet populated. Populate with gold examples to benchmark.",
            "total_records": 0,
        }

    records = []
    with open(holdout_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        logger.info(f"No valid JSON records found in '{holdout_jsonl_path}'. Skipping.")
        return {
            "status": "SKIPPED",
            "message": f"No valid JSON records found in '{holdout_jsonl_path}'.",
            "total_records": 0,
        }

    from scripts.build_holdout import load_known_training_terms, suggest_term_status
    known_terms = load_known_training_terms(
        terms_csv_path=terms_csv_path,
        annotations_jsonl_path=annotations_jsonl_path,
    )

    seen_tp, seen_fp, seen_fn = 0, 0, 0
    unseen_tp, unseen_fp, unseen_fn = 0, 0, 0
    total_gold_seen, total_gold_unseen = 0, 0
    false_positives: List[Dict[str, Any]] = []

    for rec in records:
        text = rec["text"]
        gold_ents = rec.get("entities", [])
        pred = pipeline.predict(text)
        pred_ents = pred.get("entities", [])

        gold_items = []
        for g in gold_ents:
            g_term = str(g.get("term", text[g["start"]:g["end"]])).strip()
            g_status = g.get("term_status")
            if not g_status or g_status not in ["seen", "unseen"]:
                g_status = suggest_term_status(g_term, known_terms)
            gold_items.append({
                "term": g_term.lower(),
                "label": g["label"],
                "status": g_status,
                "start": g.get("start"),
                "end": g.get("end"),
            })
            if g_status == "seen":
                total_gold_seen += 1
            else:
                total_gold_unseen += 1

        matched_preds = set()

        for g in gold_items:
            hit = False
            for p_idx, p in enumerate(pred_ents):
                p_term = p["term"].lower()
                p_label = p["category"]
                span_match = False
                if g["start"] is not None and g["end"] is not None:
                    span_match = max(g["start"], p["start"]) < min(g["end"], p["end"])
                term_match = (p_term == g["term"] or g["term"] in p_term or p_term in g["term"])

                if (span_match or term_match) and p_label == g["label"]:
                    hit = True
                    matched_preds.add(p_idx)
                    if g["status"] == "seen":
                        seen_tp += 1
                    else:
                        unseen_tp += 1
                    break

            if not hit:
                if g["status"] == "seen":
                    seen_fn += 1
                else:
                    unseen_fn += 1

        for p_idx, p in enumerate(pred_ents):
            if p_idx not in matched_preds:
                p_status = suggest_term_status(p["term"], known_terms)
                if p_status == "seen":
                    seen_fp += 1
                else:
                    unseen_fp += 1
                false_positives.append({
                    "term": p["term"],
                    "label": p["category"],
                    "confidence": p["confidence"],
                    "source": p["source"],
                    "inferred_status": p_status,
                    "sentence_snippet": text[:80] + ("..." if len(text) > 80 else ""),
                })

    def calc_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
        p = round((tp / (tp + fp)) * 100, 2) if (tp + fp) > 0 else 0.0
        r = round((tp / (tp + fn)) * 100, 2) if (tp + fn) > 0 else 0.0
        f1 = round((2 * p * r) / (p + r), 2) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1}

    seen_metrics = calc_metrics(seen_tp, seen_fp, seen_fn)
    unseen_metrics = calc_metrics(unseen_tp, unseen_fp, unseen_fn)
    total_tp = seen_tp + unseen_tp
    total_fp = seen_fp + unseen_fp
    total_fn = seen_fn + unseen_fn
    overall_metrics = calc_metrics(total_tp, total_fp, total_fn)

    return {
        "status": "EVALUATED",
        "total_records": len(records),
        "total_gold_entities": total_gold_seen + total_gold_unseen,
        "seen_terms_evaluation": {
            "total_gold_seen_entities": total_gold_seen,
            "true_positives": seen_tp,
            "false_positives": seen_fp,
            "false_negatives": seen_fn,
            "precision": seen_metrics["precision"],
            "recall": seen_metrics["recall"],
            "f1": seen_metrics["f1"],
        },
        "unseen_terms_evaluation": {
            "total_gold_unseen_entities": total_gold_unseen,
            "true_positives": unseen_tp,
            "false_positives": unseen_fp,
            "false_negatives": unseen_fn,
            "precision": unseen_metrics["precision"],
            "recall": unseen_metrics["recall"],
            "f1": unseen_metrics["f1"],
        },
        "overall_real_holdout": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": overall_metrics["precision"],
            "recall": overall_metrics["recall"],
            "f1": overall_metrics["f1"],
        },
        "false_positives_sample": false_positives[:10],
    }


def generate_full_evaluation_report(
    model_path: str = "models/ner_trf/model-best",
    terms_csv_path: str = "data/terms.csv",
    test_spacy_path: str = "data/training/test.spacy",
    holdout_jsonl_path: str = "data/test/holdout.jsonl",
    output_report_json: str = "data/evaluation_report.json",
) -> Dict[str, Any]:
    """Generates comprehensive evaluation and exports a thesis-ready summary report."""
    pipeline = HybridJournalPipeline(model_path=model_path, terms_csv_path=terms_csv_path)

    test_metrics = evaluate_test_docbin(pipeline, test_spacy_path=test_spacy_path)
    unseen_metrics = evaluate_unseen_generalization(pipeline)
    real_holdout_metrics = evaluate_real_holdout(
        pipeline,
        holdout_jsonl_path=holdout_jsonl_path,
        terms_csv_path=terms_csv_path,
    )

    full_report = {
        "held_out_test_set": test_metrics,
        "unseen_term_generalization_experiment": unseen_metrics,
        "real_world_holdout_evaluation": real_holdout_metrics,
    }

    os.makedirs(os.path.dirname(output_report_json), exist_ok=True)
    with open(output_report_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Full evaluation report generated and saved to {output_report_json}")
    return full_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = generate_full_evaluation_report()
    print("\n" + "=" * 70)
    print("REMEDIATION EVALUATION REPORT")
    print("=" * 70)
    print(json.dumps(report, indent=2))
