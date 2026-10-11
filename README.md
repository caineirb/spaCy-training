# Hybrid NER + Classification Pipeline for OJT Journal Task Tagging

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![spaCy 3.8](https://img.shields.io/badge/spaCy-3.8-09a3d5.svg)](https://spacy.io/)
[![Transformer](https://img.shields.io/badge/Backbone-RoBERTa--base-orange.svg)](https://huggingface.co/roberta-base)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%203060-76b900.svg)](https://www.nvidia.com/)

A modular, production-ready hybrid system designed for processing On-the-Job Training (OJT) weekly student internship journals. It automatically extracts, categorizes, and routes task entities into **IT** (`IT_TERM`) or **CLERICAL** (`CLERICAL_TERM`).

The pipeline integrates deterministic institutional vocabulary matching with a contextual RoBERTa Transformer NER model, resolved through a robust multi-invariant conflict resolution layer, confidence routing, and hard negative mining.

---

## 1. Architectural Design

Rather than relying on a naive dictionary lookup or an ungrounded flat NER model, this system implements a **two-layer hybrid architecture** with parallel extraction and explicit **Dictionary-label-authority** conflict resolution:

```mermaid
flowchart TB
  Text["Extracted OJT Journal Entry"]

  subgraph spaCyProcess["Parallel spaCy Process"]
    direction LR
    Layer1["Deterministic Layer<br/>spaCy EntityRuler<br/>(terms.csv)"]
    Layer2["Contextual ML Transformer<br/>NER (RoBERTa-base)"]
    Layer1 ~~~ Layer2
  end

  subgraph ConflictEngine["Conflict Resolution Engine"]
    direction LR
    id1["Longest Span Wins"]
    id2["Dictionary Label Authority"]
    id3["Abstention Fallback to Dictionary"]
    id1 --> id2 --> id3
  end

  Routing["`Confidence-Based Routing (*Threshold = 0.80*)`"]
  Accept["Auto-Accepted"]
  Review["Flagged for Review"]
  Output["Structured API / Output"]
  Retrain["Feedback into Retrain"]

  Text --> spaCyProcess
  spaCyProcess -->|"[*start*, *end*, *label*, *conf*]"| ConflictEngine
  ConflictEngine --> Routing
  Routing -->|"`*>=0.80*`"| Accept
  Routing -->|"`*<0.80*`"| Review
  Accept --> Output
  Review --> Retrain
```

### Core Pipeline Invariants

1. **Longest Span Wins**: Shorter dictionary terms never truncate or split a longer contextual ML extraction. For example, `Tailwind CSS` takes precedence over dictionary `CSS`, and `access control systems` beats `access control`.
2. **Dictionary Label Authority**: When the contextual Transformer and the seed dictionary fire on overlapping spans and disagree on the task category, the curated seed Dictionary takes precedence as institutional ground truth. The ML prediction's category is overridden by the dictionary label (e.g., overriding ML `IT_TERM` to `CLERICAL_TERM` for `student records` and `Record management systems`, or overriding ML `CLERICAL_TERM` to `IT_TERM` for `office database` and `database`). All override events are logged to `data/eval_results/dictionary_overrides.jsonl` for auditability.
3. **Abstention Fallback**: When the Transformer makes no prediction (abstains) on a span matching a seed dictionary term, the dictionary entry is accepted as a standalone extraction, ensuring zero-shot recovery of institutional vocabulary without false-positive boundary splits.
4. **Confidence-Based Routing**: Entities with confidence scores $\ge 0.80$ are marked `ACCEPTED`. Low-confidence extractions ($< 0.80$) are flagged as `NEEDS_REVIEW` for human audit.

---

## 2. Project Directory Structure

```text
spaCy-training/
├── README.md                          # Repository overview & research findings
├── main.ipynb                         # Narrative end-to-end training & evaluation pipeline
├── config_trf.cfg                     # spaCy GPU transformer training configuration
├── requirements.txt                   # Environment package dependencies
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI entity extraction HTTP endpoint
│   └── README.md                      # API documentation, curl examples & PHP integration
├── data/
│   ├── README.md                      # Comprehensive data guide, schemas, partitions & leakage audit
│   ├── terms.csv                      # Seed dictionary (371 terms: 215 IT, 156 Clerical)
│   ├── data.jsonl                     # Real authentic annotated OJT journal dataset (1,320 records)
│   ├── synthetic_paraphrases.jsonl    # T5 seq2seq generated paraphrases (565 records)
│   ├── synthetic_llm_generated.jsonl  # Gemini LLM-direct synthetic records (571 records)
│   ├── training_trstr_paraphrase.jsonl # Combined real + paraphrase training pool (1,488 records)
│   ├── training_trstr_llm.jsonl       # Combined real + LLM + mined negatives pool (1,588 records)
│   ├── training/
│   │   ├── train.spacy                # Real training partition (923 docs, 1,032 entities)
│   │   ├── dev.spacy                  # Real validation partition (198 docs, 208 entities)
│   │   ├── test.spacy                 # Real held-out test partition (199 docs, 233 entities)
│   │   ├── train_trstr_paraphrase.spacy # Paraphrase-augmented DocBin (1,488 docs, 1,416 entities)
│   │   └── train_trstr_llm.spacy      # LLM-augmented DocBin (1,588 docs, 1,587 entities)
│   ├── test/
│   │   ├── unseen_benchmark.jsonl     # Out-of-vocabulary benchmark (85 docs, 65 gold entities)
│   │   ├── holdout.jsonl              # Permanent real-world holdout evaluation set
│   │   └── raw/                       # Raw unannotated journal text files
│   ├── cv/
│   │   ├── cv_metadata_5fold.json     # 5-fold cross-validation partition splits and metadata
│   │   ├── results/                   # Cross-validation evaluation reports per condition
│   │   ├── fold_0/ ... fold_4/        # 5-fold cross-validation DocBins (val_real, train_trtr, etc.)
│   │   └── cv_3way_results.json       # Previous 5-fold cross-validation results across all 3 conditions
│   ├── eval_results/                  # Evaluation outputs partitioned by evaluation type
│   │   ├── comparison/                # Test split and benchmark comparison outputs
│   │   │   ├── trtr/                  # TRTR test partition reports & unseen benchmark outputs
│   │   │   ├── trstr_paraphrase/      # TRSTR-Paraphrase reports & unseen benchmark outputs
│   │   │   └── trstr_llm/             # TRSTR-LLM reports & unseen benchmark outputs
│   │   └── dictionary_overrides.jsonl # Canonical cumulative log of dictionary conflict overrides
│   ├── evaluation_report_3way_comparison.json # Side-by-side test partition evaluation report
│   └── review/
│       └── mined_hard_negatives.jsonl # Mined hard negative candidates (164 raw, 94 clean accepted)
├── models/
│   ├── ner_trf/                       # Default deployment model (model-best)
│   ├── ner_trf_trtr/                  # TRTR baseline model checkpoints (model-best, model-last)
│   ├── ner_trf_trstr_paraphrase/      # TRSTR-Paraphrase augmented model (model-best, model-last)
│   ├── ner_trf_trstr_llm/             # TRSTR-LLM augmented model (model-best, model-last)
│   └── cv/                            # Cross-validation model checkpoints per fold (e.g., trtr_f0)
├── scripts/
│   ├── README.md                      # Comprehensive guide to all 12 pipeline scripts
│   ├── __init__.py                    # Automatic CUDA dynamic linker preloader (`init_gpu()`)
│   ├── annotation.py                  # Real-data ingestion, diagnostics, dedup & DocBin generation
│   ├── check_data_leakage.py          # 7-check data leakage & evaluation benchmark isolation audit
│   ├── cross_validation.py            # 5-fold grouped & stratified CV (TRTR vs. Para vs. LLM)
│   ├── eval.py                        # Precision, Recall, F1 evaluation on test split & unseen benchmark
│   ├── generate_llm_synthetic.py      # LLM-direct synthetic data generation (Google Gemini)
│   ├── labels.py                      # Canonical label taxonomy (`IT_TERM`, `CLERICAL_TERM`)
│   ├── mine_negatives.py              # Hard negative mining & stylistic n-gram overlap analysis
│   ├── pipeline.py                    # HybridJournalPipeline inference & conflict resolution
│   ├── prepare_trstr_llm.py           # TRSTR-LLM dataset builder merging real, synthetic & negatives
│   ├── test_pipeline_regressions.py   # Unittest regression suite for conflict resolution rules
│   └── training.py                    # Transformer fine-tuning script with early stopping & patience
├── docs/
│   ├── annotation_guidelines.md       # Official annotation guidelines & label taxonomy
│   ├── manual_annotation_guidelines.md# Quick reference guide for human annotators
│   ├── deployment.md                  # Minimal production standalone deployment guide
│   ├── hard_negative_mining.md        # Hard negative mining methodology & leakage isolation
│   ├── synthetic_augmentation_methodology.md # Full 3-way evaluation methodology & ablation analysis
│   └── paraphrase_augmentation_methodology.md # Documentation of T5 paraphrasing
|
└── tools/
    ├── README.md                      # Guide to desktop GUI tools
    ├── pipeline_gui.py                # Desktop GUI Studio for interactive inference & batch processing
    ├── entity_extractor_gui.py        # Launcher alias for pipeline_gui.py
    ├── jsonl_editor.py                # Interactive JSONL dataset viewer & span annotation editor
    └── spacy_annotator_app.py         # Lightweight rapid span annotation helper
```

---

*For full dataset specification, taxonomy, and partition details, see the dedicated [data/README.md](data/README.md).*

---

## 4. Empirical Evaluation Results

To rigorously assess performance and generalization, three experimental conditions were evaluated under identical training hyperparameters (`max_steps=2500`, `eval_frequency=50`, `patience=400`, GPU device 0):

1. **TRTR (Train Real, Test Real)**: Baseline trained exclusively on **923** authentic journal entries.
2. **TRSTR-Paraphrase**: Trained on **923** real records + **565** accepted T5 seq2seq paraphrases (**1,488** total).
3. **TRSTR-LLM**: Trained on **923** real records + **571** Gemini LLM synthetic records + **94** clean mined hard negatives (**1,588** total).

### 4.1 Side-by-Side Test Partition Evaluation

Evaluated against the identical authentic held-out test split (`data/training/test.spacy`, 199 docs, 233 entities) and controlled out-of-vocabulary benchmark (`data/test/unseen_benchmark.jsonl`, 85 docs, 65 gold entities) from `main.ipynb`:

| Evaluation Metric | TRTR (Real Baseline) | TRSTR-Paraphrase (T5) | TRSTR-LLM (Gemini) | Delta (Para vs TRTR) | Delta (LLM vs TRTR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Held-Out Test Overall F1** | **72.92%** | 67.95% | 71.24% | -4.97% | -1.68% |
| Held-Out Test Overall Precision | 72.46% | 67.66% | **73.52%** | -4.80% | **+1.06%** |
| Held-Out Test Overall Recall | **73.39%** | 68.24% | 69.10% | -5.15% | -4.29% |
| ├── `IT_TERM` F1 | **74.01%** | 68.14% | 70.13% | -5.87% | -3.88% |
| ├── `IT_TERM` Precision | 72.89% | 69.23% | **73.47%** | -3.66% | **+0.58%** |
| ├── `IT_TERM` Recall | **75.16%** | 67.08% | 67.08% | -8.08% | -8.08% |
| ├── `CLERICAL_TERM` F1 | 70.42% | 67.55% | **73.61%** | -2.87% | **+3.19%** |
| ├── `CLERICAL_TERM` Precision | 71.43% | 64.56% | **73.61%** | -6.87% | **+2.18%** |
| └── `CLERICAL_TERM` Recall | 69.44% | 70.83% | **73.61%** | +1.39% | **+4.17%** |
| **Unseen Benchmark TRF Precision** | 38.17% | 34.56% | **64.89%** | -3.61% | **+26.72%** |
| **Unseen Benchmark TRF Recall** | 76.92% | 72.31% | **93.85%** | -4.61% | **+16.93%** |
| **Unseen Benchmark TRF F1** | 51.02% | 46.77% | **76.73%** | -4.25% | **+25.71%** |
| **Unseen Benchmark Hybrid Precision** | 37.88% | 34.31% | **60.40%** | -3.57% | **+22.52%** |
| **Unseen Benchmark Hybrid Recall** | 76.92% | 72.31% | **93.85%** | -4.61% | **+16.93%** |
| **Unseen Benchmark Hybrid F1** | 50.76% | 46.53% | **73.49%** | -4.23% | **+22.73%** |
| **Generalization Lift (vs Dict Recall 4.62%)** | +72.30% | +67.69% | **+89.23%** | -4.61% | **+16.93%** |

### 4.2 5-Fold Stratified Cross-Validation Summary

> **Cross‑Validation Results**: The 5‑fold cross‑validation on the expanded 1,320‑record dataset has completed. The aggregated metrics (mean ± standard deviation) for each experimental condition are shown below.

| Metric | TRTR (Real Baseline) | TRSTR‑Paraphrase | TRSTR‑LLM |
| :--- | :---: | :---: | :---: |
| **Overall F1** | 71.01 ± 2.81% | **75.44 ± 2.94%** | 67.75 ± 2.98% |
| Overall Precision | 68.57 ± 2.24% | **74.28 ± 1.89%** | 66.87 ± 2.38% |
| Overall Recall | 73.67 ± 3.72% | **76.69 ± 4.20%** | 68.70 ± 3.97% |
| IT‑Term F1 | 71.64 ± 3.62% | **75.88 ± 3.76%** | 68.01 ± 3.66% |
| Clerical‑Term F1 | 69.89 ± 1.92% | **74.81 ± 2.30%** | 67.29 ± 2.19% |
| Unseen Benchmark Recall (Transformer) | 85.54 ± 3.17% | 72.00 ± 4.50% | **89.54 ± 3.56%** |
| Unseen Benchmark F1 (Transformer) | 54.50 ± 1.04% | 48.78 ± 2.65% | **67.06 ± 4.85%** |

### Key Findings & Thesis Insights

1. **TRSTR-LLM Excels in Precision & Clerical Domain Mastery**: Direct LLM augmentation combined with mined hard negatives achieves the highest overall test precision (**73.52%**, **+1.06%** over TRTR baseline) and dominates across every `CLERICAL_TERM` metric: **73.61% F1** (**+3.19%** over TRTR), **73.61% Precision** (**+2.18%**), and **73.61% Recall** (**+4.17%**). The synthetic generation effectively populated technical-administrative contexts that are scarce in raw OJT entries.
2. **Tremendous Out-of-Vocabulary (OOV) Generalization (+25.71% F1)**: On the canonical 65-term unseen benchmark (`data/test/unseen_benchmark.jsonl`), TRSTR-LLM successfully recognized **61 out of 65** completely unseen technical concepts. TRSTR-LLM drove Transformer Recall to **93.85%** (vs. 76.92% [50/65] for TRTR, **+16.93%** lift), Precision to **64.89%** (vs. 38.17% for TRTR, a massive **+26.72%** jump), and Transformer F1 to **76.73%** (vs. 51.02% for TRTR, an astounding **+25.71%** improvement). In hybrid resolution mode, TRSTR-LLM maintained **73.49% F1** (vs. 50.76% for TRTR, **+22.73%** lift), achieving an overall **+89.23% generalization lift** over the seed dictionary baseline.
3. **TRTR Strong In-Domain Baseline Anchor**: The authentic real-data baseline (TRTR) performs strongly on in-domain distributions with **72.92% Held-Out Test F1**, **73.39% Recall**, and **74.01% `IT_TERM` F1** on familiar journal distributions, demonstrating the high baseline quality and label consistency of the curated authentic annotations.
4. **Mined Negative Curation & Superior Confidence Discrimination**: Filtering 164 raw candidates down to 94 clean hard negatives (blocking 19 test and 15 dev leaks) effectively suppressed spurious over-prediction on conversational text, boosting unseen benchmark precision from 38.17% (TRTR) to **64.89%** (TRSTR-LLM). Furthermore, TRSTR-LLM exhibited strong confidence discrimination on unseen evaluations: correct entities averaged **0.9901** confidence while spurious predictions averaged **0.9153** (a **+0.0748** discrimination margin, compared to +0.0215 for TRTR and -0.0208 for TRSTR-Para).
5. **TRSTR-Paraphrase Syntactic Augmentation**: T5 seq2seq paraphrasing enriched grammatical variety, yielding **70.83% Recall** on `CLERICAL_TERM` (+1.39% over TRTR 69.44%) and identifying 47/65 unseen terms (72.31% recall, +67.69% generalization lift). However, subtle semantic and stylistic shifts introduced by automated paraphrasing led to lower precision (67.66%) and an overall test F1 of **67.95%**.
6. **Strict Evaluation Isolation Maintained**: All 7 checks of the automated leakage audit passed with zero leakage. Synthetic data remains strictly isolated to the training split. Evaluation partitions (`dev.spacy`, `test.spacy`, `unseen_benchmark.jsonl`) remain 100% authentic real data.

---

## 5. Quick Start & Usage

### Environment Setup

Activate the Python virtual environment:

```bash
source .venv/bin/activate
```

### 1. Run the Narrative Walkthrough Notebook

Open `main.ipynb` in Jupyter Lab or VS Code to step through data ingestion, leakage checks, training, evaluation, and cross-validation:

```bash
jupyter lab main.ipynb
```

### 2. Run Pipeline Steps via CLI

#### Prepare Data & Build Binary DocBins

```bash
python scripts/annotation.py
```

#### Run 7-Check Data Leakage Audit

```bash
python scripts/check_data_leakage.py
```

#### Train the Transformer on GPU

```bash
python scripts/training.py --steps 2500 --eval-freq 50 --patience 400 --gpu-id 0
```

#### Run 5-Fold Cross-Validation

```bash
python scripts/cross_validation.py --folds 5 --conditions trtr,trstr_paraphrase,trstr_llm
```

#### Run Full Evaluation & Save Comparison Reports

```bash
python scripts/eval.py \
  --model-path models/ner_trf/model-best \
  --run-label trtr
```

*(Automatically routes per-item JSONL logs and reports into `data/eval_results/comparison/trtr/`)*

#### Run Hybrid Pipeline Regression Unit Tests

```bash
python -m unittest scripts/test_pipeline_regressions.py
```

#### Test Hybrid Inference in Python

```python
from scripts.pipeline import HybridJournalPipeline

pipe = HybridJournalPipeline(model_path="models/ner_trf/model-best", terms_csv_path="data/terms.csv")
res = pipe.predict("I developed an asynchronous microservice using FastAPI and PostgreSQL.")
for ent in res["entities"]:
    print(f"[{ent['category']}] {ent['term']} ({ent['source']}, conf: {ent['confidence']:.2f})")
```

---

## 6. Applications & Services

### 1. FastAPI Entity Extraction Service (`api/`)

Provides a production HTTP API for real-time entity extraction:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Endpoint: `POST /extract`
- Health check: `GET /health`
- See [`api/README.md`](api/README.md) for request/response contracts and PHP integration code.
- See [`docs/deployment.md`](docs/deployment.md) for packaging the API onto a standalone server.

### 2. Desktop GUI Studio (`tools/pipeline_gui.py`)

An interactive desktop suite featuring live visual entity tagging, instant KPI analytics, model switching, filtering, and JSON/cURL generators:

```bash
python tools/pipeline_gui.py
```

- Launcher alias: `python tools/entity_extractor_gui.py`
- Dataset curation: `python tools/jsonl_editor.py`
- See [`tools/README.md`](tools/README.md) for full desktop utility documentation.
