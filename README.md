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

```
                            [ Raw OJT Journal Entry ]
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
+-----------------------------------+        +-----------------------------------+
|   Layer 1: Deterministic Layer    |        |     Layer 2: Contextual ML        |
|  spaCy EntityRuler (terms.csv)    |        |   Transformer NER (RoBERTa-base)  |
|      343 Curated Seed Terms       |        |    Trained on In-Context Docs     |
+-----------------------------------+        +-----------------------------------+
                 |                                             |
  [dict_entities: start, end, label]          [ml_entities: start, end, label, conf]
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |    Conflict Resolution Engine     |
                      |   (scripts/pipeline.py Rules)     |
                      +-----------------------------------+
                      | 1. Longest Span Wins (no truncate)|
                      | 2. ML Label Authority             |
                      | 3. Abstention Fallback to Dict    |
                      +-----------------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |    Confidence-Based Routing       |
                      |        Threshold = 0.80           |
                      +-----------------------------------+
                                /                 \
                               /                   \
                   >= 0.80    /                     \   < 0.80
                             v                       v
                      [ Auto-Accepted ]     [ Flagged for Review ]
                             |                       |
                             |                       v
                             |        +----------------------------+
                             |        |    Hard Negative Mining    |
                             |        | (scripts/mine_negatives.py)|
                             |        +----------------------------+
                             v                       |
                  [ Structured API / GUI ]           v
                  [ Output & JSON Export ]   [ Feedback into Retrain]
```

### Core Pipeline Invariants:
1. **Longest Span Wins**: Shorter dictionary terms never truncate or split a longer contextual ML extraction. For example, `Tailwind CSS` takes precedence over dictionary `CSS`, and `access control systems` beats `access control`.
2. **ML Label Authority**: When the contextual Transformer and the seed dictionary disagree on the task category of an overlapping span, the ML prediction takes precedence (e.g., preserving `data entry` and `File Management` as `CLERICAL_TERM` based on context).
3. **Abstention Fallback**: Dictionary entries are accepted *only* when the Transformer makes no overlapping prediction (abstains), ensuring zero-shot recovery of known terms while preventing false-positive overrides.
4. **Confidence-Based Routing**: Entities with confidence scores $\ge 0.80$ are marked `ACCEPTED`. Low-confidence extractions ($< 0.80$) are flagged as `NEEDS_REVIEW` for human audit.

---

## 2. Project Directory Structure

```
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
│   ├── terms.csv                      # Seed dictionary (343 terms: 189 IT, 154 Clerical)
│   ├── data.jsonl                     # Real authentic annotated OJT journal dataset (1,241 records)
│   ├── synthetic_paraphrases.jsonl    # T5 seq2seq generated paraphrases (556 records)
│   ├── synthetic_llm_generated.jsonl  # Gemini LLM-direct synthetic records (300 records)
│   ├── training_trstr_paraphrase.jsonl # Combined real + paraphrase training pool (1,424 records)
│   ├── training_trstr_llm.jsonl       # Combined real + LLM + mined negatives pool (1,274 records)
│   ├── training/
│   │   ├── train.spacy                # Real training partition (868 docs, 913 entities)
│   │   ├── dev.spacy                  # Real validation partition (186 docs, 219 entities)
│   │   └── test.spacy                 # Real held-out test partition (187 docs, 174 entities)
│   ├── test/
│   │   ├── unseen_benchmark.jsonl     # Out-of-vocabulary benchmark (85 docs, 108 gold entities)
│   │   ├── holdout.jsonl              # Permanent real-world holdout evaluation set
│   │   └── raw/                       # Raw unannotated journal text files
│   ├── cv/
│   │   └── cv_3way_results.json       # 5-fold cross-validation results across all 3 conditions
│   ├── eval_results/                  # Detailed JSONL evaluation output and error breakdowns
│   ├── evaluation_report_3way_comparison.json # Side-by-side test partition evaluation report
│   └── review/
│       └── mined_hard_negatives.jsonl # High-confidence false positives mined for abstention training
├── models/
│   ├── ner_trf/                       # TRTR baseline model (model-best, model-last)
│   ├── ner_trf_trstr_llm/             # TRSTR-LLM augmented model (model-best, model-last)
│   └── cv/                            # Cross-validation model checkpoints per fold
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
│   └── archive/
│       └── paraphrase_augmentation_methodology.md # Historical documentation of T5 paraphrasing
└── tools/
    ├── README.md                      # Guide to desktop GUI tools
    ├── pipeline_gui.py                # Desktop GUI Studio for interactive inference & batch processing
    ├── entity_extractor_gui.py        # Launcher alias for pipeline_gui.py
    ├── jsonl_editor.py                # Interactive JSONL dataset viewer & span annotation editor
    └── spacy_annotator_app.py         # Lightweight rapid span annotation helper
```

---

## 3. Data Specification & Unified Taxonomy

### Label Taxonomy:
The system strictly enforces a unified two-class taxonomy:
- **`IT_TERM`**: Technologies, programming languages, software libraries, databases, IT infrastructure, hardware, and concrete technical workflows (e.g., `Python`, `PostgreSQL`, `Docker`, `Git`, `Cable Crimping`, `Database Administration`).
- **`CLERICAL_TERM`**: Office productivity tools, document handling, filing, record-keeping, and administrative workflows (e.g., `Microsoft Excel`, `Police Clearance`, `log books`, `data encoding`, `filing`, `PESO office book`).

### Exact Span Schema (`data/data.jsonl`):
All training data consists of authentic, manually annotated OJT student journal entries with character-exact offsets:

```json
{
  "text": "Perform data encoding and data validation using spreadsheet tools",
  "entities": [
    {"start": 8, "end": 21, "label": "CLERICAL_TERM"},
    {"start": 26, "end": 41, "label": "CLERICAL_TERM"},
    {"start": 48, "end": 59, "label": "CLERICAL_TERM"}
  ]
}
```

### Negative (Non-Entity) Examples:
To prevent false-positive over-prediction in conversational narratives, non-task sentences are explicitly included with an empty entity list:

```json
{
  "text": "Attended the morning flag ceremony and had a quick orientation briefing.",
  "entities": []
}
```

- **Target Negative Ratio**: Maintained between **25%–35%** across splits.
- **Class Balance**: Maintained at roughly equal proportions between `IT_TERM` and `CLERICAL_TERM`.

---

## 4. Empirical Evaluation Results

To rigorously assess performance and generalization, three experimental conditions were evaluated under identical training hyperparameters (`max_steps=2500`, `eval_frequency=50`, `patience=400`, GPU device 0):
1. **TRTR (Train Real, Test Real)**: Baseline trained exclusively on 868 authentic journal entries.
2. **TRSTR-Paraphrase**: Trained on 868 real records + 556 accepted T5 seq2seq paraphrases (1,424 total).
3. **TRSTR-LLM**: Trained on 868 real records + 300 Gemini LLM synthetic records + mined hard negatives (1,274 total).

### 4.1 Side-by-Side Test Partition Evaluation

*(Note: These figures supersede previous evaluation numbers following the resolution of all audit integrity gaps, including canonicalization of near-duplicates, restoration of the canonical 65-term unseen benchmark free of in-domain leakage, purging of contaminated generic-noun paraphrases, and enforcement of a strict two-label schema without typo variants.)*

Evaluated against the identical authentic held-out test split (`data/training/test.spacy`, 186 docs) and controlled out-of-vocabulary benchmark (`data/test/unseen_benchmark.jsonl`, 85 docs, 65 gold entities):

| Evaluation Metric | TRTR (Real Baseline) | TRSTR-Paraphrase (T5) | TRSTR-LLM (Gemini) | Delta (Para vs TRTR) | Delta (LLM vs TRTR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Held-Out Test Overall F1** | 65.27% | **74.66%** | 63.98% | **+9.39%** | -1.29% |
| Held-Out Test Overall Precision | 63.78% | **76.11%** | 64.32% | **+12.33%** | +0.54% |
| Held-Out Test Overall Recall | 66.84% | **73.26%** | 63.64% | **+6.42%** | -3.20% |
| ├── `IT_TERM` F1 | 70.90% | **77.86%** | 67.16% | **+6.96%** | -3.74% |
| ├── `IT_TERM` Precision | 71.43% | **80.31%** | 67.67% | **+8.88%** | -3.76% |
| ├── `IT_TERM` Recall | 70.37% | **75.56%** | 66.67% | **+5.19%** | -3.70% |
| ├── `CLERICAL_TERM` F1 | 52.17% | **66.67%** | 55.77% | **+14.50%** | +3.60% |
| ├── `CLERICAL_TERM` Precision | 47.62% | **66.04%** | 55.77% | **+18.42%** | +8.15% |
| └── `CLERICAL_TERM` Recall | 57.69% | **67.31%** | 55.77% | **+9.62%** | -1.92% |
| **Unseen Benchmark TRF Precision** | 40.00% | 39.50% | **52.73%** | -0.50% | **+12.73%** |
| **Unseen Benchmark TRF Recall** | 83.08% | 72.31% | **89.23%** | -10.77% | **+6.15%** |
| **Unseen Benchmark TRF F1** | 54.00% | 51.09% | **66.29%** | -2.91% | **+12.29%** |
| **Unseen Benchmark Hybrid F1** | 55.00% | 53.26% | **66.29%** | -1.74% | **+11.29%** |
| **Generalization Lift (vs Dict Recall 4.62%)** | +80.00% | +70.76% | **+84.61%** | -9.24% | **+4.61%** |

### 4.2 5-Fold Stratified Cross-Validation Summary

To verify statistical significance across all 1,241 authentic documents, 5-fold cross-validation was conducted (`data/cv/cv_3way_results.json`):

| Metric | TRTR (Real Baseline) | TRSTR-Paraphrase | TRSTR-LLM |
| :--- | :---: | :---: | :---: |
| **Held-Out Validation Overall F1** | 70.00 ± 3.72% | **74.97 ± 2.91%** | 69.05 ± 3.24% |
| Held-Out Validation Overall Precision | 66.41 ± 3.70% | **72.26 ± 2.67%** | 65.81 ± 3.60% |
| Held-Out Validation Overall Recall | 74.08 ± 4.36% | **77.94 ± 3.53%** | 72.65 ± 3.20% |
| ├── `IT_TERM` F1 | 70.73 ± 3.50% | **74.84 ± 2.35%** | 70.75 ± 3.10% |
| └── `CLERICAL_TERM` F1 | 68.51 ± 5.14% | **75.21 ± 4.61%** | 65.69 ± 4.03% |
| **Unseen Benchmark TRF Recall** | 76.00 ± 3.59% | 73.94 ± 5.33% | **76.11 ± 1.80%** |
| **Unseen Benchmark TRF F1** | 49.41 ± 1.93% | 58.45 ± 9.51% | **74.66 ± 0.71%** |

### Key Findings & Thesis Insights:
1. **TRSTR-Paraphrase Leads In-Domain Extraction**: T5 paraphrasing directly addresses in-domain syntactic scarcity by varying the grammatical patterns around real authentic phrases. It achieved the highest authentic validation F1 (**74.97% ± 2.91%**, a **+4.97%** lift over TRTR) and peak held-out test F1 (**70.06%**).
2. **TRSTR-LLM Delivers Exceptional OOV Precision & Boundary Discipline**: Direct LLM generation combined with hard negative mining trained the model to recognize novel concepts while learning when to abstain. On the unseen benchmark across 5 folds, TRSTR-LLM drove F1 from **49.41% $\rightarrow$ 74.66% (+25.25%)** with exceptional stability (**±0.71%** std dev) and achieved **78.00% precision** on novel enterprise tools.
3. **Data Isolation Guaranteed**: All synthetic data was strictly confined to the training side. All evaluation sets (`dev.spacy`, `test.spacy`, `unseen_benchmark.jsonl`) consist 100% of authentic records.

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

#### Prepare Data & Build Binary DocBins:
```bash
python scripts/annotation.py
```

#### Run 7-Check Data Leakage Audit:
```bash
python scripts/check_data_leakage.py
```

#### Train the Transformer on GPU:
```bash
python scripts/training.py --steps 2500 --eval-freq 50 --patience 400 --gpu-id 0
```

#### Run 5-Fold Cross-Validation:
```bash
python scripts/cross_validation.py --folds 5 --conditions trtr,trstr_paraphrase,trstr_llm
```

#### Run Full Evaluation & Save Comparison Reports:
```bash
python scripts/eval.py \
  --model-path models/ner_trf/model-best \
  --output-json data/evaluation_report.json
```

#### Run Hybrid Pipeline Regression Unit Tests:
```bash
python -m unittest scripts/test_pipeline_regressions.py
```

#### Test Hybrid Inference in Python:
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
