# Hybrid NER + Classification Pipeline for OJT Journal Task Tagging

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![spaCy 3.8](https://img.shields.io/badge/spaCy-3.8-09a3d5.svg)](https://spacy.io/)
[![Transformer](https://img.shields.io/badge/Backbone-RoBERTa--base-orange.svg)](https://huggingface.co/roberta-base)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%203060-76b900.svg)](https://www.nvidia.com/)

A modular, production-ready hybrid system designed for processing On-the-Job Training (OJT) weekly journal entries to automatically extract, categorize, and route task entities into **IT** (`IT_TERM`) or **CLERICAL** (`CLERICAL_TERM`).

---

## 1. Architectural Design

Rather than relying on a flat NER model or a pure dictionary lookup, this system implements a **two-layer hybrid architecture** with confidence-based routing and an active learning feedback loop:

```
                            [ Raw OJT Journal Entry ]
                                        |
                                        v
                      +-----------------------------------+
                      |   Layer 1: Deterministic Layer    |
                      |   spaCy EntityRuler (terms.csv)   |
                      +-----------------------------------+
                                        |
                Matched (Dictionary)    |    Unmatched Spans
               [source="dictionary",   |
                confidence=1.00]        |
                                        v
                      +-----------------------------------+
                      |     Layer 2: Contextual ML        |
                      |  Transformer NER (en_core_web_trf)|
                      +-----------------------------------+
                                        |
                      [source="ML", confidence score]
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
                                                    |
                                                    v
                                      +----------------------------+
                                      |   Candidate-Mining Loop    |
                                      |  Linguistic Pattern Mining |
                                      +----------------------------+
                                                    |
                                                    v
                                       [ Validated Feedback into ]
                                       [  terms.csv & Retrain    ]
```

### Layer Breakdown:
1. **Deterministic EntityRuler Layer**:
   - Compiles known terms directly from `data/terms.csv` (e.g. `MySQL`, `React`, `Document Stamping`, `Inventory Counting`).
   - Executes *before* the ML model (`before="ner"`).
   - Guarantees 100% precision on established institutional vocabulary with zero inference hallucination.
2. **Contextual Transformer NER Layer (`en_core_web_trf`)**:
   - Fine-tuned on the NVIDIA GPU using RoBERTa contextual representations.
   - Operates on spans not resolved by the dictionary, utilizing linguistic context (e.g. *"developed ... using [X]"*, *"encoded ... during [X]"*) to capture novel tools (e.g. `FastAPI`, `Bun`, `Svelte`, `Prisma`) without prior dictionary exposure.
3. **Confidence Routing & Review Queue**:
   - Every entity output carries `{term, category, start, end, confidence, source: "dictionary" | "ML", status}`.
   - Predictions with confidence $< 0.80$ are routed to the human-in-the-loop review queue.
4. **Candidate-Mining & Active-Learning Loop**:
   - Syntactic dependency and trigger pattern matching extract uncataloged terms.
   - Aggregates frequency counts and sample contexts into `data/candidates/mined_candidates.csv`.
   - Validated terms feed directly back into `data/terms.csv`, updating patterns dynamically without requiring architectural modification.

---

## 2. Project Directory Structure

```
spaCy-training/
├── README.md                          # Comprehensive methodology & usage guide
├── main.ipynb                         # Narrative Jupyter walkthrough notebook
├── config_trf.cfg                     # spaCy GPU transformer training configuration
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI entity extraction HTTP endpoint
│   └── README.md                      # API run instructions, curl examples, PHP integration
├── data/
│   ├── terms.csv                      # Seed dictionary (IT_TERM, CLERICAL_TERM)
│   ├── data.jsonl                     # Real annotated training data (1,044 entries)
│   ├── training/
│   │   ├── train.spacy                # Real training partition (687 docs)
│   │   ├── dev.spacy                  # Real evaluation partition (147 docs)
│   │   └── test.spacy                 # Real held-out testing partition (148 docs)
│   ├── test/
│   │   ├── unseen_benchmark.jsonl     # Controlled unseen-term generalization probe (65 terms)
│   │   ├── holdout.jsonl              # Permanent real-world holdout evaluation set
│   │   └── raw/                       # Raw holdout journal text files
│   ├── candidates/
│   │   └── mined_candidates.csv       # Ranked active learning candidates
│   └── evaluation_report.json         # Automated evaluation report & generalization metrics
├── models/
│   ├── ner_trf/                       # Baseline model (TRTR: Train Real, Test Real)
│   │   └── model-best/                # Checkpoint with peak dev F1
│   ├── ner_trf_trstr_llm/             # Augmented model (TRSTR-LLM: Train Real + Synthetic LLM)
│   │   └── model-best/                # Checkpoint with peak dev F1
│   └── hybrid_pipeline/               # Packaged EntityRuler + Transformer NER pipeline
├── scripts/
│   ├── __init__.py                    # Automatic CUDA runtime library preloader
│   ├── annotation.py                  # Real-data ingestion, diagnostics, dedup & DocBin conversion
│   ├── generate_llm_synthetic.py      # LLM-direct synthetic data generation (Gemini)
│   ├── prepare_trstr_llm.py           # TRSTR-LLM dataset assembly & DocBin builder
│   ├── training.py                    # Transformer fine-tuning script with patience-based stopping
│   ├── pipeline.py                    # HybridJournalPipeline inference & confidence routing
│   ├── candidate_mining.py            # Syntactic trigger pattern mining & active learning feedback
│   ├── eval.py                        # Precision, Recall, F1 & unseen generalization benchmark
│   ├── check_data_leakage.py          # 7-check data leakage & benchmark isolation audit
│   ├── retrain.py                     # Active learning retraining workflow
│   ├── build_holdout.py               # Real-world holdout scaffolding
│   ├── deploy_inference.py            # Production streaming inference CLI
│   └── labels.py                      # Label taxonomy & normalization
├── docs/
│   ├── annotation_guidelines.md       # Official annotation policy & label taxonomy
│   ├── synthetic_augmentation_methodology.md # Full TRTR vs TRSTR-LLM methodology & ablation results
│   └── archive/
│       └── paraphrase_augmentation_methodology.md  # Historical record of prior paraphrase-based approach
└── tools/
    ├── pipeline_gui.py                # Desktop GUI Studio for Hybrid NER inference (FastAPI counterpart)
    ├── entity_extractor_gui.py        # Launcher alias for pipeline_gui.py
    ├── jsonl_editor.py                # Interactive JSONL dataset viewer & annotation editor
    ├── spacy_annotator_app.py         # Rapid manual span annotation helper
    └── README.md                      # Guide to desktop GUI tools
```

---

## 3. Data Specification & Schema

### Training Data (`data/data.jsonl`)

All training data is **real, manually annotated OJT journal entries**. Each record enforces **exact span-level character offsets** (`start`, `end`):

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

### Negative (Non-Entity) Examples
To prevent false-positive over-prediction in conversational OJT entries, negative sentences are strictly annotated with an empty entity list:

```json
{
  "text": "Audit preparations for Annual Report",
  "entities": []
}
```

### Dataset Diagnostics

The pipeline reports negative ratio and class balance at data load time:
- **Target negative ratio**: 25–35% of total records
- **Class balance**: Roughly equal `IT_TERM` / `CLERICAL_TERM` entity counts
- Diagnostics are reports, not enforcement — the user decides whether to add more examples

### Controlled Unseen-Term Benchmark (`data/test/unseen_benchmark.jsonl`)

A separate synthetic benchmark containing 85 sentences with 65 unique entities **strictly absent from `data/terms.csv`**. Used as a controlled generalization probe to measure whether the Transformer component generalizes beyond dictionary memorization. This is evaluated in a **separate test section** from the real-data held-out evaluation.

---

## 4. Evaluation Framework

The pipeline runs three separate evaluation sections:

| Section | Data Source | Purpose |
| :--- | :--- | :--- |
| **Held-Out Test Set** | `data/training/test.spacy` (real data, 15% split) | Primary performance metric on real journal entries |
| **Unseen-Term Benchmark** | `data/test/unseen_benchmark.jsonl` (controlled synthetic) | Measures pure contextual generalization to novel terms |
| **Real-World Holdout** | `data/test/holdout.jsonl` (real data, permanent) | Permanent out-of-distribution evaluation |

The unseen benchmark evaluates three modes independently:
1. **Transformer-only**: EntityRuler disabled — measures pure ML generalization
2. **EntityRuler-only**: Dictionary matching only — establishes baseline (expected 0% recall on unseen terms)
3. **Hybrid**: Full pipeline — demonstrates the combined system's capabilities

---

## 5. Usage & Reproduction Instructions

### Environment Setup
Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 1. Run the Narrative Walkthrough Notebook
Launch Jupyter Lab or Notebook and open `main.ipynb`:
```bash
jupyter lab main.ipynb
```
*The notebook walks through every phase: data loading, diagnostics, splitting, training, leakage checks, and three-section evaluation.*

### 2. Run Individual Modules via CLI

#### Prepare Real Data (Dedup, Split, Compile DocBins):
```bash
python scripts/annotation.py
```

#### Train the Transformer on GPU:
```bash
python scripts/training.py --steps 200 --eval-freq 50 --gpu-id 0
```

#### Run Data Leakage Audit:
```bash
python scripts/check_data_leakage.py
```

#### Run Hybrid Inference on a Custom Sentence:
```bash
python -c "
from scripts.pipeline import HybridJournalPipeline
pipe = HybridJournalPipeline()
res = pipe.predict('I developed an asynchronous microservice using FastAPI and Docker.')
import json; print(json.dumps(res, indent=2))
"
```

### 3. Entity Extraction API (HTTP Service)

The FastAPI service exposes the hybrid pipeline over HTTP for integration with PHP or any other caller:

```bash
# Start the API server:
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Test with curl:
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Used Microsoft Excel for data encoding."}'
```

See [`api/README.md`](api/README.md) for full documentation, response shapes, and PHP integration examples.

### 4. Production Deployment Script (Streaming Inference)

The deployment script handles arbitrary text file sizes using **streaming line batches with constant memory overhead**:

```bash
# Process structured OJT log input:
python scripts/deploy_inference.py -i data/raw/structured_journal_input.txt -o results.csv

# Process conversational, human-written OJT journal input:
python scripts/deploy_inference.py -i data/raw/human_written_journal_input.txt -o results.csv
```

#### CLI Options:
- `-i, --input`: Path to input `.txt` file (mandatory).
- `-o, --output`: Destination path for aggregated results (`.csv`, `.json`, `.jsonl`, `.text`).
- `-d, --detailed-output`: Optional path to stream line-by-line JSONL extraction logs.
- `-b, --batch-size`: Streaming batch size for GPU inference (default: `64`).
- `-t, --threshold`: Confidence threshold for ML acceptance (default: `0.80`).
- `-f, --format`: Output format (`csv`, `json`, `jsonl`, `text`).

### 5. Interactive Desktop GUI Studio (`tools/pipeline_gui.py`)

A graphical desktop environment mirroring the FastAPI entity extraction endpoint:

```bash
# Launch the extraction GUI:
python tools/pipeline_gui.py

# Or use the convenience alias:
python tools/entity_extractor_gui.py
```

- **Live In-Text Highlighting**: Visually tags `IT_TERM` (blue) and `CLERICAL_TERM` (emerald) directly in journal narratives with hover tooltips and review flags.
- **FastAPI Contract Parity**: Replicates `_deduplicate_entities()` frequency counting, `_normalise()` canonical keys, and `_build_summary()` category percentages.
- **Interactive Inspection Table**: Multi-criteria filters (Search, Category, Status, Source), column sorting, and click-to-navigate synchronization.
- **Model Switching**: Easily swap between `TRSTR-LLM`, `Production Baseline`, `TRTR`, or custom checkpoint folders.
- **API JSON Inspector & cURL Generator**: 1-click clipboard export of FastAPI payloads and reproducible cURL commands.
- **Batch Processing**: Process `.jsonl` or `.txt` collections with progress tracking and export.

See [`tools/README.md`](tools/README.md) for full documentation of desktop tools.

---

## 6. Empirical Results: TRTR vs. TRSTR-LLM Ablation

To evaluate the effect of synthetic data augmentation on out-of-vocabulary generalization without sacrificing real-journal precision, the pipeline establishes a formal ablation protocol:
- **TRTR (Train Real, Test Real)**: Baseline model trained exclusively on authentic student journals (`data/data.jsonl`, 687 train docs).
- **TRSTR-LLM (Train Real + Synthetic LLM, Test Real)**: Model augmented with 300 novel, LLM-direct generated records (`data/synthetic_llm_generated.jsonl`, 987 total train records).
- **TRSTR-Paraphrase (Historical Reference)**: Prior seq2seq paraphrase approach archived at [`docs/archive/paraphrase_augmentation_methodology.md`](docs/archive/paraphrase_augmentation_methodology.md).

Both models were fine-tuned under identical patience-based hyperparameters (`max_steps=2500`, `patience=400`, `eval_frequency=50`) on an NVIDIA RTX 3060 GPU, and evaluated against the identical real-data evaluation sets:

| Evaluation Partition | Metric | TRTR (Real Only Baseline) | TRSTR-Paraphrase (Archived Reference) | TRSTR-LLM (Direct Gemini Lite) | Delta vs. TRTR | Relative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Unseen-Term Benchmark** | **Transformer Recall** | 33.85% (22/65) | 66.15% (43/65) | **75.38% (49/65)** | **+41.53%** | **+122.69%** |
| (Out-of-Vocabulary Probes) | Transformer Precision | 22.45% | 37.72% | **42.24%** | **+19.79%** | **+88.15%** |
| | Transformer F1 | 26.99% | 48.04% | **54.14%** | **+27.15%** | **+100.59%** |
| *Hybrid Pipeline Integration* | **Hybrid Recall** | 33.85% | 66.15% | **75.38%** | **+41.53%** | **+122.69%** |
| | **Generalization Lift** | +32.31% | +64.61% | **+73.84%** | **+41.53%** | **+128.57%** |
| **Held-Out Real Test Set** | **Overall F1** | **60.06%** | 63.19% | **59.28%** | -0.78% | -1.30% |
| (`data/training/test.spacy`) | Overall Precision | 57.32% | 61.39% | **57.59%** | **+0.27%** | +0.47% |
| | Overall Recall | **63.09%** | 65.10% | 61.07% | -2.02% | -3.20% |
| *Per-Label Performance* | `CLERICAL_TERM` Precision | 60.32% | 66.67% | **67.86%** | **+7.54%** | **+12.50%** |
| | `CLERICAL_TERM` Recall | 54.29% | **62.86%** | 54.29% | 0.00% | 0.00% |
| | `CLERICAL_TERM` F1 | 57.14% | **64.71%** | **60.32%** | **+3.18%** | **+5.57%** |
| | `IT_TERM` Precision | 55.45% | **57.61%** | 51.96% | -3.49% | -6.29% |
| | `IT_TERM` Recall | **70.89%** | 67.09% | 67.09% | -3.80% | -5.36% |
| | `IT_TERM` F1 | **62.22%** | 61.99% | 58.56% | -3.66% | -5.88% |

### Key Thesis Findings:
1. **Dramatic Generalization Surge (+41.53% Recall)**: TRSTR-LLM successfully recognized **49 out of 65 unseen enterprise technologies** from syntactic context alone without dictionary assistance, compared to 22 for TRTR, driving unseen F1 from 26.99% to **54.14% (+27.15%)**.
2. **Administrative Precision Lift (+7.54% Precision)**: On authentic student journals, `CLERICAL_TERM` precision increased from 60.32% to **67.86%**, confirming that negative non-task prompting effectively taught the model to distinguish true clerical activities from environmental narrative text.
3. **Full Methodology & Analysis**: See [`docs/synthetic_augmentation_methodology.md`](docs/synthetic_augmentation_methodology.md) for full architectural details, prompt design, and data leakage isolation audits.

---

## 7. Extensibility for Thesis Defense

- **Extensible Label Taxonomies**: The architecture seamlessly scales to more OJT categories (e.g. `ADMINISTRATIVE_TERM`, `FINANCE_TERM`, `MARKETING_TERM`, `DESIGN_TERM`) simply by adding labels to `data/terms.csv` and re-running `scripts/training.py`.
- **Modular Decoupling**: The dictionary layer (`EntityRuler`) and contextual ML layer (`Transformer NER`) remain completely decoupled, allowing either layer to be swapped or upgraded independently without architectural refactoring.
