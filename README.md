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
├── data/
│   ├── terms.csv                      # Seed dictionary (IT_TERM, CLERICAL_TERM)
│   ├── data.jsonl                     # Real annotated training data (1,044 entries)
│   ├── synthetic_paraphrases.jsonl    # T5-generated paraphrase augmentation pool (416 entries)
│   ├── synthetic_templates.jsonl      # Template-based syntactic diversification pool (132 entries)
│   ├── training_trstr.jsonl           # Combined TRSTR training pool (1,235 entries: 687 real, 548 synthetic)
│   ├── training/
│   │   ├── train.spacy                # Real training partition (687 docs)
│   │   ├── dev.spacy                  # Real evaluation partition (147 docs)
│   │   ├── test.spacy                 # Real held-out testing partition (148 docs)
│   │   └── train_trstr.spacy          # Combined TRSTR training partition (1,235 docs)
│   ├── test/
│   │   ├── unseen_benchmark.jsonl     # Controlled unseen-term generalization probe (65 terms)
│   │   ├── holdout.jsonl              # Permanent real-world holdout evaluation set
│   │   └── raw/                       # Raw holdout journal text files
│   ├── candidates/
│   │   └── mined_candidates.csv       # Ranked active learning candidates
│   └── evaluation_report.json         # Automated evaluation report & generalization metrics
├── models/
│   ├── ner_trf_trtr/                  # Pure-real model (TRTR: Train Real, Test Real)
│   │   └── model-best/                # Checkpoint with peak dev F1
│   ├── ner_trf_trstr/                 # Augmented model (TRSTR: Train Real+Synth, Test Real)
│   │   └── model-best/                # Checkpoint with peak dev F1
│   └── hybrid_pipeline/               # Packaged EntityRuler + Transformer NER pipeline
├── scripts/
│   ├── __init__.py                    # Automatic CUDA runtime library preloader
│   ├── annotation.py                  # Real-data ingestion, diagnostics, dedup & DocBin conversion
│   ├── generate_synthetic_augmentation.py # Zero-leakage T5 paraphraser & template generator
│   ├── training.py                    # Transformer fine-tuning script with patience-based stopping
│   ├── pipeline.py                    # HybridJournalPipeline inference & confidence routing
│   ├── candidate_mining.py            # Syntactic trigger pattern mining & active learning feedback
│   ├── eval.py                        # Precision, Recall, F1 & unseen generalization benchmark
│   ├── check_data_leakage.py          # 7-check data leakage & benchmark isolation audit
│   ├── retrain.py                     # Active learning retraining workflow
│   ├── build_holdout.py               # Real-world holdout scaffolding
│   ├── deploy_inference.py            # Production streaming inference CLI
│   └── labels.py                      # Label taxonomy & normalization
└── docs/
    ├── annotation_guidelines.md       # Official annotation policy & label taxonomy
    └── synthetic_augmentation_methodology.md # Full augmentation methodology & ablation report
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

### 3. Production Deployment Script (Streaming Inference)

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

---

## 6. Empirical Results: TRTR vs. TRSTR Ablation

To evaluate the effect of syntactic diversity and data volume, we conduct a controlled ablation between two training conditions under identical patience settings (`max_steps=2500`, `patience=400`):
- **TRTR (Train Real, Test Real)**: Trained purely on authentic student journal annotations (`data/data.jsonl`, 687 training records).
- **TRSTR (Train Real + Synthetic, Test Real)**: Trained on authentic data augmented with T5 paraphrases and syntactic templates (1,235 records: 687 real, 548 synthetic).

Both conditions are evaluated on the exact same real evaluation partitions (`test.spacy` and `unseen_benchmark.jsonl`):

| Evaluation Dimension | Metric | TRTR (Real Only) | TRSTR (Real + Synth) | Absolute Delta | Relative Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Held-Out Real Test Set** | **Overall F1** | 60.06% | **61.25%** | **+1.19%** | +1.98% |
| (`data/training/test.spacy`) | Overall Precision | 57.32% | 57.31% | -0.01% | -0.02% |
| | Overall Recall | 63.09% | **65.77%** | **+2.68%** | +4.25% |
| *Per-Label Performance* | `IT_TERM` F1 | **62.22%** | 60.00% | -2.22% | -3.57% |
| | `IT_TERM` Recall | **70.89%** | 68.35% | -2.54% | -3.58% |
| | `CLERICAL_TERM` F1 | 57.14% | **62.86%** | **+5.72%** | +10.01% |
| | `CLERICAL_TERM` Recall | 54.29% | **62.86%** | **+8.57%** | +15.79% |
| **Unseen Benchmark** | **Transformer Recall** | 33.85% (22/65) | **61.54% (40/65)** | **+27.69%** | **+81.80%** |
| (Out-of-Vocabulary Probes) | Transformer Precision | 22.45% | **38.10%** | **+15.65%** | +69.71% |
| | Transformer F1 | 26.99% | **47.06%** | **+20.07%** | +74.36% |
| *Hybrid Pipeline* | **Hybrid Recall** | 33.85% | **61.54%** | **+27.69%** | +81.80% |
| | **Generalization Lift** | +32.31% | **+60.00%** | **+27.69%** | +85.70% |

> [!WARNING]
> **Limitations Statement**: Synthetic augmentation was introduced to compensate for the limited volume of the authentic annotated corpus and to evaluate the impact of syntactic variety. TRSTR results demonstrate the transformer's capacity for zero-shot inductive generalization given broader linguistic variety, but should not be misconstrued as evidence that the final system was trained exclusively on authentic data. See [`docs/synthetic_augmentation_methodology.md`](docs/synthetic_augmentation_methodology.md) for full procedural documentation.

---

## 7. Extensibility for Thesis Defense

- **Extensible Label Taxonomies**: The architecture seamlessly scales to more OJT categories (e.g. `ADMINISTRATIVE_TERM`, `FINANCE_TERM`, `MARKETING_TERM`, `DESIGN_TERM`) simply by adding labels to `data/terms.csv` and re-running `scripts/training.py`.
- **Modular Decoupling**: The dictionary layer (`EntityRuler`) and contextual ML layer (`Transformer NER`) remain completely decoupled, allowing either layer to be swapped or upgraded independently without architectural refactoring.
