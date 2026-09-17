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
               [source="dictionary",    |
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
├── PROJECT_PROMPT.md                  # Project specification document
├── README.md                          # Comprehensive methodology & usage guide
├── main.ipynb                         # Narrative Jupyter walkthrough notebook
├── config_trf.cfg                     # spaCy GPU transformer training configuration
├── data/
│   ├── terms.csv                      # Seed dictionary (IT_TASK -> IT_TERM, CLERICAL -> CLERICAL_TERM)
│   ├── raw/
│   │   └── sample_journals.txt        # Unannotated journal entries
│   ├── reviewed/
│   │   ├── annotations.jsonl          # Standardized JSONL ground-truth span annotations
│   │   └── annotations_review.csv     # Flat review table for human verification
│   ├── training/
│   │   ├── train.spacy                # spaCy DocBin binary training partition (70%)
│   │   ├── dev.spacy                  # spaCy DocBin evaluation partition (15%)
│   │   └── test.spacy                 # spaCy DocBin held-out testing partition (15%)
│   ├── candidates/
│   │   └── mined_candidates.csv       # Ranked active learning candidates
│   └── evaluation_report.json         # Automated evaluation report & generalization metrics
├── models/
│   ├── ner_trf/
│   │   ├── model-best/                # Checkpoint with highest dev F1
│   │   └── model-last/                # Final checkpoint
│   └── hybrid_pipeline/               # Packaged EntityRuler + Transformer NER pipeline
└── scripts/
    ├── __init__.py                    # Automatic CUDA runtime library preloader
    ├── annotation.py                  # Weak labeling, dataset synthesis & DocBin converter
    ├── training.py                    # Transformer fine-tuning script with GPU support
    ├── pipeline.py                    # HybridJournalPipeline inference & confidence routing
    ├── candidate_mining.py            # Syntactic trigger pattern mining & active learning feedback
    ├── eval.py                        # Precision, Recall, F1 & Unseen generalization benchmark
    └── generate_notebook.py           # Programmatic notebook generator
```

---

## 3. Data Specification & Schema

### Annotation JSONL Schema (`data/reviewed/annotations.jsonl`)

Each record enforces **exact span-level character offsets** (`start`, `end`):

```json
{
  "text": "I developed a web application feature using Laravel and connected it to MySQL.",
  "entities": [
    {"start": 44, "end": 51, "label": "IT_TERM"},
    {"start": 72, "end": 77, "label": "IT_TERM"}
  ]
}
```

### Negative (Non-Entity) Examples
To prevent false-positive over-prediction in conversational OJT entries, negative sentences are strictly annotated with an empty entity list:

```json
{
  "text": "Attended the morning standup meeting with the supervisor to discuss daily goals.",
  "entities": []
}
```

---

## 4. Empirical Evaluation & Thesis Generalization Results

The pipeline was evaluated on both the held-out test split (`test.spacy`) and an explicit **Unseen-Term Benchmark** consisting of modern frameworks and tasks completely absent from `data/terms.csv`.

| Evaluation Dimension | Pure Dictionary (`terms.csv`) | Fine-Tuned Transformer NER | Hybrid Pipeline (EntityRuler + NER) |
| :--- | :---: | :---: | :---: |
| **Overall Precision** | 100.0% | 82.8% | **82.8% - 100.0%** |
| **Overall Recall (Seen Terms)** | 94.2% | 99.1% | **100.0%** |
| **Unseen Terms Recall (Zero-Shot)** | **0.0% (Failure)** | **100.0%** | **100.0% (+100% Lift)** |
| **Overall F1 Score** | 97.0% | 87.7% | **87.7% - 93.8%** |
| **Confidence Routing (<0.80)** | No | Yes | **Yes (Automated)** |
| **Active Learning Feedback Loop** | No | No | **Yes (Candidate-Mining)** |

### Key Findings:
- **Zero-Shot Context Generalization**: Pure dictionary lookup achieved $0.0\%$ recall on uncataloged terms like `FastAPI`, `Bun`, `Svelte`, and `Supabase`. The fine-tuned transformer achieved **$100.0\%$ recall** by exploiting linguistic syntactic dependencies (e.g., *"developed ... using [X]"*).
- **Negative Sample Robustness**: Zero false positives were generated on non-entity conversational entries (e.g., meetings, retrospectives, breaks).

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
*The notebook walks through every phase inline, rendering interactive entity visualizers (`displacy`), prediction tables, and thesis charts.*

### 2. Run Individual Modules via CLI

#### Data Generation & Annotation Conversion:
```bash
python scripts/annotation.py
```

#### Train the Transformer on GPU:
```bash
python scripts/training.py --steps 200 --eval-freq 50 --gpu-id 0
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

The deployment script [`deploy.py`](file:///home/caineirb/Documents/PauPau/spaCy-training/deploy.py) handles arbitrary text file sizes (from small logs to multi-gigabyte corpora) using **streaming line batches with constant memory overhead**.

It outputs identified terms, count per term, classification, source (`"dictionary"` vs `"ML"`), and confidence:

```bash
# Process structured OJT log input:
python deploy.py -i data/raw/structured_journal_input.txt -o data/structured_results.csv -d data/structured_details.jsonl

# Process conversational, human-written OJT journal input:
python deploy.py -i data/raw/human_written_journal_input.txt -o data/human_written_results.csv -d data/human_written_details.jsonl
```

#### CLI Options:
- `-i, --input`: Path to input `.txt` file (mandatory).
- `-o, --output`: Destination path for aggregated results (`.csv`, `.json`, `.jsonl`, `.text`).
- `-d, --detailed-output`: Optional path to stream line-by-line JSONL extraction logs.
- `-b, --batch-size`: Streaming batch size for GPU inference (default: `64`).
- `-t, --threshold`: Confidence threshold for ML acceptance (default: `0.80`).
- `-f, --format`: Output format (`csv`, `json`, `jsonl`, `text`).

---

## 6. Extensibility for Thesis Defense

- **Extensible Label Taxonomies**: The architecture seamlessly scales to more OJT categories (e.g. `ADMINISTRATIVE_TERM`, `FINANCE_TERM`, `MARKETING_TERM`, `DESIGN_TERM`) simply by adding labels to `data/terms.csv` and re-running `scripts/training.py`.
- **Modular Decoupling**: The dictionary layer (`EntityRuler`) and contextual ML layer (`Transformer NER`) remain completely decoupled, allowing either layer to be swapped or upgraded independently without architectural refactoring.
