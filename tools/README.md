# Tools & Interactive GUI Applications

This directory provides desktop graphical user interface (GUI) utilities for dataset annotation, JSONL curation, and interactive inference evaluation.

---

## 1. OJT Journal Entity Extraction Studio (`pipeline_gui.py`)

The primary desktop GUI tool for the Hybrid NER + Classification pipeline. It serves as the graphical counterpart to the FastAPI service ([`api/main.py`](../api/main.py)), allowing researchers and developers to inspect model predictions interactively.

### Launching the Studio

```bash
# Ensure your virtual environment is activated
source .venv/bin/activate

# Launch the extraction GUI
python tools/pipeline_gui.py

# Or use the convenience alias
python tools/entity_extractor_gui.py
```

### In-GUI Configuration

All pipeline, model, and inference parameters are configured directly within the graphical interface:
- **Model Checkpoint**: Select from pre-discovered checkpoints (`TRSTR-LLM`, `Production Baseline`, `TRTR`) or click **Browse...** to choose any custom model directory.
- **Terms Dictionary CSV**: Edit the dictionary file path (default `data/terms.csv`) or click **Browse...** to choose any custom terms CSV.
- **Inference Mode**: Select `hybrid` (EntityRuler + Transformer), `transformer_only` (ML only), or `entity_ruler_only` (Dictionary only).
- **Confidence Threshold**: Adjust the acceptance threshold (0.50 to 0.99) via the interactive spinbox to dynamically update acceptance tags.
- **Journal Text & Presets**: Type, paste, open text files, or select from curated sample presets in the dropdown.
- **Batch Processing**: Load, run, and export `.jsonl` or `.txt` batch files from the **Batch Processing** tab.

### Core Features

- **Exact FastAPI Contract Parity**:
  - Implements the exact same whitespace normalisation (`_WS_RE`), entity deduplication (`_deduplicate_entities`), and frequency/category summary statistics (`_build_summary`) as `POST /extract`.
- **Live Visual Highlighting**:
  - Automatically colors extracted entity spans directly in the journal text box:
    - **`IT_TERM`**: Soft light blue background with dark blue lettering.
    - **`CLERICAL_TERM`**: Soft emerald green background with dark forest green lettering.
    - **`NEEDS_REVIEW`**: Amber warning highlight for low-confidence ML spans (< 0.80).
  - Hovering over any span displays a live tooltip with confidence %, source, and status.
  - Clicking any tag jumps directly to the corresponding row in the entity table.
- **Analytics & KPI Dashboard**:
  - Displays instant metric cards: **Total Occurrences**, **Unique Entities**, **IT_TERM Count / %**, **CLERICAL_TERM Count / %**, **Needs Review (<0.80)**, and **Inference Latency**.
- **Interactive Entity Table**:
  - Search keyword filtering, category filtering (`All` / `IT_TERM` / `CLERICAL_TERM`), status filtering (`All` / `ACCEPTED` / `NEEDS_REVIEW`), and source filtering (`All` / `dictionary` / `ML`).
  - Column sorting by term, category, frequency, confidence, and offset.
  - Right-click context menu: Copy Term, Copy Entity JSON, or jump to text occurrence.
- **API JSON Response Viewer**:
  - Displays the exact JSON response contract matching `POST /extract`.
  - 1-click **Copy JSON** button for pasting into test suites or reports.
  - 1-click **Generate cURL** dialog to test against a live FastAPI instance.
- **Batch Processing**:
  - Load `.jsonl` or `.txt` batch files with hundreds of entries.
  - Asynchronous batch execution with a live progress bar.
  - Export all extracted entities and summary statistics to a new `.jsonl` file.
- **Multi-Model Switching**:
  - Dropdown selector pre-populated with `TRSTR-LLM (High Recall)`, `Production Baseline`, `TRTR Baseline`, and folder browsing.
  - Asynchronous background model loading with hardware acceleration status (`⚡ GPU: RTX 3060` or `💻 CPU`).

---

## 2. JSONL Entity Annotator (`jsonl_editor.py`)

A full-featured dataset curator for viewing, adding, editing, and verifying span annotations across `.jsonl` files (e.g. `data/data.jsonl`, `data/training_trstr_llm.jsonl`).

```bash
python tools/jsonl_editor.py
```

### Features
- Record-by-record navigation and search.
- Interactive token span annotation.
- Duplicate removal and label validation.
- Direct save and export.

---

## 3. spaCy Annotation Helper (`spacy_annotator_app.py`)

A lightweight scratchpad utility for rapidly annotating single sentences and copying formatted spaCy training JSON structures.

```bash
python tools/spacy_annotator_app.py
```

---

## 4. spaCy NER Evaluation Results Studio (`eval_results_gui.py`)

A native Python desktop graphical user interface (Tkinter) to scan, inspect, filter, and compare model evaluation files across `data/eval_results/` in a human-readable layout.

```bash
# Launch native desktop GUI application
python tools/eval_results_gui.py
```

### Core Features
- **File Type Primary Tabs**:
  - `Held-Out Test Set` (199 authentic documents)
  - `Unseen Benchmark (Hybrid Pipeline)` (85 modern tech stack sentences)
  - `Unseen Benchmark (Transformer Only)` (85 sentences ML-only)
  - `Unseen Benchmark (Entity Ruler Only)` (85 sentences dictionary-only)
  - `Dictionary Overrides` (18 runtime precedence audit events)
  - `3-Way Benchmark Matrix` (Comparative P/R/F1 summary)
- **Model Separation**:
  - Sub-navigation pills to switch instantly between `TRTR Baseline`, `TRSTR-Paraphrase`, and `TRSTR-LLM (High Recall)`.
- **Side-by-Side (3-Way Comparison)**:
  - Sentence-by-sentence 3-column alignment comparing TRTR, Paraphrase, and LLM predictions side-by-side with automated consensus tags (`🟢 LLM Fixed Error`, `🟡 Disagreements`, `🔴 All Failed`).
- **Scope & Error Filtering**:
  - Toggle between `All Records` and `⚠️ Errors Only`.
  - Filter chips with live counts: `All`, `✅ Correct`, `⚠️ Boundary`, `🚫 Spurious (FP)`, `❌ Missed (FN)`, `🏷️ Label Error`.
  - Filter by entity category (`IT_TERM`, `CLERICAL_TERM`) and prediction source (`ML`, `dictionary`).
- **Interactive Highlighting & Inspection**:
  - Sentences render with color-coded entity spans and hover popover tooltips (term, confidence %, source, offsets).
  - Entity breakdown comparison table for each document.
  - Expandable raw JSON inspector with 1-click clipboard copy.
