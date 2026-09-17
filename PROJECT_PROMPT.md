# Project Prompt: Hybrid NER + Classification Pipeline for OJT Journal Task Tagging

## Context

I'm building a system that processes OJT (On-the-Job Training) weekly journal entries and automatically identifies and categorizes work tasks as **IT** or **CLERICAL** (extensible to more categories later). I already have a `terms_df` / `terms.csv` dictionary of known terms (e.g. `OpenAI API`, `Microsoft Excel`, `PHP`) and access to an NVIDIA GPU for training.

Do not rely purely on the dictionary — the system must generalize to unseen terms (e.g. recognizing "Laravel" as IT even if it was never in the dictionary) by learning from **sentence context**, not just memorizing strings.

## Architecture to implement

Build a **hybrid pipeline**, not a single flat NER model:

1. **EntityRuler (deterministic layer)** — matches known terms from `terms.csv` with high precision before the ML model runs.
2. **spaCy NER (`en_core_web_trf`, fine-tuned)** — trained to detect entity spans using two labels: `IT_TERM` and `CLERICAL_TERM`. This layer is responsible for catching *unseen* terms (e.g. "Laravel", "FastAPI", "Symfony") based on the linguistic context they appear in (e.g. "developed ... using X", "implemented ... in X").
3. **Confidence-based routing** — every extracted entity should carry `{term, category, confidence, source: "ML" | "dictionary"}`. Anything below a confidence threshold (start at 0.80) is queued for human review rather than auto-accepted.
4. **Candidate-mining / active-learning loop** — when NER flags a term not in the dictionary, log it as a candidate (`candidate,count` style), route it to human validation, and feed validated terms back into both the EntityRuler patterns and the training set (retrain periodically).

## Key implementation requirements

- **Two NER labels**, not one generic `TASK` label plus a separate classifier — i.e. train the NER model directly on `IT_TERM` / `CLERICAL_TERM` spans. (Note: keep the architecture modular enough that swapping to a `TASK` + separate classifier design is a config change, not a rewrite — I may want to add more categories like `ADMINISTRATIVE`, `MARKETING`, `DESIGN`, `FINANCE` later.)
- **Span-level annotation only.** Never annotate whole sentences as one giant entity — only the exact term span (e.g. "Laravel", not the whole sentence containing it).
- **Negative examples are required.** Include journal sentences with no IT/clerical entity at all (meetings, general discussion, document review, etc.) so the model doesn't over-predict.
- **Context diversity matters more than raw volume.** For each known term, generate/collect multiple distinct sentence patterns (e.g. "developed using X", "maintained an X application", "implemented authentication using X") rather than repeating the same template.
- **Target initial dataset size:** roughly 300–500 examples per label, drawn from varied journal entries and sentence structures, plus a substantial set of negative (no-entity) examples.

## Project structure requirements

- **Main entry point must be a Jupyter notebook** (e.g. `main.ipynb`) at the project root. This notebook is the narrative/documentation layer — it should walk through the pipeline phase by phase (ingest → annotate → train → infer → mine candidates → evaluate) with markdown explanations and results inline, calling into the scripts below rather than containing implementation logic itself.
- **All reusable logic (functions, classes, methods) goes in `scripts/`** as importable Python modules (e.g. `scripts/annotation.py`, `scripts/training.py`, `scripts/pipeline.py`, `scripts/candidate_mining.py`, `scripts/eval.py`). The notebook should just import from these and call them — no function/class definitions inline in the notebook.
- **All data lives in `data/`**: `terms.csv`, the raw/reviewed JSONL annotation files, the converted spaCy binary training files, and any candidate-mining CSV outputs. Keep raw vs. reviewed vs. training-ready data in clearly separated subfolders (e.g. `data/raw/`, `data/reviewed/`, `data/training/`, `data/candidates/`).

## Deliverables I want from you

1. **Project scaffold**: `main.ipynb` at the root, a `scripts/` package for all functions/methods, a `data/` folder for `terms.csv` and all JSONL/training data (see structure requirements above), plus a README explaining the architecture.
2. **Annotation tooling**: a script/workflow to take raw OJT journal text + `terms.csv`, auto-generate *candidate* annotations via string matching (weak labeling), and output them in a human-reviewable format (CSV or JSONL) before they're trusted as training data.
3. **Training data format**: JSONL schema like:
   ```json
   {"text": "I developed a web application using Laravel.", "entities": [{"start": 38, "end": 45, "label": "IT_TERM"}]}
   ```
   Provide a converter from the reviewed CSV/JSONL into spaCy's binary training format.
4. **EntityRuler setup**: patterns generated from `terms.csv`, wired in as a pipe `before="ner"`.
5. **Fine-tuning script**: loads `en_core_web_trf`, adds the `IT_TERM` / `CLERICAL_TERM` labels, fine-tunes on the annotated dataset, uses the GPU, and saves the resulting pipeline.
6. **Inference pipeline**: takes a raw journal entry, runs EntityRuler + fine-tuned NER, merges results, attaches `confidence` and `source`, and outputs structured JSON per entity. Apply the 0.80 confidence threshold to flag low-confidence items for review.
7. **Candidate-mining script**: scans journals for entities the pipeline didn't confidently tag, extracts surrounding context patterns (e.g. "developed ... using [X]"), and outputs a ranked `candidate,count` CSV for human review.
8. **Evaluation**: precision/recall/F1 per label on a held-out set, plus a simple report of how many previously-unseen terms were correctly identified via context (to demonstrate generalization beyond the dictionary — this is the key methodological claim for my thesis).
9. **Minimal CLI or notebook** tying the whole loop together: ingest → extract → classify → flag-for-review → retrain.

## Constraints

- Python, spaCy (transformer pipeline `en_core_web_trf`), GPU-enabled training.
- Keep the dictionary/EntityRuler layer and the ML NER layer clearly decoupled so I can swap/extend either independently.
- Favor clarity and defensibility of the methodology (this feeds into an academic thesis) over cleverness — I need to be able to explain every design decision.

Start by proposing the folder structure (notebook + scripts/ + data/ as specified above) and the JSONL annotation schema, then walk through implementation phase by phase, confirming with me before moving to the next phase.