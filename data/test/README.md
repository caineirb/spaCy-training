# Real-World Holdout Evaluation Test Set (`data/test/`)

This directory houses a permanent, isolated holdout test set created from **real-world OJT journal entries**.

---

## 1. Strict Isolation Policy (Zero Data Leakage)

1. **Evaluation Only**: The contents of this directory must **NEVER** be ingested by:
   - `scripts/annotation.py` (weak labeling, synthetic dataset generation, or DocBin compilation).
   - `scripts/training.py` or included in `data/training/train.spacy` or `data/training/dev.spacy`.
   - `scripts/candidate_mining.py` or any active-learning retraining scripts (`retrain.py`).
2. **Permanent Holdout Integrity**: This test set exists exclusively to benchmark real-world generalization. Examples in `holdout.jsonl` must **never** be modified, trimmed, or adjusted after model evaluation to artificially inflate reported metrics. Changes to this file should only happen when adding newly verified, manually audited gold test sentences.

---

## 2. Directory Structure

```
data/test/
├── raw/                 # Cleaned, anonymized real journal text files (.txt), unannotated
├── holdout.jsonl        # The permanent, manually verified gold-labeled holdout test set
├── holdout_draft.jsonl  # Machine-suggested draft annotations produced by scripts/build_holdout.py (temporary/review)
└── README.md            # This documentation file
```

---

## 3. Schema & Labeling Convention (`holdout.jsonl`)

Each line in `holdout.jsonl` is a valid JSON object following this schema:

```json
{
  "text": "Scanned and organized the department's compliance certificates into Google Drive.",
  "entities": [
    {"start": 0, "end": 7, "term": "Scanned", "label": "CLERICAL_TERM", "term_status": "seen"},
    {"start": 64, "end": 76, "term": "Google Drive", "label": "IT_TERM", "term_status": "unseen"}
  ],
  "source": "real_journal_holdout",
  "notes": ""
}
```

### Understanding `term_status`:
- `"seen"`: The entity term (or a direct lemmatized/case-insensitive variant) is already present in `data/terms.csv` or appeared in the synthetic training annotations (`data/reviewed/annotations.jsonl`).
  - **Measurement Goal**: Measures the model's ability to recognize known domain terms when embedded in unstructured, natural human prose and sentence structures (*in-domain contextual robustness*).
- `"unseen"`: The entity term is novel and does **not** exist in `data/terms.csv` or training annotations (e.g. newly adopted software frameworks, internal tools, or niche office procedures).
  - **Measurement Goal**: Measures the model's true contextual generalization capability (*inductive task recognition* without dictionary memorization).
- **Negative Examples**: Sentences containing **no** IT or clerical task entities must have `"entities": []`. These sentences serve as crucial negative controls to measure real-world false-positive rates on conversational fluff, background narrative, and administrative settings.

---

## 4. Annotation & Promotion Workflow

1. Place raw real-world journal `.txt` files into `data/test/raw/`.
2. Run the helper script:
   ```bash
   python scripts/build_holdout.py
   ```
   This generates pre-annotated suggestions with candidate `term_status` tags in `data/test/holdout_draft.jsonl`.
3. Manually review, correct boundary spans, verify labels (`IT_TERM` vs `CLERICAL_TERM`), and confirm `term_status` (`seen` vs `unseen`).
4. Append or promote the verified records into `data/test/holdout.jsonl`.
5. Run evaluation:
   ```bash
   python scripts/eval.py
   ```
   The results are isolated under the `"real_world_holdout_evaluation"` section in `data/evaluation_report.json`.
