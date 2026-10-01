# Pipeline Scripts Reference (`scripts/`)

This directory contains the core Python modules and command-line scripts powering the **Hybrid NER + Classification Pipeline for OJT Journal Task Tagging**.

The scripts manage the complete machine learning lifecycle: data ingestion, synthetic augmentation, leakage audits, GPU transformer training, 5-fold cross-validation, multi-mode evaluation, hard negative mining, regression testing, and production inference.

---

## 1. Module Overview

| Script | Purpose | CLI Runnable? | Key Dependencies |
| :--- | :--- | :---: | :--- |
| [`__init__.py`](__init__.py) | Package initialization & CUDA runtime dynamic linker bootstrap (`init_gpu()`) | ❌ | `torch`, `ctypes` |
| [`annotation.py`](annotation.py) | Real-data ingestion, diagnostics, deduplication, train/dev/test splitting & DocBin compilation | ✅ | `spacy`, `srsly` |
| [`check_data_leakage.py`](check_data_leakage.py) | 7-point data leakage & evaluation benchmark isolation audit | ✅ | `spacy`, `pandas` |
| [`cross_validation.py`](cross_validation.py) | 5-fold grouped & stratified cross-validation framework (TRTR vs. TRSTR-Paraphrase vs. TRSTR-LLM) | ✅ | `spacy`, `scikit-learn` |
| [`eval.py`](eval.py) | Multi-mode evaluation (Held-Out Test, Unseen Benchmark, Holdout) & JSON report export | ✅ | `spacy`, `pandas` |
| [`generate_llm_synthetic.py`](generate_llm_synthetic.py) | LLM-direct synthetic dataset generator using Google Gemini Flash Lite | ✅ | `google-genai` |
| [`labels.py`](labels.py) | Unified label taxonomy (`IT_TERM`, `CLERICAL_TERM`) & backward-compatibility normalizer | ❌ | Core Python |
| [`mine_negatives.py`](mine_negatives.py) | Hard negative mining & stylistic n-gram overlap analysis against benchmark | ✅ | `spacy` |
| [`pipeline.py`](pipeline.py) | `HybridJournalPipeline` inference engine with ML-authority conflict resolution | ✅ | `spacy`, `spacy-transformers` |
| [`prepare_trstr_llm.py`](prepare_trstr_llm.py) | TRSTR-LLM dataset builder merging real data, Gemini synthetic logs & mined negatives | ✅ | `spacy` |
| [`test_pipeline_regressions.py`](test_pipeline_regressions.py) | Unittest suite verifying span resolution, ML label authority & abstention fallback | ✅ | `unittest` |
| [`training.py`](training.py) | GPU transformer fine-tuning wrapper with early stopping, patience & config generation | ✅ | `spacy` |

---

## 2. Detailed Script Documentation

### `scripts/__init__.py`
- **Purpose**: Automatically bootstraps the NVIDIA CUDA runtime libraries from the active Python virtual environment (`.venv/lib/python*/site-packages/nvidia/*`) into dynamic linker paths (`LD_LIBRARY_PATH` via `ctypes.CDLL`).
- **Core Function**:
  - `init_gpu(gpu_id: int = 0) -> bool`: Safe GPU initialization that sets `spacy.require_gpu(gpu_id)` and configures PyTorch CUDA memory caching.

---

### `scripts/annotation.py`
- **Purpose**: Handles authentic dataset ingestion, string-level diagnostics, weak supervision, and binary spaCy format (`.spacy`) conversions.
- **Key Functions**:
  - `load_terms_dictionary(terms_csv_path)`: Reads `data/terms.csv` into a normalized `{term: label}` lookup table.
  - `report_dataset_diagnostics(records)`: Calculates and logs total documents, entity counts, class balance ratio, and negative document percentage (target: 25%–35%).
  - `deduplicate_records(records)`: Removes exact duplicate texts and duplicate entity spans.
  - `split_and_convert_dataset(records, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15)`: Performs document-level stratified splitting and serializes to `DocBin`.
  - `prepare_real_data_pipeline()`: End-to-end execution combining loading, diagnostics, dedup, and binary serialization to `data/training/{train,dev,test}.spacy`.
- **CLI Usage**:
  ```bash
  python scripts/annotation.py
  ```

---

### `scripts/check_data_leakage.py`
- **Purpose**: Audits the entire dataset and pipeline against **7 rigorous data leakage checks** to guarantee scientific validity before training or reporting results.
- **Verification Dimensions**:
  1. **Document Duplication**: Verifies zero duplicate documents across train, dev, and test splits.
  2. **Sentence Duplication**: Verifies zero shared sentences across splits.
  3. **Terms Leakage**: Checks if any entity from `data/test/unseen_benchmark.jsonl` exists in `data/terms.csv`.
  4. **Training Annotation Leakage**: Checks if any benchmark entity appears in `data/training/train.spacy`.
  5. **EntityRuler Matchability**: Tests whether the deterministic EntityRuler can match unseen benchmark entities.
  6. **Real Holdout Isolation**: Scans `data/test/holdout.jsonl` to ensure complete isolation from training sentences.
  7. **Synthetic Pool Isolation**: Scans synthetic datasets for exact duplicates, near-duplicates, or leaked benchmark terms.
- **CLI Usage**:
  ```bash
  python scripts/check_data_leakage.py
  ```

---

### `scripts/cross_validation.py`
- **Purpose**: Executes 5-fold stratified cross-validation comparing all 3 conditions (**TRTR**, **TRSTR-Paraphrase**, **TRSTR-LLM**) across all 1,241 authentic documents.
- **Key Characteristics**:
  - Pure real evaluation: Validation folds contain strictly authentic documents.
  - Evaluation isolation: Synthetic paraphrases, Gemini records, and mined negatives are injected *only* into the training folds.
  - Evaluates both the held-out validation fold and the isolated unseen benchmark for every fold.
  - Exports a consolidated statistical summary (`Mean ± Std Dev`) to `data/cv/cv_3way_results.json`.
- **CLI Usage**:
  ```bash
  # Run full 5-fold CV across all 3 conditions:
  python scripts/cross_validation.py --folds 5 --conditions trtr,trstr_paraphrase,trstr_llm --max-steps 1500

  # Run only a specific fold (e.g., Fold 0):
  python scripts/cross_validation.py --folds-to-run 0
  ```

---

### `scripts/eval.py`
- **Purpose**: Core evaluation engine computing strict span-level Precision, Recall, and F1 across multiple datasets and pipeline modes.
- **Evaluation Sections**:
  1. **Held-Out Test Set** (`data/training/test.spacy`): Evaluates pipeline on real-world test journal entries.
  2. **Unseen-Term Benchmark** (`data/test/unseen_benchmark.jsonl`): Evaluates out-of-vocabulary generalization in 3 modes:
     - `transformer_only`: EntityRuler disabled — measures pure contextual ML generalization.
     - `entity_ruler_only`: Dictionary matching only — establishes zero-shot baseline.
     - `hybrid`: Full two-layer system with conflict resolution.
  3. **Permanent Real Holdout** (`data/test/holdout.jsonl`).
- **Artifacts Saved**:
  - Per-item prediction logs and error-only files in `data/eval_results/comparison/<condition>/` (e.g. `trtr`, `trstr_paraphrase`, `trstr_llm`).
  - Cross-validation per-fold logs in `data/eval_results/cv/<condition>/`.
  - Canonical dictionary conflict overrides in `data/eval_results/dictionary_overrides.jsonl`.
- **CLI Usage**:
  ```bash
  python scripts/eval.py \
    --model-path models/ner_trf/model-best \
    --run-label trtr
  ```

---

### `scripts/generate_llm_synthetic.py`
- **Purpose**: Generates novel, realistic OJT internship journal sentences and entity annotations using Google Gemini (`gemini-3.5-flash-lite`).
- **Design Constraints**:
  - In-context sampling of known vocabulary (`data/terms.csv`) and authentic journal style without reproducing real sentences.
  - Deterministic substring span locator ([`locate_entity_spans()`](generate_llm_synthetic.py)) with word-boundary checks to prevent tokenization offset errors.
  - Automatic non-task negative sentence prompting (~30% target) covering routine workplace scenarios (briefings, lunch breaks, cleaning).
  - Multi-tier quality filters: Generic Noun Blocklist, bare noun rejection, benchmark isolation filter.
- **CLI Usage**:
  ```bash
  # Requires GEMINI_API_KEY in environment or .env
  python scripts/generate_llm_synthetic.py --num-batches 30 --batch-size 10 --output data/synthetic_llm_generated.jsonl
  ```

---

### `scripts/labels.py`
- **Purpose**: Canonical label taxonomy definitions and backward-compatibility normalizer.
- **Canonical Labels**:
  - `IT_TERM`: Information technology tools, programming languages, databases, infrastructure, and technical tasks.
  - `CLERICAL_TERM`: Administrative records, forms, filing, office software, and clerical tasks.
- **Compatibility**: Maps legacy `IT_TASK` $\rightarrow$ `IT_TERM` and `CLERICAL` $\rightarrow$ `CLERICAL_TERM`.

---

### `scripts/mine_negatives.py`
- **Purpose**: Mines hard false-positive failure cases from unannotated authentic training text and performs stylistic n-gram overlap analysis against evaluation benchmarks.
- **Workflow**:
  1. Runs the baseline transformer model over authentic training sentences that contain zero gold entities.
  2. Flags high-confidence false-positive extractions ($\ge 0.85$).
  3. Cross-references against `data/terms.csv` to ensure genuine unannotated domain terms are not accidentally mined.
  4. Exports mined negatives to `data/review/mined_hard_negatives.jsonl`.
  5. Computes unigram, bigram, and trigram Jaccard similarity between synthetic pools and the unseen benchmark.
- **CLI Usage**:
  ```bash
  python scripts/mine_negatives.py --confidence-threshold 0.85 --output-path data/review/mined_hard_negatives.jsonl
  ```

---

### `scripts/pipeline.py`
- **Purpose**: Production inference engine implementing the `HybridJournalPipeline` class with advanced conflict resolution.
- **Conflict Resolution Rules**:
  1. **Longest Span Wins**: A shorter dictionary span never truncates or splits a longer ML span (e.g., `Tailwind CSS` over `CSS`, `access control systems` over `access control`).
  2. **ML Label Authority**: On label disagreements, the ML classification takes precedence (e.g., keeping `data entry` as `CLERICAL_TERM`).
  3. **Abstention Fallback**: Dictionary predictions are accepted *only* when the ML component makes no prediction overlapping that span.
- **Confidence Routing**:
  - Confidence $\ge 0.80$: Marked `ACCEPTED`.
  - Confidence $< 0.80$: Marked `NEEDS_REVIEW`.
- **Python Usage**:
  ```python
  from scripts.pipeline import HybridJournalPipeline

  pipe = HybridJournalPipeline(model_path="models/ner_trf/model-best", terms_csv_path="data/terms.csv")
  result = pipe.predict("Configured Docker containers and assisted in document filing.")
  print(result["entities"])
  ```

---

### `scripts/prepare_trstr_llm.py`
- **Purpose**: Combines the real training split (`data/training/train.spacy`), validated LLM-generated synthetic records (`data/synthetic_llm_generated.jsonl`), and mined hard negatives (`data/review/mined_hard_negatives.jsonl`) into the unified TRSTR-LLM training dataset.
- **Validates**:
  - Target negative ratio ($25\%\text{--}35\%$).
  - Class balance between `IT_TERM` and `CLERICAL_TERM`.
- **CLI Usage**:
  ```bash
  python scripts/prepare_trstr_llm.py
  ```

---

### `scripts/test_pipeline_regressions.py`
- **Purpose**: Automated regression unit tests validating the hybrid pipeline's conflict resolution logic.
- **Test Invariants**:
  - `test_longest_span_ml_over_dict`: ML longer span takes priority over shorter dictionary sub-span.
  - `test_ml_label_authority_on_disagreement`: ML label takes priority over dictionary label on disagreement.
  - `test_dictionary_used_only_when_ml_abstains`: Dictionary matches accepted only on ML abstention.
  - `test_dict_longer_than_partial_ml_span`: When dictionary span is longer than a partial ML span, longer span wins with ML label authority.
- **CLI Usage**:
  ```bash
  python -m unittest scripts/test_pipeline_regressions.py
  ```

---

### `scripts/training.py`
- **Purpose**: Fine-tunes the spaCy RoBERTa transformer pipeline on GPU using early stopping.
- **Key Parameters**:
  - `--steps`: Maximum training steps (default: `2500`).
  - `--eval-freq`: Evaluation frequency on dev DocBin (default: `50`).
  - `--patience`: Early stopping patience steps without dev improvement (default: `400`).
  - `--gpu-id`: GPU device ID (default: `0`).
- **CLI Usage**:
  ```bash
  python scripts/training.py \
    --train-path data/training/train.spacy \
    --dev-path data/training/dev.spacy \
    --output-dir models/ner_trf \
    --steps 2500 \
    --eval-freq 50 \
    --patience 400 \
    --gpu-id 0
  ```
