# Dataset Overview & Directory Contents

## 1. Dataset Specification & Unified Taxonomy

> **Note:** The full dataset documentation, schema definitions, and audit guarantees are provided here for quick reference. See the original project README for a concise link.

### Summary of Dataset Assets

The system enforces a unified binary taxonomy distinguishing technical workflows from administrative procedures:

- **`IT_TERM`**: Technologies, programming languages, software libraries, databases, IT infrastructure, hardware, and concrete technical workflows (e.g., `Python`, `PostgreSQL`, `Docker`, `Git`, `Cable Crimping`, `Database Administration`).
- **`CLERICAL_TERM`**: Office productivity tools, document handling, filing, record‑keeping, and administrative workflows (e.g., `Microsoft Excel`, `Police Clearance`, `log books`, `data encoding`, `filing`, `PESO office book`).

Across the complete authentic dataset ([`data.jsonl`](data.jsonl), **1,320** records), annotations contain **1,474** total entities (**933** `IT_TERM` and **541** `CLERICAL_TERM`, a balanced **1.72:1** ratio).

### Partition Splits & DocBin Summary

| Partition / Split          | File Path                                                               | Documents | Positive | Negative (Ratio) | Valid Entities |
| :---                       | :---                                                                    |   :---:   |   :---:  |      :---:       |      :---:     |
| **Authentic Full Corpus**  | [`data.jsonl`](data.jsonl)                                              | 1,320     | 868      | 452 (34.2%)      | 1,474          |
| ├── **Real Train Split**   | [`training/train.spacy`](training/train.spacy)                           | 923       | 606      | 317 (34.3%)      | 1,032          |
| ├── **Real Validation Split** | [`training/dev.spacy`](training/dev.spacy)                           | 198       | 128      | 70 (35.4%)       | 208            |
| └── **Held‑Out Test Split**   | [`training/test.spacy`](training/test.spacy)                       | 199       | 134      | 65 (32.7%)       | 233            |
| **TRSTR‑Paraphrase Pool**  | [`training/train_trstr_paraphrase.spacy`](training/train_trstr_paraphrase.spacy) | 1,488 | 855 | 633 (42.5%) | 1,416 |
| **TRSTR‑LLM Pool**         | [`training/train_trstr_llm.spacy`](training/train_trstr_llm.spacy)       | 1,588     | 1,000    | 588 (37.0%)      | 1,587          |
| **OOV Unseen Benchmark**   | [`test/unseen_benchmark.jsonl`](test/unseen_benchmark.jsonl)           | 85        | 85       | 0 (0.0%)         | 65 *gold*      |

*For complete schema definitions, negative‑sample guidelines, and audit guarantees, see the top‑level [README.md](../README.md).*.

## 2. Directory Contents

```text
spaCy-training/
├── data/
│   ├── terms.csv                     # Seed dictionary (371 terms: 215 IT, 156 Clerical)
│   ├── data.jsonl                    # Real authentic OJT journal dataset (1,320 records)
│   ├── synthetic_paraphrases.jsonl   # T5‑generated paraphrases (565 records)
│   ├── synthetic_llm_generated.jsonl # Gemini‑generated synthetic records (571 records)
│   ├── training_trstr_paraphrase.jsonl # Combined real + paraphrase pool (1,488 records)
│   ├── training_trstr_llm.jsonl      # Combined real + LLM + mined negatives pool (1,588 records)
│   ├── training/                     # DocBin partitions
│   │   ├── train.spacy                # Real train partition (923 docs, 1,032 entities)
│   │   ├── dev.spacy                  # Real validation partition (198 docs, 208 entities)
│   │   ├── test.spacy                 # Real held‑out test partition (199 docs, 233 entities)
│   │   ├── train_trstr_paraphrase.spacy # Paraphrase‑augmented DocBin
│   │   └── train_trstr_llm.spacy      # LLM‑augmented DocBin
│   ├── test/                         # Evaluation benchmarks
│   │   ├── unseen_benchmark.jsonl     # Out‑of‑vocabulary benchmark (85 docs, 65 gold entities)
│   │   ├── holdout.jsonl              # Permanent real‑world holdout set
│   │   └── raw/                       # Raw unannotated journal text files
│   ├── cv/                           # 5‑fold cross‑validation artefacts
│   │   ├── cv_metadata_5fold.json     # Fold split metadata
│   │   ├── results/                   # Per‑fold evaluation reports (JSON)
│   │   ├── fold_0/ … fold_4/          # DocBin splits per fold
│   │   └── cv_3way_results.json       # **Aggregated 3‑way CV results** (means ± std)
│   ├── eval_results/                 # Evaluation outputs per condition
│   │   ├── comparison/                # Test‑split & benchmark comparison reports
│   │   │   ├── trtr/                  # TRTR condition reports
│   │   │   ├── trstr_paraphrase/      # Paraphrase‑augmented reports
│   │   │   └── trstr_llm/             # LLM‑augmented reports
│   │   └── dictionary_overrides.jsonl # Log of dictionary‑label overrides
│   └── review/                       # Mined hard‑negative candidates
│       └── mined_hard_negatives.jsonl # 164 raw candidates, 94 clean accepted
```

## 3. Cross‑Validation Summary (Updated)

The latest 5‑fold cross‑validation on the **expanded 1,320‑record dataset** yields the following aggregated metrics (mean ± standard deviation) for each experimental condition:

| Metric                     | TRTR (Real Baseline) | TRSTR‑Paraphrase | TRSTR‑LLM |
| :---                       | :---:                | :---:           | :---:    |
| **Overall F1**             | 71.01 ± 2.81%        | 75.44 ± 2.94%   | 67.75 ± 2.98% |
| Overall Precision          | 68.57 ± 2.24%        | 74.28 ± 1.89%   | 66.87 ± 2.38% |
| Overall Recall             | 73.67 ± 3.72%        | 76.69 ± 4.20%   | 68.70 ± 3.97% |
| IT‑Term F1                 | 71.64 ± 3.62%        | 75.88 ± 3.76%   | 68.01 ± 3.66% |
| Clerical‑Term F1           | 69.89 ± 1.92%        | 74.81 ± 2.30%   | 67.29 ± 2.19% |
| Unseen Benchmark Recall (Transformer) | 85.54 ± 3.17% | 72.00 ± 4.50% | 89.54 ± 3.56% |
| Unseen Benchmark F1 (Transformer) | 54.50 ± 1.04% | 48.78 ± 2.65% | 67.06 ± 4.85% |

These figures complement the **single‑run test‑partition results** already displayed in the main README (Section 4). They provide a robust view of model stability across folds.

---

*For a deeper dive into the evaluation methodology, see the `docs/` folder and the main project README.*
