# Hard Negative Mining & Stylistic Overlap Analysis

## 1. Overview & Motivation

In domain-specific Named Entity Recognition (NER) for internship and operational journals, models frequently suffer from **extractive false-positive bias**. When trained predominantly on positive examples or high-density technical logs, the transformer learns to associate common operational nouns (such as administrative events, generic equipment, or routine activities) with entity labels.

To counteract this bias without manual relabeling, the pipeline incorporates **Hard Negative Mining** ([`scripts/mine_negatives.py`](../scripts/mine_negatives.py)):
- Automatically discovers authentic non-entity sentences where the baseline model predicts spurious entities with high confidence ($\ge 0.85$).
- Ingests these failure cases into the training pool as true negatives (`entities: []`) during dataset preparation ([`scripts/prepare_trstr_llm.py`](../scripts/prepare_trstr_llm.py)).
- Calibrates the precision of the resulting model (TRSTR-LLM) so that the transformer learns when to **abstain** from extracting generic phrases.

---

## 2. Hard Negative Mining Pipeline

```
┌────────────────────────────────┐
│  Authentic Data Repository    │
│       (data/data.jsonl)        │
└──────────────┬─────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Zero-Leakage Partition Filter                                  │
│  - Exclude data/training/test.spacy   (Held-Out Test Set)       │
│  - Exclude data/training/dev.spacy    (Validation Set)          │
│  - Exclude data/test/unseen_benchmark.jsonl (OOV Benchmark)     │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Candidate Negative Selection                                    │
│  - Gold entities == 0                                            │
│  - Strictly within the training partition pool                   │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Transformer Model Inference & Error Mining                      │
│  - Run pipeline in transformer_only mode                         │
│  - Identify predicted entities with confidence >= threshold     │
│  - Filter out overlapping spans if any gold entity exists        │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Deduplication & Review Export                                   │
│  - Deduplicate across (text, false_positive_term) keys           │
│  - Export to data/review/mined_hard_negatives.jsonl              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Strict Zero-Data-Leakage Protocol

A primary risk in active error mining is accidentally evaluating or leaking held-out evaluation sentences into the training distribution. The mining script enforces strict isolation programmatically:

1. **Test & Dev Split Exclusion**: The script dynamically loads texts from `data/training/test.spacy` and `data/training/dev.spacy` using `DocBin`. Any sentence whose text matches a test or dev document (case-insensitively, stripped) is strictly excluded from candidate mining.
2. **Benchmark Isolation**: Texts from `data/test/unseen_benchmark.jsonl` are parsed and added to the exclusion set, preventing benchmark contamination.
3. **Training Partition Confinement**: Only authentic sentences designated for the training split are scanned for model confusion.

---

## 4. Downstream Integration in TRSTR-LLM

Once mined, candidates in [`data/review/mined_hard_negatives.jsonl`](../data/review/mined_hard_negatives.jsonl) are processed by [`scripts/prepare_trstr_llm.py`](../scripts/prepare_trstr_llm.py) during Phase 12 of [`main.ipynb`](../main.ipynb):

1. **Accidental Entity Filtering**:
   - Each mined candidate is cross-referenced against [`data/terms.csv`](../data/terms.csv).
   - If a sentence contains an actual cataloged domain term (which may have been missed during original manual annotation), it is filtered out to avoid training the model to ignore genuine domain terms.
2. **Target Negative Ratio Balancing**:
   - Mined negatives are merged with authentic training records and LLM-generated synthetic records.
   - The dataset builder monitors the composite **Negative Ratio** ($25\%\text{--}35\%$ target band) to maintain class balance between `IT_TERM`, `CLERICAL_TERM`, and non-entity context.

---

## 5. Stylistic Overlap Analysis

To guarantee that synthetic data does not memorize or mimic the evaluation benchmark, [`scripts/mine_negatives.py`](../scripts/mine_negatives.py) calculates quantitative lexical and n-gram overlap metrics:

- **Vocabulary Jaccard Similarity**: Measures token-level intersection between synthetic templates and the unseen benchmark.
- **Bigram & Trigram Jaccard Similarity**: Evaluates whether sequential phrasing or multi-token templates from the benchmark were copied into synthetic training sentences. Low values ($\approx 0.00\text{--}0.03$) mathematically verify stylistic independence.
- **Token Length Diagnostics**: Compares sentence length distributions between generated records and evaluation partitions.

---

## 6. Execution & CLI Options

The mining script can be executed as a standalone module or called within larger automated workflows:

```bash
python scripts/mine_negatives.py [OPTIONS]
```

### Supported Arguments

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model-path` | `str` | Auto-detected | Path to the spaCy model checkpoint to audit for false positives. |
| `--confidence-threshold` | `float` | `0.85` | Minimum model confidence required to classify an extraction as a hard false positive. |
| `--data-path` | `str` | `data/data.jsonl` | Path to authentic source annotations. |
| `--test-spacy` | `str` | `data/training/test.spacy` | Path to held-out test DocBin for leakage exclusion. |
| `--dev-spacy` | `str` | `data/training/dev.spacy` | Path to dev DocBin for leakage exclusion. |
| `--unseen-benchmark` | `str` | `data/test/unseen_benchmark.jsonl` | Path to unseen benchmark for leakage exclusion. |
| `--output-path` | `str` | `data/review/mined_hard_negatives.jsonl` | Output JSONL destination. |

---

## 7. Output Specifications

### 7.1 Mined Records Schema (`mined_hard_negatives.jsonl`)

Each line output to `data/review/mined_hard_negatives.jsonl` is a JSON record structured as follows:

```json
{
  "text": "<full sentence context>",
  "false_positive_term": "<extracted term>",
  "false_positive_category": "<IT_TERM | CLERICAL_TERM>",
  "confidence": 0.9854,
  "gold_entity_count": 0
}
```

### 7.2 Summary Output Format

Upon execution completion, the script reports summary statistics:

1. **Mining Summary**:
   - `total_instances_mined`: Count of unique high-confidence false positive spans discovered.
   - `unique_fp_terms`: Number of distinct terms erroneously predicted.
   - `top_fp_terms`: Frequency-ranked list of top false-positive tokens.
   - `model_used`: Checkpoint evaluated during mining.
2. **Stylistic Overlap Report**:
   - `synthetic_records` / `benchmark_records`: Sample counts compared.
   - `vocab_overlap_pct`: Percent of benchmark vocabulary present in the synthetic pool.
   - `bigram_jaccard_similarity`: Sequential pair overlap index.
   - `trigram_jaccard_similarity`: Template copying indicator index.
