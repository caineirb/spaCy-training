# Synthetic Data Augmentation Methodology & 3-Way Comparative Evaluation

**Document Version:** 2.0.0  
**Project:** Hybrid NER + Classification Pipeline for OJT Journal Task Tagging  
**Last Updated:** October 2026  

---

## 1. Overview & Methodological Motivation

In domain-specific Named Entity Recognition (NER) for internship and operational journals, models frequently face a dual bottleneck:

1. **In-Domain Syntactic Scarcity**: Authentic student journals often express similar tasks using repetitive phrasing, which limits the contextual variety available to a Transformer backbone during training.
2. **Out-of-Vocabulary (OOV) Fragility**: Authentic student logs document only the specific local tools and tasks present at their host companies. When novel enterprise software or administrative procedures appear in unseen production journals, a model that relies solely on dictionary memorization fails to generalize.

To investigate whether synthetic data augmentation can solve both challenges without causing harmful distributional shift or hallucination, this project establishes a formal 3-way empirical comparison across three conditions:

- **Condition 1: TRTR (Train Real, Test Real)**: Baseline model trained exclusively on authentic, manually annotated OJT student journal data (`data/data.jsonl`, 868 training records).
- **Condition 2: TRSTR-Paraphrase (Train Real + Synthetic Paraphrase, Test Real)**: Model trained on authentic records augmented with 556 high-confidence T5 seq2seq sentence paraphrases (`data/synthetic_paraphrases.jsonl`, 1,424 total training records).
- **Condition 3: TRSTR-LLM (Train Real + Synthetic LLM, Test Real)**: Model trained on authentic records augmented with 300 novel, LLM-direct generated records from Gemini Flash Lite (`data/synthetic_llm_generated.jsonl`) plus curated mined hard negatives (`data/review/mined_hard_negatives.jsonl`, 1,274 total training records).

> [!IMPORTANT]
> **Strict Evaluation Isolation**: Synthetic data exists **strictly on the training side**. All validation and test sets—including the real held-out test split (`data/training/test.spacy`, 187 docs), the real validation split (`data/training/dev.spacy`, 186 docs), and the controlled unseen-term benchmark (`data/test/unseen_benchmark.jsonl`, 85 sentences)—consist exclusively of authentic, isolated records. Synthetic data never touches the evaluation side.

---

## 2. Augmentation Approaches

### 2.1 Condition 2: Seq2Seq Paraphrasing (T5-Base)

The paraphrase augmentation pipeline utilizes `Vamsi/T5_Paraphrase_Paws` to generate syntactic variations of authentic student sentences while retaining ground-truth entity spans:

- **Filtering**: Paraphrases are accepted only if all original entity surface strings appear intact with valid word boundaries.
- **Span Tracking**: Substring matching re-identifies exact character offsets without distorting span boundaries.
- **Characteristic**: Excellent at generating stylistic variations of known phrasing, expanding in-domain syntactic diversity.

### 2.2 Condition 3: Direct LLM Generation (Google Gemini)

To move beyond the syntactic envelope of existing sentences, `scripts/generate_llm_synthetic.py` utilizes Gemini Flash Lite (`gemini-3.5-flash-lite`) via `google-genai`:

- **Real-Data Grounding**: Prompts dynamically sample 10 `IT_TERM` and 10 `CLERICAL_TERM` examples from `data/terms.csv`, alongside 3 positive and 2 negative authentic student sentences purely for tone and stylistic reference.
- **Anti-Copy Constraint**: The model is instructed to write novel, realistic internship log narratives across diverse workplace settings (LGUs, schools, corporate offices, healthcare, IT agencies).
- **Span Extraction via Substring Matching**: The LLM outputs surface entity text strings without character offsets. Exact character offsets `[start:end]` are calculated deterministically by [`locate_entity_spans()`](../scripts/generate_llm_synthetic.py) using word boundaries, avoiding generative tokenization offset errors.
- **Negative-Example Prompting**: Explicitly prompts for non-task workplace prose (commuting, orientation, cafeteria lunches, workstation cleaning) to maintain a ~30% negative ratio and reduce false-positive extractive bias.
- **Hard Negative Mining Integration**: Leverages [`scripts/mine_negatives.py`](../scripts/mine_negatives.py) to mine high-confidence false-positive predictions on unannotated authentic training text and ingest them as explicit negatives (`entities: []`).

---

## 3. Quality & Anti-Leakage Guardrails

Every generated candidate must satisfy automated filters before inclusion:

1. **Generic Noun Blocklist**: Blocks overly broad or ambiguous terms with low historical entity rates (`program`, `encode`, `encoded`, `office documents`, `coordination`, `orientation`, `deployment`).
2. **Bare Generic Noun Rejection**: Rejects isolated generic nouns (`system`, `database`, `software`, `code`, `computer`, `documents`, `files`, `records`, `forms`).
3. **Evaluation Benchmark Isolation**: Cross-references candidate entities against the 85 unseen benchmark terms (`data/test/unseen_benchmark.jsonl`). Any candidate containing an unseen term is dropped to guarantee zero benchmark contamination.
4. **Cross-Split Deduplication**: Enforces zero exact or near-duplicate sentences against `train.spacy`, `dev.spacy`, `test.spacy`, and `holdout.jsonl`.
5. **7-Check Data Leakage Audit**: Verified via [`scripts/check_data_leakage.py`](../scripts/check_data_leakage.py).

---

## 4. Dataset Composition Across Conditions

| Dataset / Partition | Total Docs | Total Entities | IT_TERM | CLERICAL_TERM | Composition / Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Authentic Dataset (`data.jsonl`)** | 1,241 | 1,306 | 722 | 584 | Complete authentic corpus |
| ├── **Real Train Split (`train.spacy`)** | 868 | 913 | 509 | 404 | 70% authentic training pool |
| ├── **Real Dev Split (`dev.spacy`)** | 186 | 219 | 120 | 99 | 15% authentic validation pool |
| └── **Real Test Split (`test.spacy`)** | 187 | 174 | 121 | 53 | 15% authentic held-out test pool |
| **Synthetic Paraphrases** | 556 | 622 | 344 | 278 | Accepted T5 paraphrases of train split |
| **Synthetic LLM Generated** | 300 | 402 | 231 | 171 | Gemini-generated synthetic records |
| **Controlled Unseen Benchmark** | 85 | 108 | 58 | 50 | Pure out-of-vocabulary evaluation |

### Training Pools by Condition

- **TRTR Pool**: 868 real records (100% authentic).
- **TRSTR-Paraphrase Pool**: 1,424 records (868 real + 556 paraphrases).
- **TRSTR-LLM Pool**: 1,274 records (868 real + 300 LLM synthetic + mined hard negatives).

---

## 5. Empirical Results: Held-Out Test Evaluation

All models were fine-tuned using identical hyperparameters (`max_steps=2500`, `eval_frequency=50`, `patience=400`, GPU device 0) on the NVIDIA RTX 3060 Laptop GPU and evaluated against the identical held-out test set (`test.spacy`, 187 docs) and unseen benchmark (`unseen_benchmark.jsonl`, 85 docs, 108 entities):

| Evaluation Metric | TRTR (Real Baseline) | TRSTR-Paraphrase (T5) | TRSTR-LLM (Gemini) | Delta (Para vs TRTR) | Delta (LLM vs TRTR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Held-Out Test Overall F1** | **67.63%** | **70.06%** | **64.25%** | **+2.43%** | -3.38% |
| Held-Out Test Overall Precision | 68.02% | **68.89%** | 62.50% | **+0.87%** | -5.52% |
| Held-Out Test Overall Recall | 67.24% | **71.26%** | 66.09% | **+4.02%** | -1.15% |
| ├── `IT_TERM` F1 | 67.80% | **70.78%** | 66.95% | **+2.98%** | -0.85% |
| ├── `IT_TERM` Precision | 69.57% | **70.49%** | 67.80% | **+0.92%** | -1.77% |
| ├── `IT_TERM` Recall | 66.12% | **71.07%** | 66.12% | **+4.95%** | 0.00% |
| ├── `CLERICAL_TERM` F1 | 67.27% | **68.47%** | 58.82% | **+1.20%** | -8.45% |
| ├── `CLERICAL_TERM` Precision | 64.91% | **65.52%** | 53.03% | **+0.61%** | -11.88% |
| └── `CLERICAL_TERM` Recall | 69.81% | **71.70%** | 66.04% | **+1.89%** | -3.77% |
| **Unseen Benchmark TRF Precision** | 64.52% | 65.35% | **78.00%** | +0.83% | **+13.48%** |
| **Unseen Benchmark TRF Recall** | 74.07% | **76.85%** | 72.22% | **+2.78%** | -1.85% |
| **Unseen Benchmark TRF F1** | 68.97% | 70.64% | **75.00%** | +1.67% | **+6.03%** |
| **Unseen Benchmark Hybrid F1** | 68.97% | 70.64% | **75.60%** | +1.67% | **+6.63%** |
| **Generalization Lift** | +66.66% | **+69.44%** | +65.74% | **+2.78%** | -0.92% |

---

## 6. Empirical Results: 5-Fold Cross-Validation

To verify that findings are statistically robust and not artifacts of a single random split, a rigorous 5-fold cross-validation experiment was executed across all 1,241 authentic documents using [`scripts/cross_validation.py`](../scripts/cross_validation.py):

| Metric | TRTR (Real Baseline) | TRSTR-Paraphrase | TRSTR-LLM |
| :--- | :---: | :---: | :---: |
| **Held-Out Validation Overall F1** | 70.00 ± 3.72% | **74.97 ± 2.91%** | 69.05 ± 3.24% |
| Held-Out Validation Overall Precision | 66.41 ± 3.70% | **72.26 ± 2.67%** | 65.81 ± 3.60% |
| Held-Out Validation Overall Recall | 74.08 ± 4.36% | **77.94 ± 3.53%** | 72.65 ± 3.20% |
| ├── `IT_TERM` F1 | 70.73 ± 3.50% | **74.84 ± 2.35%** | 70.75 ± 3.10% |
| └── `CLERICAL_TERM` F1 | 68.51 ± 5.14% | **75.21 ± 4.61%** | 65.69 ± 4.03% |
| **Unseen Benchmark TRF Recall** | 76.00 ± 3.59% | 73.94 ± 5.33% | **76.11 ± 1.80%** |
| **Unseen Benchmark TRF F1** | 49.41 ± 1.93% | 58.45 ± 9.51% | **74.66 ± 0.71%** |

---

## 7. Analytical Conclusions & Thesis Discussion

1. **TRSTR-Paraphrase Dominates In-Domain Generalization**:
   - Paraphrasing directly targets the in-domain syntactic bottleneck by rephrasing authentic student expressions without changing vocabulary.
   - Resulted in the highest overall validation F1 across 5 folds (**74.97% ± 2.91%**, a **+4.97%** improvement over TRTR) with balanced gains across both `IT_TERM` (+4.11%) and `CLERICAL_TERM` (+6.70%).
   - Achieved highest held-out test F1 (**70.06%**) and highest generalization lift on the unseen benchmark (**+69.44%**).

2. **TRSTR-LLM Delivers Exceptional OOV Precision & Stability**:
   - Direct LLM generation introduces novel contexts and strict negative examples that train the model when to *abstain*.
   - Across the 5 folds, TRSTR-LLM boosted unseen benchmark F1 from **49.41% to 74.66% (+25.25%)** with remarkable consistency (standard deviation of only **±0.71%**).
   - On the held-out test split, TRSTR-LLM recorded **78.00% precision** on the unseen benchmark (+13.48% over TRTR), preventing the hallucinated false positives common in open-domain NER.
   - However, the introduction of standardized, grammatically complete LLM phrasing causes a mild distributional penalty on informal, telegraphic student writing, lowering held-out authentic test F1 to 64.25%.

3. **Methodological Takeaway**:
   - For **maximizing in-domain performance on authentic student logs**, T5-based paraphrase augmentation (`TRSTR-Paraphrase`) is the superior strategy.
   - For **robust out-of-vocabulary entity extraction in unconstrained environments**, direct LLM augmentation with hard negative mining (`TRSTR-LLM`) offers unmatched precision and calibration.
   - The hybrid pipeline's deterministic EntityRuler layer and ML-authority conflict resolution ensure that dictionary-known entities maintain 100% precision regardless of the ML model selected.
