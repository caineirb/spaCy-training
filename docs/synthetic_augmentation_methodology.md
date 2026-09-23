# Synthetic Data Augmentation Methodology & TRTR vs. TRSTR Ablation

## 1. Overview & Methodological Context

To evaluate the impact of training corpus size and syntactic diversity on out-of-vocabulary generalization, this project establishes a formal ablation protocol comparing two distinct conditions:
- **TRTR (Train Real, Test Real)**: Baseline model trained exclusively on authentic, manually annotated OJT student journal data (`data/data.jsonl`, 687 training records).
- **TRSTR (Train Real + Synthetic, Test Real)**: Model trained on the authentic dataset augmented with controlled, paraphrase-based and template-based synthetic sentences (1,235 total records: 687 real, 548 synthetic).

> [!IMPORTANT]
> **Strict Evaluation Isolation**: Synthetic data appears **strictly on the training side of TRSTR**. All validation and test sets—including the real held-out test split (`test.spacy`), the real validation set (`dev.spacy`), and the controlled unseen-term benchmark (`unseen_benchmark.jsonl`)—consist solely of authentic records. Synthetic data never touches the evaluation side.

---

## 2. Augmentation Generation Pipeline

### 2.1 Paraphrase-Based Augmentation (Primary Component)
- **Model**: `humarin/chatgpt_paraphraser_on_T5_base` (T5-base seq2seq model fine-tuned for diverse paraphrasing) executed locally on CUDA (RTX 3060 Laptop GPU).
- **Generation Settings**: Beam search with `num_beams=8`, `num_return_sequences=5`, `no_repeat_ngram_size=2`, and `max_length=96`.
- **Literal Span Preservation**: For each candidate paraphrase, an exact boundary-sensitive substring match checks for the literal entity text. If any entity text is altered, truncated, or matched within another word, the candidate is discarded. Start and end offsets are dynamically recomputed.
- **Task Stability Guardrail**: Automated filters reject candidates exhibiting cross-domain lexical leakage (e.g., clerical tasks receiving IT programming keywords such as `scripting`, `coding`, `debugging`) or abnormal sentence length ratios ($< 0.5\times$ or $> 2.0\times$).
- **Negative Sentence Paraphrasing**: Authentic negative records (`entities: []`) are paraphrased naturally into non-entity sentences without hallucinating domain terms, preserving the negative ratio.

### 2.2 Template-Based Diversification (Secondary Supplement)
- **Purpose**: Exposes the model to underrepresented terminology from `data/terms.csv` and observed vocabulary within diverse syntactic frames.
- **Syntactic Frames**: Includes active voice, passive voice, fronted prepositional phrases, and imperative/gerund logbook formats.
- **Label Partitioning**: Clearly tagged with `"augmentation_type": "template"` in `data/training_trstr.jsonl` to ensure traceability.

---

## 3. Dataset Composition

The combined TRSTR training pool maintains established project guardrails (25–35% negative ratio, balanced classes):

| Split Component | Records | Percentage | Positive Records | Negative Records | Negative Ratio | IT_TERM Count | CLERICAL_TERM Count | Balance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Real Training Split** | 687 | 55.6% | 484 | 203 | 29.5% | 345 | 354 | 1.03 : 1 |
| **Paraphrase Augmentation** | 416 | 33.7% | 311 | 105 | 25.2% | 201 | 225 | 1.12 : 1 |
| **Template Supplement** | 132 | 10.7% | 122 | 10 | 7.6% | 62 | 62 | 1.00 : 1 |
| **Total TRSTR Pool** | **1,235** | **100.0%** | **917** | **318** | **25.7%** | **608** | **641** | **1.05 : 1** |

---

## 4. Leakage Audit & Evaluation Guardrails

Every synthetic record was audited via `scripts/check_data_leakage.py` against all evaluation partitions. The audit verifies 7 strict conditions:

1. **Check 1 (Document Duplication)**: 0 duplicate documents between train, dev, and test.
2. **Check 2 (Sentence Duplication)**: 0 duplicate sentences across train, dev, and test.
3. **Check 3 (Terms CSV Leakage)**: 0 unseen benchmark terms appear in `data/terms.csv`.
4. **Check 4 (Training Annotation Leakage)**: 0 unseen benchmark terms appear in `data/data.jsonl`.
5. **Check 5 (EntityRuler Matchability)**: 0 unseen benchmark terms matchable by dictionary EntityRuler.
6. **Check 6 (Holdout Isolation)**: 0 sentences in holdout partitions appear in training sets.
7. **Check 7 (Synthetic Pool Isolation)**: 
   - 0 synthetic sentences match evaluation sentences exactly.
   - 0 synthetic sentences have normalized string similarity $\ge 0.70$ (SequenceMatcher ratio) to any sentence in `test.spacy`, `dev.spacy`, `unseen_benchmark.jsonl`, or `holdout.jsonl`.
   - 0 synthetic records contain unseen benchmark vocabulary.

---

## 5. Empirical Results: TRTR vs. TRSTR Ablation

Both conditions were evaluated under identical hyperparameter conditions (`max_steps=2500`, `patience=400`, `eval_frequency=50`) on the exact same real evaluation partitions:

| Evaluation Dimension | Metric | TRTR (Real Only) | TRSTR (Real + Synth) | Absolute Delta | Relative Change |
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
| *Pipeline Integration* | **Hybrid Pipeline Recall** | 33.85% | **61.54%** | **+27.69%** | +81.80% |
| | **Generalization Lift** | +32.31% | **+60.00%** | **+27.69%** | +85.70% |

---

## 6. Analytical Findings

1. **Substantial Generalization Lift on Unseen Vocabulary**: Synthetic augmentation nearly doubled the Transformer's zero-shot inductive recall on unseen enterprise terms (**33.85% $\rightarrow$ 61.54%**, +27.69% absolute gain), with precision increasing from 22.45% to 38.10%. Syntactic variation provided critical contextual diversity, helping the self-attention heads recognize verb-object configurations independent of specific surface tokens.
2. **Held-Out Test Improvement Driven by Clerical Normalization**: On the real student journal test set, overall recall improved from 63.09% to 65.77% (+2.68%) and overall F1 rose from 60.06% to 61.25% (+1.19%). This improvement was concentrated in `CLERICAL_TERM`, where recall increased from 54.29% to 62.86% (+8.57%), overcoming the prior bottleneck where clerical tasks were frequently confused with narrative text.
3. **Realistic Ceiling**: While TRSTR significantly outperformed TRTR, it did not achieve the artificial 100% recall observed in earlier synthetic-only experiments. This demonstrates that real-world language contains inherent distributional variance and syntactic irregularities that require grounded evaluation rather than templated proxies.

---

## 7. Limitations Statement

> [!WARNING]
> **Thesis Methodological Disclaimer**: Synthetic augmentation was introduced specifically to compensate for the limited volume of the authentic annotated corpus and to isolate the effect of syntactic diversity on model generalization. The results obtained under the TRSTR condition should be interpreted strictly as empirical evidence of the transformer architecture's capacity to generalize given increased linguistic diversity, and **not** as a claim that the final system was trained exclusively on authentic student journals. Real-world deployment benchmarks should reference TRTR as the authentic baseline and TRSTR as the augmented potential ceiling.
