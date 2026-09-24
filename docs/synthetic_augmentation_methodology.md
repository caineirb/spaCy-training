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

### 2.3 Quality Guardrail: Generic-Noun Blocklist & Relabeling Filter
- **Upstream Contamination Mitigation**: An audit of early paraphrase generations identified that generic nouns/verbs surviving in `data.jsonl` (e.g., `"system workflows"`, `"data requirements"`, `"encode"`, `"formatting"`, `"program"`, `"technical"`, `"layouts"`, `"debugging"`) were being amplified into synthetic data.
- **Selective Augmentation Filtering**: To preserve authentic training annotations while preventing synthetic distortion, a strict 34-term blocklist (`GENERIC_NOUN_BLOCKLIST`) filters out borderline activities and concepts during synthetic candidate generation. Any paraphrase whose only entities were generic is dropped entirely.
- **Label Corrections**: Specific ambiguous terms (such as `"printing"` originally misclassified as `IT_TERM`) are automatically normalized to their correct domain (`CLERICAL_TERM`) via `ENTITY_RELABEL_MAP`.
- **Zero Drift Verification**: Verified that zero generic nouns remain in `data/synthetic_paraphrases.jsonl` and all 7 isolation checks pass.

---

## 3. Dataset Composition

The combined TRSTR training pool maintains established project guardrails (balanced classes, preserved negative ratio):

| Split Component | Records | Percentage | Positive Records | Negative Records | Negative Ratio | IT_TERM Count | CLERICAL_TERM Count | Balance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Real Training Split** | 687 | 39.3% | 484 | 203 | 29.5% | 349 | 350 | 1.00 : 1 |
| **Paraphrase Augmentation** | 928 | 53.1% | 727 | 201 | 21.7% | 441 | 548 | 1.24 : 1 |
| **Template Supplement** | 132 | 7.6% | 124 | 8 | 6.1% | 62 | 62 | 1.00 : 1 |
| **Total TRSTR Pool** | **1,747** | **100.0%** | **1,335** | **412** | **23.6%** | **852** | **960** | **1.13 : 1** |

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
| **Held-Out Real Test Set** | **Overall F1** | 60.06% | **63.19%** | **+3.13%** | +5.21% |
| (`data/training/test.spacy`) | Overall Precision | 57.32% | **61.39%** | **+4.07%** | +7.10% |
| | Overall Recall | 63.09% | **65.10%** | **+2.01%** | +3.19% |
| *Per-Label Performance* | `IT_TERM` F1 | **62.22%** | 61.99% | -0.23% | -0.37% |
| | `IT_TERM` Precision | 55.45% | **57.61%** | **+2.16%** | +3.90% |
| | `IT_TERM` Recall | **70.89%** | 67.09% | -3.80% | -5.36% |
| | `CLERICAL_TERM` F1 | 57.14% | **64.71%** | **+7.57%** | +13.25% |
| | `CLERICAL_TERM` Precision | 60.32% | **66.67%** | **+6.35%** | +10.53% |
| | `CLERICAL_TERM` Recall | 54.29% | **62.86%** | **+8.57%** | +15.79% |
| **Unseen Benchmark** | **Transformer Recall** | 33.85% (22/65) | **66.15% (43/65)** | **+32.30%** | **+95.42%** |
| (Out-of-Vocabulary Probes) | Transformer Precision | 22.45% | **37.72%** | **+15.27%** | +68.02% |
| | Transformer F1 | 26.99% | **48.04%** | **+21.05%** | +77.99% |
| *Pipeline Integration* | **Hybrid Pipeline Recall** | 33.85% | **66.15%** | **+32.30%** | +95.42% |
| | **Generalization Lift** | +32.31% | **+64.61%** | **+32.30%** | +100.0% |

---

## 6. Analytical Findings

1. **Dramatic Generalization Lift on Unseen Vocabulary**: Synthetic augmentation nearly doubled the Transformer's zero-shot inductive recall on unseen enterprise terms (**33.85% $\rightarrow$ 66.15%**, identifying 43 out of 65 out-of-vocabulary terms vs. 22 for TRTR, a +32.30% absolute gain), with Transformer F1 nearly doubling from 26.99% to 48.04%. Syntactic variation provided critical contextual diversity, helping self-attention heads recognize verb-object configurations independent of specific surface tokens.
2. **Precision Surge via Clean Filtering**: Following the removal of generic noun noise (`"program"`, `"system workflows"`, `"debugging"`, etc.), real held-out test precision jumped from 57.32% to **61.39% (+4.07%)**, demonstrating that filtering generic concepts directly prevented the model from making spurious false-positive predictions on everyday logbook vocabulary.
3. **Major Gains in Clerical Normalization**: On the real student journal test set, `CLERICAL_TERM` F1 surged from 57.14% to **64.71% (+7.57%)**, driven by simultaneous increases in precision (60.32% $\rightarrow$ 66.67%) and recall (54.29% $\rightarrow$ 62.86%), overcoming the prior bottleneck where clerical tasks were frequently confused with narrative text.
4. **Realistic Ceiling**: While TRSTR significantly outperformed TRTR, it did not achieve the artificial 100% recall observed in earlier synthetic-only experiments. This demonstrates that real-world language contains inherent distributional variance and syntactic irregularities that require grounded evaluation rather than templated proxies.

---

## 7. Limitations Statement

> [!WARNING]
> **Thesis Methodological Disclaimer**: Synthetic augmentation was introduced specifically to compensate for the limited volume of the authentic annotated corpus and to isolate the effect of syntactic diversity on model generalization. The results obtained under the TRSTR condition should be interpreted strictly as empirical evidence of the transformer architecture's capacity to generalize given increased linguistic diversity, and **not** as a claim that the final system was trained exclusively on authentic student journals. Real-world deployment benchmarks should reference TRTR as the authentic baseline and TRSTR as the augmented potential ceiling.
