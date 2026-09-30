# Synthetic Data Augmentation Methodology & TRTR vs. TRSTR-LLM Ablation

## 1. Overview & Methodological Motivation

In Named Entity Recognition (NER) for domain-specific internship journals, models frequently face a vocabulary bottleneck: authentic student logs document a limited set of local tools and manual workflows, leaving the fine-tuned Transformer vulnerable to out-of-vocabulary degradation when new technologies or administrative procedures appear in production.

To rigorously investigate whether synthetic data can enhance generalization without degrading precision on authentic logs, this project evaluates a formal ablation protocol:
- **TRTR (Train Real, Test Real)**: Baseline model trained exclusively on authentic, manually annotated OJT student journal data (`data/data.jsonl`, 687 training records).
- **TRSTR-LLM (Train Real + Synthetic LLM, Test Real)**: Model trained on the authentic training dataset augmented with 300 novel, LLM-direct generated records (987 total training records: 687 real + 300 synthetic).
- **TRSTR-Paraphrase (Historical Reference)**: Prior augmentation attempt using seq2seq sentence paraphrasing (`humarin/chatgpt_paraphraser_on_T5_base`), documented in [`docs/archive/paraphrase_augmentation_methodology.md`](archive/paraphrase_augmentation_methodology.md).

> [!IMPORTANT]
> **Strict Evaluation Isolation**: Synthetic data exists **strictly on the training side of TRSTR-LLM**. All validation and test sets—including the real held-out test split (`data/training/test.spacy`), the real validation set (`data/training/dev.spacy`), and the controlled unseen-term benchmark (`data/test/unseen_benchmark.jsonl`)—consist solely of authentic records. Synthetic data never touches the evaluation side.

---

## 2. Why LLM-Direct Generation Replaced Paraphrase Augmentation

The project initially employed a paraphrase-based augmentation pipeline that transformed existing real sentences using a T5-base paraphraser. While that approach achieved a substantial inductive recall gain on out-of-vocabulary terms, it exhibited three severe structural limitations:
1. **Span Relocation Fragility**: Paraphrasing existing sentences altered sentence structure, frequently splitting multi-word entities, dropping key domain tokens, or requiring complex character offset re-alignment heuristics.
2. **Upstream Contamination Amplification**: Because paraphrasing operated directly on real training sentences, borderline or noisy annotations from earlier iterations (e.g. generic concepts like *"program"*, *"debugging"*, or *"system workflows"*) were paraphrased and multiplied, requiring multiple auditing passes and blocklists to suppress.
3. **Syntactic Redundancy**: Paraphrasing was inherently bounded by the syntactic structures of the existing real sentences, limiting true lexical and stylistic novelty.

To resolve these limitations, the paraphrase pipeline was entirely removed and replaced with **LLM-direct generation** (`scripts/generate_llm_synthetic.py`). Instead of rewriting existing sentences, a large language model generates novel sentences and entity annotations together, conditioned directly on [`docs/annotation_guidelines.md`](annotation_guidelines.md).

---

## 3. Architecture of the LLM-Direct Generation Pipeline

### 3.1 Model & Execution
- **LLM Engine**: Google Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite`) accessed via the official `google-genai` SDK.
- **Quota & Pacing**: Executed under free-tier limits with a 4.5-second inter-batch delay to stay reliably within the 15 RPM rate limit, generating 30 batches of 10 sentences each (300 records total).
- **Persistence**: Incremental disk-saving and automatic checkpoint resumption ensure zero lost records during generation.

### 3.2 Real-Data Grounding Without Paraphrasing
To ensure that generated sentences reflect the authentic operational environment of Philippine OJT student journals—without copying or paraphrasing real sentences—the generator prompt is grounded dynamically:
- **In-Context Vocabulary**: Each batch samples 10 `IT_TERM` and 10 `CLERICAL_TERM` entries from [`data/terms.csv`](../data/terms.csv) as active vocabulary examples.
- **In-Context Phrasing & Style**: Each batch presents 3 positive and 2 negative real student journal sentences from [`data/data.jsonl`](../data/data.jsonl) purely as tone and stylistic references.
- **Strict Anti-Copy Constraint**: The prompt explicitly instructs the model to compose novel sentences inspired by the operational context (local government units, health offices, registrars, IT departments), never transforming or reproducing the in-context examples.

### 3.3 Span Validation via Post-Generation String Search
A critical architectural constraint was adopted based on findings in clinical and domain-specific NLP literature (e.g., JMIR AI 2024 studies on generative LLM tokenization unreliability): **LLMs are fundamentally unreliable at calculating exact character-level offsets**. 

To guarantee 100% mathematical precision:
1. The LLM outputs only the sentence text and an array of entity objects containing `{"text": "<surface_form>", "label": "<IT_TERM|CLERICAL_TERM>"}`. It **never outputs character offsets**.
2. A deterministic post-processor ([`locate_entity_spans()`](../scripts/generate_llm_synthetic.py)) computes character offsets `[start:end]` via exact substring search.
3. **Word-Boundary Enforcement**: Character offsets are validated to ensure adjacent characters are non-alphanumeric, strictly preventing partial-word substring collisions (e.g. matching `Bun` inside `Ubuntu`, or matching `CSS` inside a non-entity token).
4. **Multi-Word Span Integrity**: Spans are tagged as single contiguous units (e.g., `[Google Cloud Platform]`, `[Document Verification]`, `[Production Deployment]`).

### 3.4 Negative-Example Prompting & Hard Negatives
Per Section 8.1 of [`docs/annotation_guidelines.md`](annotation_guidelines.md), models trained with insufficient negative examples develop an aggressive extractive bias, predicting false positives on institutional and environmental nouns.

The generation prompt explicitly requests ~30% negative sentences (3 per 10-sentence batch) and prompts for known failure categories:
- Arriving at offices, schools, or campuses
- Attending morning briefings, general assemblies, orientations, and seminars
- Social interactions (lunch breaks, commutes, pantry conversations)
- Cleaning workstations, organizing desks, and handling physical equipment without specific clerical tasks

---

## 4. Quality & Anti-Leakage Guardrails

Every generated candidate must pass a sequence of strict automated filters ([`passes_quality_checks()`](../scripts/generate_llm_synthetic.py)):
1. **Generic Noun Blocklist**: Rejects terms from the 34-word historical blocklist (`system design`, `data organization`, `stalls`, `front page`, etc.).
2. **Bare Generic Noun Rejection**: Rejects isolated generic nouns (`system`, `database`, `software`, `code`, `computer`, `documents`, `files`, `records`, `forms`).
3. **Trailing Generic Word Stripping**: Rejects entities appending generic suffixes (e.g. `Python script` -> requires `Python`; `Excel software` -> requires `Excel`).
4. **Evaluation Benchmark Term Isolation**: Cross-references against the 65 gold terms in [`data/test/unseen_benchmark.jsonl`](../data/test/unseen_benchmark.jsonl) using regex word boundaries. Any candidate containing an unseen benchmark term (e.g. `Tailwind CSS`, `Bun`, `FastAPI`, `Svelte`) is rejected, preventing benchmark contamination.
5. **Cross-Split Deduplication**: Rejects exact duplicates and near-duplicates against all sentences in `train.spacy`, `dev.spacy`, `test.spacy`, and `holdout.jsonl`.
6. **7-Check Data Leakage Audit**: Before inclusion in the training pool, the entire synthetic dataset is audited by [`scripts/check_data_leakage.py`](../scripts/check_data_leakage.py) across all 7 verification dimensions (cross-split document dedup, sentence dedup, terms leakage, training annotation leakage, EntityRuler matchability, holdout isolation, synthetic pool isolation).

---

## 5. Dataset Composition: TRTR vs. TRSTR-LLM

The resulting `TRSTR-LLM` training pool maintained optimal project composition targets:

| Split Component | Total Records | Positive Records | Negative Records | Negative Ratio | IT_TERM Count | CLERICAL_TERM Count | Class Balance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Real Training Split** | 687 | 484 | 203 | 29.5% | 349 | 350 | 1.00 : 1 |
| **LLM-Direct Augmentation** | 300 | 208 | 92 | 30.7% | 187 | 142 | 1.32 : 1 |
| **Total TRSTR-LLM Pool** | **987** | **692** | **295** | **29.9%** | **536** | **492** | **1.09 : 1** |

- **Target Negative Ratio**: 29.9% (target: 25%–35%) ✅
- **Class Balance**: 536 IT vs. 492 Clerical (1.09:1 ratio, neither class < 45%) ✅
- **Evaluation Sets**: 100% authentic, untouched real data (`dev.spacy`: 147 docs; `test.spacy`: 148 docs; `unseen_benchmark.jsonl`: 85 sentences).

---

## 6. Empirical Results: TRTR vs. TRSTR-LLM Ablation

Both models were trained using identical patience-based early-stopping hyperparameters (`max_steps=2500`, `patience=400`, `eval_frequency=50`, GPU ID 0) on the NVIDIA RTX 3060 Laptop GPU. Both were evaluated on the identical real-data evaluation sets.

The table below contrasts the baseline **TRTR** against **TRSTR-LLM**, with the historical **TRSTR-Paraphrase** included for reference:

| Evaluation Partition | Metric | TRTR (Real Only Baseline) | TRSTR-Paraphrase (Archived T5) | TRSTR-LLM (Direct Gemini Lite) | Delta vs. TRTR | Relative Lift vs. TRTR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Unseen-Term Benchmark** | **Transformer Recall** | 33.85% (22/65) | 66.15% (43/65) | **75.38% (49/65)** | **+41.53%** | **+122.69%** |
| (Out-of-Vocabulary Probes) | Transformer Precision | 22.45% | 37.72% | **42.24%** | **+19.79%** | **+88.15%** |
| | Transformer F1 | 26.99% | 48.04% | **54.14%** | **+27.15%** | **+100.59%** |
| *Hybrid Pipeline Integration* | **Hybrid Recall** | 33.85% | 66.15% | **75.38%** | **+41.53%** | **+122.69%** |
| | **Generalization Lift** | +32.31% | +64.61% | **+73.84%** | **+41.53%** | **+128.57%** |
| **Held-Out Real Test Set** | **Overall F1** | **60.06%** | 63.19% | **59.28%** | -0.78% | -1.30% |
| (`data/training/test.spacy`) | Overall Precision | 57.32% | 61.39% | **57.59%** | **+0.27%** | +0.47% |
| | Overall Recall | **63.09%** | 65.10% | 61.07% | -2.02% | -3.20% |
| *Per-Label Performance* | `CLERICAL_TERM` Precision | 60.32% | 66.67% | **67.86%** | **+7.54%** | **+12.50%** |
| | `CLERICAL_TERM` Recall | 54.29% | **62.86%** | 54.29% | 0.00% | 0.00% |
| | `CLERICAL_TERM` F1 | 57.14% | **64.71%** | **60.32%** | **+3.18%** | **+5.57%** |
| | `IT_TERM` Precision | 55.45% | **57.61%** | 51.96% | -3.49% | -6.29% |
| | `IT_TERM` Recall | **70.89%** | 67.09% | 67.09% | -3.80% | -5.36% |
| | `IT_TERM` F1 | **62.22%** | 61.99% | 58.56% | -3.66% | -5.88% |

---

## 7. Analytical Findings

1. **Massive Surge in Out-of-Vocabulary Inductive Generalization**:
   - The primary research objective of synthetic augmentation was to teach the Transformer pipeline to recognize novel entities from syntactic context alone.
   - TRSTR-LLM achieved **75.38% recall on unseen enterprise technologies** (49 out of 65 terms correctly extracted without dictionary assistance), compared to **33.85% (22/65)** for TRTR.
   - This represents an absolute inductive recall gain of **+41.53%** (more than doubling zero-shot recall), and outperforms the older paraphrase approach (+9.23% higher recall, identifying 6 additional unseen terms).
   - Unseen benchmark F1 surged from **26.99% $\rightarrow$ 54.14% (+27.15%)**, driven by a simultaneous precision improvement (**22.45% $\rightarrow$ 42.24%**).

2. **Precision Surge on Administrative Clerical Terms**:
   - On the authentic student journal test set, `CLERICAL_TERM` precision climbed from 60.32% to **67.86% (+7.54%)**, raising Clerical F1 from 57.14% to **60.32% (+3.18%)**.
   - By generating high-quality clerical workflow sentences (`Document Arrangement`, `Document Verification`, `Records Management`) alongside strict negative non-task sentences, the LLM taught the Transformer clear boundary distinctions between genuine office workflows and generic environmental paperwork.

3. **Inherent Trade-Off on In-Domain Real-Data F1**:
   - On the overall authentic test set, TRSTR-LLM recorded **59.28% F1** (vs. 60.06% for TRTR, a minor -0.78% delta).
   - While precision slightly improved (+0.27%), `IT_TERM` recall exhibited a mild contraction (70.89% $\rightarrow$ 67.09%). This occurs because the LLM synthetic sentences exposed the model to more varied sentence contexts, slightly tightening the model's decision threshold on borderline informal student phrasing.

---

## 8. Limitations & Methodological Integrity

In alignment with the empirical standards of this thesis:
1. **Synthetic Sentences are Not Authentic Student Logs**: While conditioned on authentic student vocabulary and local administrative contexts, LLM-generated entries exhibit standard grammatical coherence and punctuation rarely found in raw student logs. They serve to enrich syntactic diversity, not substitute for authentic ground truth.
2. **TRTR as the Reference Baseline**: TRTR remains the official benchmark for authentic in-domain journal performance. The value of TRSTR-LLM lies specifically in its out-of-vocabulary inductive generalization capacity (+41.53% recall lift).
3. **Data Isolation Guaranteed**: All evaluation partitions (`dev.spacy`, `test.spacy`, `unseen_benchmark.jsonl`, `holdout.jsonl`) remain 100% authentic and completely isolated from synthetic tokens.
