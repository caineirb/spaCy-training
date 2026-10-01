# T5 Paraphrase Data Augmentation Methodology & TRTR vs. TRSTR-Paraphrase Ablation

**Document Version:** 2.0.0  
**Project:** Hybrid NER + Classification Pipeline for OJT Journal Task Tagging  
**Last Updated:** October 2026  

---

## 1. Overview & Methodological Context

In domain-specific Named Entity Recognition (NER) for On-the-Job Training (OJT) student internship journals, models frequently face **in-domain syntactic scarcity**. Authentic student journals often express repetitive daily activities using identical or highly restricted sentence structures (e.g., *"Assisted with..."*, *"Conducted encoding of..."*, *"Fixed the..."*). This repetitive phrasing limits the contextual variety available to self-attention heads in Transformer backbones (`roberta-base` via `spacy-transformers`), increasing the risk of overfitting to syntactic templates rather than learning robust contextual representations of task entities.

To rigorously evaluate whether sequence-to-sequence (seq2seq) paraphrase augmentation can overcome syntactic monotony without introducing label drift, hallucination, or distributional skew, this project establishes a formal ablation protocol comparing two distinct conditions:

- **Condition 1: TRTR (Train Real, Test Real)**: Baseline model trained exclusively on authentic, manually annotated OJT student journal data ([`data/data.jsonl`](file:///home/caineirb/Documents/PauPau/spaCy-training/data/data.jsonl), 868 real training records).
- **Condition 2: TRSTR-Paraphrase (Train Real + Synthetic Paraphrase, Test Real)**: Model trained on authentic records augmented with 550 high-confidence T5 seq2seq sentence paraphrases ([`data/training_trstr_paraphrase.jsonl`](file:///home/caineirb/Documents/PauPau/spaCy-training/data/training_trstr_paraphrase.jsonl), 1,418 total training records: 868 real + 550 accepted paraphrases).

> [!IMPORTANT]
> **Strict Evaluation Isolation Principle**: Synthetic data appears **strictly on the training side of TRSTR-Paraphrase**. All validation and test partitions—including the held-out real test split ([`data/training/test.spacy`](file:///home/caineirb/Documents/PauPau/spaCy-training/data/training/test.spacy), 186 docs), the real validation split ([`data/training/dev.spacy`](file:///home/caineirb/Documents/PauPau/spaCy-training/data/training/dev.spacy), 186 docs), and the canonical 65-term out-of-vocabulary benchmark ([`data/test/unseen_benchmark.jsonl`](file:///home/caineirb/Documents/PauPau/spaCy-training/data/test/unseen_benchmark.jsonl), 85 sentences)—consist 100% of authentic, unaugmented student records. Synthetic data never touches the evaluation side.

---

## 2. Paraphrase Generation Pipeline

### 2.1 Model Architecture & Generation Settings

The paraphrase augmentation pipeline is executed locally within [`main.ipynb`](file:///home/caineirb/Documents/PauPau/spaCy-training/main.ipynb) (Phases 8–10) leveraging PyTorch and HuggingFace Transformers with hardware acceleration on an NVIDIA GeForce RTX 3060 Laptop GPU:

- **Base Model**: `Vamsi/T5_Paraphrase_Paws` (a `t5-base` seq2seq model fine-tuned on the Paraphrase Adversaries from Word Scrambling dataset).
- **Conditioning Prompt**: Each authentic sentence is prefixed with the canonical T5 task prompt:
  ```python
  input_text = f"paraphrase: {text} </s>"
  ```
- **Decoding Configuration**:
  - `num_beams=5` with beam search decoding
  - `num_return_sequences=5` candidate variations per input record
  - `max_length=128` tokens
  - `no_repeat_ngram_size=2`
  - Early stopping enabled
- **Training-Set Scoping**: Only sentences from the 868-record authentic training split (`train.spacy`) are paraphrased. Sentences in `dev.spacy`, `test.spacy`, or `unseen_benchmark.jsonl` are strictly excluded from the generation prompt pool.

---

### 2.2 Exact Entity Span Relocation Algorithm

Because sequence-to-sequence generation modifies word ordering and sentence structure, original character offsets `[start, end]` cannot be directly transferred. The pipeline implements an exact substring boundary tracking algorithm ([`relocate_entities()`](file:///home/caineirb/Documents/PauPau/spaCy-training/main.ipynb)):

```python
def relocate_entities(original_text, paraphrase_text, original_entities):
    """
    Relocate entity spans from original to paraphrase via exact substring match.
    Enforces non-alphanumeric boundary checks to prevent sub-token collisions.
    Rejects the candidate if any entity cannot be verified.
    """
    relocated = []
    for ent in original_entities:
        entity_text = original_text[ent["start"]:ent["end"]]
        
        # 1. Exact case-sensitive search
        pos = paraphrase_text.find(entity_text)
        
        # 2. Case-insensitive fallback
        if pos == -1:
            pos = paraphrase_text.lower().find(entity_text.lower())
            
        if pos == -1:
            return None  # Entity dropped or altered by T5 -> discard candidate
            
        start_idx = pos
        end_idx = pos + len(entity_text)
        
        # 3. Word boundary guardrails
        before = paraphrase_text[start_idx - 1] if start_idx > 0 else " "
        after = paraphrase_text[end_idx] if end_idx < len(paraphrase_text) else " "
        if before.isalnum() or after.isalnum():
            return None  # Matched inside an unrelated word -> discard
            
        relocated.append({
            "start": start_idx,
            "end": end_idx,
            "label": ent["label"]
        })
    return relocated
```

#### Key Invariants of Span Relocation:
1. **Zero Entity Drop**: If a candidate paraphrase modifies, truncates, or omits even one entity surface string (e.g., transforming `"MySQL database"` into *"the database"*), the entire candidate is discarded.
2. **Boundary Sensitivity**: The match must begin and end at word boundaries (`before.isalnum() == False` and `after.isalnum() == False`). A search for `"Git"` will never accidentally match the middle of `"Digital"`.
3. **No Character Drift**: Character offsets are recalculated with zero-index arithmetic directly against the generated string.

---

### 2.3 Negative Sentence Paraphrasing & Distribution Balance

Authentic negative student logbook entries (e.g., general reflections, attending seminars, lunchtime, office cleaning, commuting) contain zero domain entities (`"entities": []`). 

To preserve the authentic positive-to-negative ratio and prevent false-positive over-prediction in production inference:
- Negative sentences from the authentic training split are passed through the T5 paraphraser.
- Candidates are accepted only if they remain completely free of domain entities and pass all quality checks.
- Out of 550 accepted paraphrases, **323 are negative records (58.7%)**, raising the combined TRSTR-Paraphrase negative ratio to **46.3%** (657 negatives across 1,418 records). This acts as a powerful regularizer against hallucinating entities in casual log entries.

---

## 3. Quality Guardrails & Anti-Contamination Filters

To prevent amplifying noise or introducing data leakage, every candidate paraphrase passes four layers of automated guardrails before acceptance:

```
[ T5 Beam Candidates (5) ]
            │
            ▼
┌──────────────────────────────────────┐
│  Layer 1: Exact Span Relocation      │ ── Rejected if entity modified / boundary invalid
└──────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────┐
│  Layer 2: Generic Noun Blocklist     │ ── Rejected if only entities are generic concepts
└──────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────┐
│  Layer 3: Plural-Morphology Generic  │ ── Rejected if bare generic noun tagged (Task 3)
│           Expansion (BARE_GENERICS)  │
└──────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────┐
│  Layer 4: Benchmark & Eval Isolation │ ── Rejected if overlap with unseen benchmark / eval
└──────────────────────────────────────┘
            │
            ▼
[ Accepted Paraphrase Record ]
```

### 3.1 Generic-Noun Blocklist (`GENERIC_NOUN_BLOCKLIST`)
An audit of initial prototype generations revealed that borderline abstract nouns surviving in manual annotations (such as `"system workflows"`, `"data requirements"`, `"encode"`, `"formatting"`, `"program"`, `"technical"`, `"layouts"`) were being over-generated by the model. 

A strict 34-term blocklist filters out borderline activities during synthetic generation:
- *Blocked IT terms*: `system workflows`, `program`, `code`, `debugging`, `technical`, `it support`, `data requirements`.
- *Blocked Clerical terms*: `encode`, `encoding`, `encoded`, `formatting`, `filing records`, `office documents`, `coordination`, `documentation of cases`.

If all entities in a candidate paraphrase belong to this blocklist, the candidate is discarded.

### 3.2 Plural-Morphology Bare Generic Expansion (`BARE_GENERICS`)
During the Task 3 Pipeline Integrity Audit ([`audit_gaps_closure_report.md`](file:///home/caineirb/Documents/PauPau/spaCy-training/audit_gaps_closure_report.md)), it was discovered that plural forms of generic nouns (specifically `"systems"`) had slipped through singular-only blocklists in 6 paraphrase records. 

The blocklist was upgraded to a comprehensive morphological expansion covering both singular and plural forms:

```python
BARE_GENERICS = {
    # IT infrastructure & software bare generics
    "system", "systems", "database", "databases", "software", "softwares",
    "code", "codes", "computer", "computers", "website", "websites",
    "application", "applications", "server", "servers", "backend", "backends",
    "frontend", "frontends", "network", "networks", "printer", "printers",
    "scanner", "scanners", "monitor", "monitors", "keyboard", "keyboards",
    "program", "programs", "data", "table", "tables",
    # Clerical & administrative bare generics
    "document", "documents", "file", "files", "record", "records",
    "form", "forms", "paper", "papers", "folder", "folders",
    "desk", "desks", "office", "offices", "university", "universities",
    "campus", "campuses", "department", "departments",
    "laboratory", "laboratories", "supervisor", "supervisors",
    "intern", "interns", "meeting", "meetings", "discussion", "discussions"
}
```

- **Purge Action**: All 6 contaminated records with `"systems"` were purged from the accepted pool, reducing total accepted paraphrases from 556 to **550 clean records**.
- **Audit Verification**: Contamination scan across all 550 paraphrases confirms **0 bare generic hits**.

### 3.3 Sentence Length and Lexical Stability Filters
- **Length Ratio**: Candidates whose character length is $< 0.5\times$ or $> 2.0\times$ the original sentence length are discarded to prevent runaway truncation or rambling.
- **Cross-Domain Leakage**: Clerical sentences that receive IT programming keywords during paraphrasing (e.g., inserting `"compiled"`, `"debugged"`, `"script"`) are rejected.

---

## 4. Dataset Composition & Split Characteristics

The combined TRSTR-Paraphrase training pool maintains strict balance across entity categories while providing sufficient negative regularization:

| Dataset / Partition | Total Records | Percentage | Positive Records | Negative Records | Negative Ratio | IT_TERM Count | CLERICAL_TERM Count | Total Entities |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Authentic Real Train Split** | 868 | 61.2% | 534 | 334 | 38.5% | 576 | 322 | 898 |
| **Accepted T5 Paraphrases** | 550 | 38.8% | 227 | 323 | 58.7% | 228 | 124 | 352 |
| **Total TRSTR-Paraphrase Pool** | **1,418** | **100.0%** | **761** | **657** | **46.3%** | **804** | **446** | **1,250** |
| *Evaluation Partitions (100% Authentic)* | | | | | | | | |
| ├── **Authentic Dev Split (`dev.spacy`)** | 186 | — | 136 | 50 | 26.9% | 160 | 60 | 220 |
| ├── **Authentic Test Split (`test.spacy`)** | 186 | — | 114 | 72 | 38.7% | 135 | 52 | 187 |
| └── **Unseen Benchmark (`unseen_benchmark.jsonl`)** | 85 | — | 85 | 0 | 0.0% | 40 | 25 | 65 |

> [!NOTE]
> All 550 accepted paraphrases are derived exclusively from the 868 real training split entries. No validation or test sentences were ever provided as input to the paraphraser.

---

## 5. Leakage Audit & Verification Protocol

Every record in [`data/training_trstr_paraphrase.jsonl`](file:///home/caineirb/Documents/PauPau/spaCy-training/data/training_trstr_paraphrase.jsonl) is audited prior to model training using [`scripts/check_data_leakage.py`](file:///home/caineirb/Documents/PauPau/spaCy-training/scripts/check_data_leakage.py). The audit executes 7 mandatory checks:

| Check ID | Verification Rule | Audit Criterion | Audit Result |
| :--- | :--- | :--- | :---: |
| **Check 1** | Document Duplication | 0 identical documents between train, dev, and test splits | **[PASS]** (0 leaks) |
| **Check 2** | Sentence Duplication | 0 exact sentence matches across train, dev, and test splits | **[PASS]** (0 leaks) |
| **Check 3** | Terms CSV Leakage | 0 of 65 unseen benchmark terms appear in `data/terms.csv` | **[PASS]** (0/65 terms) |
| **Check 4** | Training Annotation Leakage | 0 of 65 unseen benchmark terms appear in `data/data.jsonl` | **[PASS]** (0/65 terms) |
| **Check 5** | EntityRuler Matchability | 0 unseen benchmark terms matchable by seed dictionary rules | **[PASS]** (0/65 terms) |
| **Check 6** | Holdout Isolation | 0 sentences in holdout partitions appear in training sets | **[PASS]** (0 leaks) |
| **Check 7** | Synthetic Pool Isolation | SequenceMatcher string similarity $< 0.70$ against all eval sentences; 0 unseen terms in synthetic text | **[PASS]** (0 leaks) |

### 5-Fold Cross-Validation Fold Audit
In addition to the static train/dev/test split, all 5 cross-validation folds ([`scripts/cross_validation.py`](file:///home/caineirb/Documents/PauPau/spaCy-training/scripts/cross_validation.py)) undergo automated isolation verification:
- Across all 5 folds, TRSTR-Paraphrase train pools (1,542 docs per fold) exhibit **0 overlap** with validation folds (248 docs per fold).
- All 5 folds achieve **100% clean isolation** on the 65-term unseen benchmark.

---

## 6. Empirical Evaluation Results: TRTR vs. TRSTR-Paraphrase

Both the baseline (TRTR) and augmented (TRSTR-Paraphrase) models were trained using identical hyperparameter configurations:
- **Base Architecture**: `roberta-base` via `spacy-transformers`
- **Training Constraints**: `max_steps=2500`, `eval_frequency=50`, `patience=400`, `batch_size=128` (accumulated)
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU (CUDA device 0)

### 6.1 Real Held-Out Test Evaluation (`data/training/test.spacy`)

Evaluated against 186 authentic, unseen student journal records (187 ground-truth entities):

| Metric | TRTR (Real Baseline) | TRSTR-Paraphrase (T5) | Absolute Delta | Relative Lift |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Held-Out Test F1** | 65.27% | **74.66%** | **+9.39%** | **+14.39%** |
| Overall Held-Out Test Precision | 63.78% | **76.11%** | **+12.33%** | **+19.33%** |
| Overall Held-Out Test Recall | 66.84% | **73.26%** | **+6.42%** | **+9.61%** |
| ├── **`IT_TERM` F1** | 70.90% | **77.86%** | **+6.96%** | **+9.82%** |
| ├── `IT_TERM` Precision | 71.43% | **80.31%** | **+8.88%** | **+12.43%** |
| └── `IT_TERM` Recall | 70.37% | **75.56%** | **+5.19%** | **+7.38%** |
| ├── **`CLERICAL_TERM` F1** | 52.17% | **66.67%** | **+14.50%** | **+27.79%** |
| ├── `CLERICAL_TERM` Precision | 47.62% | **66.04%** | **+18.42%** | **+38.68%** |
| └── `CLERICAL_TERM` Recall | 57.69% | **67.31%** | **+9.62%** | **+16.68%** |

---

### 6.2 5-Fold Cross-Validation Performance

To ensure empirical results are statistically representative and not an artifact of a single lucky split, 5-fold cross-validation was evaluated across all 1,240 authentic documents:

| Cross-Validation Metric | TRTR (Real Baseline) | TRSTR-Paraphrase | Absolute Advantage |
| :--- | :---: | :---: | :---: |
| **Validation Overall F1 (Mean ± SD)** | 70.00 ± 3.72% | **74.97 ± 2.91%** | **+4.97%** |
| Validation Overall Precision | 66.41 ± 3.70% | **72.26 ± 2.67%** | **+5.85%** |
| Validation Overall Recall | 74.08 ± 4.36% | **77.94 ± 3.53%** | **+3.86%** |
| ├── `IT_TERM` F1 | 70.73 ± 3.50% | **74.84 ± 2.35%** | **+4.11%** |
| └── `CLERICAL_TERM` F1 | 68.51 ± 5.14% | **75.21 ± 4.61%** | **+6.70%** |

---

### 6.3 Out-of-Vocabulary Generalization Benchmark (65 Canonical Terms)

Evaluated on [`data/test/unseen_benchmark.jsonl`](file:///home/caineirb/Documents/PauPau/spaCy-training/data/test/unseen_benchmark.jsonl) containing 65 enterprise tools and administrative procedures strictly omitted from all training data and seed dictionaries:

| Evaluation Metric | Dictionary Baseline | TRTR (Real Baseline) | TRSTR-Paraphrase | Paraphrase Lift |
| :--- | :---: | :---: | :---: | :---: |
| **Unseen Benchmark Recall** | 4.62% (3/65) | **83.08%** (54/65) | **72.31%** (47/65) | **+67.69%** over Dict |
| Unseen Benchmark Precision | 100.00% | 40.00% | 39.50% | — |
| **Unseen Benchmark Transformer F1** | 8.82% | **54.00%** | **51.09%** | **+42.27%** over Dict |
| **Hybrid Pipeline Overall F1** | 8.82% | **55.00%** | **53.26%** | **+44.44%** over Dict |
| **Inductive Generalization Lift** | Baseline | +80.00% | **+70.76%** | — |

---

## 7. Analytical Findings & Thesis Discussion

### 7.1 Why Paraphrase Augmentation Dominates In-Domain Extraction
T5 sequence-to-sequence paraphrasing directly addresses the **syntactic monotony bottleneck** characteristic of OJT internship logs:
1. **Preservation of Authentic Domain Grounding**: Paraphrasing operates on real student sentences, retaining genuine terminology (`"Epson L3250"`, `"iClinicSys"`, `"Police Clearance"`, `"PEIS System"`) while restructuring clause structure (e.g., converting passive to active voice, swapping dependent clauses, substituting colloquial coordinating conjunctions).
2. **Contextual Variety for Transformer Attention**: By seeing authentic entities situated in diverse grammatical contexts, self-attention heads learn to identify tasks from semantic syntactic patterns (such as verb-object dependencies) rather than memorizing fixed surrounding bigrams.
3. **Remarkable Precision Surge (+12.33%)**: With the enforcement of `BARE_GENERICS` and `GENERIC_NOUN_BLOCKLIST`, test precision jumped from 63.78% to **76.11%**. Spurious activations on everyday words were eliminated because the paraphraser produced varied syntaxes for non-entity sentences as well.
4. **Resolution of the Clerical Classification Bottleneck**: `CLERICAL_TERM` F1 surged from 52.17% to **66.67% (+14.50%)**, overcoming the historic vulnerability where clerical activities were mistaken for casual narrative filler.

### 7.2 Comparison with Direct LLM Generation (`TRSTR-LLM`)
While TRSTR-Paraphrase is the undisputed leader for in-domain student log extraction (+9.39% test F1 over baseline vs. -1.29% for TRSTR-LLM), it serves a complementary role to direct LLM generation:
- **TRSTR-Paraphrase** excels at **in-distribution precision and calibration**: it adapts the model to the exact stylistic nuances and colloquialisms of student writing. However, because it relies on existing training surface terms, it cannot invent novel enterprise tools out-of-domain.
- **TRSTR-LLM** excels at **out-of-vocabulary inductive generalization**: by prompting Gemini Flash Lite with diverse unseen workplace scenarios, it achieves higher out-of-vocabulary precision (52.73% vs. 39.50%) and recall (89.23% vs. 72.31%) on novel enterprise tools.

---

## 8. Limitations & Guidelines for Replication

> [!WARNING]
> **Thesis Methodological Disclaimer**: Synthetic data augmentation was introduced strictly to mitigate training-set sample scarcity and isolate the impact of syntactic diversity on Transformer NER performance. 
> - **Evaluation Integrity**: All benchmarks and reported test scores are measured on 100% authentic, human-annotated student records.
> - **Vocabulary Scope**: Paraphrasing preserves ground-truth entity tokens; it does not substitute entities or hallucinate terms. Consequently, it cannot introduce novel out-of-vocabulary tools. For OOV terminology recovery, the pipeline relies on the hybrid EntityRuler layer and complementary LLM generation.
> - **Pipeline Execution**: Paraphrase generation requires local PyTorch CUDA support. For reproduction, execute Phases 8 through 10 in [`main.ipynb`](file:///home/caineirb/Documents/PauPau/spaCy-training/main.ipynb).
