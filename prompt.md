# TRSTR-LLM: Model Performance Improvement Recommendations

*Generated: September 2024*

---

## 1. Problem Statement

Based on the empirical results from the LLM-direct synthetic pipeline (TRSTR-LLM) and per-item error analysis of the held-out test set (`data/eval_results/held_out_test_trstr_llm.jsonl`), three systematic error sources were identified:

1. **Dictionary label conflicts** — terms in `data/terms.csv` assigned to the wrong label, causing the EntityRuler to fire at full confidence with the wrong label and override the transformer.
2. **Dictionary over-breadth** — generic/event terms forced at full confidence, generating false positives that the transformer could have correctly avoided.
3. **ML span boundary errors and over-extraction** — the transformer produces spurious entities on generic/environmental noun phrases, partly because negative-example coverage in training data is insufficient.

---

## 2. Changes Applied

### 2.1 Dictionary Hygiene Fix (`data/terms.csv`)

**Backup:** `git diff data/terms.csv` shows all changes (fully reversible via `git checkout data/terms.csv`).

| Action | Term(s) | Reason |
| :--- | :--- | :--- |
| **Relabeled** | `Data Entry` (IT_TERM → CLERICAL_TERM) | Gold labels in test set annotate `data entry` as `CLERICAL_TERM`; EntityRuler was forcing IT_TERM at conf 1.0 — a guaranteed error. |
| **Removed** | `System Design`, `System Planning`, `System Features`, `System Modules`, `System Interface`, `System Performance`, `System Stability` | Over-broad generic "system" concepts. Guidelines §3.1/§5 explicitly forbid tagging bare `system` constructs. Appeared in ML and dictionary false positives in eval results. |
| **Removed** | `Application Development`, `Application Management`, `Family Profiling Application`, `Nutrimap System`, `Inquiry System` | Generic "application" / "system" compound terms — same issue as above. |
| **Removed** | `QR Code` | Generic artifact / infrastructure term. Appeared as a dictionary false positive in held-out test eval. |
| **Removed** | `IT Fest`, `Research Summit`, `Capstone Project`, `Capstone Study`, `Charter Day`, `BNS Meeting`, `BNS Activities` | Events / institutional activities, not specific software tools. Should be context-decided by the transformer. |
| **Kept** | `System Development`, `System Testing`, `System Debugging`, `System Deployment`, `System Maintenance`, `System Encryption` | These are concrete technical task phrases (per annotation guidelines §3.1 "Concrete Technical Tasks"), not bare generic concepts. |

**Net change:** 291 → 270 terms (19 removed, 1 relabeled). Balance: 148 IT_TERM, 122 CLERICAL_TERM.

### 2.2 Automation Scripts

Two scripts were added to ensure dictionary hygiene is maintained across the active learning loop.

#### `scripts/audit_dictionary.py`

Audits `data/terms.csv` against two sources of ground truth:

1. **Gold label conflicts:** Checks every term in `terms.csv` against annotated spans in `data/data.jsonl` (and `data/test/holdout.jsonl`). Flags any term whose dictionary label differs from how it is actually annotated in gold data. A conflict is a guaranteed inference error because the EntityRuler fires at conf 1.0 and overrides the transformer.
2. **Generic/event term detection:** Applies the annotation guidelines' forbidden categories to flag over-broad terms (`System *`, `Application *`, event names) that should be context-decided rather than string-forced.

**Usage:**

```bash
# Read-only audit (reports findings without modifying files)
python scripts/audit_dictionary.py

# Auto-apply fixes (creates timestamped backup first)
python scripts/audit_dictionary.py --apply

# Custom paths
python scripts/audit_dictionary.py --terms-csv data/terms.csv
```

**Workflow:** Run this script after every active learning iteration (`python scripts/retrain.py --dry-run`) to catch new label conflicts before they enter the next training cycle.

---

## 3. Remaining Recommendations

### 3.1 Complete `scripts/mine_negatives.py` *(not yet implemented)*

The most impactful next step for reducing ML false positives. Run the pipeline in `transformer_only` mode over:

- `data/data.jsonl` (gold-annotated records) → classify any ML-predicted span with zero gold overlap as a candidate false positive
- `data/test/holdout.jsonl` (real-world records) → surface new spurious patterns

Produces:

- `data/fp_mining/candidates.jsonl` — per-sentence candidate negatives with span/label/source/confidence
- `data/fp_mining/fp_terms.json` — aggregated term counts, sorted by frequency
- `data/fp_mining/blocklist_additions.txt` — feed directly into `generate_llm_synthetic.py`'s `GENERIC_NOUN_BLOCKLIST` set

**Priority:** High. Directly addresses the residual ML false positives on held-out test (37 ML FPs vs. 6 dictionary FPs).

### 3.2 Re-tune Confidence Threshold on Dev Set

The `confidence_threshold = 0.80` in `HybridJournalPipeline.__init__` is a constant. With the new dictionary (fewer dictionary FPs), a lower threshold (e.g. 0.70) could allow more true transformer positives through while still filtering spurious ones. Evaluate on `data/training/dev.spacy` with a sweep (0.60–0.90) to find the optimal operating point before the next retrain.

**Key metric to optimize:** Real held-out test F1, not unseen benchmark recall (the latter is already strong at 54.14%).

### 3.3 Increase Synthetic Hard Negative Volume

Current negative ratio is 29.9% (at the lower end of the 25–35% target band). Increase toward 33–35% and specifically inject negative sentences containing the top FP surface forms (`enrollment processes`, `certificates`, `Calendar management`, `IT knowledge`, `LAN cable`, `designing`) so the transformer learns to suppress them.

Update the generation prompt in `scripts/generate_llm_synthetic.py` to add a "topics to include as NEGATIVE only" section populated from the mined FP terms.

### 3.4 More Training Data (Longer-term)

The current combined pool is 987 records. For a `roberta-base` transformer learning 2 fine-grained labels on short OJT sentences, this is modest. Consider:

- **Active learning annotation round** on the highest-uncertainty real records (use `scripts/retrain.py` with `--dry-run` to identify novel terms first)
- Increase synthetic generation from 300 → 600 records while maintaining strict deduplication

### 3.5 Architectural Tune *(lower priority — only after data fixes)*

`config_trf.cfg` uses `TransitionBasedParser.v2` with `hidden_width=64, maxout_pieces=2, use_upper=false`. Increasing to `hidden_width=128, use_upper=true` would give the parser more representational capacity for boundary disambiguation. However, this amplifies whatever signal is in the data — only pursue after the data fixes above are applied and verified, otherwise you optimize an already-overfit model.
