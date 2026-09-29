# Data-Driven Term Rates & Blocklist/Dictionary Remediation (Task 2)

## 1. Methodology & Decision Rules

For every candidate term (current blocklist, evaluation false positives/negatives, and dictionary entries), we compute its occurrences across the authentic **real train + dev dataset (834 documents)**:

$$\text{Entity Rate} = \frac{\text{Annotated Entity Occurrences}}{\text{Total Surface Occurrences in Text}}$$

### Decision Thresholds:
1. **Restore to Synthetic Augmentation (`RESTORE_TO_DATA`)**: Term is in the current blocklist, but has $\ge 2$ real occurrences and $\ge 50.0\%$ entity rate. Real human annotators consistently tag this term as an entity; removing it caused severe False Negatives (e.g. `coding`, `debugging`, `formatting`).
2. **Maintain in Blocklist (`MAINTAIN_BLOCK`)**: Term has $< 50.0\%$ entity rate (mostly narrative prose, rarely an entity). Kept blocked to avoid amplifying annotation noise.
3. **Relabel Dictionary (`RELABEL_DICTIONARY`)**: Term exists in `data/terms.csv`, but its dictionary label contradicts the majority ground-truth label in authentic annotations (e.g., `data entry` forced to `IT_TERM` while real annotators tag 100% `CLERICAL_TERM`).
4. **Remove / Downgrade from Dictionary (`REMOVE_FROM_DICTIONARY`)**: Term exists in `data/terms.csv`, but appears primarily as un-annotated narrative text (entity rate $< 25.0\%$ with $\ge 4$ occurrences), causing persistent False Positives with confidence 1.0.

---

## 2. Summary of Changed Terms

| Term | Current Blocklist | Dictionary Label | Total Occur. | Annotated | Entity Rate | Real IT | Real Clerical | Recommendation | Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `program` | Yes | - | 21 | 10 | 47.6% | 2 | 8 | **MAINTAIN_BLOCK** | Entity rate 47.6% (10/21). Overwhelmingly general narrative prose. |
| `encoded` | Yes | - | 13 | 3 | 23.1% | 0 | 3 | **MAINTAIN_BLOCK** | Entity rate 23.1% (3/13). Overwhelmingly general narrative prose. |
| `office documents` | Yes | - | 12 | 3 | 25.0% | 1 | 2 | **MAINTAIN_BLOCK** | Entity rate 25.0% (3/12). Overwhelmingly general narrative prose. |
| `encode` | Yes | - | 9 | 4 | 44.4% | 0 | 4 | **MAINTAIN_BLOCK** | Entity rate 44.4% (4/9). Overwhelmingly general narrative prose. |
| `coordination` | Yes | - | 8 | 1 | 12.5% | 0 | 1 | **MAINTAIN_BLOCK** | Entity rate 12.5% (1/8). Overwhelmingly general narrative prose. |
| `front page` | Yes | - | 7 | 2 | 28.6% | 0 | 2 | **MAINTAIN_BLOCK** | Entity rate 28.6% (2/7). Overwhelmingly general narrative prose. |
| `orientation` | Yes | - | 7 | 1 | 14.3% | 0 | 1 | **MAINTAIN_BLOCK** | Entity rate 14.3% (1/7). Overwhelmingly general narrative prose. |
| `deployment` | Yes | - | 6 | 2 | 33.3% | 2 | 0 | **MAINTAIN_BLOCK** | Entity rate 33.3% (2/6). Overwhelmingly general narrative prose. |
| `copies` | Yes | - | 3 | 1 | 33.3% | 0 | 1 | **MAINTAIN_BLOCK** | Entity rate 33.3% (1/3). Overwhelmingly general narrative prose. |
| `data entry` | No | IT_TERM | 10 | 8 | 80.0% | 1 | 7 | **RELABEL_DICTIONARY** | Dictionary has IT_TERM, but real annotations are 1 IT vs 7 Clerical (Majority: CLERICAL_TERM). |
| `filing system` | No | CLERICAL_TERM | 3 | 1 | 33.3% | 1 | 0 | **RELABEL_DICTIONARY** | Dictionary has CLERICAL_TERM, but real annotations are 1 IT vs 0 Clerical (Majority: IT_TERM). |
| `tarpaulin design` | No | IT_TERM | 3 | 2 | 66.7% | 0 | 2 | **RELABEL_DICTIONARY** | Dictionary has IT_TERM, but real annotations are 0 IT vs 2 Clerical (Majority: CLERICAL_TERM). |
| `record management` | No | CLERICAL_TERM | 2 | 2 | 100.0% | 1 | 1 | **RELABEL_DICTIONARY** | Dictionary has CLERICAL_TERM, but real annotations are 1 IT vs 1 Clerical (Majority: IT_TERM). |
| `technical` | Yes | - | 22 | 14 | 63.6% | 14 | 0 | **RESTORE_TO_DATA** | Entity rate 63.6% (14/22) on real data. Annotators consistently treat as entity. |
| `coding` | Yes | - | 18 | 15 | 83.3% | 15 | 0 | **RESTORE_TO_DATA** | Entity rate 83.3% (15/18) on real data. Annotators consistently treat as entity. |
| `requirements` | Yes | - | 9 | 7 | 77.8% | 5 | 2 | **RESTORE_TO_DATA** | Entity rate 77.8% (7/9) on real data. Annotators consistently treat as entity. |
| `accounts` | Yes | - | 8 | 7 | 87.5% | 2 | 5 | **RESTORE_TO_DATA** | Entity rate 87.5% (7/8) on real data. Annotators consistently treat as entity. |
| `survey` | Yes | - | 6 | 3 | 50.0% | 0 | 3 | **RESTORE_TO_DATA** | Entity rate 50.0% (3/6) on real data. Annotators consistently treat as entity. |
| `system development` | Yes | IT_TERM | 6 | 5 | 83.3% | 5 | 0 | **RESTORE_TO_DATA** | Entity rate 83.3% (5/6) on real data. Annotators consistently treat as entity. |
| `debugging` | Yes | - | 5 | 5 | 100.0% | 5 | 0 | **RESTORE_TO_DATA** | Entity rate 100.0% (5/5) on real data. Annotators consistently treat as entity. |
| `drivers` | Yes | - | 4 | 4 | 100.0% | 4 | 0 | **RESTORE_TO_DATA** | Entity rate 100.0% (4/4) on real data. Annotators consistently treat as entity. |
| `system design` | Yes | IT_TERM | 4 | 4 | 100.0% | 4 | 0 | **RESTORE_TO_DATA** | Entity rate 100.0% (4/4) on real data. Annotators consistently treat as entity. |
| `formatting` | Yes | - | 3 | 2 | 66.7% | 0 | 2 | **RESTORE_TO_DATA** | Entity rate 66.7% (2/3) on real data. Annotators consistently treat as entity. |
| `network` | Yes | - | 3 | 3 | 100.0% | 3 | 0 | **RESTORE_TO_DATA** | Entity rate 100.0% (3/3) on real data. Annotators consistently treat as entity. |
| `notices` | Yes | - | 3 | 3 | 100.0% | 0 | 3 | **RESTORE_TO_DATA** | Entity rate 100.0% (3/3) on real data. Annotators consistently treat as entity. |
| `data organization` | Yes | CLERICAL_TERM | 2 | 2 | 100.0% | 0 | 2 | **RESTORE_TO_DATA** | Entity rate 100.0% (2/2) on real data. Annotators consistently treat as entity. |
| `layouts` | Yes | - | 2 | 1 | 50.0% | 0 | 1 | **RESTORE_TO_DATA** | Entity rate 50.0% (1/2) on real data. Annotators consistently treat as entity. |
| `office systems` | Yes | - | 2 | 2 | 100.0% | 2 | 0 | **RESTORE_TO_DATA** | Entity rate 100.0% (2/2) on real data. Annotators consistently treat as entity. |
| `policies` | Yes | - | 2 | 1 | 50.0% | 0 | 1 | **RESTORE_TO_DATA** | Entity rate 50.0% (1/2) on real data. Annotators consistently treat as entity. |
| `system exploration` | Yes | - | 2 | 1 | 50.0% | 1 | 0 | **RESTORE_TO_DATA** | Entity rate 50.0% (1/2) on real data. Annotators consistently treat as entity. |

---

## 3. Detailed Rates for All Blocklist Terms

| Term | Total Text Occurrences | Annotated Occurrences | Entity Rate | Real IT Count | Real Clerical Count | Action Taken |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `data organization` | 2 | 2 | 100.0% | 0 | 2 | **RESTORE** |
| `data quality` | 1 | 1 | 100.0% | 0 | 1 | **BLOCK** |
| `data requirements` | 1 | 1 | 100.0% | 1 | 0 | **BLOCK** |
| `debugging` | 5 | 5 | 100.0% | 5 | 0 | **RESTORE** |
| `drivers` | 4 | 4 | 100.0% | 4 | 0 | **RESTORE** |
| `field trials` | 1 | 1 | 100.0% | 0 | 1 | **BLOCK** |
| `issued` | 1 | 1 | 100.0% | 0 | 1 | **BLOCK** |
| `network` | 3 | 3 | 100.0% | 3 | 0 | **RESTORE** |
| `notices` | 3 | 3 | 100.0% | 0 | 3 | **RESTORE** |
| `office systems` | 2 | 2 | 100.0% | 2 | 0 | **RESTORE** |
| `proctoring` | 1 | 1 | 100.0% | 0 | 1 | **BLOCK** |
| `reference numbers` | 1 | 1 | 100.0% | 0 | 1 | **BLOCK** |
| `stalls` | 1 | 1 | 100.0% | 0 | 1 | **BLOCK** |
| `system design` | 4 | 4 | 100.0% | 4 | 0 | **RESTORE** |
| `system workflows` | 1 | 1 | 100.0% | 1 | 0 | **BLOCK** |
| `accounts` | 8 | 7 | 87.5% | 2 | 5 | **RESTORE** |
| `coding` | 18 | 15 | 83.3% | 15 | 0 | **RESTORE** |
| `system development` | 6 | 5 | 83.3% | 5 | 0 | **RESTORE** |
| `requirements` | 9 | 7 | 77.8% | 5 | 2 | **RESTORE** |
| `formatting` | 3 | 2 | 66.7% | 0 | 2 | **RESTORE** |
| `technical` | 22 | 14 | 63.6% | 14 | 0 | **RESTORE** |
| `layouts` | 2 | 1 | 50.0% | 0 | 1 | **RESTORE** |
| `policies` | 2 | 1 | 50.0% | 0 | 1 | **RESTORE** |
| `survey` | 6 | 3 | 50.0% | 0 | 3 | **RESTORE** |
| `system exploration` | 2 | 1 | 50.0% | 1 | 0 | **RESTORE** |
| `program` | 21 | 10 | 47.6% | 2 | 8 | **BLOCK** |
| `encode` | 9 | 4 | 44.4% | 0 | 4 | **BLOCK** |
| `copies` | 3 | 1 | 33.3% | 0 | 1 | **BLOCK** |
| `deployment` | 6 | 2 | 33.3% | 2 | 0 | **BLOCK** |
| `front page` | 7 | 2 | 28.6% | 0 | 2 | **BLOCK** |
| `office documents` | 12 | 3 | 25.0% | 1 | 2 | **BLOCK** |
| `encoded` | 13 | 3 | 23.1% | 0 | 3 | **BLOCK** |
| `orientation` | 7 | 1 | 14.3% | 0 | 1 | **BLOCK** |
| `coordination` | 8 | 1 | 12.5% | 0 | 1 | **BLOCK** |
