# Proposed Relabels and Annotation Audit (Task 6)

*Generated from error analysis of `data/eval_results/held_out_test_trstr_llm_errors_only.jsonl`. No ground-truth data has been altered.*

---

## 1. Group (a): Malformed Gold Spans

Spans containing leading conjunctions, dangling adverbs, punctuation artifacts, or over-extended multi-clause text.

| Context Sentence | Current Gold Span | Gold Label | Defect Detected | Proposed Cleaned Span | Proposed Label |
| :--- | :--- | :---: | :--- | :--- | :---: |
| `NBSC Guard House Installing CAT6 UTP Cable and Camera` | **"Installing CAT6 UTP Cable and Camera"** | `IT_TERM` | Over-extended compound action clause | **"CAT6 UTP Cable"** | `IT_TERM` |
| `I was assigned to assist in filing and organizing office documents.` | **"and organizing office documents"** | `CLERICAL_TERM` | Leading conjunction (e.g. 'and ...') | **"organizing office documents"** | `CLERICAL_TERM` |
| `Through this task, I practiced organizing information systematically` | **"information systematically"** | `IT_TERM` | Dangling adverb or adjective modifier | **"information"** | `IT_TERM` |
| `Feb 24 I made the navbar dynamic and mobile-friendly with a drawer menu.` | **"navbar dynamic"** | `IT_TERM` | Dangling adverb or adjective modifier | **"navbar"** | `IT_TERM` |

---

## 2. Group (b): IT vs. CLERICAL Label Conflicts

Genuine ambiguity where ground-truth and model disagree on whether the activity is technical (IT) or administrative (Clerical).

| Context Sentence | Entity Term | Gold Label | Model Predicted | Model Conf. | Recommended Resolution | Rationale |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `I manage the data entry for new Iligan dealer memberships and enjoy the str...` | **"data entry"** | `CLERICAL_TERM` | `IT_TERM` | 1.0000 | `CLERICAL_TERM` | Data entry and filing are core clerical duties per guideline §3.2 |
| `elping PNP File Management` | **"File Management"** | `CLERICAL_TERM` | `IT_TERM` | 1.0000 | `CLERICAL_TERM` | Data entry and filing are core clerical duties per guideline §3.2 |
| `Coordinated with staff regarding member information and system concerns` | **"member information"** | `CLERICAL_TERM` | `IT_TERM` | 0.7016 | `CLERICAL_TERM` | Retain gold annotation standard |
| `This task improved my understanding of regulatory compliance` | **"regulatory compliance"** | `CLERICAL_TERM` | `IT_TERM` | 0.9850 | `CLERICAL_TERM` | Retain gold annotation standard |
| `I realized the importance of access control systems` | **"access control systems"** | `CLERICAL_TERM` | `IT_TERM` | 1.0000 | `CLERICAL_TERM` | Retain gold annotation standard |
| `Layout a program flow for the Local Internship Program (LIP) orientation` | **"program flow"** | `CLERICAL_TERM` | `IT_TERM` | 0.9999 | `CLERICAL_TERM` | Retain gold annotation standard |
| `Through proper procedures, I understood compliance requirements` | **"compliance requirements"** | `CLERICAL_TERM` | `IT_TERM` | 0.9704 | `CLERICAL_TERM` | Retain gold annotation standard |
| `Coding Master List on the computer.` | **"Master List"** | `CLERICAL_TERM` | `IT_TERM` | 0.0264 | `CLERICAL_TERM` | Retain gold annotation standard |
| `I learned how to coordinate data collection with all of us assigned to diff...` | **"data collection"** | `IT_TERM` | `CLERICAL_TERM` | 0.9999 | `IT_TERM` | Retain gold annotation standard |
| `Updated and encoded CIF (Customer Information File) records` | **"CIF (Customer Information File) records"** | `CLERICAL_TERM` | `IT_TERM` | 0.0225 | `CLERICAL_TERM` | Retain gold annotation standard |
| `Monday (April 6, 2026): I improved the ABC signage that was completed last ...` | **"ABC signage"** | `CLERICAL_TERM` | `IT_TERM` | 0.2370 | `CLERICAL_TERM` | Retain gold annotation standard |
| `This taught me incident tracking procedures` | **"incident tracking"** | `CLERICAL_TERM` | `IT_TERM` | 0.5300 | `CLERICAL_TERM` | Retain gold annotation standard |
| `I verified customer information to ensure consistency between documents and...` | **"system records"** | `CLERICAL_TERM` | `IT_TERM` | 0.9976 | `CLERICAL_TERM` | Retain gold annotation standard |

---

## 3. Group (c): Genuine Model Errors (Sample)

A total of **115** genuine model errors were identified (clean FPs and FNs). Top instances:

| Type | Term | Context Snippet | Gold Label | Predicted Label | Confidence |
| :---: | :--- | :--- | :---: | :---: | :---: |
| `correct` | **"system capacity"** | `Load testing evaluated system capacity` | `IT_TERM` | `IT_TERM` | 0.9108 |
| `false_positive` | **"Load testing"** | `Load testing evaluated system capacity` | `-` | `IT_TERM` | 0.9754 |
| `false_positive` | **"IT Fest"** | `1st IT Fest` | `-` | `IT_TERM` | 1.0000 |
| `false_positive` | **"application system"** | `I manage the data entry for new Iligan dealer memberships and enjoy th...` | `-` | `IT_TERM` | 0.9958 |
| `false_positive` | **"name stickers"** | `Created and attached name stickers for office folders and records.` | `-` | `CLERICAL_TERM` | 0.8890 |
| `boundary_error` | **"Canva Editing"** | `Canva Editing allowed me to express creativity while still contributin...` | `IT_TERM` | `IT_TERM` | 1.0000 |
| `false_negative` | **"book"** | `THURSDAY (MARCH 26, 2026) We make some changes the book. We carefully ...` | `CLERICAL_TERM` | `-` | - |
| `false_positive` | **"QR code"** | `QR code system implementation enhanced enrollment processes` | `-` | `IT_TERM` | 1.0000 |
| `false_positive` | **"enrollment processes"** | `QR code system implementation enhanced enrollment processes` | `-` | `IT_TERM` | 0.9166 |
| `false_negative` | **"giving"** | `One of the tasks I handled was preparing and giving certificates` | `CLERICAL_TERM` | `-` | - |
| `false_positive` | **"certificates"** | `One of the tasks I handled was preparing and giving certificates` | `-` | `CLERICAL_TERM` | 0.9977 |
| `correct` | **"internet connection"** | `Tuesday (March 31, 2026): I assisted in troubleshooting the internet c...` | `IT_TERM` | `IT_TERM` | 0.9849 |
| `correct` | **"Starlink connection"** | `Tuesday (March 31, 2026): I assisted in troubleshooting the internet c...` | `IT_TERM` | `IT_TERM` | 0.9999 |
| `correct` | **"LAN cable"** | `Tuesday (March 31, 2026): I assisted in troubleshooting the internet c...` | `IT_TERM` | `IT_TERM` | 0.9999 |
| `false_negative` | **"router"** | `Tuesday (March 31, 2026): I assisted in troubleshooting the internet c...` | `IT_TERM` | `-` | - |
| `correct` | **"Starlink system"** | `Tuesday (March 31, 2026): I assisted in troubleshooting the internet c...` | `IT_TERM` | `IT_TERM` | 0.9999 |
| `correct` | **"connection issue"** | `Tuesday (March 31, 2026): I assisted in troubleshooting the internet c...` | `IT_TERM` | `IT_TERM` | 0.9454 |
| `false_positive` | **"internet access"** | `Tuesday (March 31, 2026): I assisted in troubleshooting the internet c...` | `-` | `IT_TERM` | 0.4934 |
| `boundary_error` | **"signage/flow"** | `Monday (March 30, 2026): I was tasked by Ma'am Cheng to layout and pri...` | `CLERICAL_TERM` | `CLERICAL_TERM` | 0.9999 |
| `correct` | **"budget amounts"** | `Monday (March 30, 2026): I was tasked by Ma'am Cheng to layout and pri...` | `CLERICAL_TERM` | `CLERICAL_TERM` | 0.9999 |
| `false_negative` | **"Annual Gender and Development (GAD) Plan and Budget for CY 2027"** | `Monday (March 30, 2026): I was tasked by Ma'am Cheng to layout and pri...` | `CLERICAL_TERM` | `-` | - |
| `false_positive` | **"sample expiration"** | `Worked with the system with sample expiration dates.` | `-` | `CLERICAL_TERM` | 0.7894 |
| `correct` | **"system concerns"** | `Coordinated with staff regarding member information and system concern...` | `IT_TERM` | `IT_TERM` | 0.9975 |
| `false_negative` | **"Paperwork's"** | `Paperwork's/Printer Troubleshooting` | `CLERICAL_TERM` | `-` | - |
| `correct` | **"Printer Troubleshooting"** | `Paperwork's/Printer Troubleshooting` | `IT_TERM` | `IT_TERM` | 1.0000 |

---

## 4. Annotation Guideline Draft: Generic Activity Terms & Ambiguity Resolution

### 4.1 Activity & Process Terms (`coding`, `debugging`, `organizing`, `teamwork`)

1. **Concrete Technical Actions (`IT_TERM`)**:
   - Terms describing direct software engineering, programming, or technical diagnostic activities (**`coding`**, **`debugging`**, **`troubleshooting`**, **`system development`**, **`testing`**) MUST be annotated as `IT_TERM` when used to denote technical work.
   - *Rule*: If the activity directly interacts with code, software logic, or computer hardware, classify as `IT_TERM`.

2. **Administrative & Organizational Actions (`CLERICAL_TERM`)**:
   - Terms describing document handling, record structuring, and office workflow (**`organizing office documents`**, **`formatting files`**, **`filing`**, **`sorting records`**) MUST be annotated as `CLERICAL_TERM`.
   - *Rule*: Bare generic actions like `organizing` or `managing` without an entity object are narrative verbs (do not tag). When modifying office materials (`organizing files`), tag the specific object/compound.

3. **General Interpersonal / Soft Skills (NOT ENTITIES)**:
   - Broad collaborative or workplace terms (**`teamwork`**, **`collaboration`**, **`communication`**, **`coordination`**) MUST NOT be tagged as entities.
   - *Rationale*: These describe interpersonal dynamics, not domain-specific IT or clerical tools/deliverables.

### 4.2 IT vs. CLERICAL Boundary Disambiguation

| Term / Task | Correct Label | Boundary Rule |
| :--- | :---: | :--- |
| **Data Entry** | `CLERICAL_TERM` | Manual transcription, keying data into forms/Excel/spreadsheets is administrative. Only data architecture or ETL pipeline scripts qualify as IT. |
| **File Management** | `CLERICAL_TERM` | Organizing, locating, and archiving physical or desktop office files is clerical. Database schema storage is IT. |
| **Printing (Routine)** | `CLERICAL_TERM` | Operating office printers to produce reports, booklets, or notices is clerical. |
| **Printer Setup & Diagnostics** | `IT_TERM` | Installing printer drivers, configuring network ports, and clearing hardware faults is IT. |
| **Record-Keeping & Compliance** | `CLERICAL_TERM` | Maintaining attendance logs, assessment checklists, and official memos is clerical. |
| **Access Control Systems** | `IT_TERM` (or `CLERICAL_TERM` if physical) | Physical key/visitor log access is clerical; electronic biometric / RFID / credential security is IT. |

