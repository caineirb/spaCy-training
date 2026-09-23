# Annotation Guidelines & Label Taxonomy

**Document Version:** 1.1.0  
**Project:** Hybrid NER + Classification Pipeline for OJT Journal Task Tagging  
**Last Updated:** September 2026  

---

## 1. Executive Summary & Purpose

This document establishes the official annotation policy and label taxonomy for extracting task entities from On-the-Job Training (OJT) student internship journals. It defines the labeling standards used across:
- Seed dictionary compilation (`data/terms.csv`)
- Real-world annotated journal dataset (`data/data.jsonl`)
- spaCy training/dev/test datasets (`data/training/*.spacy`)
- Controlled unseen-term generalization benchmark (`data/test/unseen_benchmark.jsonl`)
- Permanent real-world holdout evaluation sets (`data/test/holdout.jsonl`)
- Production inference routing and review (`scripts/pipeline.py`, `scripts/deploy_inference.py`)

Every annotator and contributor must adhere to these guidelines to ensure consistency, eliminate annotation drift, and prevent data contamination.

---

## 2. Unified Label Taxonomy & Canonical Labels

The project operates on a single, unified canonical label taxonomy across all stages of the pipeline:

| Label | File / Component | Purpose |
| :--- | :--- | :--- |
| `IT_TERM` | `data/terms.csv`, `data/data.jsonl`, `*.spacy`, Inference Spans | Information technology technologies, programming languages, software tools, databases, infrastructure, and technical tasks. |
| `CLERICAL_TERM` | `data/terms.csv`, `data/data.jsonl`, `*.spacy`, Inference Spans | Clerical and office productivity tools, document handling, filing, record-keeping, and administrative workflows. |

### Retirement of the Legacy Two-Tier Scheme
Earlier iterations of the project (Phase 1–2) maintained a distinction between dictionary-level categories (`IT_TASK`, `CLERICAL`) and NER span labels (`IT_TERM`, `CLERICAL_TERM`). This two-tier separation has been formally retired:
- `data/terms.csv` has been directly normalized so all entries carry either `IT_TERM` or `CLERICAL_TERM`.
- `data/data.jsonl`, binary DocBins (`train.spacy`, `dev.spacy`, `test.spacy`), and model checkpoints exclusively use `IT_TERM` and `CLERICAL_TERM`.
- [`scripts/labels.py`](file:///home/caineirb/Documents/PauPau/spaCy-training/scripts/labels.py) is retained strictly as a backward-compatibility normalization shim (mapping legacy `IT_TASK` $\rightarrow$ `IT_TERM` and `CLERICAL` $\rightarrow$ `CLERICAL_TERM`) to ensure that external seed datasets or historical CSVs can still be ingested safely without code changes.

---

## 3. Label Definitions & Inclusion Criteria

### 3.1. `IT_TERM` (Information Technology Tasks & Tools)

An `IT_TERM` is a specific technology, software tool, programming language, library, framework, database, infrastructure component, hardware device, or concrete technical operational activity performed during an IT internship.

#### What to Tag as `IT_TERM`:
1. **Programming Languages & Runtimes:**  
   `Python`, `TypeScript`, `JavaScript`, `Java`, `C#`, `C++`, `PHP`, `Go`, `Rust`, `SQL`, `HTML`, `CSS`, `Node.js`, `Deno`, `Bun`.
2. **Frameworks & Libraries:**  
   `React`, `Angular`, `Vue.js`, `Svelte`, `Next.js`, `Nuxt`, `Django`, `Flask`, `FastAPI`, `Laravel`, `Spring Boot`, `Express`, `Tailwind CSS`, `Bootstrap`.
3. **Databases, ORMs & Backend Services:**  
   `PostgreSQL`, `MySQL`, `MongoDB`, `Redis`, `SQLite`, `Prisma`, `Supabase`, `Cassandra`, `ClickHouse`, `GraphQL`.
4. **DevOps, Cloud & Infrastructure Tools:**  
   `Docker`, `Kubernetes`, `Git`, `GitHub`, `GitLab`, `Terraform`, `AWS`, `Azure`, `GCP`, `Prometheus`, `Grafana`, `ArgoCD`, `Helm`.
5. **Testing, Build & Quality Tools:**  
   `Postman`, `JMeter`, `Playwright`, `Cypress`, `Jest`, `Pytest`, `Vite`, `Webpack`, `SonarQube`.
6. **Concrete Technical Tasks & Workflows:**  
   `Database Administration`, `Database Optimization`, `Network Troubleshooting`, `Cable Crimping`, `PC Assembly`, `Operating System Installation`, `API Integration`, `Continuous Integration`, `Unit Testing`, `Static Code Analysis`, `Data Validation`, `Bug Fixing`.

#### What NOT to Tag as `IT_TERM`:
- Generic technical nouns without tool or task specificity (e.g., `database`, `backend`, `frontend`, `software`, `server`, `code`, `system`, `data`, `application`, `website`).
- General workplace nouns (e.g., `computer`, `monitor`, `keyboard`, `laptop`, `screen`).
- General technical concepts lacking operational context (e.g., `authentication`, `dashboard`, `schema`, `function`, `variable`, `loop`).

---

### 3.2. `CLERICAL_TERM` (Clerical & Administrative Tasks & Tools)

A `CLERICAL_TERM` is an office productivity application, physical or digital filing system, record-keeping workflow, administrative duty, or routine organizational process performed during an office internship.

#### What to Tag as `CLERICAL_TERM`:
1. **Office Software & Digital Productivity Suites:**  
   `Microsoft Excel`, `Microsoft Word`, `Microsoft PowerPoint`, `Google Sheets`, `Google Docs`, `Google Slides`, `Google Forms`, `LibreOffice Calc`, `Canva`.
2. **Document Management & Processing Tasks:**  
   `Document Filing`, `Document Sorting`, `Document Archiving`, `Document Routing`, `Document Verification`, `Data Encoding`, `Transcription`, `Proofreading`, `Scanning`, `Printing`, `Photocopying`.
3. **Office & Administrative Operations:**  
   `Laminating`, `General Cleaning`, `Inventory Checking`, `Stock Counting`, `Office Supplies Replenishment`, `Receiving Documents`, `Logbook Recording`.
4. **Front-Desk & Public Reception Workflows:**  
   `Visitor Log Entry`, `Visitor Escorting`, `Phone Call Routing`, `Inquiry Handling`, `Appointment Scheduling`, `Queue Management`.
5. **Institutional & Legal Clerical Procedures:**  
   `Curriculum Verification`, `Graduation Clearance`, `Thesis Defense Scheduling`, `Petty Cash Voucher`, `Transcript Notarization`, `Diploma Archiving`, `Subpoena Tracking`, `Police Clearance Processing`.

#### What NOT to Tag as `CLERICAL_TERM`:
- Generic administrative nouns (e.g., `paperwork`, `files`, `folder`, `documents`, `records`, `forms`, `notes`, `sheet`, `page`, `envelope`).
- General office furniture or equipment (e.g., `desk`, `chair`, `cabinet`, `printer`, `scanner`, `shelf`, `whiteboard`).
- General non-task social interactions (e.g., `meeting`, `lunch`, `greeting`, `conversation`, `discussion`).

---

## 4. Specific Tagging Policies & Edge Cases

### 4.1. The "Specific Tool / Specific Task, Not Generic Concept" Rule
A central lesson from past remediation passes is the strict exclusion of bare generic concepts:
- **Rule:** Do not tag abstract architectural components or conceptual categories.
- **Examples:**
  - ❌ *"I connected to the `database`."* → Do not tag `database`.
  - ❌ *"I configured user `authentication`."* → Do not tag `authentication`.
  - ❌ *"I designed the executive `dashboard`."* → Do not tag `dashboard`.
  - ❌ *"I pushed code to the `backend`."* → Do not tag `backend`.
  - ✅ *"I connected to `PostgreSQL`."* → Tag `PostgreSQL` as `IT_TERM`.
  - ✅ *"I configured `Supabase Auth`."* → Tag `Supabase Auth` as `IT_TERM`.
  - ✅ *"I performed `Database Optimization`."* → Tag `Database Optimization` as `IT_TERM`.

---

### 4.2. Dual Scope: Named Tools vs. Concrete Task/Activity Phrases
The project intentionally recognizes two distinct styles of terms:
1. **Named Software / Technologies** (e.g., `Python`, `Docker`, `Microsoft Excel`).
2. **Concrete Activity / Workflow Phrases** (e.g., `Photocopying`, `General Cleaning`, `Cable Crimping`, `Document Filing`, `Data Encoding`).

**Policy:**  
Both categories are valid and intentional. An OJT journal entry frequently documents manual or procedural tasks that do not involve a brand-name software tool. However, an activity phrase must describe a **concrete, recognizable job action** (verbal noun or gerund-based action), not an environmental setting.

---

### 4.3. Bare Brand Names vs. Full Suite Names
Student journal entries vary in how software is referenced:
- **Policy:** Tag the exact span as written by the author up to its complete entity boundary.
  - If the text writes *"Microsoft Excel"*, tag the entire span `[Microsoft Excel]` as `CLERICAL_TERM`.
  - If the text writes bare *"Excel"*, tag `[Excel]` as `CLERICAL_TERM`.
  - If the text writes *"Google Sheets"*, tag the entire span `[Google Sheets]` as `CLERICAL_TERM`.
- **Constraint:** Do not expand the span to include words that are not part of the tool name (e.g., in *"opened Excel software"*, tag only `Excel`, not `Excel software`).

---

### 4.4. Multi-Word Terms & Boundary Contiguity
- Multi-word entities must be tagged as a **single, contiguous span**.
  - Example: `[Tailwind CSS]` (one span), NOT `[Tailwind]` + `[CSS]`.
  - Example: `[Curriculum Verification]` (one span), NOT `[Curriculum]` + `[Verification]`.
  - Example: `[Microsoft SQL Server]` (one span).
- Punctuation that is part of the name must be included: `Vue.js`, `Node.js`, `C#`, `C++`, `ASP.NET`, `CI/CD`.

---

### 4.5. Abbreviations & Acronyms
- Standard, unambiguous technical acronyms are tagged when used in a technical context:
  - `API`, `SDK`, `IDE`, `REST`, `SQL`, `CLI`, `VPN`, `SSH`, `FTP`, `LAN`, `VLAN`.
- If an acronym is ambiguous and used in a non-technical sense, do NOT tag it.

---

### 4.6. Dual-Natured Tools & Context-Dependent Labeling
Certain productivity applications (most notably `Microsoft Excel` and `Google Sheets`) span both administrative office tasks and technical automation workflows:
- **Dictionary Exclusion Policy:** Dual-natured tools are excluded from the deterministic EntityRuler dictionary (`data/terms.csv`) so that they are never forcefully labeled by string match. The fine-tuned Transformer resolves them by context alone.
- **Clerical Operations (Default):** When the software is used for routine data entry, organizing records, tabulating logs, or preparing office reports, tag `[Microsoft Excel]` or `[Excel]` as `CLERICAL_TERM`.
  - ✅ *"Encoded student attendance records in `[Microsoft Excel]`."* → Tag `Microsoft Excel` as `CLERICAL_TERM`.
  - ✅ *"Organized vendor evaluation files in `[Excel]`."* → Tag `Excel` as `CLERICAL_TERM`.
- **Automation, Scripting & Engineering:** When the sentence describes automating workflows, writing scripts, or programming inside the application, tag the **specific technical construct** as `IT_TERM`, rather than the bare suite name:
  - ✅ *"Automated weekly report consolidation using `[VBA macros]`."* → Tag `VBA macros` as `IT_TERM`.
  - ✅ *"Developed custom data validation scripts in `[Google Apps Script]`."* → Tag `Google Apps Script` as `IT_TERM`.

---

## 5. Hard-Negative Exclusion Rules (Institutional & Environmental Words)

To eliminate false-positive extraction of context nouns, the following rule is strictly enforced:

> **Rule:** Institutional, environmental, physical-space, and personnel nouns MUST NEVER be tagged as `IT_TERM` or `CLERICAL_TERM`, regardless of context.

### Hard-Negative Category Reference Table

| Category | Forbidden Nouns (Never Tag) | Incorrect Prediction | Correct Annotation |
| :--- | :--- | :--- | :--- |
| **Academic & Institutional** | `university`, `campus`, `college`, `school`, `academy`, `institute`, `faculty` | `"arrived at the [university]"` | No entity |
| **Offices & Departments** | `office`, `MIS office`, `accounting office`, `registrar`, `department`, `division` | `"walked to the [MIS office]"` | No entity |
| **Physical Spaces & Rooms** | `laboratory`, `computer lab`, `workstation`, `auditorium`, `conference room`, `cafeteria`, `hall` | `"cleaned the [computer lab]"` | `[General Cleaning]` if phrase present, but never `[computer lab]` |
| **People & Roles** | `intern`, `student`, `supervisor`, `mentor`, `colleague`, `staff`, `personnel`, `clerk` | `"met with my [supervisor]"` | No entity |
| **Generic Artifacts** | `computer`, `monitor`, `desk`, `chair`, `files`, `folder`, `paper`, `document`, `binder` | `"organized [green folders]"` | No entity |

### Negative Sentence Examples (Zero Entities to Tag)
- *"Arrived at the university early and met with the department supervisor in the MIS office."* → **Zero entities**.
- *"Attended the weekly intern alignment meeting in conference room B."* → **Zero entities**.
- *"Wiped down the workstation desks and turned off the laboratory computers."* → **Zero entities**.
- *"Helped fellow student interns locate the cafeteria on campus."* → **Zero entities**.

---

## 6. Annotator Decision Workflow

```
                  [Token or Phrase in Sentence]
                                |
             Is it an institutional noun, room,
               personnel title, or bare noun?
               (e.g., university, office, intern, files)
                                |
                   YES ---------+--------- NO
                    |                       |
              [DO NOT TAG]           Does it denote a:
                                     (1) Specific software, tool, language, or
                                     (2) Concrete technical or clerical task?
                                            |
                               YES ---------+--------- NO
                                |                       |
                       Which category?             [DO NOT TAG]
                                |
             +------------------+------------------+
             |                                     |
       Technical / IT                       Clerical / Office
             |                                     |
     Tag as [IT_TERM]                    Tag as [CLERICAL_TERM]
```

---

## 7. Span Boundary Rules

1. **Exact Word Boundaries:** Span starts at the first character of the term and ends at the character immediately after the last word:
   - Correct: `[start: 47, end: 59]` → `"Tailwind CSS"`
   - Incorrect: `" Tailwind CSS "` (includes whitespace)
2. **No Grammatical Particles:** Do not include determiners (`the`, `a`), prepositions (`using`, `with`, `in`), or trailing conjunctions:
   - Correct: *"using [Docker] containers"*
   - Incorrect: *"using [the Docker containers]"*
3. **Compound Noun Attachment:** Include version or descriptive qualifiers only if they form the proper product name:
   - Correct: `[Python 3]`, `[Vue.js 3]`, `[Microsoft Excel 2019]`
   - Correct: `[PostgreSQL] database` (tag `PostgreSQL`, leave `database` out)

---

## 8. Dataset Composition Targets

To maintain balanced generalization and prevent spurious false-positive predictions, the dataset builder ([`scripts/annotation.py`](../scripts/annotation.py)) evaluates two composition diagnostics upon loading and splitting:

### 8.1. Negative-Example Ratio (Target: 25–35%)
- **Target Range:** 25% to 35% of all records in the dataset should contain zero entities (`"entities": []`).
- **Empirical Rationale:** Project experiments revealed that when the negative ratio fell below ~20%, the transformer model developed an aggressive extractive bias, triggering false-positive extractions on institutional and environmental nouns (e.g. erroneously tagging `computer lab`, `department`, or `supervisor`). Negative examples teach the model when *not* to predict.
- **Operational Guidance:** This is a diagnostic metric reported during `prepare_real_data_pipeline()`. It guides human annotators on whether additional un-tagged journal sentences (e.g. orientation days, commute logs, general administrative announcements) should be ingested.

### 8.2. Class Balance (Target: ~1:1, neither class < 45%)
- **Target Ratio:** Roughly equal representation of `IT_TERM` and `CLERICAL_TERM` entity counts (imbalance ratio $\le 1.5:1$, with neither class dropping below ~45% of total entities).
- **Empirical Rationale:** Marked class imbalance directly degrades precision and recall on the minority class. Historically, under-representation of `CLERICAL_TERM` caused lower F1 on administrative workflows compared to software tools.
- **Operational Guidance:** Class balance diagnostics are reported across Full, Train, Dev, and Test splits. Annotators should prioritize collecting entries for the under-represented category when the ratio exceeds 1.5:1.

---

## 9. Real-Data Provenance & Anonymization Policy

Following the retirement of synthetic training data, all project training, validation, and holdout data originate exclusively from **authentic OJT student journal submissions** stored in `data/data.jsonl`.

### 9.1. Mandatory Anonymization Policy
Before any real student journal submission is appended to `data/data.jsonl` or `data/test/holdout.jsonl`, annotators must enforce strict privacy sanitization:
1. **Personal Names (Strictly Forbidden):**
   - The student author's personal name must be omitted or removed.
   - Names of specific individuals mentioned in the journal text (supervisors, mentors, staff, teachers, fellow students, e.g. *"Ma'am Kathy"*, *"Sir Alex"*, *"Dr. Ramos"*) must be removed or replaced with neutral role nouns (e.g. *"the supervisor"*, *"the office head"*).
2. **Office and Department Names (Permitted):**
   - Public institutional, municipal, or agency titles (e.g. *"Municipal Health Office"*, *"Department of Agrarian Reform"*, *"Probation and Parole Administration"*) are acceptable to retain as realistic operational context.
   - However, annotators must adhere to Section 5: office and agency names **must never be tagged as entities**.
3. **Contact Details & Private Identifiers (Strictly Forbidden):**
   - Phone numbers, email addresses, student identification numbers, and exact home addresses must be scrubbed prior to ingestion.

### 9.2. Ingestion Checklist for New Records
When contributing new journal records to `data/data.jsonl`:
- [ ] Record conforms to standard schema: `{"text": "...", "entities": [{"start": ..., "end": ..., "label": "..."}]}`.
- [ ] Personal names and sensitive PII scrubbed.
- [ ] Spans precisely match character offsets $[start, end)$ against original text.
- [ ] Labels are exclusively `IT_TERM` or `CLERICAL_TERM`.
- [ ] Hard-negative nouns (offices, equipment, rooms) remain unannotated.
- [ ] Record does not conflict with existing sentences in `data/test/holdout.jsonl`.

---

## 10. Version History & Changelog

- **v1.0.0 (September 2026):** Initial Phase 2 specification defining the two-tier label taxonomy (`IT_TASK`/`CLERICAL` $\rightarrow$ `IT_TERM`/`CLERICAL_TERM`), hard-negative exclusion rules, and initial span boundary criteria.
- **v1.1.0 (September 2026):** Unified the taxonomy to a single-tier model (`IT_TERM`/`CLERICAL_TERM`), retired the legacy two-tier scheme, updated stale file references from synthetic `data/reviewed/annotations.jsonl` to real `data/data.jsonl`, added Dataset Composition Targets (negative ratio and class balance), and established the Real-Data Provenance & Anonymization Policy.
