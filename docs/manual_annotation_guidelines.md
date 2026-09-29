# Annotation Guidelines: `IT_TERM` and `CLERICAL_TERM`

## 1. What you should do

Read sentences from the OJT journals and mark the **terms** that belong to one of the categories:

| Label | Meaning in one line |
| :--- | :--- |
| `IT_TERM` | Computers, software, networks, programming, systems, and technical support |
| `CLERICAL_TERM` | Paperwork, records, forms, filing, data encoding, and office administration |
 
The marked terms are used to train a model. **The model learns from your consistency**, so a rule applied the same way every time is more valuable than a clever decision made once.
 
**The three things that matter most**
 
1. **Label every occurrence.** If a term is labeled in one sentence, label it in every sentence where it appears in the same meaning. You must be consistent in all of the items in the dataset. Missing/Improper labels are the biggest quality problem. Example, if one sentence have `Microsoft Excel` as its term, then all sentences with the **same term** must also have `Microsoft Excel`, not just `Microsoft` nor `Excel`.
2. **When unsure, talk with your groupmates. Do not skip silently nor just label it randomly.**
3. **Never edit the text.** Do not fix typos, spacing, capitalization or line breaks.

---

## 2. Quick reference card
 
1. Read the **whole sentence** before marking anything.
2. Label **specific** terms, not generic words that could appear in any workplace.
3. Never label a **verb**. Label the noun phrase or an approved task name.
4. Use the **shortest span** that still names the concept: no leading verbs, articles, "and", or punctuation.
5. If a longer term contains a shorter one, label the **longest meaningful term** (`Tailwind CSS`, not `CSS`).
6. Spans start and end on **whole words**, and never overlap.
7. Same string, same label, unless the sentence clearly changes the meaning.
8. Choose the label with the **tie-breakers** in Section 5, then the **hard-case table** in Section 6.
9. Sentence has no terms? Leave it empty, but only after you have re-read it once.

---

## 3. Labels in detail
 
### `IT_TERM`
 
Hardware, software and tools used for computing, networks, programming and system work, technical concepts, and technical support activities.
 
- **Devices and hardware:** `router`, `projector`, `CCTV`, `CAT6 UTP Cable`, `printer`, `computer units`
- **Software and tools:** `Git`, `Tailwind CSS`, `Canva`, `database`
- **Systems and concepts:** `system design`, `UI/UX`, `data security`, `authentication system`
- **Technical tasks:** `coding`, `debugging`, `troubleshooting`, `preventive maintenance`, `device setup`

### `CLERICAL_TERM`
 
Paperwork, records, forms and administrative procedures, plus office software used for records and paperwork.
 
- **Documents and records:** `Police Clearance`, `log books`, `contracts`, `received documents`, `passbooks`, `PESO office book`
- **Clerical tasks:** `data entry`, `encoding`, `filing`, `bookkeeping`, `file management`
- **Office software for records work:** `Excel`, `Microsoft Excel`, `Word`
- **Administrative concepts:** `regulatory compliance`, `incident tracking`

---
 
## 4. What counts as a term
 
### 4.1 The specificity test
 
Ask: **"Could this word or phrase appear in almost any workplace, unchanged?"**
 
- **Yes → it is generic. Do not label it.**
- **No, it names a particular tool, device, system, document type, or IT/clerical technique → label it.**

### 4.2 Generic words: do not label these on their own
 
`program` · `technical` · `issues` · `errors` · `procedures` · `communication` · `teamwork` · `collaboration` · `requirements` · `suggestions` · `information` · `data` · `records` · `files` · `documents` · `papers` · `system` · `equipment` · `devices` · `tools` · `tasks`
 
**A generic word becomes a term when a specific modifier makes it name something particular:**

| Generic (do not label) | Specific (label) |
| :--- | :--- |
| documents | `received documents`, `routed documents`, `medical documentation` |
| data | `evaluation data`, `data security` |
| system | `authentication system`, `record-keeping system` |
| testing | `load testing` |
| scanning | `vulnerability scanning`, `document scanning` |
| editing | `Canva Editing`, `video editing` |

### 4.3 Task words: verbs versus task names
 
- **Never label a verb form:** `encoded`, `designed`, `attended`, `monitored`, `collaborated`, `synced`, `edited`.
    - This only means the verb itself is never part of the span. The object after it is judged on its own, using Section 4.2 and 4.4: `encoded data` → nothing (`data` is generic); `encoded evaluation data` → `[evaluation data]{CLERICAL_TERM}`; `encoded new employees into the database` → `[database]{IT_TERM}`.
- **Label these task names, even alone** (they are established technical or clerical terms):

|`IT_TERM` | `CLERICAL_TERM` |
| :--- | :---| 
| `coding`, `debugging`, `troubleshooting` | `encoding`, `data entry`, `filing`, `bookkeeping`, `formatting` |

- **Do not label ordinary work verbs or gerunds on their own:** `organizing`, `arranging`, `sorting`, `managing`, `updating`, `planning`, `printing`, `scanning`, `assisting`, `giving`. Label them only when a specific modifier makes a term (Section 4.2).

**For any general or activity word, look it up in Section 11 first.** It gives the decision, the label, and examples for each term that was previously labeled inconsistently.
 
### 4.4 Ask before labeling: is it in the sentence?
 
Only label what the sentence actually says. Do not label something because it is "probably meant" or because you know the surrounding project.
 
---
 
## 5. Choosing the label
 
Apply these in order and stop at the first that decides it.
 
1. **What kind of thing is it?** Software, hardware, network, code or a technical system → `IT_TERM`. Paper, record, form, document or an administrative procedure → `CLERICAL_TERM`.
2. **Container versus content.** The system, software or database that *stores* things → `IT_TERM`. The records, documents or entries *stored* → `CLERICAL_TERM`.
   - `record-keeping system` → `IT_TERM`; `system records` → `CLERICAL_TERM`
3. **Task words follow the lists in Section 4.3.**
4. **Meaning decides, not the word.** Check the hard-case table (Section 6).
5. **Still 50/50?** Decide as a team. **Do not guess.**

---

## 6. Hard cases (fixed answers)
 
These are terms that were labeled inconsistently in earlier rounds. Use the answer given here every time.
 
| Term | Label | Note |
| :--- | :--- | :--- |
| `data entry`, `encoding` | `CLERICAL_TERM` | Typing or transferring information into records |
| `file management`, `filing` | `CLERICAL_TERM` | Handling paper or digital files as records |
| `record-keeping system` | `IT_TERM` | The system that holds records |
| `system records` | `CLERICAL_TERM` | The records themselves |
| `data collection` | `IT_TERM` | Does this just collect data or data encoding -> `CLERICAL_TERM`, or like web scrapping -> `IT_TERM`? |
| `incident tracking` | `CLERICAL_TERM` | |
| `regulatory compliance`, `compliance requirements` | `CLERICAL_TERM` | |
| `PPMP` | `CLERICAL_TERM` | `IT_TERM` only when part of a system name (e.g. "PPMP system") |
| `Excel`, `Microsoft Excel`, `Word` | `CLERICAL_TERM` | Office software for records work |
| `Canva`, `Git`, `Tailwind CSS`, `CSS` | `IT_TERM` | Design and development tools |
| `accounts` | by meaning | Login or user accounts → `IT_TERM`; financial or ledger accounts → `CLERICAL_TERM` |
| `security`, `data security` | `IT_TERM` | |
| `database`, `office database` | `IT_TERM` | Label it, even in sentences about records |
| `printer`, `scanner`, `projector` | `IT_TERM` | Devices. The words `printing` and `scanning` alone are not labeled |
 
If a term is not in this table and Section 5 does not decide it, decide with your groupmates.
 
---

## 7. Span rules
 
### 7.1 Boundaries
 
Use the **shortest span that still names the concept**. Never include:
 
- leading verbs or gerunds (`Installing`, `Processing`, `Configure`)
- leading articles, possessives or conjunctions (`the`, `and`, `their`)
- trailing punctuation, quotation marks or `'s`
Spans **start and end on whole words**.
 
| Sentence fragment | Correct | Wrong | Why |
| :--- | :--- | :--- | :--- |
| Processing Police Clearance | `Police Clearance` | `Processing Police Clearance` | leading verb |
| Installing CAT6 UTP Cable and Camera | `CAT6 UTP Cable`, `Camera` | one long span | leading verb and a list (7.3) |
| using Tailwind CSS | `Tailwind CSS` | `CSS` | longest term wins |
| working on the PESO office book | `PESO office book` | `the PESO office book` or `book` | article; longest term |
| assist in filing and organizing | `filing` | `ling and organizing` | whole words only; not a verb-led phrase |
| the printer's functionality | `printer` | `printer's` | exclude `'s` |
 
### 7.2 Longest meaningful term, no overlaps
 
- If a longer term contains a shorter one, label the longer term only: `peso office book`, not `book`.
- Never label the same characters twice. Never nest one span inside another. Terms must not overlap.

### 7.3 Lists and "and"
 
- **Independent items joined by "and" or commas get separate spans:** `CAT6 UTP Cable and Camera` → two spans `CAT6 UTP Cable` and `Camera`.
- **A shared head noun stays as one span:** `Internal and External Log Books` → one span.

### 7.4 Names, abbreviations and parentheses
 
- **Names of specific systems or documents:** label in full, even if long (`Vehicular Record Expiration Alert System`).
- **Abbreviations:** label them (`PC`, `CCTV`, `LAN`, `PPMP`, `CIF`).
- **Parenthetical abbreviations inside a name stay in the span:** `Annual Investment Program (AIP) System`. Where the abbreviation appears alone elsewhere, label it alone.

### 7.5 Case, headings and typos
 
- Capitalization does not matter: `PC`, `pc` and `Pc` are the same. Label ALL-CAPS headings the same as normal text.
- Do not correct typos or spelling. Label the term as written.

---

## 8. Flags and special cases
 
| Situation | What to do |
| :--- | :--- |
| You cannot decide the label or whether it is a term | Discuss with group. Do not leave it as a plain negative. |
| Same term appears twice in a sentence | Label both occurrences. |
| The sentence has a stray `\n` or extra whitespace with no other issue | Replace it with a single space, then label normally. Do not change anything else. |
| The sentence is otherwise garbled, cut off, or missing words | Do not label it and do not rewrite or guess the missing text. |
| The sentence is an exact duplicate of another | Label it once. Do not add the duplicate. |
| You disagree with this guideline | Follow the guideline, add a note, and raise it at the next calibration meeting. |
 
**A sentence is a true negative only if** you have re-read it and no specific IT or clerical term is present.
 
---
 
### 9. Before you submit
 
- [ ] I labeled every occurrence of each term.
- [ ] No span starts or ends mid-word.
- [ ] No span starts with a verb, article, "and", or punctuation.
- [ ] I did not label a generic word on its own.
- [ ] Every doubtful item was discussed.
- [ ] I did not edit the text.

---
 
## 10. Worked examples
 
`[term]{LABEL}`; sentences with no brackets have no terms.
 
| # | Sentence | Annotation | Why |
| :-: | :--- | :--- | :--- |
| 1 | Encoded and organized vendor evaluation data using Microsoft Excel. | `[evaluation data]{CLERICAL_TERM}`, `[Microsoft Excel]{CLERICAL_TERM}` | `Encoded`, `organized` are verbs. Modifier makes `data` specific. |
| 2 | Performed basic internet connection troubleshooting. | `[internet connection troubleshooting]{IT_TERM}` | One technical task phrase. |
| 3 | Sorted, classified, and arranged office documents, contracts, and papers. | `[contracts]{CLERICAL_TERM}` | `documents` and `papers` are generic; the verbs are not labeled. |
| 4 | Processing Police Clearance | `[Police Clearance]{CLERICAL_TERM}` | Exclude leading `Processing`. |
| 5 | Continued working on the PESO office book | `[PESO office book]{CLERICAL_TERM}` | Longest term, not `book`. |
| 6 | Managed code versioning with Git | `[code versioning]{IT_TERM}`, `[Git]{IT_TERM}` | Two independent terms. |
| 7 | User authentication system protected data security | `[authentication system]{IT_TERM}`, `[data security]{IT_TERM}` | Label both; the `User` modifier is not needed. |
| 8 | Managed and updated records in the office database. | `[office database]{IT_TERM}` | `database` is specific; `records` is generic. |
| 9 | Set up speaker, projector and microphone for the event. | `[speaker]{IT_TERM}`, `[projector]{IT_TERM}`, `[microphone]{IT_TERM}` | Three separate devices. |
| 10 | Corrected errors in reading data from Excel | `[Excel]{CLERICAL_TERM}` | `errors` and `data` are generic. |
| 11 | Through proper documentation, I understood audit trails | discuss `audit trails` | Could be IT logs or clerical records. Flag it; do not guess. |
| 12 | Attended the orientation for new trainees. | *(no terms)* | `Attended` is a verb; `orientation` is generic. |
 
---
 
## 11. General and activity terms
 
These are words that earlier rounds labeled in some sentences and skipped in others. **Look the word up here before deciding.** The rule applies to every form of the word, but remember that verb forms are never labeled (`encode`, `encoded` → no; `encoding` → yes).
 
### 11.1 Three questions, in order
 
1. **Is it a verb form?** (`encoded`, `designed`, `attended`, `assist`) → **do not label.**
2. **Is it on the Group 1 list?** → **label it, even alone.**
3. **Does a specific modifier or object turn it into a particular term?** (Group 2) → **label the whole phrase.** Otherwise → **do not label** (Group 3).
Still looks technical or clerical but is not listed → discuss with group (Section 8).
 
### 11.2 Group 1: always label (task names)
 
| Term | Label | Label it when | Do NOT label when |
| :--- | :--- | :--- | :--- |
| `coding`, `programming` | `IT_TERM` | writing software, including "coding phase" | not about software (`dress code`, `color coding`) |
| `debugging` | `IT_TERM` | always | |
| `troubleshooting` | `IT_TERM` | technical problems with devices, networks or software; label the object too if it is a specific device (`troubleshooting printers`) | |
| `encoding` | `CLERICAL_TERM` | entering or transferring information into records or spreadsheets | technical sense (video or character encoding) → discuss |
| `data entry` | `CLERICAL_TERM` | always | |
| `filing` | `CLERICAL_TERM` | putting documents or records in order or storage | legal or general sense (`filing a complaint`) |
| `bookkeeping` | `CLERICAL_TERM` | always | |
| `formatting` | by meaning | document layout → `CLERICAL_TERM`; drive or disk → `IT_TERM` | |
 
### 11.3 Group 2: label only the whole phrase, never the bare word
 
| Bare word (do NOT label) | Label this phrase | Label | Note |
| :--- | :--- | :--- | :--- |
| `testing` | `load testing`, `system testing`, `unit testing` | `IT_TERM` | `Assisted in testing` → nothing |
| `scanning` | `vulnerability scanning` | `IT_TERM` | |
| | `document scanning` | `CLERICAL_TERM` | paperwork digitization |
| `printing` | `3D printing` | `IT_TERM` | other compounds (`tarpaulin printing`) → discuss with group |
| `editing` | `video editing`, `photo editing`, `Canva Editing` | `IT_TERM` | `editing documents` → nothing |
| `design`, `designing` | `system design`, `UI design`, `database design`, `UI/UX` | `IT_TERM` | graphic or print design (`ID design`, `layout`, `design elements`) → discuss with group |
| `deployment` | `system deployment`, `web deployment` | `IT_TERM` | |
| `updates`, `updating` | `system updates`, `software updates` | `IT_TERM` | `updating records` → nothing |
| `assessment` | `security assessment`, `vulnerability assessment` | `IT_TERM` | |
| `technical` | `technical support` | `IT_TERM` | see 11.5 |
 
### 11.4 Group 3: never label (generic words)
 
- **Activities and soft skills:** `organizing`, `arranging`, `sorting`, `managing`, `assisting`, `giving`, `coordination`, `communication`, `teamwork`, `collaboration`, `planning`, `presentation`
- **Events and training:** `orientation`, `lectures`, `safety demonstrations`, `attended`
- **Generic nouns:** `program`, `technical`, `issues`, `errors`, `procedures`, `requirements`, `suggestions`, `paper`, `ink`, `book`, `components`, `functionalities`, `office documents`

### 11.5 Special cases
 
- **Two generic words together stay generic.** `technical problems`, `technical duties`, `technical challenges`, `technical equipment` → **nothing**. The exceptions are the phrases listed in Groups 1 and 2, the hard-case table (Section 6), and real device or system names (`computer units`, `internet connection`).
- **`program`:** label only when it is part of the full name of an IT system (`Annual Investment Program (AIP) System`). `the program` or `training program` → nothing.
- **`book`:** label only named record books (`PESO office book`, `log books`, `passbooks`). If the sentence says only `the book` and you cannot tell which one → discuss with group.
- **`office documents`:** generic → nothing. Specific kinds are terms: `received documents`, `routed documents`, `physical documents`, `important documents`.
- **Document-type nouns are terms even alone:** `contracts`, `forms`, `passbooks`, `log books`, `ID cards`, `vendor profiles`, `remarks`, `Police Clearance` → `CLERICAL_TERM`. The umbrella words `documents`, `papers`, `files`, `records`, `information` are not.
- **Supplies and materials:** `ink`, `paper` → nothing. `tarpaulin` → discuss with group.
- **`IS` and `is`:** the word `is` is never labeled. An uppercase `IS` that seems to mean "information system" → `IT_TERM`.
- **`PC`:** `IT_TERM` whenever it means a computer (`PC setup`, `PC assembling`).
- **`clearance` alone:** nothing. Label the full document name: `Police Clearance`.
- **`audit trails`:** discuss (could be IT logs or clerical records) or do not label.

### 11.6 Examples
 
| Sentence | Annotation | Why |
| :--- | :--- | :--- |
| Continued coding the Profiling & Inventory System | `[coding]{IT_TERM}`, `[Profiling & Inventory System]{IT_TERM}` | Group 1 task name; a named system is labeled in full. |
| Assisted in troubleshooting printers, reconnecting devices, and refilling printer ink. | `[troubleshooting]{IT_TERM}`, `[printers]{IT_TERM}` | `devices` is generic; `printer ink` is a supply. |
| Assisted in the filing and organization of the office documents. | `[filing]{CLERICAL_TERM}` | `organization` and `office documents` are generic. |
| Ran load testing on the system. | `[load testing]{IT_TERM}` | The modifier makes `testing` specific. |
| Assisted in testing. | *(no terms)* | Bare `testing`. |
| Assisted with formatting a USB drive. | `[formatting]{IT_TERM}`, `[USB drive]{IT_TERM}` | `formatting` follows the meaning: a drive, not a document. |
| Adjusted the formatting of the form. | `[formatting]{CLERICAL_TERM}` | Document layout. |
| Solved technical problems with computers and printers. | `[computers]{IT_TERM}`, `[printers]{IT_TERM}` | `technical problems` is two generic words. |
| Good communication and teamwork helped the team. | *(no terms)* | Soft skills are never labeled. |
| Encoded evaluation data into Excel. | `[evaluation data]{CLERICAL_TERM}`, `[Excel]{CLERICAL_TERM}` | `Encoded` is a verb, so it is not labeled. |
