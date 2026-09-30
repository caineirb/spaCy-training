"""
Label and Gold Annotation Audit Script (Task 6).

Categorizes held-out test errors into three distinct groups:
  Group (a): Gold spans that look malformed (leading conjunctions, dangling adverbs, punctuation, clipped tokens, over-extended phrases)
  Group (b): IT/CLERICAL label conflicts (ambiguous boundary cases where model and gold disagree on category)
  Group (c): Genuine model errors (clean true FPs and FNs)

Outputs docs/proposed_relabels.md with proposed changes and an annotation guideline draft.
DOES NOT modify any ground-truth data.
"""

import os
import sys
import json
import re
import logging
from typing import List, Dict, Any, Set, Tuple

logger = logging.getLogger("ojt_pipeline.audit_labels")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ERROR_FILES = [
    "data/eval_results/held_out_test_trstr_llm_errors_only.jsonl",
    "data/eval_results/held_out_test_trstr_errors_only.jsonl",
]

OUTPUT_DOC = "docs/proposed_relabels.md"


def is_malformed_span(term: str, full_text: str = "") -> Tuple[bool, str]:
    """Detects structural and grammatical defects in entity spans."""
    t = term.strip()
    lower_t = t.lower()

    # Leading conjunctions
    if re.match(r"^(and|or|but|with|also|then)\s+", lower_t):
        return True, "Leading conjunction (e.g. 'and ...')"

    # Dangling adverbs / prepositions
    if re.search(r"\b(systematically|dynamic|continuously|efficiently|regularly)\b", lower_t):
        return True, "Dangling adverb or adjective modifier"

    # Clipped words (e.g. 'ocuments')
    if lower_t.startswith("ocuments"):
        return True, "Clipped word token ('ocuments' missing leading 'd')"

    # Over-extended compound phrases with verb + multiple objects
    if lower_t.startswith("installing ") and " and " in lower_t:
        return True, "Over-extended compound action clause"

    # Trailing dangling prepositions / conjunctions
    if re.search(r"\s+(and|or|for|to|with|in|of|by)$", lower_t):
        return True, "Trailing dangling preposition/conjunction"

    # Punctuation artifacts
    if re.search(r"^[,.;:\"'(\[]|[,.;:\"')\]]$", t):
        return True, "Enclosed or dangling punctuation mark"

    # Known specific malformed phrases from error audit
    known_malformed = {
        "and organizing office documents",
        "information systematically",
        "navbar dynamic",
        "installing cat6 utp cable and camera",
        "and technical equipment",
        "two access point configure",
    }
    if lower_t in known_malformed:
        return True, "Flagged malformed gold annotation"

    return False, ""


def audit_test_errors() -> Dict[str, List[Dict[str, Any]]]:
    """Parses test error files and splits into Group A, B, and C."""
    error_file = None
    for ef in ERROR_FILES:
        if os.path.exists(ef):
            error_file = ef
            break

    if not error_file:
        raise FileNotFoundError(f"No test error files found among {ERROR_FILES}")

    logger.info(f"Auditing test errors from: {error_file}")

    group_a_malformed = []
    group_b_label_conflicts = []
    group_c_genuine_errors = []

    seen_errors = set()

    with open(error_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            entity_results = rec.get("entity_results", [])

            for er in entity_results:
                status = er.get("status")
                gt = er.get("gold_term")
                gl = er.get("gold_label")
                pt = er.get("pred_term")
                pl = er.get("pred_label")
                conf = er.get("pred_confidence")

                error_key = (text, gt, pt, gl, pl, status)
                if error_key in seen_errors:
                    continue
                seen_errors.add(error_key)

                item = {
                    "text": text,
                    "status": status,
                    "gold_term": gt,
                    "gold_label": gl,
                    "pred_term": pt,
                    "pred_label": pl,
                    "confidence": conf,
                }

                # Check Group A: Malformed gold span
                if gt:
                    malformed, reason = is_malformed_span(gt, text)
                    if malformed:
                        item["malformed_reason"] = reason
                        group_a_malformed.append(item)
                        continue

                # Check Group B: IT vs CLERICAL label conflict
                if status == "label_error" or (gt and pt and gl != pl and gl and pl):
                    group_b_label_conflicts.append(item)
                    continue

                # Group C: Genuine model error (clean FP or clean FN or boundary)
                group_c_genuine_errors.append(item)

    logger.info(
        f"Audit complete: Group A (Malformed)={len(group_a_malformed)}, "
        f"Group B (Label Conflicts)={len(group_b_label_conflicts)}, "
        f"Group C (Genuine Errors)={len(group_c_genuine_errors)}"
    )

    return {
        "group_a": group_a_malformed,
        "group_b": group_b_label_conflicts,
        "group_c": group_c_genuine_errors,
        "error_file": error_file,
    }


def generate_markdown_report(audit_data: Dict[str, Any]) -> str:
    """Generates docs/proposed_relabels.md with structured tables and guideline drafts."""
    group_a = audit_data["group_a"]
    group_b = audit_data["group_b"]
    group_c = audit_data["group_c"]
    error_file = audit_data["error_file"]

    lines = [
        "# Proposed Relabels and Annotation Audit (Task 6)",
        "",
        f"*Generated from error analysis of `{error_file}`. No ground-truth data has been altered.*",
        "",
        "---",
        "",
        "## 1. Group (a): Malformed Gold Spans",
        "",
        "Spans containing leading conjunctions, dangling adverbs, punctuation artifacts, or over-extended multi-clause text.",
        "",
        "| Context Sentence | Current Gold Span | Gold Label | Defect Detected | Proposed Cleaned Span | Proposed Label |",
        "| :--- | :--- | :---: | :--- | :--- | :---: |",
    ]

    for item in group_a:
        gt = item["gold_term"] or ""
        gl = item["gold_label"] or ""
        reason = item.get("malformed_reason", "Malformed span")
        text_snippet = item["text"][:80] + ("..." if len(item["text"]) > 80 else "")

        # Compute clean proposal
        clean_prop = gt
        if clean_prop.lower().startswith("and "):
            clean_prop = clean_prop[4:].strip()
        elif clean_prop.lower().startswith("with "):
            clean_prop = clean_prop[5:].strip()
        elif clean_prop.lower().startswith("installing cat6 utp cable and camera"):
            clean_prop = "CAT6 UTP Cable"
        elif clean_prop == "ocuments":
            clean_prop = "documents"
        elif "dynamic" in clean_prop.lower():
            clean_prop = clean_prop.replace("dynamic", "").strip()
        elif "systematically" in clean_prop.lower():
            clean_prop = clean_prop.replace("systematically", "").strip()

        lines.append(
            f"| `{text_snippet}` | **\"{gt}\"** | `{gl}` | {reason} | **\"{clean_prop}\"** | `{gl}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Group (b): IT vs. CLERICAL Label Conflicts",
        "",
        "Genuine ambiguity where ground-truth and model disagree on whether the activity is technical (IT) or administrative (Clerical).",
        "",
        "| Context Sentence | Entity Term | Gold Label | Model Predicted | Model Conf. | Recommended Resolution | Rationale |",
        "| :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
    ])

    for item in group_b:
        term = item["gold_term"] or item["pred_term"] or ""
        gl = item["gold_label"] or ""
        pl = item["pred_label"] or ""
        conf = f"{item['confidence']:.4f}" if item["confidence"] else "-"
        text_snippet = item["text"][:75] + ("..." if len(item["text"]) > 75 else "")

        rec = gl
        rat = "Retain gold annotation standard"
        if term.lower() in ["data entry", "file management", "record management"]:
            rec = "CLERICAL_TERM"
            rat = "Data entry and filing are core clerical duties per guideline §3.2"
        elif term.lower() in ["printing", "printer connection"]:
            rec = "CLERICAL_TERM" if "office" in text_snippet.lower() else "IT_TERM"
            rat = "Physical printing is clerical; driver/network configuration is IT"

        lines.append(
            f"| `{text_snippet}` | **\"{term}\"** | `{gl}` | `{pl}` | {conf} | `{rec}` | {rat} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Group (c): Genuine Model Errors (Sample)",
        "",
        f"A total of **{len(group_c)}** genuine model errors were identified (clean FPs and FNs). Top instances:",
        "",
        "| Type | Term | Context Snippet | Gold Label | Predicted Label | Confidence |",
        "| :---: | :--- | :--- | :---: | :---: | :---: |",
    ])

    for item in group_c[:25]:
        st = item["status"]
        term = item["gold_term"] or item["pred_term"] or ""
        gl = item["gold_label"] or "-"
        pl = item["pred_label"] or "-"
        conf = f"{item['confidence']:.4f}" if item["confidence"] else "-"
        snippet = item["text"][:70] + ("..." if len(item["text"]) > 70 else "")
        lines.append(f"| `{st}` | **\"{term}\"** | `{snippet}` | `{gl}` | `{pl}` | {conf} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Annotation Guideline Draft: Generic Activity Terms & Ambiguity Resolution",
        "",
        "### 4.1 Activity & Process Terms (`coding`, `debugging`, `organizing`, `teamwork`)",
        "",
        "1. **Concrete Technical Actions (`IT_TERM`)**:",
        "   - Terms describing direct software engineering, programming, or technical diagnostic activities (**`coding`**, **`debugging`**, **`troubleshooting`**, **`system development`**, **`testing`**) MUST be annotated as `IT_TERM` when used to denote technical work.",
        "   - *Rule*: If the activity directly interacts with code, software logic, or computer hardware, classify as `IT_TERM`.",
        "",
        "2. **Administrative & Organizational Actions (`CLERICAL_TERM`)**:",
        "   - Terms describing document handling, record structuring, and office workflow (**`organizing office documents`**, **`formatting files`**, **`filing`**, **`sorting records`**) MUST be annotated as `CLERICAL_TERM`.",
        "   - *Rule*: Bare generic actions like `organizing` or `managing` without an entity object are narrative verbs (do not tag). When modifying office materials (`organizing files`), tag the specific object/compound.",
        "",
        "3. **General Interpersonal / Soft Skills (NOT ENTITIES)**:",
        "   - Broad collaborative or workplace terms (**`teamwork`**, **`collaboration`**, **`communication`**, **`coordination`**) MUST NOT be tagged as entities.",
        "   - *Rationale*: These describe interpersonal dynamics, not domain-specific IT or clerical tools/deliverables.",
        "",
        "### 4.2 IT vs. CLERICAL Boundary Disambiguation",
        "",
        "| Term / Task | Correct Label | Boundary Rule |",
        "| :--- | :---: | :--- |",
        "| **Data Entry** | `CLERICAL_TERM` | Manual transcription, keying data into forms/Excel/spreadsheets is administrative. Only data architecture or ETL pipeline scripts qualify as IT. |",
        "| **File Management** | `CLERICAL_TERM` | Organizing, locating, and archiving physical or desktop office files is clerical. Database schema storage is IT. |",
        "| **Printing (Routine)** | `CLERICAL_TERM` | Operating office printers to produce reports, booklets, or notices is clerical. |",
        "| **Printer Setup & Diagnostics** | `IT_TERM` | Installing printer drivers, configuring network ports, and clearing hardware faults is IT. |",
        "| **Record-Keeping & Compliance** | `CLERICAL_TERM` | Maintaining attendance logs, assessment checklists, and official memos is clerical. |",
        "| **Access Control Systems** | `IT_TERM` (or `CLERICAL_TERM` if physical) | Physical key/visitor log access is clerical; electronic biometric / RFID / credential security is IT. |",
        "",
    ])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    audit_data = audit_test_errors()
    report_content = generate_markdown_report(audit_data)
    os.makedirs(os.path.dirname(OUTPUT_DOC), exist_ok=True)
    with open(OUTPUT_DOC, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Audit report saved to: {OUTPUT_DOC}")
