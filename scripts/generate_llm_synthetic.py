"""
LLM-Direct Synthetic Data Generation for OJT Journal NER.

Replaces the prior paraphrase-based augmentation pipeline. Instead of transforming
existing real sentences, this script has a large language model (Google Gemini)
generate novel sentence + entity pairs directly, conditioned on:
  1. docs/annotation_guidelines.md — label definitions, hard-case rules, span rules
  2. data/terms.csv — the project's entity dictionary for vocabulary grounding
  3. data/data.jsonl — rotating samples of real sentences for style/pattern reference

Critical design constraint (per JMIR findings on LLM span-offset unreliability):
  The LLM returns entity TEXT and LABEL only — never character offsets.
  Offsets are computed afterward via exact string search with word-boundary validation.

Output: data/synthetic_llm_generated.jsonl
"""

import os
import sys
import json
import re
import random
import time
import argparse
from typing import List, Dict, Any, Tuple, Set, Optional

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from google import genai

SEED = 42
random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
TERMS_CSV = "data/terms.csv"
REAL_DATA = "data/data.jsonl"
OUTPUT_PATH = "data/synthetic_llm_generated.jsonl"
GUIDELINES_PATH = "docs/annotation_guidelines.md"

# ── Generation parameters ────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 10       # Sentences per LLM call
DEFAULT_NUM_BATCHES = 50      # Total batches to generate
GEMINI_MODEL = "gemini-3.5-flash-lite"
NEGATIVE_RATIO_TARGET = 0.30  # 25-35% target per Section 8 of guidelines
RETRY_LIMIT = 3
RETRY_DELAY = 5               # Seconds between retries

# ── Data-Driven Generic Noun Blocklist (Task 2) ─────────────────────────────
# Based on authentic train+dev entity rates:
# Terms with >=50% entity rate (coding, debugging, formatting, technical, etc.)
# are RESTORED because real human annotators consistently tag them.
# Terms with <50% entity rate are BLOCKED to avoid noise propagation.
GENERIC_NOUN_BLOCKLIST = {
    # Low-rate generic narrative terms (< 50% entity rate in real data)
    "program",            # 47.6% (10/21) - overwhelmingly general narrative prose
    "encode",             # 44.4% (4/9) - narrative action verb
    "encoded",            # 23.1% (3/13) - narrative action verb
    "office documents",   # 25.0% (3/12) - generic narrative prose
    "coordination",       # 12.5% (1/8) - general soft skill
    "orientation",        # 14.3% (1/7) - general activity
    "deployment",         # 33.3% (2/6) - ambiguous event/milestone
    "front page",         # 28.6% (2/7) - document layout location
    "copies",             # 33.3% (1/3) - generic noun
    "issued",             # 0.0% (0/1) - past verb
    "stalls",             # 0.0% - physical market stalls
    "proctoring",         # 0.0% - exam monitoring
    "reference numbers",  # 0.0% - generic clerical artifact
    "field trials",       # 0.0% - non-software field trial
    "data quality",       # 0.0% - abstract concept
    "system workflows",   # 0.0% - abstract concept
    "data requirements",  # 0.0% - abstract concept
    "system exploration", # 0.0% - abstract concept
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_terms_by_label(path: str = TERMS_CSV) -> Dict[str, List[str]]:
    """Load terms.csv into {label: [term, ...]} mapping."""
    terms: Dict[str, List[str]] = {"IT_TERM": [], "CLERICAL_TERM": []}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                term, label = parts[0].strip(), parts[1].strip()
                if label in terms:
                    terms[label].append(term)
    return terms


def load_real_data(path: str = REAL_DATA) -> List[Dict[str, Any]]:
    """Load real annotated records from data.jsonl."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_style_examples(
    real_records: List[Dict[str, Any]],
    n_positive: int = 5,
    n_negative: int = 2,
) -> str:
    """Build a rotating sample of real sentences for in-context style reference.

    Returns a formatted string showing example sentences with their entity annotations.
    Samples are randomized each call to provide variety across batches.
    """
    positive = [r for r in real_records if r.get("entities")]
    negative = [r for r in real_records if not r.get("entities")]

    pos_sample = random.sample(positive, min(n_positive, len(positive)))
    neg_sample = random.sample(negative, min(n_negative, len(negative)))

    lines = ["### Style Reference (real OJT journal sentences — do NOT copy or paraphrase these):\n"]
    for r in pos_sample:
        text = r["text"]
        ents = []
        for e in r["entities"]:
            term = text[e["start"]:e["end"]]
            ents.append(f'  - "{term}" → {e["label"]}')
        lines.append(f'Sentence: "{text}"')
        lines.append("Entities:")
        lines.extend(ents)
        lines.append("")

    for r in neg_sample:
        lines.append(f'Sentence: "{r["text"]}"')
        lines.append("Entities: (none — negative example)")
        lines.append("")

    return "\n".join(lines)


def sample_vocabulary(
    terms_by_label: Dict[str, List[str]],
    n_per_label: int = 15,
) -> str:
    """Sample a subset of dictionary terms for vocabulary grounding."""
    lines = ["### Available Vocabulary (sample from data/terms.csv):\n"]
    for label, terms in terms_by_label.items():
        sample = random.sample(terms, min(n_per_label, len(terms)))
        lines.append(f"**{label}**: {', '.join(sample)}")
    lines.append("")
    return "\n".join(lines)


# ── Prompt construction ──────────────────────────────────────────────────────

def build_system_prompt() -> str:
    """Build the system instruction for the LLM."""
    return """You are a data generation assistant for an NER (Named Entity Recognition) training pipeline.
Your task is to generate realistic OJT (On-the-Job Training) student journal entry sentences
with accurate entity annotations.

You operate under strict labeling rules defined in the project's annotation guidelines.
Follow ALL rules exactly — especially the hard-negative exclusion rules and the
"specific tool / specific task, not generic concept" rule."""


def build_generation_prompt(
    terms_by_label: Dict[str, List[str]],
    real_records: List[Dict[str, Any]],
    batch_size: int = 10,
    negative_count: int = 3,
) -> str:
    """Build the full generation prompt for one batch."""
    vocab_section = sample_vocabulary(terms_by_label)
    style_section = sample_style_examples(real_records)
    positive_count = batch_size - negative_count

    return f"""Generate exactly {batch_size} unique OJT journal entry sentences for NER training data.

## Requirements:
- Generate {positive_count} POSITIVE sentences (containing at least one IT_TERM or CLERICAL_TERM entity)
- Generate {negative_count} NEGATIVE sentences (containing ZERO entities — describe general activities,
  social interactions, orientation, commute, meetings, etc.)
- Write sentences that sound like real Filipino college student OJT journal entries
- Vary sentence structures: simple, compound, and complex sentences
- Mix first-person and third-person perspectives
- Include both short (1 sentence) and medium-length (2-3 sentences) entries

## Label Definitions:
**IT_TERM**: Specific technology, software tool, programming language, library, framework,
database, infrastructure component, or concrete technical task.
Examples: Python, Docker, REST API, Database Optimization, Network Troubleshooting, Git

**CLERICAL_TERM**: Office productivity application, document handling task, filing system,
record-keeping workflow, or administrative duty.
Examples: Microsoft Excel, Document Filing, Data Encoding, Inventory Checking, Photocopying

## CRITICAL RULES (violations will invalidate the data):
1. **NO generic nouns as entities**: Never tag "database", "system", "software", "code",
   "computer", "website", "application", "server", "backend", "frontend", "data",
   "documents", "files", "records", "forms", "program"
2. **NO institutional/environmental nouns**: Never tag "office", "university", "campus",
   "department", "laboratory", "computer lab", "workstation", "supervisor", "intern"
3. **NO office equipment as entities**: Never tag "printer", "scanner", "monitor",
   "keyboard", "desk", "chair"
4. **Multi-word terms must be complete**: "Google Cloud Platform" (correct), not "Google" + "Cloud Platform"
5. **Tag exact surface form as written**: If "Excel" appears, the entity text is "Excel".
   If "Microsoft Excel" appears, the entity text is "Microsoft Excel".
6. **Entities must be contiguous spans**: No split spans.
7. **Do NOT append generic words to tool names**: Tag "Python" (not "Python script"), "Docker" (not "Docker container"), "Excel" (not "Excel software").
8. **DO NOT use held-out benchmark terms**: Do NOT mention or tag: Tailwind CSS, Bun, Svelte, Supabase, Vite, Next.js, Rust, GraphQL, Prisma, Pinia, Astro, Deno, NestJS, Celery, Poetry.

## For NEGATIVE sentences, include these types (hard negatives):
- Arriving at the office/university
- Meeting with supervisors or colleagues
- Attending orientations, seminars, or workshops
- General observations about the workplace
- Commute or routine non-task activities
- Cleaning desks, organizing physical spaces (without naming specific clerical tasks)

{vocab_section}

{style_section}

## Output Format:
Return a JSON array. Each element must have exactly two fields:
- "text": the journal sentence (string)
- "entities": array of objects, each with "text" (string) and "label" (string: "IT_TERM" or "CLERICAL_TERM")
  For negative sentences, use an empty array [].

Example output format:
```json
[
  {{"text": "Configured the project's CI/CD pipeline using GitHub Actions.", "entities": [{{"text": "CI/CD", "label": "IT_TERM"}}, {{"text": "GitHub Actions", "label": "IT_TERM"}}]}},
  {{"text": "Attended the weekly team meeting and discussed progress.", "entities": []}}
]
```

Generate exactly {batch_size} sentences. Return ONLY the JSON array, no other text."""


# ── Span location (post-LLM offset computation) ─────────────────────────────

def locate_entity_spans(
    text: str,
    entities: List[Dict[str, str]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Locate entity text spans in the sentence via exact string search with
    word-boundary validation.

    Adapted from the prior pipeline's relocate_entities() function.
    The LLM returns entity text + label only; this function computes the
    character offsets.

    Returns:
        (success, located_entities) where located_entities have start/end/label fields.
        Returns (False, []) if any entity text cannot be found or has boundary issues.
    """
    located: List[Dict[str, Any]] = []
    used_ranges: List[Tuple[int, int]] = []

    for ent in entities:
        surface = ent["text"].strip()
        raw_lbl = ent.get("label", "").strip()
        label = "IT_TERM" if "IT" in raw_lbl.upper() else "CLERICAL_TERM"

        if not surface:
            return False, []

        # Search for the entity text, skipping already-used ranges
        search_start = 0
        found = False
        while search_start < len(text):
            pos = text.find(surface, search_start)
            if pos == -1:
                break

            start_idx = pos
            end_idx = pos + len(surface)

            # Word-boundary check: character before start and after end
            # must not be alphanumeric (prevents partial-word matches)
            before = text[start_idx - 1] if start_idx > 0 else " "
            after = text[end_idx] if end_idx < len(text) else " "

            if before.isalnum() or after.isalnum():
                search_start = pos + 1
                continue

            # Check this range isn't already taken by another entity
            overlap = False
            for (us, ue) in used_ranges:
                if max(start_idx, us) < min(end_idx, ue):
                    overlap = True
                    break

            if overlap:
                search_start = pos + 1
                continue

            located.append({
                "start": start_idx,
                "end": end_idx,
                "label": label,
            })
            used_ranges.append((start_idx, end_idx))
            found = True
            break

        if not found:
            return False, []

    # Sort by start offset and verify no overlaps
    located.sort(key=lambda e: e["start"])
    for i in range(len(located) - 1):
        if located[i]["end"] > located[i + 1]["start"]:
            return False, []

    return True, located


# ── Quality filters ──────────────────────────────────────────────────────────

def passes_quality_checks(
    text: str,
    entities: List[Dict[str, Any]],
    eval_sentences: Optional[Set[str]] = None,
    unseen_benchmark_terms: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """Validate a single generated record against quality criteria.

    Returns:
        (passes, reason) where reason explains rejection if passes is False.
    """
    # 1. Text must be non-empty and reasonable length
    if not text or len(text.strip()) < 10:
        return False, "Text too short"
    if len(text) > 1000:
        return False, "Text too long"

    # 2. Check for generic noun blocklist in entity terms
    for ent in entities:
        term = text[ent["start"]:ent["end"]]
        if term.lower().strip() in GENERIC_NOUN_BLOCKLIST:
            return False, f"Blocked generic noun: '{term}'"

    # 3. Check for bare generic nouns tagged as entities
    bare_generics = {
        "system", "database", "software", "code", "computer", "website",
        "application", "server", "backend", "frontend", "data", "documents",
        "files", "records", "forms", "program", "network", "printer",
        "scanner", "monitor", "keyboard", "desk", "office", "university",
        "campus", "department", "laboratory", "supervisor", "intern",
        "meeting", "discussion", "conversation",
    }
    for ent in entities:
        term = text[ent["start"]:ent["end"]].strip().lower()
        if term in bare_generics:
            return False, f"Bare generic noun tagged: '{term}'"

    # 4. Check for trailing generic words on entities (e.g. 'Python script', 'Excel software')
    for ent in entities:
        term = text[ent["start"]:ent["end"]].strip()
        t_low = term.lower()
        if t_low.endswith(" script") and t_low not in {"javascript", "typescript", "google apps script"}:
            return False, f"Entity contains trailing generic noun: '{term}'"
        if t_low.endswith(" software") or t_low.endswith(" tool"):
            return False, f"Entity contains trailing generic noun: '{term}'"

    # 5. Check for unseen benchmark term leakage (strict evaluation isolation)
    if unseen_benchmark_terms:
        text_lower = text.lower()
        for term in unseen_benchmark_terms:
            pattern = r"\b" + re.escape(term.lower()) + r"\b"
            if re.search(pattern, text_lower):
                return False, f"Contains unseen benchmark term: '{term}'"

    # 6. Verify entity text matches what's at the offsets
    for ent in entities:
        actual = text[ent["start"]:ent["end"]]
        if not actual.strip():
            return False, "Empty entity span"

    # 7. Check against evaluation set for near-duplicates
    if eval_sentences:
        text_lower = text.strip().lower()
        if text_lower in eval_sentences:
            return False, "Exact duplicate of evaluation sentence"

    return True, "OK"


# ── LLM interaction ──────────────────────────────────────────────────────────

def call_gemini(
    client: genai.Client,
    prompt: str,
    system_prompt: str,
    model: str = GEMINI_MODEL,
) -> Optional[str]:
    """Call Gemini and return the text response. Tries fallback models if needed."""
    candidate_models = [model]

    for m in candidate_models:
        for attempt in range(RETRY_LIMIT):
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.9,
                        top_p=0.95,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                    ),
                )
                if response.text:
                    return response.text
            except Exception as e:
                err_str = str(e)
                print(f"  [WARN] Gemini API error ({m}, attempt {attempt + 1}/{RETRY_LIMIT}): {e}")
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    time.sleep(15)
                elif attempt < RETRY_LIMIT - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def parse_llm_response(raw: str) -> List[Dict[str, Any]]:
    """Parse the LLM's JSON response into a list of records."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    return []


# ── Main generation pipeline ────────────────────────────────────────────────

def run_generation(
    num_batches: int = DEFAULT_NUM_BATCHES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_path: str = OUTPUT_PATH,
    model: str = GEMINI_MODEL,
) -> List[Dict[str, Any]]:
    """Run the full LLM-direct generation pipeline."""

    # 1. Initialize Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("  Set it with: export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    print(f"Initialized Gemini client (model: {model})")

    # 2. Load grounding data
    terms_by_label = load_terms_by_label()
    real_records = load_real_data()
    print(f"Loaded {sum(len(v) for v in terms_by_label.values())} terms from {TERMS_CSV}")
    print(f"Loaded {len(real_records)} real records from {REAL_DATA}")

    # 3. Build system prompt
    system_prompt = build_system_prompt()

    # 4. Load evaluation sentences for dedup checking
    eval_sentences: Set[str] = set()
    for split in ["dev", "test"]:
        path = f"data/training/{split}.spacy"
        if os.path.exists(path):
            import spacy
            from spacy.tokens import DocBin
            nlp = spacy.blank("en")
            db = DocBin().from_disk(path)
            for doc in db.get_docs(nlp.vocab):
                eval_sentences.add(doc.text.strip().lower())

    for eval_path in ["data/test/unseen_benchmark.jsonl", "data/test/holdout.jsonl"]:
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            eval_sentences.add(rec["text"].strip().lower())
                        except Exception:
                            pass

    # Also add all real training sentences to prevent near-copies
    real_sentences = {r["text"].strip().lower() for r in real_records}
    eval_sentences.update(real_sentences)

    # Load unseen benchmark terms for strict evaluation isolation
    unseen_benchmark_terms: Set[str] = set()
    unseen_path = "data/test/unseen_benchmark.jsonl"
    if os.path.exists(unseen_path):
        with open(unseen_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        for ent in rec.get("entities", []):
                            t = ent.get("term", rec["text"][ent["start"]:ent["end"]] if "start" in ent else "")
                            if t:
                                unseen_benchmark_terms.add(t.strip())
                    except Exception:
                        pass

    print(f"Loaded {len(eval_sentences)} sentences for deduplication guard")
    print(f"Loaded {len(unseen_benchmark_terms)} unseen benchmark terms for isolation guard")

    # 5. Generate batches
    all_records: List[Dict[str, Any]] = []
    total_rejected = 0
    total_span_failures = 0
    seen_texts: Set[str] = set()

    # Resume from existing records if present
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        all_records.append(rec)
                        seen_texts.add(rec["text"].strip().lower())
                    except Exception:
                        pass
        print(f"Resuming with {len(all_records)} existing records from {output_path}")

    target_total = num_batches * batch_size
    target_negatives = int(target_total * NEGATIVE_RATIO_TARGET)
    neg_per_batch = max(1, target_negatives // num_batches)

    print(f"\n{'='*60}")
    print(f"Target: {target_total} total records ({num_batches} batches × {batch_size})")
    print(f"Current: {len(all_records)} records (need {max(0, target_total - len(all_records))} more)")
    print(f"Target negative ratio: {NEGATIVE_RATIO_TARGET*100:.0f}% (~{neg_per_batch} per batch)")
    print(f"{'='*60}\n")

    batch_idx = len(all_records) // batch_size
    attempt_idx = 0
    max_attempts = num_batches * 3

    while len(all_records) < target_total and attempt_idx < max_attempts:
        attempt_idx += 1
        prompt = build_generation_prompt(
            terms_by_label, real_records,
            batch_size=batch_size,
            negative_count=neg_per_batch,
        )

        print(f"Batch {batch_idx + 1}/{num_batches} (attempt {attempt_idx}): ", end="", flush=True)

        raw_response = call_gemini(client, prompt, system_prompt, model=model)
        if not raw_response:
            print("FAILED (no response, retrying in 5s)")
            time.sleep(5)
            continue

        parsed = parse_llm_response(raw_response)
        if not parsed:
            print("FAILED (parse error, retrying in 3s)")
            time.sleep(3)
            continue

        batch_accepted = 0
        for item in parsed:
            text = item.get("text", "").strip()
            raw_entities = item.get("entities", [])

            if not text:
                total_rejected += 1
                continue

            # Dedup within this run
            text_lower = text.lower()
            if text_lower in seen_texts:
                total_rejected += 1
                continue

            # Locate spans
            if raw_entities:
                success, located = locate_entity_spans(text, raw_entities)
                if not success:
                    total_span_failures += 1
                    continue
            else:
                located = []

            # Quality check
            passes, reason = passes_quality_checks(
                text, located,
                eval_sentences=eval_sentences,
                unseen_benchmark_terms=unseen_benchmark_terms,
            )
            if not passes:
                total_rejected += 1
                continue

            # Accept
            record = {
                "text": text,
                "entities": [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in located],
                "augmentation_type": "llm_generated",
            }
            all_records.append(record)
            seen_texts.add(text_lower)
            batch_accepted += 1

        print(f"accepted {batch_accepted}/{len(parsed)} (total: {len(all_records)})")
        if batch_accepted > 0:
            batch_idx += 1
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for record in all_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Rate limiting (15 RPM free tier limit for gemini-3.5-flash-lite)
        time.sleep(4.5)

    # 6. Report statistics
    positive_records = [r for r in all_records if r["entities"]]
    negative_records = [r for r in all_records if not r["entities"]]

    it_count = sum(1 for r in positive_records for e in r["entities"] if e["label"] == "IT_TERM")
    clerical_count = sum(1 for r in positive_records for e in r["entities"] if e["label"] == "CLERICAL_TERM")
    total_entities = it_count + clerical_count

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total accepted records : {len(all_records)}")
    print(f"  Positive records     : {len(positive_records)}")
    print(f"  Negative records     : {len(negative_records)} ({len(negative_records)/max(len(all_records),1)*100:.1f}%)")
    print(f"  Total entities       : {total_entities}")
    print(f"    IT_TERM            : {it_count} ({it_count/max(total_entities,1)*100:.1f}%)")
    print(f"    CLERICAL_TERM      : {clerical_count} ({clerical_count/max(total_entities,1)*100:.1f}%)")
    print(f"  Rejected records     : {total_rejected}")
    print(f"  Span location fails  : {total_span_failures}")

    # Check composition targets
    neg_ratio = len(negative_records) / max(len(all_records), 1)
    if neg_ratio < 0.25 or neg_ratio > 0.35:
        print(f"\n  [WARN] Negative ratio {neg_ratio*100:.1f}% outside target range (25-35%)")

    if total_entities > 0:
        it_pct = it_count / total_entities
        clerical_pct = clerical_count / total_entities
        if it_pct < 0.45 or clerical_pct < 0.45:
            print(f"  [WARN] Class imbalance: IT={it_pct*100:.1f}%, CLERICAL={clerical_pct*100:.1f}% (target: neither <45%)")

    # 7. Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records)} records to {output_path}")
    return all_records


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic NER training data using LLM-direct generation."
    )
    parser.add_argument(
        "--batches", "-b", type=int, default=DEFAULT_NUM_BATCHES,
        help=f"Number of generation batches (default: {DEFAULT_NUM_BATCHES})"
    )
    parser.add_argument(
        "--batch-size", "-s", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Sentences per batch (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=OUTPUT_PATH,
        help=f"Output JSONL path (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=GEMINI_MODEL,
        help=f"Gemini model to use (default: {GEMINI_MODEL})"
    )
    args = parser.parse_args()

    global GEMINI_MODEL_OVERRIDE
    if args.model != GEMINI_MODEL:
        GEMINI_MODEL = args.model

    run_generation(
        num_batches=args.batches,
        batch_size=args.batch_size,
        output_path=args.output,
        model=args.model,
    )
