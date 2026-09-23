import json
import re
import difflib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_data(path="data/data.jsonl"):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def check_task_stability(orig_text, para_text, labels):
    """
    Automated check to detect task/label drift:
    - Flags if clerical records suddenly introduce IT coding/scripting keywords.
    - Flags if IT records suddenly introduce clerical filing/paperwork keywords.
    - Flags if length drastically diverges.
    """
    orig_lower = orig_text.lower()
    para_lower = para_text.lower()

    # Ratio check
    len_ratio = len(para_text) / max(len(orig_text), 1)
    if len_ratio < 0.5 or len_ratio > 2.0:
        return False, "Length ratio out of bounds"

    # IT keywords that shouldn't appear in purely clerical tasks
    it_keywords = ["coding", "scripting", "python", "javascript", "sql query", "programming", "compiled", "vba macro", "debugging", "git commit"]
    # Clerical keywords that shouldn't appear in purely IT tasks
    clerical_keywords = ["photocopying", "paper filing", "filing documents", "hard copy", "shredding", "stapling"]

    has_it = "IT_TERM" in labels
    has_clerical = "CLERICAL_TERM" in labels

    if has_clerical and not has_it:
        for kw in it_keywords:
            if kw in para_lower and kw not in orig_lower:
                return False, f"Clerical task contaminated by IT keyword '{kw}'"

    if has_it and not has_clerical:
        for kw in clerical_keywords:
            if kw in para_lower and kw not in orig_lower:
                return False, f"IT task contaminated by clerical keyword '{kw}'"

    return True, "Passed"

def relocate_entities(orig_text, para_text, orig_entities):
    """
    Verifies that all entity surface texts survive intact and non-overlapping in the paraphrase.
    Returns (success, new_entities).
    """
    new_entities = []
    # Sort entities by start position
    sorted_ents = sorted(orig_entities, key=lambda x: x["start"])
    search_start = 0

    for ent in sorted_ents:
        surface = orig_text[ent["start"]:ent["end"]]
        # Find occurrences in paraphrase
        pos = para_text.find(surface, search_start)
        if pos == -1:
            # If not found after search_start, search from 0 (in case word order swapped)
            pos = para_text.find(surface)
            if pos == -1:
                return False, None

        # Check for word boundary around surface text
        start_idx = pos
        end_idx = pos + len(surface)

        # Boundary check: ensure not matching inside a longer word
        before = para_text[start_idx-1] if start_idx > 0 else " "
        after = para_text[end_idx] if end_idx < len(para_text) else " "
        if before.isalnum() or after.isalnum():
            # Partial word match (e.g. 'Excel' matching inside 'Excellent')
            return False, None

        new_entities.append({
            "start": start_idx,
            "end": end_idx,
            "label": ent["label"],
            "surface": surface
        })
        search_start = end_idx

    # Check for overlaps
    new_entities.sort(key=lambda x: x["start"])
    for i in range(len(new_entities) - 1):
        if new_entities[i]["end"] > new_entities[i+1]["start"]:
            return False, None  # Overlapping entities

    return True, new_entities

def run_sample_generation():
    records = load_data()
    
    # Stratified selection: pick diverse records
    pos_records = [r for r in records if len(r.get("entities", [])) > 0]
    neg_records = [r for r in records if len(r.get("entities", [])) == 0]

    # Select 40 positive (20 IT, 20 Clerical / mixed) and 10 negative
    it_records = [r for r in pos_records if any(e["label"] == "IT_TERM" for e in r["entities"])]
    clerical_records = [r for r in pos_records if all(e["label"] == "CLERICAL_TERM" for e in r["entities"])]

    selected_records = []
    # Interleave to ensure diversity
    step_it = max(1, len(it_records) // 25)
    for i in range(0, min(len(it_records), 25 * step_it), step_it):
        selected_records.append(it_records[i])
        if len(selected_records) >= 25:
            break

    step_cl = max(1, len(clerical_records) // 20)
    for i in range(0, min(len(clerical_records), 20 * step_cl), step_cl):
        selected_records.append(clerical_records[i])
        if len(selected_records) >= 45:
            break

    print(f"Loading T5 paraphraser model...")
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    
    for idx, rec in enumerate(selected_records):
        orig_text = rec["text"]
        entities = rec["entities"]
        labels = [e["label"] for e in entities]
        prompt = f"paraphrase: {orig_text}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=96,
            num_beams=8,
            num_return_sequences=6,
            no_repeat_ngram_size=2,
            temperature=1.0
        )
        
        candidates = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
        
        # Deduplicate and filter candidates
        valid_paras = []
        for cand in candidates:
            # Check not identical
            if cand.lower() == orig_text.lower():
                continue
            # Check entity preservation
            ok_ent, new_ents = relocate_entities(orig_text, cand, entities)
            if not ok_ent:
                continue
            # Check task stability
            ok_task, reason = check_task_stability(orig_text, cand, labels)
            if not ok_task:
                continue
            valid_paras.append((cand, new_ents))

        if valid_paras:
            # Pick best candidate (most varied word order or highest token difference)
            best_cand, best_ents = valid_paras[0]
            results.append({
                "record_id": idx + 1,
                "original_text": orig_text,
                "original_entities": [{"text": orig_text[e["start"]:e["end"]], "label": e["label"], "start": e["start"], "end": e["end"]} for e in entities],
                "paraphrase_text": best_cand,
                "paraphrase_entities": [{"text": best_cand[e["start"]:e["end"]], "label": e["label"], "start": e["start"], "end": e["end"]} for e in best_ents],
                "all_valid_count": len(valid_paras)
            })

    print(f"Generated {len(results)} valid paraphrased records out of {len(selected_records)} selected.")
    
    with open("scratch/sample_paraphrases.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved sample to scratch/sample_paraphrases.json")

if __name__ == "__main__":
    run_sample_generation()
