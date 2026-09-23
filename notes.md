> [!Note]
> Populate `data/test` with actual real data from the student documentation.
> also `data/raw`. Thanks

```python
# ====================================================
# 11. RETURN JSON TO PHP
# ====================================================

result = {
    "success":
        True,

    "content":
        text,

    # Raw spaCy entities are returned only
    # for debugging / verification.
    "spacy_entities":
        spacy_entities,

    # These are the ONLY entities that
    # should be displayed by the PHP page.
    "entities":
        matched_entities,

    "predefined_matches":
        matched_entities,

    "entity_count":
        len(matched_entities),

    "summary":
        summary

}

print(
    json.dumps(
        result,
        ensure_ascii=False
    )
)
```
