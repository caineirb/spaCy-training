# Standalone API Deployment Guide

This guide describes how to isolate and deploy the FastAPI entity extraction service (`api/`) onto a production server, VM, or separate directory without copying unnecessary training datasets, raw data, or research notebooks.

---

## 1. Minimal File Checklist

To run inference independently, you only need **4 core components**:

| Component | Repository Path | Destination Path | Purpose |
|---|---|---|---|
| **API Application** | `api/` | `my-api-service/api/` | FastAPI endpoints, request schemas, lifespan handlers. |
| **Pipeline Scripts** | `scripts/__init__.py`<br>`scripts/pipeline.py`<br>`scripts/annotation.py`<br>`scripts/labels.py` | `my-api-service/scripts/` | `HybridJournalPipeline`, conflict resolution, dictionary loaders, and label mappings. |
| **Terms Dictionary** | `data/terms.csv` | `my-api-service/data/terms.csv` | Exact-match terms dictionary used by EntityRuler. |
| **Trained Model** | `models/ner_trf/model-best/`<br>*(or chosen best model)* | `my-api-service/models/ner_trf/model-best/` | Fine-tuned transformer weights, vocabulary, tokenizer, and config. |

### Files You Can Safely Exclude:
- `data/data.jsonl`, `data/synthetic_*.jsonl`, `data/review/` (training datasets).
- `data/test/` (benchmark evaluation datasets).
- `main.ipynb`, `scratch/`, `tools/` (training pipelines, GUIs, exploratory scripts).
- `scripts/mine_negatives.py`, `scripts/cross_validation.py`, `scripts/training.py` (training-only scripts).

---

## 2. Recommended Directory Structure

Preserving this layout ensures default relative paths in `api/main.py` and `scripts/pipeline.py` work out-of-the-box:

```text
my-api-service/
├── api/
│   ├── __init__.py
│   └── main.py
├── scripts/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── annotation.py
│   └── labels.py
├── data/
│   └── terms.csv
├── models/
│   └── ner_trf/
│       └── model-best/
└── requirements.txt
```

---

## 3. Automated Copy Command

Run the following bash script from the root of `spaCy-training` to package the deployment files into a target destination:

```bash
# Set your target directory
DEST="/path/to/my-api-service"

# Create required subdirectories
mkdir -p "$DEST/api" "$DEST/scripts" "$DEST/data" "$DEST/models/ner_trf"

# 1. Copy API package
cp -r api/* "$DEST/api/"

# 2. Copy inference scripts
cp scripts/__init__.py "$DEST/scripts/"
cp scripts/pipeline.py "$DEST/scripts/"
cp scripts/annotation.py "$DEST/scripts/"
cp scripts/labels.py "$DEST/scripts/"

# 3. Copy terms dictionary
cp data/terms.csv "$DEST/data/"

# 4. Copy trained model weights
cp -r models/ner_trf/model-best "$DEST/models/ner_trf/"

echo "API service assets successfully copied to $DEST"
```

> [!NOTE]
> If deploying an alternative model checkpoint (e.g., `models/ner_trf_trstr_llm/model-best`), copy that directory instead and set the `MODEL_PATH` environment variable.

---

## 4. Environment & Dependencies

On the target machine, create a clean virtual environment and install the required dependencies:

```bash
cd /path/to/my-api-service

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install minimal runtime dependencies
pip install --upgrade pip
pip install \
    "fastapi>=0.100.0" \
    "uvicorn[standard]>=0.23.0" \
    "pydantic>=2.0.0" \
    "spacy>=3.7.0" \
    "spacy-transformers>=1.3.0" \
    "torch" \
    "pandas" \
    "transformers" \
    "sentencepiece"
```

A minimal `requirements.txt` file:
```text
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
spacy>=3.7.0
spacy-transformers>=1.3.0
torch
pandas
transformers
sentencepiece
```

---

## 5. Running the Service

### Development / Local Test:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

> [!TIP]
> **CPU-Only Mode by Design**: The inference pipeline runs in CPU-only mode (`use_gpu=False`). This eliminates GPU driver and CUDA toolkit requirements on production hosts while delivering predictable ~90–165 ms latencies for typical journal logs.

---

## 6. Verifying Deployment

### Health Check (`GET /health`)

> [!NOTE]
> On startup, loading the transformer weights and vocabulary into CPU memory takes **5–10 seconds per worker**. During this startup window, `/health` acts as a readiness probe and returns `503 Pipeline not loaded yet.` Wait until you see `Pipeline ready.` in your server console before querying `/health`.

```bash
curl http://localhost:8000/health
```
**Expected Response (once ready):**
```json
{
  "status": "ok",
  "model_path": "models/ner_trf/model-best"
}
```

### Entity Extraction Test (`POST /extract`)
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Assisted with data entry and developed microservices using FastAPI."}'
```
**Expected Response:**
```json
{
  "success": true,
  "content": "Assisted with data entry and developed microservices using FastAPI.",
  "entities": [
    {
      "term": "data entry",
      "category": "CLERICAL_TERM",
      "start": 14,
      "end": 24,
      "confidence": 1.0,
      "source": "dictionary",
      "status": "ACCEPTED",
      "frequency": 1
    },
    {
      "term": "FastAPI",
      "category": "IT_TERM",
      "start": 59,
      "end": 66,
      "confidence": 0.88,
      "source": "ML",
      "status": "ACCEPTED",
      "frequency": 1
    }
  ],
  "entity_count": 2,
  "summary": {
    "total_occurrences": 2,
    "unique_entities": 2,
    "IT_TERM": 1,
    "CLERICAL_TERM": 1,
    "IT_TERM_percentage": 50.0,
    "CLERICAL_TERM_percentage": 50.0
  }
}
```

---

## 7. Environment Variables Reference

| Variable | Default Value | Description |
|---|---|---|
| `MODEL_PATH` | `models/ner_trf/model-best` | Filesystem path to the trained spaCy transformer model. |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

To point to a different model:
```bash
export MODEL_PATH="/var/models/ner_trf_trstr_llm/model-best"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
