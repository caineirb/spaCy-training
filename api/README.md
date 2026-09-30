# Entity Extraction API

FastAPI service wrapping `HybridJournalPipeline` for HTTP-based entity extraction.

## Quick Start

```bash
# From the project root, with the virtual environment activated:
source .venv/bin/activate

# Install dependencies (if not already):
pip install fastapi uvicorn[standard]

# Start the server:
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The server loads the hybrid pipeline model at startup (this takes a few seconds on first launch). Once the `Pipeline ready.` log appears, the API is ready to accept requests.

## Endpoints

### `GET /health`

Readiness probe — confirms the model is loaded.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_path": "models/ner_trf/model-best"
}
```

### `POST /extract`

Runs hybrid entity extraction on the submitted text. Returns deduplicated entities with frequency counts and a category breakdown summary.

**Request:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Used Microsoft Excel for data encoding. Configured a REST API using FastAPI."}'
```

**Response shape:**
```json
{
  "success": true,
  "content": "Used Microsoft Excel for data encoding. Configured a REST API using FastAPI.",
  "entities": [
    {
      "term": "Microsoft Excel",
      "category": "CLERICAL_TERM",
      "start": 5,
      "end": 20,
      "confidence": 1.0,
      "source": "dictionary",
      "status": "ACCEPTED",
      "frequency": 1
    },
    {
      "term": "data encoding",
      "category": "CLERICAL_TERM",
      "start": 25,
      "end": 38,
      "confidence": 1.0,
      "source": "dictionary",
      "status": "ACCEPTED",
      "frequency": 1
    },
    {
      "term": "REST API",
      "category": "IT_TERM",
      "start": 55,
      "end": 63,
      "confidence": 1.0,
      "source": "dictionary",
      "status": "ACCEPTED",
      "frequency": 1
    },
    {
      "term": "FastAPI",
      "category": "IT_TERM",
      "start": 70,
      "end": 77,
      "confidence": 0.87,
      "source": "ML",
      "status": "ACCEPTED",
      "frequency": 1
    }
  ],
  "entity_count": 4,
  "summary": {
    "total_occurrences": 4,
    "unique_entities": 4,
    "IT_TERM": 2,
    "CLERICAL_TERM": 2,
    "IT_TERM_percentage": 50.0,
    "CLERICAL_TERM_percentage": 50.0
  }
}
```

### Duplicate entity deduplication

When the same entity appears multiple times in the input text, the API returns a single record with `"frequency": N` instead of N separate records:

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Used Microsoft Excel for encoding. Also used Microsoft Excel for reports."}'
```

Returns one `Microsoft Excel` entity with `"frequency": 2`.

### Error responses

**Empty text (400):**
```json
{
  "success": false,
  "error": "Invalid request.",
  "details": "..."
}
```

**Pipeline failure (500):**
```json
{
  "success": false,
  "error": "Entity extraction failed.",
  "details": "..."
}
```

## Architecture & Deployment

### CPU-Only by Design
The API service explicitly runs in **CPU-only mode** (`use_gpu=False`).
- **Rationale**: Keeps the service lightweight for web server deployment, avoids competition with GPU training/retraining tasks, and guarantees zero CUDA/GPU library dependencies on production hosts without dedicated GPUs.
- **Latency Benchmark**:
  - Typical OJT journal paragraph (~150–200 characters): **~120–190 ms** on CPU.
  - Previous GPU latency: **~40–70 ms**.
  - While CPU inference is ~2–3× slower than GPU, sub-200ms latency is well within interactive web UI tolerances for asynchronous HTTP requests.

## Configuration

| Setting | Default | Description |
|---|---|---|
| Device target | `CPU` (`use_gpu=False`) | API runs strictly on CPU by design |
| Max text length | 100,000 chars | Rejects payloads exceeding this limit with a 400 error |
| Pipeline mode | `hybrid` | EntityRuler + Transformer NER combined |
| Confidence threshold | 0.80 | ML predictions below this are flagged `NEEDS_REVIEW` |

## PHP Integration

From PHP, call the endpoint using `file_get_contents` or `curl`:

```php
$response = file_get_contents('http://localhost:8000/extract', false,
    stream_context_create([
        'http' => [
            'method'  => 'POST',
            'header'  => 'Content-Type: application/json',
            'content' => json_encode(['text' => $journalText]),
        ]
    ])
);
$result = json_decode($response, true);
if ($result['success']) {
    foreach ($result['entities'] as $entity) {
        // ...
    }
}
```
