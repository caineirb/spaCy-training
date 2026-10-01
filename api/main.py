"""
FastAPI service wrapping HybridJournalPipeline for HTTP-based entity extraction.

Startup loads the model once; each POST /extract request runs inference,
deduplicates entities by (term, category) with frequency counts, and returns
a response shaped for the PHP frontend (or any caller).
"""

import os
import sys
import re
import logging
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Path bootstrap — ensure project root is importable regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline import HybridJournalPipeline

logger = logging.getLogger("ojt_pipeline.api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# 100 000 characters ≈ 40 pages of dense text — generous enough for real
# journal entries, small enough to prevent the endpoint from hanging on
# pathological multi-MB payloads.  Transformer tokenisers also have their
# own internal limits (512 tokens for most BERT-family models), so extremely
# long text wouldn't improve results anyway.
MAX_TEXT_LENGTH = 100_000

# Whitespace normalisation regex — mirrors the lowercase + collapse-whitespace
# approach already used by the pipeline's EntityRuler (phrase_matcher_attr=LOWER)
# and the _lower_terms_set dictionary lookup.
_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase and collapse internal whitespace for dedup key generation."""
    return _WS_RE.sub(" ", text.strip().lower())


# ---------------------------------------------------------------------------
# Global pipeline reference (populated at startup)
# ---------------------------------------------------------------------------
_pipeline: Optional[HybridJournalPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the heavy pipeline once when the server starts."""
    global _pipeline
    model_path = os.environ.get("MODEL_PATH", "models/ner_trf/model-best")
    logger.info(f"Loading HybridJournalPipeline from '{model_path}' (CPU-only mode) at startup …")
    _pipeline = HybridJournalPipeline(model_path=model_path, use_gpu=False)
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down — releasing pipeline resources.")
    _pipeline = None


app = FastAPI(
    title="OJT Journal Entity Extraction API",
    description="HTTP wrapper around HybridJournalPipeline for entity extraction.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ExtractionRequest(BaseModel):
    """JSON body for POST /extract."""
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text must not be empty or whitespace-only.")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deduplicate_entities(entities: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Group raw entity records by normalised (term, category), count frequency.

    For each group the record with the earliest ``start`` offset is kept as the
    representative.  The ``start`` and ``end`` offsets in the output refer to the
    *first* occurrence of the entity in the text so callers can still highlight
    it if desired.
    """
    groups: dict[tuple[str, str], dict] = {}
    freq: dict[tuple[str, str], int] = defaultdict(int)

    for ent in entities:
        key = (_normalise(ent["term"]), ent["category"])
        freq[key] += 1
        if key not in groups:
            groups[key] = dict(ent)  # shallow copy of first occurrence

    deduped = []
    for key, record in groups.items():
        record["frequency"] = freq[key]
        deduped.append(record)

    # Sort by start offset of first occurrence for a stable, readable order
    deduped.sort(key=lambda e: e["start"])
    return deduped


def _build_summary(entities: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Build category breakdown summary matching the old PHP contract."""
    unique = len(entities)
    total_occurrences = sum(e.get("frequency", 1) for e in entities)

    category_counts: dict[str, int] = defaultdict(int)
    for e in entities:
        category_counts[e["category"]] += 1

    it_count = category_counts.get("IT_TERM", 0)
    clerical_count = category_counts.get("CLERICAL_TERM", 0)

    summary: Dict[str, Any] = {
        "total_occurrences": total_occurrences,
        "unique_entities": unique,
        "IT_TERM": it_count,
        "CLERICAL_TERM": clerical_count,
    }

    if unique > 0:
        summary["IT_TERM_percentage"] = round(it_count / unique * 100, 1)
        summary["CLERICAL_TERM_percentage"] = round(clerical_count / unique * 100, 1)
    else:
        summary["IT_TERM_percentage"] = 0.0
        summary["CLERICAL_TERM_percentage"] = 0.0

    return summary


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Readiness probe — confirms the model is loaded and ready to serve."""
    if _pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "Pipeline not loaded yet."},
        )
    return {"status": "ok", "model_path": _pipeline.model_path}


@app.post("/extract")
async def extract_entities(req: ExtractionRequest):
    """Run hybrid entity extraction on the submitted text.

    Returns deduplicated entities with frequency counts, category breakdown
    summary, and the original input text — shaped to match the PHP frontend's
    expected JSON contract.
    """
    # --- Guard: pipeline must be loaded ---
    if _pipeline is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Pipeline is not loaded. Try again shortly.",
            },
        )

    # --- Guard: text length ---
    if len(req.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "Text exceeds maximum allowed length.",
                "details": f"Received {len(req.text)} characters; limit is {MAX_TEXT_LENGTH}.",
            },
        )

    # --- Run inference ---
    try:
        raw_result = _pipeline.predict(req.text, mode="hybrid")
    except Exception as exc:
        logger.exception("Pipeline inference failed.")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Entity extraction failed.",
                "details": str(exc),
            },
        )

    # --- Deduplicate & summarise ---
    deduped = _deduplicate_entities(raw_result["entities"])
    summary = _build_summary(deduped)

    return {
        "success": True,
        "content": req.text,
        "entities": deduped,
        "entity_count": len(deduped),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Pydantic validation error handler — returns 400 matching the old fail() shape
# ---------------------------------------------------------------------------
from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Invalid request.",
            "details": str(exc),
        },
    )
