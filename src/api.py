"""PayeeShield REST API.  Run from project root:  uvicorn src.api:app --port 8000"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import math

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scorer import PayeeShieldScorer  # noqa: E402

state: dict = {"scorer": None, "error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state["scorer"] = PayeeShieldScorer()
    except Exception as e:  # keep serving /health so the failure is visible
        state["error"] = str(e)
    yield


app = FastAPI(title="PayeeShield", version="0.1.0", lifespan=lifespan,
              description="Real-time mule-payee risk scoring for UPI payments")


def _json_safe(o):
    """NaN/Infinity are valid Python-JSON input but not valid JSON output; stringify them."""
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    # Default handler crashes (500) when the offending input is NaN, e.g. {"amount": NaN}.
    return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))})


class Txn(BaseModel):
    txn_id: Optional[str] = None
    amount: float = Field(..., ge=0, description="INR")
    hour: int = Field(..., ge=0, le=23)
    payee_sim_reachable: Literal[0, 1]
    payee_sim_age_days: float = Field(..., ge=0)
    payee_sim_changes_90d: int = Field(..., ge=0)
    payee_vpa_age_days: float = Field(..., ge=0)
    payee_unique_senders_24h: int = Field(..., ge=0)
    payee_outflow_minutes: float = Field(..., ge=0)
    first_time_payee: Literal[0, 1]
    payer_on_call: Literal[0, 1]


class ScoreOut(BaseModel):
    txn_id: Optional[str]
    risk_score: int
    decision: Literal["ALLOW", "WARN", "BLOCK"]
    reasons: list[str]
    latency_ms: float
    model_version: str


def _scorer() -> PayeeShieldScorer:
    if state["scorer"] is None:
        raise HTTPException(503, f"model not loaded: {state['error']}")
    return state["scorer"]


def _score(s: PayeeShieldScorer, t: Txn) -> dict:
    try:
        return s.score(t.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/health")
def health():
    s = state["scorer"]
    return {"status": "ok" if s else "degraded", "model_loaded": s is not None,
            "model_version": s.model_version if s else None, "error": state["error"]}


@app.post("/score", response_model=ScoreOut)
def score(txn: Txn):
    return _score(_scorer(), txn)


@app.post("/score/batch", response_model=list[ScoreOut])
def score_batch(txns: list[Txn]):
    try:
        return _scorer().score_batch([t.model_dump() for t in txns])
    except ValueError as e:
        raise HTTPException(422, str(e))
