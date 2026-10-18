"""Real-time PayeeShield scorer: model probability + transparent reason codes."""
from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path

import joblib
import pandas as pd

try:
    from .features import BLOCK_AT, FEATURES, PATHS, WARN_AT
except ImportError:  # run from inside src/
    from features import BLOCK_AT, FEATURES, PATHS, WARN_AT

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {"payee_sim_reachable", "first_time_payee", "payer_on_call"}


def validate(txn: dict) -> dict:
    """Return a clean {feature: float} dict or raise ValueError."""
    if not isinstance(txn, dict):
        raise ValueError("transaction must be a JSON object / dict")
    missing = [f for f in FEATURES if f not in txn]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    clean = {}
    for f in FEATURES:
        v = txn[f]
        if isinstance(v, (str, bytes)) or v is None:
            raise ValueError(f"{f} must be numeric, got {v!r}")
        try:
            x = float(v)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{f} must be numeric, got {v!r}") from None
        if not math.isfinite(x):
            raise ValueError(f"{f} must be a finite number, got {v!r}")
        if x < 0:
            raise ValueError(f"{f} must be non-negative, got {v!r}")
        if f == "hour" and (x > 23 or x != int(x)):
            raise ValueError(f"hour must be an integer 0-23, got {v!r}")
        if f in FLAGS and x not in (0.0, 1.0):
            raise ValueError(f"{f} must be 0 or 1, got {v!r}")
        clean[f] = x
    return clean


def _plural(n: float, word: str) -> str:
    n = int(round(n))
    return f"{n} {word}{'' if n == 1 else 's'}"


def reason_codes(t: dict) -> list[str]:
    """Transparent rules, most severe first. `t` is a validated feature dict."""
    young_vpa = t["payee_vpa_age_days"] < 30
    unreachable = t["payee_sim_reachable"] == 0
    first = t["first_time_payee"] == 1
    rules = [  # (severity, fires, text)
        (100, unreachable, "Payee's mobile number is unreachable on the telecom network"),
        (95, t["payer_on_call"] == 1 and first,
         "You are on a call while paying a new payee — common in 'digital arrest' scams"),
        (90, t["payee_unique_senders_24h"] >= 15 and (young_vpa or unreachable),
         f"{int(t['payee_unique_senders_24h'])} different people paid this payee in 24h"),
        (85, t["payee_outflow_minutes"] < 10,
         f"Money leaves payee account within {_plural(t['payee_outflow_minutes'], 'minute')}"),
        (80, young_vpa, f"Payee UPI ID created {_plural(t['payee_vpa_age_days'], 'day')} ago"),
        (75, t["payee_sim_changes_90d"] >= 2,
         f"Payee SIM changed {int(t['payee_sim_changes_90d'])} times in 90 days"),
        (70, t["payee_sim_age_days"] < 15,
         f"Payee SIM only {_plural(t['payee_sim_age_days'], 'day')} old"),
        (60, t["amount"] >= 10000 and first, f"Large amount (₹{t['amount']:,.0f}) to a new payee"),
        (10, first, "First payment to this payee"),
    ]
    return [text for _, fires, text in sorted(rules, key=lambda r: -r[0]) if fires]


def decide(score: int) -> str:
    return "BLOCK" if score >= BLOCK_AT else "WARN" if score >= WARN_AT else "ALLOW"


class PayeeShieldScorer:
    def __init__(self, model_path: str | os.PathLike | None = None):
        path = model_path or os.environ.get("PAYEESHIELD_MODEL") or ROOT / PATHS["model"]
        self.model_path = Path(path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found at {self.model_path} (run src/train.py first)")
        self.model = joblib.load(self.model_path)
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:8]
        self.model_version = f"{self.model_path.stem}-{digest}"
        self._predict(pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES))  # warm-up

    def _predict(self, df: pd.DataFrame):
        return self.model.predict_proba(df)[:, 1]

    def _result(self, txn: dict, t: dict, prob: float, t0: float) -> dict:
        risk = max(0, min(100, round(float(prob) * 100)))
        decision = decide(risk)
        # On ALLOW, don't list warning reasons the model has already weighed and
        # dismissed (e.g. a busy legit merchant) — it would contradict the decision.
        reasons = [] if decision == "ALLOW" else reason_codes(t)[:4]
        if not reasons:
            reasons = ["No risk signals found"] if decision == "ALLOW" else \
                ["Unusual payee behaviour pattern detected by the model"]
        return {
            "txn_id": txn.get("txn_id"),
            "risk_score": risk,
            "decision": decision,
            "reasons": reasons,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
            "model_version": self.model_version,
        }

    def score(self, txn: dict) -> dict:
        t0 = time.perf_counter()
        t = validate(txn)
        prob = self._predict(pd.DataFrame([t], columns=FEATURES))[0]
        return self._result(txn, t, prob, t0)

    def score_batch(self, txns: list[dict]) -> list[dict]:
        """Vectorised: one model call for the whole batch; latency_ms is the batch total."""
        if not txns:
            return []
        t0 = time.perf_counter()
        clean = []
        for i, txn in enumerate(txns):
            try:
                clean.append(validate(txn))
            except ValueError as e:
                raise ValueError(f"item {i}: {e}") from None
        probs = self._predict(pd.DataFrame(clean, columns=FEATURES))
        return [self._result(txn, t, p, t0) for txn, t, p in zip(txns, clean, probs)]
