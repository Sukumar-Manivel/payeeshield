"""Simulate a live UPI stream through PayeeShield and report latency / catch rate.

    python src/stream_sim.py [-n 2000] [--quiet] [--json out.json] [--model PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURES, LABEL, PATHS  # noqa: E402
from src.scorer import PayeeShieldScorer  # noqa: E402

COLOR = {"WARN": "\033[33m", "BLOCK": "\033[31m"}
RESET = "\033[0m"


def random_rows(n: int, seed: int = 7) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "txn_id": [f"SIM{i:06d}" for i in range(n)],
        "amount": r.lognormal(6, 1.5, n).round(2),
        "hour": r.integers(0, 24, n),
        "payee_sim_reachable": (r.random(n) > 0.05).astype(int),
        "payee_sim_age_days": r.integers(0, 3000, n),
        "payee_sim_changes_90d": r.poisson(0.2, n),
        "payee_vpa_age_days": r.integers(0, 2000, n),
        "payee_unique_senders_24h": r.poisson(3, n),
        "payee_outflow_minutes": r.exponential(600, n).round(1),
        "first_time_payee": (r.random(n) < 0.3).astype(int),
        "payer_on_call": (r.random(n) < 0.05).astype(int),
    })


def load(n: int) -> pd.DataFrame:
    # Fresh, unseen payments: same generator, different seed than training (42),
    # so the catch rate isn't inflated by rows the model was trained on.
    try:
        from src.data_gen import generate
        df = generate(n=max(n, 300), seed=2026)
    except Exception:
        path = ROOT / PATHS["data"]
        df = pd.read_csv(path) if path.exists() else random_rows(max(n, 300))
    if "txn_id" not in df:
        df["txn_id"] = [f"T{i:07d}" for i in range(len(df))]
    return df.head(n)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=2000, help="transactions to stream")
    ap.add_argument("--json", metavar="OUT", help="write summary JSON here")
    ap.add_argument("--model", help="model path (default models/model.joblib)")
    ap.add_argument("--quiet", action="store_true", help="don't print the flagged feed")
    a = ap.parse_args(argv)

    scorer = PayeeShieldScorer(a.model)
    df = load(a.n)
    has_label = LABEL in df
    rows = df[["txn_id", *FEATURES]].to_dict("records")
    labels = df[LABEL].astype(int).tolist() if has_label else [None] * len(rows)
    tty = sys.stdout.isatty()

    lat, counts = [], {"ALLOW": 0, "WARN": 0, "BLOCK": 0}
    caught = missed = false_alarms = 0
    t0 = time.perf_counter()
    for row, y in zip(rows, labels):
        row["txn_id"] = str(row["txn_id"])
        out = scorer.score(row)
        lat.append(out["latency_ms"])
        d = out["decision"]
        counts[d] += 1
        flagged = d != "ALLOW"
        if y is not None:
            caught += y == 1 and flagged
            missed += y == 1 and not flagged
            false_alarms += y == 0 and flagged
        if flagged and not a.quiet:
            tag = f"{COLOR[d]}{d:<5}{RESET}" if tty else f"{d:<5}"
            truth = "" if y is None else ("  [fraud]" if y else "  [legit]")
            print(f"{tag} {out['risk_score']:>3}  {out['txn_id']:<12} ₹{row['amount']:>12,.2f}  "
                  f"{out['latency_ms']:.2f}ms  {'; '.join(out['reasons'][:2])}{truth}")
    wall = time.perf_counter() - t0

    p50, p95, p99 = np.percentile(lat, [50, 95, 99]) if lat else (0, 0, 0)
    summary = {
        "model_version": scorer.model_version, "transactions": len(rows),
        "wall_seconds": round(wall, 3), "throughput_tps": round(len(rows) / wall, 1) if wall else 0,
        "latency_ms": {"p50": round(p50, 3), "p95": round(p95, 3), "p99": round(p99, 3)},
        "decisions": counts,
    }
    if has_label:
        frauds = caught + missed
        summary["fraud"] = {"total": frauds, "caught": caught, "missed": missed,
                            "catch_rate": round(caught / frauds, 4) if frauds else None,
                            "false_alarms": false_alarms}

    print("\n=== PayeeShield stream summary ===")
    print(f"model {summary['model_version']}  |  {len(rows)} txns in {wall:.2f}s "
          f"-> {summary['throughput_tps']} txn/s")
    print(f"latency ms  p50 {p50:.2f}  p95 {p95:.2f}  p99 {p99:.2f}")
    print("decisions   " + "  ".join(f"{k} {v}" for k, v in counts.items()))
    if has_label:
        f = summary["fraud"]
        rate = f"{f['catch_rate']:.1%}" if f["catch_rate"] is not None else "n/a"
        print(f"fraud       caught {caught}/{caught + missed} ({rate})  missed {missed}  "
              f"false alarms {false_alarms}")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=2))
        print(f"summary written to {a.json}")
    return summary


if __name__ == "__main__":
    main()
