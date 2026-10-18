"""Train the PayeeShield model and write metrics.

Main model : HistGradientBoosting on all FEATURES (payment + payee-side signals).
Baseline   : LogisticRegression on transaction-side features only, to show the
             lift that payee-side (receiver) signals give.

Run: python3 src/train.py  -> writes PATHS["model"] and PATHS["metrics"]
"""
from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import BLOCK_AT, FEATURES, LABEL, PATHS, WARN_AT  # noqa: E402

TXN_SIDE = ["amount", "hour", "first_time_payee", "payer_on_call"]


def at_threshold(y, prob, score_cut):
    """Precision/recall/F1 when flagging risk_score = round(prob*100) >= score_cut."""
    pred = (np.round(prob * 100) >= score_cut).astype(int)
    return {"threshold_risk_score": score_cut,
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4),
            "flag_rate": round(pred.mean(), 4)}


def main():
    df = pd.read_csv(ROOT / PATHS["data"])
    X, y = df[FEATURES], df[LABEL]
    X_tr, X_te, y_tr, y_te, _, sc_te = train_test_split(
        X, y, df["scenario"], test_size=0.2, stratify=y, random_state=42)

    model = Pipeline([
        # Selects FEATURES by name (any extra/reordered columns are fine); no custom code,
        # so the pickle loads from any module without importing train.py.
        ("select", ColumnTransformer([("f", "passthrough", FEATURES)])),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=42)),
    ])
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0

    baseline = Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(class_weight="balanced", max_iter=1000))])
    Xb_tr, Xb_te = X_tr[TXN_SIDE].assign(amount=np.log1p(X_tr.amount)), X_te[TXN_SIDE].assign(amount=np.log1p(X_te.amount))
    baseline.fit(Xb_tr, y_tr)
    pb = baseline.predict_proba(Xb_te)[:, 1]

    p = model.predict_proba(X_te)[:, 1]
    warn_pred = np.round(p * 100) >= WARN_AT
    per_scenario = {s: round(float(warn_pred[(sc_te == s).values & (y_te == 1).values].mean()), 4)
                    for s in sorted(sc_te[y_te == 1].unique()) if not s.startswith("legit")}
    # False-alarm rate on legit segments at WARN
    legit_fpr = {s: round(float(warn_pred[(sc_te == s).values & (y_te == 0).values].mean()), 4)
                 for s in sorted(sc_te.unique()) if s.startswith("legit")}

    # Permutation importance (PR-AUC drop) on a 5k test sample
    samp = X_te.sample(5000, random_state=42)
    pi = permutation_importance(model, samp, y_te.loc[samp.index], scoring="average_precision",
                                n_repeats=5, random_state=42, n_jobs=-1)
    importance = sorted(({"feature": f, "importance": round(float(m), 4)}
                         for f, m in zip(FEATURES, pi.importances_mean)),
                        key=lambda r: -r["importance"])

    # Single-row latency, as the API would call it
    rows = [X_te.iloc[[i]] for i in range(500)]
    model.predict_proba(rows[0])  # warm-up
    t0 = time.perf_counter()
    for r in rows:
        model.predict_proba(r)
    latency_ms = (time.perf_counter() - t0) / len(rows) * 1000

    metrics = {
        "roc_auc": round(roc_auc_score(y_te, p), 4),
        "pr_auc": round(average_precision_score(y_te, p), 4),
        "block": at_threshold(y_te, p, BLOCK_AT),
        "warn": at_threshold(y_te, p, WARN_AT),
        "baseline_features": TXN_SIDE,
        "baseline_roc_auc": round(roc_auc_score(y_te, pb), 4),
        "baseline_pr_auc": round(average_precision_score(y_te, pb), 4),
        "recall_by_scenario_at_warn": per_scenario,
        "false_alarm_rate_by_legit_segment_at_warn": legit_fpr,
        "test_rows": int(len(y_te)),
        "fraud_rate": round(float(y.mean()), 4),
        "train_seconds": round(train_s, 2),
        "feature_importance": importance,
        "predict_latency_ms_single_row": round(latency_ms, 3),
    }
    out = ROOT / PATHS["model"]
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    (ROOT / PATHS["metrics"]).write_text(json.dumps(metrics, indent=2))

    print(f"ROC-AUC {metrics['roc_auc']}  PR-AUC {metrics['pr_auc']}  "
          f"(baseline txn-only: ROC {metrics['baseline_roc_auc']}, PR {metrics['baseline_pr_auc']})")
    for k in ("warn", "block"):
        m = metrics[k]
        print(f"{k.upper():5} >= {m['threshold_risk_score']}: P={m['precision']} R={m['recall']} "
              f"F1={m['f1']} flagged={m['flag_rate']:.2%}")
    print("recall@WARN by scenario:", per_scenario)
    print("top features:", [r["feature"] for r in importance[:5]])
    print(f"latency {latency_ms:.2f} ms/row, train {train_s:.1f}s -> {out}")


if __name__ == "__main__":
    main()
