"""Synthetic UPI transaction generator for PayeeShield.

Each row is one payment attempt, described by the payment itself (amount, hour,
payer context) and by the RECEIVING account (SIM, VPA, inflow/outflow behaviour).
Segments overlap on purpose so no single feature separates fraud from legit.

Run: python3 src/data_gen.py   -> writes PATHS["data"]
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURES, LABEL, PATHS  # noqa: E402

FRAUD_RATE = 0.03
# Share of each fraud scenario among fraud rows
FRAUD_MIX = {"dead_sim_mule": 0.35, "digital_arrest": 0.15,
             "fake_refund_qr": 0.30, "sim_swap_mule": 0.20}
# Share of each legit segment among legit rows
LEGIT_MIX = {"legit_p2p": 0.78, "legit_merchant": 0.15, "legit_new_shop": 0.07}


def _hours(rng, n, evening_bias):
    """Daytime-heavy hour distribution; evening_bias shifts mass to 19-02h."""
    base = np.array([1, .6, .4, .3, .3, .5, 1.5, 3, 4, 5, 5, 5,
                     5, 5, 4.5, 4.5, 4.5, 5, 5.5, 5.5, 5, 4, 3, 2], float)
    late = np.zeros(24); late[[19, 20, 21, 22, 23, 0, 1, 2]] = 1
    p = base / base.sum() * (1 - evening_bias) + late / late.sum() * evening_bias
    return rng.choice(24, size=n, p=p)


def _lognorm(rng, n, median, sigma, lo=1, hi=500_000):
    return np.clip(rng.lognormal(np.log(median), sigma, n), lo, hi).round(2)


def _bern(rng, n, p):
    return (rng.random(n) < p).astype(int)


def _segment(rng, n, name):
    """Return a dict of feature arrays for n rows of one segment."""
    d = {}
    if name.startswith("legit"):
        merchant = name == "legit_merchant"
        new_shop = name == "legit_new_shop"
        d["amount"] = _lognorm(rng, n, 350 if merchant else 600, 1.2)
        d["hour"] = _hours(rng, n, 0.15)
        d["payee_sim_reachable"] = _bern(rng, n, 0.98)
        # Mostly old SIMs; ~8% recently bought phones/SIMs
        old_sim = rng.gamma(2.0, 600, n) + 30
        d["payee_sim_age_days"] = np.where(rng.random(n) < 0.08,
                                           rng.integers(3, 90, n), old_sim)
        d["payee_sim_changes_90d"] = rng.choice([0, 1, 2], n, p=[.93, .06, .01])
        vpa_old = rng.gamma(2.0, 500, n) + 10
        vpa_young = rng.integers(1, 60, n)  # new UPI users exist too
        d["payee_vpa_age_days"] = np.where(
            rng.random(n) < (0.6 if new_shop else 0.10), vpa_young, vpa_old)
        if merchant or new_shop:
            # Shops: many payers, frequent sweeps to bank/supplier
            d["payee_unique_senders_24h"] = rng.negative_binomial(2, 2 / (2 + (40 if merchant else 12)), n)
            d["payee_outflow_minutes"] = _lognorm(rng, n, 90, 1.3, 1, 10_000)
            d["first_time_payee"] = _bern(rng, n, 0.55 if new_shop else 0.35)
        else:
            d["payee_unique_senders_24h"] = rng.poisson(1.2, n)
            d["payee_outflow_minutes"] = _lognorm(rng, n, 1500, 1.2, 1, 20_000)
            d["first_time_payee"] = _bern(rng, n, 0.12)
        d["payer_on_call"] = _bern(rng, n, 0.05)
        return d

    # ---- fraud scenarios: each has a core pattern plus partial normality ----
    d["hour"] = _hours(rng, n, 0.30)
    d["payer_on_call"] = _bern(rng, n, 0.10)
    d["first_time_payee"] = _bern(rng, n, 0.80)
    d["payee_sim_reachable"] = _bern(rng, n, 0.90)
    d["payee_sim_age_days"] = rng.gamma(1.5, 300, n) + 5
    d["payee_sim_changes_90d"] = rng.choice([0, 1, 2], n, p=[.75, .2, .05])
    d["payee_vpa_age_days"] = rng.gamma(1.2, 60, n) + 1
    d["payee_unique_senders_24h"] = rng.negative_binomial(2, 2 / (2 + 6), n)
    d["payee_outflow_minutes"] = _lognorm(rng, n, 120, 1.4, 1, 10_000)
    d["amount"] = _lognorm(rng, n, 1500, 1.1)

    if name == "dead_sim_mule":
        d["payee_sim_reachable"] = _bern(rng, n, 0.25)       # mostly dead SIMs
        d["payee_unique_senders_24h"] = rng.negative_binomial(2, 2 / (2 + 14), n)
        d["payee_outflow_minutes"] = _lognorm(rng, n, 15, 1.3, 1, 5_000)
        d["payee_vpa_age_days"] = rng.gamma(1.2, 40, n) + 1
    elif name == "digital_arrest":
        d["payer_on_call"] = _bern(rng, n, 0.85)              # victim kept on video call
        d["amount"] = _lognorm(rng, n, 80_000, 0.8, 20_000, 500_000)
        d["first_time_payee"] = _bern(rng, n, 0.95)
        d["payee_vpa_age_days"] = rng.gamma(1.2, 30, n) + 1
        d["hour"] = _hours(rng, n, 0.10)                       # happens in office hours too
    elif name == "fake_refund_qr":
        d["amount"] = _lognorm(rng, n, 900, 0.9, 1, 25_000)
        d["payee_sim_reachable"] = _bern(rng, n, 0.97)        # SIM looks fine
        d["payee_vpa_age_days"] = rng.gamma(1.2, 25, n) + 1
        d["payee_unique_senders_24h"] = rng.negative_binomial(2, 2 / (2 + 4), n)
    elif name == "sim_swap_mule":
        d["payee_sim_changes_90d"] = rng.choice([1, 2, 3, 4], n, p=[.35, .35, .2, .1])
        d["payee_sim_age_days"] = rng.integers(1, 30, n)
        d["payee_vpa_age_days"] = rng.gamma(2.0, 300, n) + 30  # hijacked OLD account
        d["payee_outflow_minutes"] = _lognorm(rng, n, 30, 1.3, 1, 5_000)
        d["amount"] = _lognorm(rng, n, 4000, 1.1)

    # ~20% of fraud rows "look normal" on one random payee-side signal
    camo = rng.random(n) < 0.20
    d["payee_vpa_age_days"] = np.where(camo, rng.gamma(2.0, 500, n) + 10, d["payee_vpa_age_days"])
    camo = rng.random(n) < 0.20
    d["payee_outflow_minutes"] = np.where(camo, _lognorm(rng, n, 1200, 1.2, 1, 20_000), d["payee_outflow_minutes"])
    return d


def generate(n=50_000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * FRAUD_RATE))
    plan = {k: int(round(v * (n - n_fraud))) for k, v in LEGIT_MIX.items()}
    plan.update({k: int(round(v * n_fraud)) for k, v in FRAUD_MIX.items()})

    parts = []
    for name, cnt in plan.items():
        seg = pd.DataFrame(_segment(rng, cnt, name))
        seg["scenario"] = name
        seg[LABEL] = 0 if name.startswith("legit") else 1
        parts.append(seg)
    df = pd.concat(parts, ignore_index=True)

    # Integer-valued columns
    for c in ["hour", "payee_sim_reachable", "payee_sim_changes_90d",
              "payee_unique_senders_24h", "first_time_payee", "payer_on_call"]:
        df[c] = df[c].astype(int)
    for c in ["payee_sim_age_days", "payee_vpa_age_days"]:
        df[c] = df[c].round().astype(int)
    df["payee_outflow_minutes"] = df["payee_outflow_minutes"].round(1)

    # Label noise: ~2% of fraud mislabeled legit (unreported), ~0.15% legit flagged fraud (bad reports)
    flip = np.where(df[LABEL] == 1, rng.random(len(df)) < 0.02, rng.random(len(df)) < 0.0015)
    df.loc[flip, LABEL] = 1 - df.loc[flip, LABEL]

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "txn_id", [f"TXN{i:07d}" for i in range(len(df))])
    return df[["txn_id"] + FEATURES + [LABEL, "scenario"]]


if __name__ == "__main__":
    out = ROOT / PATHS["data"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows to {out}  fraud_rate={df[LABEL].mean():.4f}")
    print(df.groupby("scenario")[LABEL].agg(["count", "mean"]).round(3))
