"""Three-transaction story for the PayeeShield demo.   python src/demo_story.py [--model PATH]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scorer import PayeeShieldScorer  # noqa: E402

STORIES = [
    ("The ₹1 test",
     "A ₹1 'test' payment to a scammer's number that is out of service. "
     "UPI only checked the SIM at registration, so it went through.",
     dict(txn_id="DEMO-1", amount=1, hour=22, payee_sim_reachable=0, payee_sim_age_days=6,
          payee_sim_changes_90d=2, payee_vpa_age_days=3, payee_unique_senders_24h=40,
          payee_outflow_minutes=2, first_time_payee=1, payer_on_call=0)),
    ("Digital arrest",
     "A 'police officer' on a video call orders the victim to transfer ₹1,50,000 "
     "to a 'verification account' created 5 days ago.",
     dict(txn_id="DEMO-2", amount=150000, hour=14, payee_sim_reachable=1, payee_sim_age_days=9,
          payee_sim_changes_90d=1, payee_vpa_age_days=5, payee_unique_senders_24h=6,
          payee_outflow_minutes=5, first_time_payee=1, payer_on_call=1)),
    ("Everyday payment",
     "₹250 to the neighbourhood kirana store you've paid many times; UPI ID is 3 years old.",
     dict(txn_id="DEMO-3", amount=250, hour=19, payee_sim_reachable=1, payee_sim_age_days=1400,
          payee_sim_changes_90d=0, payee_vpa_age_days=1100, payee_unique_senders_24h=55,
          payee_outflow_minutes=900, first_time_payee=0, payer_on_call=0)),
]
def inr(x: float) -> str:
    """Indian digit grouping: 150000 -> 1,50,000."""
    s = f"{int(x)}"
    head, tail = s[:-3], s[-3:]
    while len(head) > 2:
        head, tail = head[:-2], head[-2:] + "," + tail
    return f"₹{head},{tail}" if head else f"₹{tail}"


ICON = {"ALLOW": "[ OK ]", "WARN": "[WARN]", "BLOCK": "[STOP]"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model path (default models/model.joblib)")
    scorer = PayeeShieldScorer(ap.parse_args(argv).model)
    print(f"PayeeShield demo  (model {scorer.model_version})\n")
    for i, (title, story, txn) in enumerate(STORIES, 1):
        r = scorer.score(txn)
        bar = "#" * (r["risk_score"] // 5) + "." * (20 - r["risk_score"] // 5)
        print(f"{i}. {title} — {inr(txn['amount'])}")
        print(f"   {story}")
        print(f"   {ICON[r['decision']]} {r['decision']:<5}  risk {r['risk_score']:>3}/100  "
              f"[{bar}]  {r['latency_ms']:.2f} ms")
        for reason in r["reasons"]:
            print(f"     - {reason}")
        print()


if __name__ == "__main__":
    main()
