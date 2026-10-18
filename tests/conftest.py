import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.demo_story import STORIES  # noqa: E402

DEAD_SIM = dict(STORIES[0][2])
DIGITAL_ARREST = dict(STORIES[1][2])
NORMAL = dict(txn_id="T-NORMAL", amount=250, hour=19, payee_sim_reachable=1, payee_sim_age_days=1400,
              payee_sim_changes_90d=0, payee_vpa_age_days=1095, payee_unique_senders_24h=2,
              payee_outflow_minutes=900, first_time_payee=0, payer_on_call=0)
MERCHANT = dict(txn_id="T-MERCH", amount=300, hour=12, payee_sim_reachable=1, payee_sim_age_days=2000,
                payee_sim_changes_90d=0, payee_vpa_age_days=1500, payee_unique_senders_24h=200,
                payee_outflow_minutes=5, first_time_payee=0, payer_on_call=0)


@pytest.fixture(scope="session")
def scorer():
    from src.scorer import PayeeShieldScorer
    return PayeeShieldScorer()
