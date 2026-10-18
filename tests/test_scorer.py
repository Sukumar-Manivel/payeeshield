import math

import pytest

from conftest import DEAD_SIM, DIGITAL_ARREST, MERCHANT, NORMAL
from src.features import BLOCK_AT, FEATURES, WARN_AT
from src.scorer import decide, reason_codes, validate


def test_score_shape_and_types(scorer):
    r = scorer.score(NORMAL)
    assert set(r) == {"txn_id", "risk_score", "decision", "reasons", "latency_ms", "model_version"}
    assert r["txn_id"] == "T-NORMAL"
    assert isinstance(r["risk_score"], int) and not isinstance(r["risk_score"], bool)
    assert 0 <= r["risk_score"] <= 100
    assert r["decision"] in {"ALLOW", "WARN", "BLOCK"}
    assert isinstance(r["reasons"], list) and all(isinstance(x, str) for x in r["reasons"])
    assert isinstance(r["latency_ms"], float) and r["latency_ms"] >= 0
    assert isinstance(r["model_version"], str) and r["model_version"]
    assert r["decision"] == decide(r["risk_score"])


@pytest.mark.parametrize("score,expected", [
    (0, "ALLOW"), (39, "ALLOW"), (40, "WARN"), (74, "WARN"), (75, "BLOCK"), (100, "BLOCK")])
def test_decide_thresholds(score, expected):
    assert WARN_AT == 40 and BLOCK_AT == 75
    assert decide(score) == expected


@pytest.mark.parametrize("field,value", [
    ("amount", "250"), ("amount", None), ("amount", float("nan")), ("amount", float("inf")),
    ("amount", -1), ("amount", 10 ** 400), ("hour", 24), ("hour", -1), ("hour", 3.5),
    ("first_time_payee", 2), ("payer_on_call", 0.5), ("payee_sim_reachable", -1),
])
def test_validate_rejects_bad_values(field, value):
    with pytest.raises(ValueError):
        validate(dict(NORMAL, **{field: value}))


def test_validate_missing_field():
    t = dict(NORMAL); del t["amount"]
    with pytest.raises(ValueError, match="missing.*amount"):
        validate(t)


def test_validate_non_dict():
    with pytest.raises(ValueError):
        validate([1, 2, 3])


def test_validate_returns_all_features_as_floats():
    v = validate(NORMAL)
    assert list(v) == FEATURES and all(isinstance(x, float) and math.isfinite(x) for x in v.values())


def test_score_raises_valueerror(scorer):
    with pytest.raises(ValueError):
        scorer.score(dict(NORMAL, hour=24))


def test_reasons_capped_at_four(scorer):
    # every rule fires here
    t = dict(DEAD_SIM, amount=50000, payer_on_call=1)
    assert len(reason_codes(validate(t))) > 4
    assert len(scorer.score(t)["reasons"]) == 4


def test_clean_txn_says_no_risk_signals(scorer):
    r = scorer.score(NORMAL)
    assert r["decision"] == "ALLOW"
    assert r["reasons"] == ["No risk signals found"]


def test_dead_sim_rupee_one_blocked(scorer):
    r = scorer.score(DEAD_SIM)
    assert r["decision"] == "BLOCK" and r["risk_score"] >= BLOCK_AT
    assert any("unreachable" in x for x in r["reasons"])


def test_normal_repeat_payment_allowed(scorer):
    r = scorer.score(NORMAL)
    assert r["decision"] == "ALLOW" and r["risk_score"] < WARN_AT


def test_digital_arrest_blocked(scorer):
    r = scorer.score(DIGITAL_ARREST)
    assert DIGITAL_ARREST["amount"] == 150000
    assert r["decision"] == "BLOCK"
    assert any("digital arrest" in x.lower() for x in r["reasons"])


def test_legit_merchant_not_flagged_for_many_senders(scorer):
    r = scorer.score(MERCHANT)
    assert not any("different people paid" in x for x in r["reasons"])
    assert r["decision"] != "BLOCK"


def test_batch_matches_single(scorer):
    txns = [DEAD_SIM, DIGITAL_ARREST, NORMAL]
    batch = scorer.score_batch(txns)
    assert [b["txn_id"] for b in batch] == [t["txn_id"] for t in txns]
    for t, b in zip(txns, batch):
        s = scorer.score(t)
        assert (s["risk_score"], s["decision"], s["reasons"]) == (b["risk_score"], b["decision"], b["reasons"])
    assert scorer.score_batch([]) == []


def test_batch_error_names_item(scorer):
    with pytest.raises(ValueError, match="item 1"):
        scorer.score_batch([NORMAL, dict(NORMAL, hour=24)])
