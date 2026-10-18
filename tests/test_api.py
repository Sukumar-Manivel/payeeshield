import json

import pytest
from fastapi.testclient import TestClient

from conftest import DEAD_SIM, DIGITAL_ARREST, NORMAL
from src.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs lifespan (loads model)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["model_loaded"] is True and j["model_version"]


def test_score_dead_sim(client):
    r = client.post("/score", json=DEAD_SIM)
    assert r.status_code == 200
    j = r.json()
    assert j["decision"] == "BLOCK" and j["txn_id"] == "DEMO-1"
    assert isinstance(j["risk_score"], int) and 1 <= len(j["reasons"]) <= 4
    assert isinstance(j["latency_ms"], float)


def test_score_without_txn_id(client):
    t = dict(NORMAL); del t["txn_id"]
    r = client.post("/score", json=t)
    assert r.status_code == 200 and r.json()["txn_id"] is None


@pytest.mark.parametrize("field,value", [
    ("amount", "abc"), ("amount", -5), ("hour", 24), ("first_time_payee", 2),
    ("payee_sim_reachable", 3), ("payee_unique_senders_24h", -1)])
def test_score_422_bad_values(client, field, value):
    r = client.post("/score", json=dict(NORMAL, **{field: value}))
    assert r.status_code == 422


def test_score_422_missing_field(client):
    t = dict(NORMAL); del t["payer_on_call"]
    assert client.post("/score", json=t).status_code == 422


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_score_422_nan_inf(client, literal):
    body = json.dumps(dict(NORMAL, amount=0)).replace('"amount": 0', f'"amount": {literal}')
    r = client.post("/score", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text


def test_batch(client):
    r = client.post("/score/batch", json=[DEAD_SIM, DIGITAL_ARREST, NORMAL])
    assert r.status_code == 200
    out = r.json()
    assert [o["decision"] for o in out] == ["BLOCK", "BLOCK", "ALLOW"]
    assert client.post("/score/batch", json=[]).json() == []


def test_batch_422(client):
    assert client.post("/score/batch", json=[NORMAL, dict(NORMAL, hour=99)]).status_code == 422
    assert client.post("/score/batch", json=NORMAL).status_code == 422
