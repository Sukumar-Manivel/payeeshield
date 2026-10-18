import json

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src.data_gen import generate
from src.features import FEATURES, LABEL, PATHS
from src.scorer import validate


@pytest.fixture(scope="module")
def small():
    return generate(n=4000, seed=123)


def test_deterministic(small):
    pd.testing.assert_frame_equal(small, generate(n=4000, seed=123))
    assert not generate(n=4000, seed=124).equals(small)


def test_columns(small):
    assert set(FEATURES + [LABEL]) <= set(small.columns)
    assert list(small.columns) == ["txn_id"] + FEATURES + [LABEL, "scenario"]
    assert small[FEATURES].notna().all().all()


def test_fraud_rate(small):
    assert 0.02 <= small[LABEL].mean() <= 0.05


def test_rows_pass_scorer_validation(small):
    for row in small[FEATURES].head(500).to_dict("records"):
        validate(row)


def test_metrics():
    p = ROOT / PATHS["metrics"]
    assert p.exists()
    m = json.loads(p.read_text())
    assert m["roc_auc"] > m["baseline_roc_auc"]
    assert m["pr_auc"] > m["baseline_pr_auc"]
    assert 0.5 < m["roc_auc"] <= 1


def test_committed_data_matches_generator():
    p = ROOT / PATHS["data"]
    if not p.exists():
        pytest.skip("no data file")
    head = pd.read_csv(p, nrows=200)
    pd.testing.assert_frame_equal(head, generate().head(200), check_dtype=False)
