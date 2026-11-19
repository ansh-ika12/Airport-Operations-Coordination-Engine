"""Contract tests for Ojashwee's ml/evaluate.py and ml/baselines.py (PRD 5.6).
Skipped automatically if her files are not in this repo yet."""
import numpy as np
import pytest


def test_classification_metrics():
    ev = pytest.importorskip("ml.evaluate")
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])          # perfectly separated
    m = ev.classification_metrics(y, p)
    assert {"roc_auc", "precision", "recall", "f1"} <= set(m)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_regression_metrics():
    ev = pytest.importorskip("ml.evaluate")
    y = np.array([0.0, 10.0, 20.0])
    assert ev.regression_metrics(y, y)["mae"] == pytest.approx(0.0)
    m = ev.regression_metrics(y, y + 5)
    assert m["mae"] == pytest.approx(5.0)
    assert m["rmse"] == pytest.approx(5.0)


def test_naive_mean_prediction():
    bl = pytest.importorskip("ml.baselines")
    out = bl.naive_mean_prediction(np.array([0.0, 10.0, 20.0]), 4)
    assert len(out) == 4
    assert np.allclose(out, 10.0)


def test_logistic_baseline_fits_and_predicts(toy_df):
    bl = pytest.importorskip("ml.baselines")
    num, cat = ["month", "dep_hour", "distance"], ["airline", "origin"]
    pipe = bl.build_logistic_baseline(num, cat)
    pipe.fit(toy_df[num + cat], toy_df["delayed"])
    proba = pipe.predict_proba(toy_df[num + cat])[:, 1]
    assert len(proba) == len(toy_df)
    assert proba.min() >= 0 and proba.max() <= 1