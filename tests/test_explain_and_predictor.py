import datetime

import numpy as np
import pandas as pd
import pytest

from config import MODEL_DIR, RISK_HIGH_MIN, RISK_LOW_MAX

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "clf.joblib").exists(),
    reason="Trained model files not found. Run 03_train.py first.")

REQUIRED_KEYS = {"delay_probability", "expected_delay_minutes", "risk_level",
                 "top_factors", "weather_source", "estimate", "disclaimer"}


def sample_flight(**changes):
    day = (datetime.date.today() + datetime.timedelta(days=5)).isoformat()
    flight = {"airline": "DL", "origin": "ATL", "dest": "LAX", "flight_date": day,
              "sched_dep_time": 1730, "sched_arr_time": 1910,
              "sched_duration_min": 280, "distance_miles": 1947}
    flight.update(changes)
    return flight


# ---------------- explain.py ----------------
def test_feature_labels():
    from ml.explain import feature_label
    assert feature_label("o_wind_gusts_10m") == "Origin wind gusts"
    assert feature_label("d_precipitation") == "Destination precipitation"
    assert feature_label("dep_hour") == "Scheduled departure hour"


@pytest.fixture(scope="module")
def explainer_and_data():
    import joblib
    from ml.explain import _test_sample, build_explainer
    clf = joblib.load(MODEL_DIR / "clf.joblib")
    X, meta = _test_sample(n=200)
    return clf, build_explainer(clf), X, meta


def test_shap_values_shape_and_additivity(explainer_and_data):
    clf, ex, X, meta = explainer_and_data
    sv = np.asarray(ex.shap_values(X.iloc[:20]))
    assert sv.shape == (20, len(meta["features"]))
    # LightGBM's own contributions must add up to the model's raw score
    contrib = np.asarray(clf.booster_.predict(X.iloc[:50], pred_contrib=True))
    raw = np.asarray(clf.booster_.predict(X.iloc[:50], raw_score=True))
    assert np.abs(contrib.sum(axis=1) - raw).max() < 1e-4


def test_top_factors_structure(explainer_and_data):
    from ml.explain import top_factors
    _, ex, X, _ = explainer_and_data
    row = X.iloc[[0]]
    factors = top_factors(ex, row, row.iloc[0].to_dict(), n=5)
    assert len(factors) == 5
    for f in factors:
        assert set(f) == {"factor", "value", "effect", "impact"}
        assert f["effect"] in ("increases delay risk", "reduces delay risk")
    impacts = [abs(f["impact"]) for f in factors]
    assert impacts == sorted(impacts, reverse=True)


# ---------------- predictor.py ----------------
def test_predict_flight_schema():
    from ml.predictor import predict_flight
    r = predict_flight(sample_flight())
    assert REQUIRED_KEYS <= set(r)
    assert 0.0 <= r["delay_probability"] <= 1.0
    assert r["expected_delay_minutes"] >= 0.0
    assert r["risk_level"] in ("Low", "Medium", "High")
    assert len(r["top_factors"]) == 5
    assert r["estimate"] is True
    assert isinstance(r["disclaimer"], str) and r["disclaimer"]
    assert r["weather_source"] in ("forecast", "partly or fully unavailable")


def test_risk_level_matches_cutoffs():
    from ml.predictor import predict_flight, risk_level
    r = predict_flight(sample_flight())
    assert r["risk_level"] == risk_level(r["delay_probability"])
    assert risk_level(RISK_LOW_MAX - 0.001) == "Low"
    assert risk_level(RISK_LOW_MAX) == "Medium"
    assert risk_level(RISK_HIGH_MIN - 0.001) == "Medium"
    assert risk_level(RISK_HIGH_MIN) == "High"


def test_unknown_airport_is_rejected():
    from ml.predictor import predict_flight
    with pytest.raises(ValueError):
        predict_flight(sample_flight(origin="XXX"))


def test_missing_field_is_rejected():
    from ml.predictor import predict_flight
    flight = sample_flight()
    del flight["distance_miles"]
    with pytest.raises(ValueError):
        predict_flight(flight)


def test_weather_failure_still_gives_an_answer(monkeypatch):
    import ml.predictor as p

    def boom(*args, **kwargs):
        raise ConnectionError("simulated outage")

    monkeypatch.setattr(p, "_team_fetch_forecast", None)
    monkeypatch.setattr(p.requests, "get", boom)
    r = p.predict_flight(sample_flight())
    assert r["weather_source"] == "partly or fully unavailable"
    assert REQUIRED_KEYS <= set(r)
    assert 0.0 <= r["delay_probability"] <= 1.0