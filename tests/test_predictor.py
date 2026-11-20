"""Schema test for Anshika's ml/predictor.py predict_flight (PRD 5.9).
Skipped until the trained models exist in models/ (they come via Google Drive)."""
import pytest

from config import MODEL_DIR

FLIGHT = {
    "airline": "DL", "origin": "ATL", "dest": "ORD",
    "flight_date": "2026-11-25",
    "sched_dep_time": 1430, "sched_arr_time": 1530,   # local HHMM
    "sched_duration_min": 120, "distance_miles": 606,
}
REQUIRED = {"delay_probability", "expected_delay_minutes", "risk_level",
            "top_factors", "weather_source", "estimate", "disclaimer"}


@pytest.fixture
def predictor():
    if not (MODEL_DIR / "meta.joblib").exists():
        pytest.skip("models/ not found: get the trained artifacts from Anshika")
    return pytest.importorskip("ml.predictor")


def check_schema(out):
    assert REQUIRED <= set(out), f"missing fields: {REQUIRED - set(out)}"
    assert 0 <= out["delay_probability"] <= 1
    assert out["expected_delay_minutes"] >= 0
    assert out["risk_level"] in ("Low", "Medium", "High")
    assert out["weather_source"] in ("forecast", "partly or fully unavailable")
    assert out["estimate"] is True
    assert isinstance(out["disclaimer"], str) and out["disclaimer"].strip()
    assert len(out["top_factors"]) == 5
    for f in out["top_factors"]:
        assert {"factor", "value", "effect", "impact"} <= set(f)
        assert f["effect"] in ("increases delay risk", "reduces delay risk")


def test_predict_flight_schema(predictor):
    check_schema(predictor.predict_flight(dict(FLIGHT)))


def test_predict_flight_survives_weather_failure(predictor, monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("simulated outage")

    monkeypatch.setattr(requests, "get", boom)
    out = predictor.predict_flight(dict(FLIGHT))
    check_schema(out)
    assert out["weather_source"] == "partly or fully unavailable"