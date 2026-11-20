"""Contract test for Suvanikhaa's ml/weather_forecast.py (PRD 5.6, task S-1).
The network is replaced by a function that always fails, so this never calls Open-Meteo."""
from datetime import datetime

import numpy as np
import pytest

from config import WEATHER_VARS


def test_forecast_fails_safely(monkeypatch):
    wf = pytest.importorskip("ml.weather_forecast")
    import requests

    def boom(*args, **kwargs):
        raise requests.exceptions.Timeout("simulated timeout")

    monkeypatch.setattr(requests, "get", boom)

    values, ok = wf.fetch_forecast(33.64, -84.43, datetime(2026, 11, 25, 14, 0), "o_")

    assert ok is False
    assert set(values) == {"o_" + v for v in WEATHER_VARS}
    assert all(np.isnan(v) for v in values.values())