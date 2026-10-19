import logging

import holidays
import joblib
import numpy as np
import pandas as pd
import requests

from config import *

log = logging.getLogger("predictor")

# ---------- Use teammates' modules when they exist; otherwise use built-in fallbacks ----------
try:
    from ml.weather_forecast import fetch_forecast as _team_fetch_forecast
except ImportError:
    _team_fetch_forecast = None
try:
    from ml.explain import build_explainer as _team_build_explainer
    from ml.explain import top_factors as _team_top_factors
except ImportError:
    _team_build_explainer = _team_top_factors = None

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DISCLAIMER = ("Estimate for decision support only. Based on historical patterns and forecast "
              "weather; a person makes the final decision.")
REQUIRED = ["airline", "origin", "dest", "flight_date", "sched_dep_time",
            "sched_arr_time", "sched_duration_min", "distance_miles"]

LABELS = {
    "airline_delay_rate": "Airline's historical delay rate",
    "origin_delay_rate": "Origin airport's historical delay rate",
    "dest_delay_rate": "Destination airport's historical delay rate",
    "route_delay_rate": "Route's historical delay rate",
    "origin_dep_count": "Departures scheduled at origin that hour",
    "dep_hour": "Scheduled departure hour", "day_of_week": "Day of week",
    "month": "Month", "is_holiday": "Holiday", "distance": "Distance",
    "sched_duration": "Scheduled duration", "airline": "Airline",
    "origin": "Origin airport", "dest": "Destination airport",
}
WX_LABELS = {"temperature_2m": "temperature", "precipitation": "precipitation",
             "cloud_cover": "cloud cover", "wind_speed_10m": "wind speed",
             "wind_gusts_10m": "wind gusts", "weather_code": "weather condition"}

_STATE = {}


def _load():
    """Load the saved model files once."""
    if not _STATE:
        _STATE["clf"] = joblib.load(MODEL_DIR / "clf.joblib")
        _STATE["reg"] = joblib.load(MODEL_DIR / "reg.joblib")
        _STATE["cal"] = joblib.load(MODEL_DIR / "calibrator.joblib")
        _STATE["meta"] = joblib.load(MODEL_DIR / "meta.joblib")
    return _STATE


def _label(feature):
    if feature in LABELS:
        return LABELS[feature]
    side = "Origin" if feature.startswith("o_") else "Destination"
    return f"{side} {WX_LABELS.get(feature[2:], feature[2:])}"


def _fallback_fetch_forecast(lat, lon, when, prefix):
    empty = {f"{prefix}{v}": np.nan for v in WEATHER_VARS}
    try:
        day = when.date().isoformat()
        r = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
            "start_date": day, "end_date": day, "timezone": "auto"}, timeout=10)
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        row = h[h["time"] == when]
        if row.empty:
            return empty, False
        return {f"{prefix}{v}": float(row.iloc[0][v]) for v in WEATHER_VARS}, True
    except Exception:
        return empty, False


def _forecast(lat, lon, when, prefix):
    if _team_fetch_forecast is not None:
        try:
            return _team_fetch_forecast(lat, lon, when, prefix)
        except Exception as e:
            log.warning("Teammate forecast client failed, using fallback: %s", e)
    return _fallback_fetch_forecast(lat, lon, when, prefix)


def risk_level(p):
    if p < RISK_LOW_MAX:
        return "Low"
    return "High" if p >= RISK_HIGH_MIN else "Medium"


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    return v


def _factors(clf, X, values, n=5):
    if _team_top_factors is not None and _team_build_explainer is not None:
        try:
            if "explainer" not in _STATE:
                _STATE["explainer"] = _team_build_explainer(clf)
            return _team_top_factors(_STATE["explainer"], X, values, n)
        except Exception as e:
            log.warning("Teammate explainer failed, using built-in fallback: %s", e)
    # Fallback: LightGBM's own per-feature contributions (log-odds of the raw score; last column is the bias)
    contrib = clf.booster_.predict(X, pred_contrib=True)[0][:-1]
    names = list(X.columns)
    out = []
    for i in np.argsort(-np.abs(contrib))[:n]:
        out.append({"factor": _label(names[i]), "value": values.get(names[i]),
                    "effect": "increases delay risk" if contrib[i] > 0 else "reduces delay risk",
                    "impact": round(float(contrib[i]), 3)})
    return out


def predict_flight(flight: dict) -> dict:
    """Estimate delay probability, expected minutes, risk level and top factors for one flight."""
    missing = [k for k in REQUIRED if k not in flight]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    s = _load()
    clf, reg, cal, meta = s["clf"], s["reg"], s["cal"], s["meta"]

    origin, dest = flight["origin"], flight["dest"]
    coords = meta["airport_coords"]
    if origin not in coords or dest not in coords:
        raise ValueError(f"Origin and destination must be one of: {sorted(coords)}")

    date = pd.Timestamp(flight["flight_date"]).normalize()
    dep_hour = (int(flight["sched_dep_time"]) // 100) % 24
    arr_hour = (int(flight["sched_arr_time"]) // 100) % 24
    overnight = 24 if int(flight["sched_arr_time"]) < int(flight["sched_dep_time"]) else 0
    dep_dt = date + pd.Timedelta(hours=dep_hour)
    arr_dt = date + pd.Timedelta(hours=arr_hour + overnight)

    wx_o, ok_o = _forecast(*coords[origin], dep_dt, "o_")
    wx_d, ok_d = _forecast(*coords[dest], arr_dt, "d_")

    h = meta["history"]
    g = h["global"]
    cong = flight.get("origin_dep_count")
    if cong is None:
        cong = meta["congestion_median"].get(f"{origin}|{dep_hour}", np.nan)

    row = {
        "airline": flight["airline"], "origin": origin, "dest": dest,
        "month": int(date.month), "day_of_week": int(date.dayofweek) + 1,   # BTS: 1 = Monday
        "dep_hour": dep_hour,
        "is_holiday": int(date.date() in holidays.US(years=[date.year])),
        "distance": float(flight["distance_miles"]),
        "sched_duration": float(flight["sched_duration_min"]),
        "origin_dep_count": cong,
        "airline_delay_rate": h["airline"].get(flight["airline"], g),
        "origin_delay_rate": h["origin"].get(origin, g),
        "dest_delay_rate": h["dest"].get(dest, g),
        "route_delay_rate": h["route"].get(f"{origin}_{dest}", g),
        **wx_o, **wx_d,
    }
    X = pd.DataFrame([row])[meta["features"]]
    for c, categories in meta["cats"].items():
        X[c] = pd.Categorical(X[c], categories=categories)

    raw = float(clf.predict_proba(X)[0, 1])
    prob = float(np.clip(cal.predict([raw])[0], 0.0, 1.0))       # calibrated probability
    minutes = max(0.0, float(reg.predict(X)[0]))
    values = {k: _clean(v) for k, v in row.items()}

    return {
        "delay_probability": round(prob, 3),
        "expected_delay_minutes": round(minutes, 1),
        "risk_level": risk_level(prob),
        "top_factors": _factors(clf, X, values),
        "weather_source": "forecast" if (ok_o and ok_d) else "partly or fully unavailable",
        "estimate": True,
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    import datetime
    import json

    day = (datetime.date.today() + datetime.timedelta(days=3)).isoformat()
    example = {"airline": "DL", "origin": "ATL", "dest": "LAX", "flight_date": day,
               "sched_dep_time": 1730, "sched_arr_time": 1910,
               "sched_duration_min": 280, "distance_miles": 1947}
    print(json.dumps(predict_flight(example), indent=2))