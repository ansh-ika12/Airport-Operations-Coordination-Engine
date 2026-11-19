"""TEST-ONLY generator of FAKE flights in the PRD 5.4 schema. Never use for results.

Run from flight-delay-ml/:
    python tests/toy_data.py
    python tests/toy_data.py --leaky
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AIRPORTS = ["ATL", "ORD", "DFW", "DEN", "JFK", "LAX", "SFO", "SEA", "MIA", "BOS"]
AIRLINES = ["AA", "DL", "UA", "WN", "B6", "AS"]


def make_toy(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    origin = rng.choice(AIRPORTS, n)
    dest = rng.choice(AIRPORTS, n)
    dest = np.where(dest == origin, "ATL", dest)
    origin = np.where(origin == dest, "ORD", origin)

    df = pd.DataFrame({
        "flight_date": dates,
        "tail_number": ["N" + str(x) for x in rng.integers(100, 999, n)],
        "route": [o + "_" + d for o, d in zip(origin, dest)],
        "airline": rng.choice(AIRLINES, n),
        "origin": origin,
        "dest": dest,
        "month": dates.month.astype("int64"),
        "day_of_week": dates.dayofweek.astype("int64") + 1,
        "dep_hour": rng.integers(5, 23, n),
        "is_holiday": 0,
        "distance": rng.uniform(200, 2500, n),
        "sched_duration": rng.uniform(60, 360, n),
        "origin_dep_count": rng.integers(5, 60, n),
    })
    for prefix in ("o_", "d_"):
        df[prefix + "temperature_2m"] = rng.normal(15, 10, n)
        df[prefix + "precipitation"] = rng.exponential(0.3, n)
        df[prefix + "cloud_cover"] = rng.uniform(0, 100, n)
        df[prefix + "wind_speed_10m"] = rng.uniform(0, 40, n)
        df[prefix + "wind_gusts_10m"] = rng.uniform(0, 70, n)
        df[prefix + "weather_code"] = rng.choice([0, 1, 3, 61, 71], n).astype(float)

    delayed = (rng.random(n) < 0.22).astype(int)
    on_time = np.round(np.clip(rng.normal(0, 8, n), 0, 15))
    late = np.round(np.clip(16 + rng.exponential(35, n), 16, 300))
    df["delayed"] = delayed
    df["delay_min"] = np.where(delayed == 1, late, on_time).astype(float)
    return df


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    df = make_toy()
    if "--leaky" in sys.argv:
        df["dep_delay"] = df["delay_min"] + 2
        df["taxi_out"] = 12.0
        out = DATA / "toy_flights_leaky.parquet"
    else:
        out = DATA / "toy_flights.parquet"
    df.to_parquet(out, index=False)
    print(f"TOY (fake) data written: {out}  rows={len(df)}")