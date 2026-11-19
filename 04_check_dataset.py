"""Checks a flights dataset against PRD 5.4 (schema) and 5.10 (leakage).

Run from flight-delay-ml/:
    python 04_check_dataset.py [path-to-parquet]
"""
import sys
from pathlib import Path

import pandas as pd
from pandas.api import types as pt

from config import DATASET_FILE, AIRPORTS, ALL_FEATURES

SCHEMA = {
    "flight_date": "datetime", "tail_number": "string", "route": "string",
    "airline": "string", "origin": "string", "dest": "string",
    "month": "int", "day_of_week": "int", "dep_hour": "int", "is_holiday": "int",
    "distance": "number", "sched_duration": "number", "origin_dep_count": "number",
    "delayed": "int", "delay_min": "number",
}
for _p in ("o_", "d_"):
    for _v in ["temperature_2m", "precipitation", "cloud_cover",
            "wind_speed_10m", "wind_gusts_10m", "weather_code"]:
        SCHEMA[_p + _v] = "number"
WEATHER_COLS = [c for c in SCHEMA if c.startswith(("o_", "d_"))]

LEAKY_PATTERNS = [
    "dep_delay", "departure_delay", "depdelay", "taxi", "wheels",
    "arr_time", "arrtime", "actual", "cancel", "divert",
    "carrier_delay", "carrierdelay", "weather_delay", "weatherdelay",
    "nas_delay", "nasdelay", "security_delay", "late_aircraft", "lateaircraft",
    "arr_delay", "arrdelay",
]

results = []


def report(name, status, details=()):
    results.append((name, status))
    print(f"\n[{status}] {name}")
    for line in details:
        print("    - " + line)


def kind_ok(series, kind):
    if kind == "datetime":
        return pt.is_datetime64_any_dtype(series)
    if kind == "string":
        return pt.is_string_dtype(series) or pt.is_object_dtype(series)
    if kind == "int":
        return pt.is_integer_dtype(series)
    return pt.is_numeric_dtype(series) and not pt.is_bool_dtype(series)


def check_schema(df):
    problems, warns = [], []
    missing = [c for c in SCHEMA if c not in df.columns]
    extra = [c for c in df.columns if c not in SCHEMA]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        warns.append(f"extra columns not in PRD 5.4: {extra}")

    for col, kind in SCHEMA.items():
        if col in df.columns and not kind_ok(df[col], kind):
            problems.append(f"column '{col}' should be {kind}, found {df[col].dtype}")

    if not missing:
        rules = {
            "month outside 1..12": ~df["month"].between(1, 12),
            "day_of_week outside 1..7": ~df["day_of_week"].between(1, 7),
            "dep_hour outside 0..23": ~df["dep_hour"].between(0, 23),
            "is_holiday not 0/1": ~df["is_holiday"].isin([0, 1]),
            "delayed not 0/1": ~df["delayed"].isin([0, 1]),
            "delay_min outside 0..300": ~df["delay_min"].between(0, 300),
            "origin not in AIRPORTS": ~df["origin"].isin(AIRPORTS),
            "dest not in AIRPORTS": ~df["dest"].isin(AIRPORTS),
            "origin equals dest": df["origin"] == df["dest"],
            "route != origin_dest": df["route"] != (df["origin"] + "_" + df["dest"]),
            "delayed inconsistent with delay_min>15":
                (df["delay_min"] > 15).astype(int) != df["delayed"],
            "distance <= 0": df["distance"] <= 0,
            "sched_duration <= 0": df["sched_duration"] <= 0,
        }
        for text, bad in rules.items():
            n = int(bad.sum())
            if n:
                problems.append(f"{n} rows: {text}")
        if pt.is_datetime64_any_dtype(df["flight_date"]):
            if (df["flight_date"].dt.month != df["month"]).any():
                problems.append("flight_date month does not match month column")

    status = "FAIL" if problems else ("WARN" if warns else "PASS")
    report("Schema and value checks", status, problems + warns)


def check_leakage(df):
    problems, warns = [], []
    names = set(df.columns) | set(ALL_FEATURES)
    for name in sorted(names):
        for pat in LEAKY_PATTERNS:
            if pat in name.lower():
                problems.append(f"'{name}' looks leaky (matches '{pat}')")
                break

    if "delayed" in df.columns:
        cols = [c for c in ALL_FEATURES if c in df.columns
                and pt.is_numeric_dtype(df[c]) and c != "delayed"]
        constant = [c for c in cols if df[c].nunique(dropna=True) <= 1]
        for c in constant:
            warns.append(f"'{c}' is constant (one value only); check how it was built")
        cols = [c for c in cols if c not in constant]
        if cols:
            corr = df[cols].corrwith(df["delayed"]).abs().sort_values(ascending=False)
            for name, val in corr.items():
                if val > 0.5:
                    warns.append(f"'{name}' has |correlation| {val:.2f} with label; investigate")

    status = "FAIL" if problems else ("WARN" if warns else "PASS")
    report("Leakage check (PRD 5.10)", status, problems + warns)


def check_missing(df):
    pct = (df.isna().mean() * 100).round(2)
    pct = pct[pct > 0].sort_values(ascending=False)
    problems, warns = [], []
    for col, p in pct.items():
        if col in WEATHER_COLS:
            (problems if p > 5 else warns).append(f"{col}: {p}% missing (weather limit is 5%)")
        elif col == "tail_number":
            warns.append(f"{col}: {p}% missing (not a model feature)")
        else:
            problems.append(f"{col}: {p}% missing (must be 0%)")
    details = problems + warns if len(pct) else ["no missing values in any column"]
    status = "FAIL" if problems else ("WARN" if warns else "PASS")
    report("Missing values", status, details)


def check_rates(df):
    problems = []
    print(f"\nRows: {len(df):,}   Overall delay rate: {df['delayed'].mean() * 100:.1f}%")

    by_month = df.groupby("month").agg(rows=("delayed", "size"),
                                    delay_rate=("delayed", "mean")).round(3)
    print("\nBy month:")
    print(by_month.to_string())

    by_airport = df.groupby("origin").agg(rows=("delayed", "size"),
                                        delay_rate=("delayed", "mean")).round(3)
    print("\nBy origin airport:")
    print(by_airport.to_string())

    absent_months = [m for m in range(1, 13) if m not in by_month.index]
    absent_airports = [a for a in AIRPORTS if a not in by_airport.index]
    if absent_months:
        problems.append(f"months with zero flights: {absent_months}")
    if absent_airports:
        problems.append(f"airports with zero departures: {absent_airports}")
    report("Coverage by month and airport", "FAIL" if problems else "PASS", problems)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_FILE
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)
    df = pd.read_parquet(path)
    print(f"Checking: {path}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    check_schema(df)
    check_leakage(df)
    check_missing(df)
    check_rates(df)

    print("\n" + "=" * 50)
    for name, status in results:
        print(f"{status:5s} {name}")
    failed = any(s == "FAIL" for _, s in results)
    print("OVERALL:", "FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()