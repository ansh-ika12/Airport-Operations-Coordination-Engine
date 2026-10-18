import zipfile

import holidays
import pandas as pd

from config import *

# Only scheduled/pre-departure columns plus the labels.
# No DepDelay, TaxiOut, WheelsOff, ActualElapsedTime or delay-cause columns (they leak the answer).
COLS = ["Month", "DayOfWeek", "FlightDate", "Reporting_Airline", "Tail_Number",
        "Origin", "Dest", "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
        "Distance", "ArrDelay", "Cancelled", "Diverted"]

RENAME = {"Month": "month", "DayOfWeek": "day_of_week", "FlightDate": "flight_date",
          "Reporting_Airline": "airline", "Tail_Number": "tail_number",
          "Origin": "origin", "Dest": "dest", "Distance": "distance",
          "CRSElapsedTime": "sched_duration"}

SAMPLE_FILE = DATA / "sample_flights.parquet"
SAMPLE_SIZE = 20_000


def read_bts_zip(path):
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as f:
            return pd.read_csv(f, usecols=COLS, low_memory=False)


# ---------- 1. Load BTS (one month at a time to save memory) ----------
zips = sorted(BTS_DIR.glob("*.zip"))
if len(zips) != 12:
    print(f"WARNING: expected 12 BTS zip files, found {len(zips)}.")

frames = []
for p in zips:
    print("Reading", p.name)
    d = read_bts_zip(p)
    frames.append(d[d["Origin"].isin(AIRPORTS)])   # keep all destinations for congestion counts
df = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
print("Flights leaving the 10 airports:", len(df))

# ---------- 2. Basic cleaning and local hours ----------
df = df.dropna(subset=["CRSDepTime", "CRSArrTime", "sched_duration"])
df["flight_date"] = pd.to_datetime(df["flight_date"])
df["dep_hour"] = ((df["CRSDepTime"] // 100) % 24).astype(int)
df["arr_hour"] = ((df["CRSArrTime"] // 100) % 24).astype(int)

# ---------- 3. Congestion: scheduled departures from the origin in that hour ----------
# Counted before filtering so it includes flights to every destination.
cong = (df.groupby(["origin", "flight_date", "dep_hour"]).size()
          .rename("origin_dep_count").reset_index())
df = df.merge(cong, on=["origin", "flight_date", "dep_hour"], how="left")

# ---------- 4. Keep the 10-airport pairs, drop cancelled and diverted flights ----------
df = df[df["dest"].isin(AIRPORTS) & (df["Cancelled"] == 0) & (df["Diverted"] == 0)]
df = df.dropna(subset=["ArrDelay"]).copy()
print("Completed flights between the 10 airports:", len(df))

# ---------- 5. Local timestamps for the weather join (BTS times are local) ----------
overnight = (df["CRSArrTime"] < df["CRSDepTime"]).astype(int)
df["dep_dt"] = df["flight_date"] + pd.to_timedelta(df["dep_hour"], unit="h")
df["arr_dt"] = df["flight_date"] + pd.to_timedelta(df["arr_hour"] + 24 * overnight, unit="h")

wx = pd.read_parquet(WEATHER_FILE)
wx["time"] = wx["time"].astype("datetime64[ns]")
wx = wx.drop_duplicates(["airport", "time"])    # guards against repeated hours at daylight-saving changes
wx_o = wx.add_prefix("o_").rename(columns={"o_airport": "origin", "o_time": "dep_dt"})
wx_d = wx.add_prefix("d_").rename(columns={"d_airport": "dest", "d_time": "arr_dt"})
df = df.merge(wx_o, on=["origin", "dep_dt"], how="left")
df = df.merge(wx_d, on=["dest", "arr_dt"], how="left")

# ---------- 6. Holiday flag ----------
us_hol = set(holidays.US(years=[YEAR]).keys())
df["is_holiday"] = df["flight_date"].dt.date.isin(us_hol).astype(int)

# ---------- 7. Labels ----------
df["delayed"] = (df["ArrDelay"] > DELAY_THRESHOLD_MIN).astype(int)
df["delay_min"] = df["ArrDelay"].clip(lower=0, upper=300)   # delay length; early arrivals = 0
df["route"] = df["origin"] + "_" + df["dest"]

# ---------- 8. Save the final table (schema from the team PRD, Section 5.4) ----------
keep = (["flight_date", "tail_number", "route", "dep_hour", "delayed", "delay_min"]
        + CAT_FEATURES
        + [c for c in NUM_FEATURES if c not in HISTORY_FEATURES])
keep = list(dict.fromkeys(keep))
final = df[keep].reset_index(drop=True)
DATA.mkdir(exist_ok=True)
final.to_parquet(DATASET_FILE)

# ---------- 9. Sample file for teammates (for testing code, NOT for judging model quality) ----------
sample = final.sample(n=min(SAMPLE_SIZE, len(final)), random_state=42)
sample.to_parquet(SAMPLE_FILE)

# ---------- 10. Summary ----------
print("\n===== SUMMARY =====")
print("Rows:", len(final), "| columns:", final.shape[1])
print("Delayed rate:", round(final["delayed"].mean(), 3))
print("Weather missing (origin):", round(final["o_temperature_2m"].isna().mean(), 4))
print("Weather missing (dest):  ", round(final["d_temperature_2m"].isna().mean(), 4))
print("\nRows and delay rate by month:")
print(final.groupby("month")["delayed"].agg(rows="size", delay_rate="mean").round(3))
print("\nSaved:", DATASET_FILE)
print("Saved:", SAMPLE_FILE, f"({len(sample)} rows)")