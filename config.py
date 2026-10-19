from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
BTS_DIR = DATA / "bts"
WEATHER_FILE = DATA / "weather.parquet"
AIRPORTS_FILE = DATA / "airports_selected.csv"
DATASET_FILE = DATA / "flights_model.parquet"
MODEL_DIR = ROOT / "models"

YEAR = 2023
AIRPORTS = ["ATL", "ORD", "DFW", "DEN", "JFK", "LAX", "SFO", "SEA", "MIA", "BOS"]
DELAY_THRESHOLD_MIN = 15

TRAIN_MONTHS = list(range(1, 10))
VAL_MONTHS = [10]
TEST_MONTHS = [11, 12]

# Provisional cutoffs for CALIBRATED probabilities; final values come from Ojashwee's analysis.
RISK_LOW_MAX = 0.15
RISK_HIGH_MIN = 0.30

WEATHER_VARS = ["temperature_2m", "precipitation", "cloud_cover",
                "wind_speed_10m", "wind_gusts_10m", "weather_code"]
WEATHER_FEATURES = [f"{p}{v}" for p in ("o_", "d_") for v in WEATHER_VARS]

CAT_FEATURES = ["airline", "origin", "dest"]
HISTORY_FEATURES = ["airline_delay_rate", "origin_delay_rate",
                    "dest_delay_rate", "route_delay_rate"]
NUM_FEATURES = (["month", "day_of_week", "dep_hour", "is_holiday", "distance",
                 "sched_duration", "origin_dep_count"]
                + WEATHER_FEATURES + HISTORY_FEATURES)
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES