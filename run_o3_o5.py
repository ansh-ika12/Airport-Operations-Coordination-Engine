# RUN ONCE data/ is ready. Do not commit.
import numpy as np
import pandas as pd

from config import (DATASET_FILE, DATA, TRAIN_MONTHS, TEST_MONTHS,
                    ALL_FEATURES, CAT_FEATURES, NUM_FEATURES)
from ml.baselines import build_logistic_baseline, naive_mean_prediction
from ml.evaluate import classification_metrics, regression_metrics, comparison_table
from ml.risk_cutoffs import suggest_cutoffs

df = pd.read_parquet(DATASET_FILE)
train = df[df["month"].isin(TRAIN_MONTHS)].copy()
test = df[df["month"].isin(TEST_MONTHS)].copy()

# History features from TRAINING months only (same logic as 03_train.py)
history = {
    "global": float(train["delayed"].mean()),
    "airline": train.groupby("airline")["delayed"].mean().to_dict(),
    "origin": train.groupby("origin")["delayed"].mean().to_dict(),
    "dest": train.groupby("dest")["delayed"].mean().to_dict(),
    "route": train.groupby("route")["delayed"].mean().to_dict(),
}


def add_history(d):
    d = d.copy()
    for key in ["airline", "origin", "dest", "route"]:
        d[f"{key}_delay_rate"] = d[key].map(history[key]).fillna(history["global"])
    return d


train, test = add_history(train), add_history(test)
y = test["delayed"]

# O-3: logistic baseline trained on training months, scored on test months
baseline = build_logistic_baseline(NUM_FEATURES, CAT_FEATURES)
baseline.fit(train[ALL_FEATURES], train["delayed"])
base_proba = baseline.predict_proba(test[ALL_FEATURES])[:, 1]

naive_minutes = naive_mean_prediction(train["delay_min"], len(test))
naive_reg = regression_metrics(test["delay_min"], naive_minutes)

# LightGBM predictions from Anshika's file
lgb = pd.read_parquet(DATA / "test_predictions.parquet")
assert len(lgb) == len(test), "Row count differs from test months - check dataset version"
assert (lgb["delayed"].to_numpy() == test["delayed"].to_numpy()).all(), \
    "Rows differ from the test months - check dataset version"

# Like-for-like: both models use class weighting, so compare their raw scores at 0.50.
# ROC-AUC does not depend on the threshold and is the main comparison.
base_cls = classification_metrics(y, base_proba, threshold=0.5)
lgb_raw_cls = classification_metrics(y, lgb["proba_raw"], threshold=0.5)
lgb_cal_cls = classification_metrics(y, lgb["proba"], threshold=0.20)
lgb_reg = regression_metrics(lgb["delay_min"], lgb["pred_minutes"])

# O-5: comparison table
results = {
    "Logistic baseline (class-weighted, threshold 0.50)": base_cls,
    "LightGBM raw (class-weighted, threshold 0.50)": {**lgb_raw_cls, **lgb_reg},
    "LightGBM calibrated (threshold 0.20)": lgb_cal_cls,
    "Naive mean (minutes only)": naive_reg,
}
table = comparison_table(results)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(table.round(4).to_string())
print("\nCompare ROC-AUC across models. Precision and recall depend on the threshold used.")
table.to_csv(DATA / "comparison_table.csv")

# O-4: risk cutoffs from the calibrated LightGBM probabilities
cut = suggest_cutoffs(lgb["delayed"], lgb["proba"])
print("\nRecommended low_max:", cut["low_max"], "high_min:", cut["high_min"])
print(cut["note"])
cut["table"].to_csv(DATA / "cutoff_table.csv", index=False)

t = cut["table"]
for name, low, high in [("Recommended", cut["low_max"], cut["high_min"]),
                        ("Provisional (config.py)", 0.15, 0.30)]:
    rows = t[np.isclose(t["low_max"], low) & np.isclose(t["high_min"], high)]
    print(f"\n{name}: low_max={low}, high_min={high}")
    print(rows[["level", "share_of_flights", "precision", "recall"]].round(3).to_string(index=False))
print("\n(For Low, 'precision' is the share of Low flights that were really delayed.)")
