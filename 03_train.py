import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)

from config import *

# Use Ojashwee's metrics module once it exists; until then use this identical fallback.
try:
    from ml.evaluate import classification_metrics
except ImportError:
    def classification_metrics(y_true, proba, threshold=0.5):
        pred = (np.asarray(proba) >= threshold).astype(int)
        return {"roc_auc": roc_auc_score(y_true, proba),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0)}


def show(name, y, p, thr=0.5):
    m = classification_metrics(y, p, threshold=thr)
    print(f"{name:<30} AUC={m['roc_auc']:.3f}  precision={m['precision']:.3f}  "
          f"recall={m['recall']:.3f}  F1={m['f1']:.3f}")


def reg_metrics(y, p):
    return mean_absolute_error(y, p), float(np.sqrt(mean_squared_error(y, p)))


# ---------- 0. Classifier settings: from the tuning step if available ----------
DEFAULTS = {"num_leaves": 63, "min_child_samples": 20, "learning_rate": 0.05}
PARAMS_FILE = ROOT / "best_params.json"
params = dict(DEFAULTS)
if PARAMS_FILE.exists():
    saved = json.load(open(PARAMS_FILE))
    params.update({k: saved[k] for k in DEFAULTS if k in saved})
print("Classifier settings:", params)

# ---------- 1. Load and split by time (never random) ----------
df = pd.read_parquet(DATASET_FILE)
train = df[df["month"].isin(TRAIN_MONTHS)].copy()
val = df[df["month"].isin(VAL_MONTHS)].copy()
test = df[df["month"].isin(TEST_MONTHS)].copy()
for name, d in [("train", train), ("validation", val), ("test", test)]:
    print(f"{name:<11} {len(d):>8,} rows | delayed rate {d['delayed'].mean():.3f}")

# ---------- 2. History features: computed on TRAINING months only ----------
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


train, val, test = add_history(train), add_history(val), add_history(test)

cats = {c: sorted(train[c].unique()) for c in CAT_FEATURES}


def to_lgb(d):
    X = d[ALL_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


X_tr, X_va, X_te = to_lgb(train), to_lgb(val), to_lgb(test)
y_tr, y_va, y_te = train["delayed"], val["delayed"], test["delayed"]

# ---------- 3. LightGBM classifier ----------
# Class imbalance: scale_pos_weight. Early stopping tracks AUC ONLY (metric="auc"),
# because weighted log-loss gets worse immediately and would stop training after one tree.
pos_rate = y_tr.mean()
clf = lgb.LGBMClassifier(
    n_estimators=3000, learning_rate=params["learning_rate"],
    num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    scale_pos_weight=(1 - pos_rate) / pos_rate,
    metric="auc", random_state=42, verbose=-1,
)
print("\nTraining classifier... (a few minutes)")
clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
print("Classifier best iteration:", clf.best_iteration_)

raw_va = clf.predict_proba(X_va)[:, 1]
raw_te = clf.predict_proba(X_te)[:, 1]

# ---------- 4. Calibration: turn weighted scores into real probabilities ----------
# Fitted on the VALIDATION months only. The test months stay untouched for evaluation.
calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
calibrator.fit(raw_va, y_va)
cal_te = calibrator.predict(raw_te)

print("\n=== Delay probability on TEST months ===")
show("Raw scores, threshold 0.50", y_te, raw_te, 0.5)
for thr in (0.15, 0.20, 0.30):
    show(f"Calibrated, threshold {thr:.2f}", y_te, cal_te, thr)
print("\nAverage probability on test:  raw", round(float(raw_te.mean()), 3),
      "| calibrated", round(float(cal_te.mean()), 3),
      "| actual delayed rate", round(float(y_te.mean()), 3))

bins = [0, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
table = (pd.DataFrame({"p": cal_te, "y": y_te.values})
         .assign(bin=lambda t: pd.cut(t["p"], bins, include_lowest=True))
         .groupby("bin", observed=True)
         .agg(flights=("y", "size"), mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
         .round(3))
print("\n=== Calibration check on test (mean_predicted should be close to actual_rate) ===")
print(table)

gain = pd.Series(clf.booster_.feature_importance(importance_type="gain"), index=ALL_FEATURES)
print("\n=== Top 10 features by importance (% of total gain) ===")
print((100 * gain / gain.sum()).sort_values(ascending=False).head(10).round(1))

# ---------- 5. LightGBM regressor: expected delay minutes ----------
reg = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    metric="rmse", random_state=42, verbose=-1,
)
print("\nTraining regressor... (a few minutes)")
reg.fit(X_tr, train["delay_min"], eval_set=[(X_va, val["delay_min"])], eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
print("Regressor best iteration:", reg.best_iteration_)

y_te_min = test["delay_min"]
pred_min = np.clip(reg.predict(X_te), 0, None)
print("\n=== Delay minutes on TEST months ===")
for name, p in [("Predict 0 minutes", np.zeros(len(test))),
                ("Predict training mean", np.full(len(test), train["delay_min"].mean())),
                ("LightGBM regressor", pred_min)]:
    mae, rmse = reg_metrics(y_te_min, p)
    print(f"{name:<24} MAE={mae:.2f}  RMSE={rmse:.2f}")
print("Average delay minutes on test: actual", round(float(y_te_min.mean()), 2),
      "| predicted", round(float(pred_min.mean()), 2))

# ---------- 6. Save everything teammates and the Backend need ----------
ap = pd.read_csv(AIRPORTS_FILE)
coords = {r.iata_code: (float(r.latitude_deg), float(r.longitude_deg)) for r in ap.itertuples()}
cong = train.groupby(["origin", "dep_hour"])["origin_dep_count"].median()
meta = {
    "features": ALL_FEATURES,
    "cats": cats,
    "history": history,
    "airport_coords": coords,
    "congestion_median": {f"{o}|{h}": float(v) for (o, h), v in cong.items()},
}
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(clf, MODEL_DIR / "clf.joblib")
joblib.dump(reg, MODEL_DIR / "reg.joblib")
joblib.dump(calibrator, MODEL_DIR / "calibrator.joblib")
joblib.dump(meta, MODEL_DIR / "meta.joblib")

# Test-month predictions for Ojashwee (risk cutoffs) and Kamakshi (plots)
out = test[["flight_date", "airline", "origin", "dest", "delayed", "delay_min"]].copy()
out["proba_raw"] = raw_te
out["proba"] = cal_te            # calibrated probability: use this one
out["pred_minutes"] = pred_min
out.to_parquet(DATA / "test_predictions.parquet")

print("\nSaved: models/clf.joblib, reg.joblib, calibrator.joblib, meta.joblib and "
      "data/test_predictions.parquet")