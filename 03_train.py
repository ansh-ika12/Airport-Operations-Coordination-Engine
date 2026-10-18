import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def show(name, y, p):
    m = classification_metrics(y, p)
    print(f"{name:<12} AUC={m['roc_auc']:.3f}  precision={m['precision']:.3f}  "
          f"recall={m['recall']:.3f}  F1={m['f1']:.3f}")


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

# ---------- 3. LightGBM classifier (class imbalance handled with scale_pos_weight) ----------
pos_rate = y_tr.mean()
clf = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    scale_pos_weight=(1 - pos_rate) / pos_rate,
    random_state=42, verbose=-1,
)
print("\nTraining... (this can take a few minutes)")
clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)])
print("Best iteration:", clf.best_iteration_)

# ---------- 4. Evaluate on validation and TEST months ----------
p_va = clf.predict_proba(X_va)[:, 1]
p_te = clf.predict_proba(X_te)[:, 1]
print("\n=== Delay probability (threshold 0.5) ===")
show("Validation", y_va, p_va)
show("Test", y_te, p_te)

# ---------- 5. Which features matter most (by gain) ----------
gain = pd.Series(clf.booster_.feature_importance(importance_type="gain"),
                 index=ALL_FEATURES)
print("\n=== Top 10 features by importance (% of total gain) ===")
print((100 * gain / gain.sum()).sort_values(ascending=False).head(10).round(1))

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
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, MODEL_DIR / "clf.joblib")
joblib.dump(meta, MODEL_DIR / "meta.joblib")

# Test-month probabilities, so Ojashwee (risk cutoffs) and Kamakshi (plots) can work without the model
out = test[["flight_date", "airline", "origin", "dest", "delayed", "delay_min"]].copy()
out["proba"] = p_te
out.to_parquet(DATA / "test_predictions.parquet")

print("\nSaved: models/clf.joblib, models/meta.joblib, data/test_predictions.parquet")