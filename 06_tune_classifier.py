import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import *

PARAMS_FILE = ROOT / "best_params.json"
MAX_ROUNDS = 3000
MIN_GAIN = 0.001          # adopt new settings only if validation AUC improves by at least this

# ---------- Same split and history features as 03_train.py ----------
df = pd.read_parquet(DATASET_FILE)
train = df[df["month"].isin(TRAIN_MONTHS)].copy()
val = df[df["month"].isin(VAL_MONTHS)].copy()
test = df[df["month"].isin(TEST_MONTHS)].copy()

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

pos_rate = y_tr.mean()
BASE = dict(objective="binary", metric="auc", bagging_fraction=0.8, bagging_freq=1,
            feature_fraction=0.8, scale_pos_weight=(1 - pos_rate) / pos_rate,
            seed=42, verbose=-1)


def fit(num_leaves, min_child_samples, learning_rate):
    params = {**BASE, "num_leaves": num_leaves, "min_child_samples": min_child_samples,
              "learning_rate": learning_rate}
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    t0 = time.time()
    booster = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS,
                        valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
    return booster, booster.best_score["val"]["auc"], booster.best_iteration, time.time() - t0


rows, boosters = [], {}


def run(nl, mcs, lr):
    print(f"Training num_leaves={nl}, min_child_samples={mcs}, learning_rate={lr} ...", flush=True)
    booster, auc, best_it, secs = fit(nl, mcs, lr)
    key = (nl, mcs, lr)
    boosters[key] = booster
    rows.append({"num_leaves": nl, "min_child_samples": mcs, "learning_rate": lr,
                 "val_auc": round(auc, 4), "best_iteration": best_it, "seconds": round(secs)})


# ---------- 1. Grid at learning_rate 0.05 ----------
for nl in (15, 31, 63):
    for mcs in (20, 100, 300):
        run(nl, mcs, 0.05)

# ---------- 2. Best grid point again with a smaller learning rate ----------
best_grid = max(rows, key=lambda r: r["val_auc"])
run(best_grid["num_leaves"], best_grid["min_child_samples"], 0.03)

table = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
pd.set_option("display.width", 200)
print("\n=== Validation AUC by setting (best first) ===")
print(table.to_string(index=False))

# ---------- 3. Decide, using VALIDATION only ----------
current = (63, 20, 0.05)                      # the settings 03_train.py used so far
cur_val = next(r["val_auc"] for r in rows
               if (r["num_leaves"], r["min_child_samples"], r["learning_rate"]) == current)
best = table.iloc[0]
best_key = (int(best["num_leaves"]), int(best["min_child_samples"]), float(best["learning_rate"]))
gain = float(best["val_auc"]) - cur_val
adopt = gain >= MIN_GAIN
chosen_key = best_key if adopt else current

# ---------- 4. One look at the TEST months, for the report only ----------
def test_auc(key):
    b = boosters[key]
    return roc_auc_score(y_te, b.predict(X_te, num_iteration=b.best_iteration))


print("\n=== Test AUC (not used for choosing) ===")
print(f"Current settings {current}: {test_auc(current):.4f}")
print(f"Best on validation {best_key}: {test_auc(best_key):.4f}")
print(f"\nValidation gain over current: {gain:+.4f} (adopt only if at least {MIN_GAIN})")

params = {"num_leaves": chosen_key[0], "min_child_samples": chosen_key[1],
          "learning_rate": chosen_key[2], "val_auc": round(float(
              next(r["val_auc"] for r in rows
                   if (r["num_leaves"], r["min_child_samples"], r["learning_rate"]) == chosen_key)), 4)}
json.dump(params, open(PARAMS_FILE, "w"), indent=2)
print("Decision:", "ADOPT new settings" if adopt else "KEEP current settings (gain too small)")
print("Saved", PARAMS_FILE, params)