import numpy as np
import pandas as pd
import lightgbm as lgb

from config import *

ROUNDS = 400
CHECKPOINTS = [1, 10, 50, 100, 200, 400]

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
y_tr, y_va, y_te = train["delayed"], val["delayed"], test["delayed"]
pos_rate = y_tr.mean()

BASE = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
            bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8,
            scale_pos_weight=(1 - pos_rate) / pos_rate, seed=42, verbose=-1)

STRONG_REG = dict(num_leaves=15, min_child_samples=500, lambda_l2=10.0,
                  learning_rate=0.03, feature_fraction=0.7)

# name: (features to drop, parameter changes)
VARIANTS = {
    "1. all features (current)": ([], {}),
    "2. drop month": (["month"], {}),
    "3. drop month + stronger regularization": (["month"], STRONG_REG),
    "4. drop month + all weather": (["month"] + WEATHER_FEATURES, STRONG_REG),
}


def prep(d, feats):
    X = d[feats].copy()
    for c in CAT_FEATURES:
        if c in feats:
            X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


rows = []
for name, (drop, extra) in VARIANTS.items():
    print("Training variant:", name)
    feats = [f for f in ALL_FEATURES if f not in drop]
    dtr = lgb.Dataset(prep(train, feats), label=y_tr)
    dva = lgb.Dataset(prep(val, feats), label=y_va, reference=dtr)
    dte = lgb.Dataset(prep(test, feats), label=y_te, reference=dtr)
    ev = {}
    lgb.train({**BASE, **extra}, dtr, num_boost_round=ROUNDS,
              valid_sets=[dva, dte], valid_names=["val", "test"],
              callbacks=[lgb.record_evaluation(ev)])
    va, te = np.array(ev["val"]["auc"]), np.array(ev["test"]["auc"])
    best = int(va.argmax())
    row = {"variant": name, "best_val_round": best + 1,
           "val_auc": round(va[best], 3), "test_auc_at_best": round(te[best], 3)}
    for k in CHECKPOINTS:
        row[f"test@{k}"] = round(te[k - 1], 3)
    rows.append(row)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("\n=== AUC comparison ===")
print(pd.DataFrame(rows).to_string(index=False))
print("\nNote: choose settings using the validation AUC (val_auc), not the test columns.")
print("The test columns are only here to show whether the curve keeps improving or collapses.")