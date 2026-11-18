import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic_baseline(num_features, cat_features):
    """Logistic regression baseline for delay probability.

    Pipeline: median imputation + scaling for numeric columns,
    most-frequent imputation + one-hot encoding for categorical columns,
    then LogisticRegression with balanced class weights.
    Input is a DataFrame with the columns in num_features and cat_features.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, list(num_features)),
        ("cat", categorical, list(cat_features)),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def naive_mean_prediction(train_minutes, n):
    """Predict the training mean delay for every flight (n predictions)."""
    mean = float(np.mean(np.asarray(train_minutes, dtype=float)))
    return np.full(n, mean)
