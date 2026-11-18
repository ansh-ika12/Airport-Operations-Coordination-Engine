import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


def classification_metrics(y_true, proba, threshold=0.5):
    """Return ROC-AUC, precision, recall and F1 for a classifier.

    y_true: true labels (0/1)
    proba: predicted probability of the positive class (delayed = 1)
    threshold: cutoff used to turn probabilities into 0/1 predictions
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def regression_metrics(y_true, y_pred):
    """Return MAE and RMSE for delay minutes."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def comparison_table(results):
    """Build a comparison table.

    results: {model name: metrics dict}
    Returns a DataFrame with one row per model.
    """
    return pd.DataFrame.from_dict(results, orient="index")
