import numpy as np
import pandas as pd

PROVISIONAL_LOW_MAX = 0.15
PROVISIONAL_HIGH_MIN = 0.30


def _level_rows(y_true, proba, low_max, high_min):
    """One row per risk level for a given pair of cutoffs."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    total_delayed = max(int(y.sum()), 1)
    masks = {
        "Low": p < low_max,
        "Medium": (p >= low_max) & (p < high_min),
        "High": p >= high_min,
    }
    rows = []
    for level, mask in masks.items():
        n = int(mask.sum())
        delayed = int(y[mask].sum())
        rows.append({
            "level": level,
            "share_of_flights": n / len(y),
            "precision": delayed / n if n else np.nan,
            "recall": delayed / total_delayed,
        })
    return rows


def suggest_cutoffs(y_true, proba, low_grid=None, high_grid=None,
                    min_high_share=0.05, max_high_share=0.25):
    """Propose Low/Medium/High cutoffs for calibrated probabilities.

    Returns a dict with low_max, high_min, table (one row per cutoff pair and
    level, with share_of_flights, precision and recall) and a note.
    Recommendation: among pairs where High covers between min_high_share and
    max_high_share of flights, pick the one with the highest High precision
    (ties: higher recall). Falls back to the provisional cutoffs otherwise.
    """
    if low_grid is None:
        low_grid = np.round(np.arange(0.05, 0.31, 0.01), 2)
    if high_grid is None:
        high_grid = np.round(np.arange(0.15, 0.61, 0.01), 2)

    records = []
    for low in low_grid:
        for high in high_grid:
            if high <= low:
                continue
            for row in _level_rows(y_true, proba, low, high):
                records.append({"low_max": float(low), "high_min": float(high), **row})
    table = pd.DataFrame(records)

    high_rows = table[table["level"] == "High"]
    ok = high_rows[
        (high_rows["share_of_flights"] >= min_high_share)
        & (high_rows["share_of_flights"] <= max_high_share)
        & high_rows["precision"].notna()
    ]
    if ok.empty:
        low_max, high_min = PROVISIONAL_LOW_MAX, PROVISIONAL_HIGH_MIN
        note = "No cutoff pair met the High-share range. Provisional values kept."
    else:
        best = ok.sort_values(["precision", "recall"], ascending=False).iloc[0]
        low_max, high_min = float(best["low_max"]), float(best["high_min"])
        note = "Best High-level precision within the High-share range."

    return {"low_max": low_max, "high_min": high_min, "table": table, "note": note}
