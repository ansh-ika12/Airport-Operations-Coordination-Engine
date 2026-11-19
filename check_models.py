from pathlib import Path
import joblib

from config import MODEL_DIR

for name in ["clf.joblib", "reg.joblib", "calibrator.joblib", "meta.joblib"]:
    p = Path(MODEL_DIR) / name
    if p.exists():
        print(f"OK       {name:20s} {p.stat().st_size / 1e6:8.2f} MB")
    else:
        print(f"MISSING  {name}")

meta = joblib.load(Path(MODEL_DIR) / "meta.joblib")
print("meta keys:", list(meta.keys()))