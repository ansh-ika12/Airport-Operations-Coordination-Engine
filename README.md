# Airport-Operations-Coordination-Engine

## Train the delay model

From the repository root, run:

```powershell
python data\03_train.py
```

This trains from `data\flights_model.parquet` and creates `models\clf.joblib`
and `models\meta.joblib`. The `models` directory and generated model files are
not tracked by Git; rerun training to regenerate them.