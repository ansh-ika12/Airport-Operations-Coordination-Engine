# Airport-Operations-Coordination-Engine

## Train the delay model

From the repository root, run:

```powershell
python 01_download_data.py
python 02_build_dataset.py
python 04_check_dataset.py
python 03_train.py
python run_o3_o5.py
python -m pytest tests -v
```

The download, dataset-build, and training scripts should be run in order.
Training creates `models\clf.joblib`, `models\reg.joblib`,
`models\calibrator.joblib`, `models\meta.joblib`, and
`data\test_predictions.parquet`. `run_o3_o5.py` evaluates the baseline and
produces the comparison and cutoff tables. Generated data and model files are
not tracked by Git.