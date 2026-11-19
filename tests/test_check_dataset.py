"""Tests for 04_check_dataset.py (runs it as a script, like a user would)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_checker(path):
    return subprocess.run([sys.executable, "04_check_dataset.py", str(path)],
                          cwd=ROOT, capture_output=True, text=True)


def test_clean_data_passes(toy_df, tmp_path):
    f = tmp_path / "clean.parquet"
    toy_df.to_parquet(f, index=False)
    r = run_checker(f)
    assert r.returncode == 0, r.stdout
    assert "OVERALL: PASS" in r.stdout


def test_leaky_column_fails(toy_df, tmp_path):
    df = toy_df.copy()
    df["dep_delay"] = df["delay_min"] + 2
    f = tmp_path / "leaky.parquet"
    df.to_parquet(f, index=False)
    r = run_checker(f)
    assert r.returncode == 1
    assert "dep_delay" in r.stdout


def test_missing_column_fails(toy_df, tmp_path):
    f = tmp_path / "missing.parquet"
    toy_df.drop(columns=["distance"]).to_parquet(f, index=False)
    assert run_checker(f).returncode == 1


def test_inconsistent_label_fails(toy_df, tmp_path):
    df = toy_df.copy()
    df["delayed"] = 1 - df["delayed"]      # label no longer matches delay_min > 15
    f = tmp_path / "badlabel.parquet"
    df.to_parquet(f, index=False)
    assert run_checker(f).returncode == 1