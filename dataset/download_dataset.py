"""
Dataset Acquisition Script: Pima Indians Diabetes Database
==========================================================
Downloads the authentic Pima Indians Diabetes dataset (National Institute of
Diabetes and Digestive and Kidney Diseases / UCI / OpenML ID 37) and verifies
its schema and integrity.

Usage:
    python dataset/download_dataset.py

This script NEVER generates synthetic or placeholder data. If all download
sources are unreachable, it exits with an error and displays manual download
instructions.
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

EXPECTED_ROWS = 768
EXPECTED_POSITIVE_COUNT = 268
EXPECTED_NEGATIVE_COUNT = 500

DATASET_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATASET_DIR / "pima_indians_diabetes.csv"

# Canonical public mirrors of the original NIDDK/UCI Pima Indians Diabetes CSV
CSV_SOURCES = [
    (
        "Plotly / UCI Pima Indians Diabetes Mirror (with header)",
        "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        True,
    ),
    (
        "Jason Brownlee / UCI Machine Learning Datasets Mirror (no header)",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        False,
    ),
]


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the loaded DataFrame matches the authentic Pima dataset."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns: {list(df.columns)}. Expected: {EXPECTED_COLUMNS}"
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected row count: {len(df)}. Expected exactly {EXPECTED_ROWS} records."
        )

    # Coerce numeric types cleanly
    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Convert integer-valued columns cleanly
    for int_col in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "Age", "Outcome"]:
        df[int_col] = df[int_col].astype(int)

    pos_count = int((df["Outcome"] == 1).sum())
    neg_count = int((df["Outcome"] == 0).sum())
    if pos_count != EXPECTED_POSITIVE_COUNT or neg_count != EXPECTED_NEGATIVE_COUNT:
        raise ValueError(
            f"Unexpected class balance (positive={pos_count}, negative={neg_count}). "
            f"Expected positive={EXPECTED_POSITIVE_COUNT}, negative={EXPECTED_NEGATIVE_COUNT}."
        )

    return df


def try_download_from_mirrors() -> tuple[pd.DataFrame, str] | None:
    """Attempt to download the CSV from verified public UCI mirrors."""
    for name, url, has_header in CSV_SOURCES:
        print(f"Attempting download from: {name}\n  URL: {url}")
        try:
            req = Request(url, headers={"User-Agent": "AIDL-Disease-Prediction-Dataset-Fetcher/1.0"})
            with urlopen(req, timeout=20) as resp:
                raw_bytes = resp.read()
            text = raw_bytes.decode("utf-8")
            if has_header:
                df = pd.read_csv(io.StringIO(text))
            else:
                df = pd.read_csv(io.StringIO(text), header=None, names=EXPECTED_COLUMNS)
            df = validate_dataframe(df)
            return df, name
        except (URLError, TimeoutError, ValueError, OSError) as exc:
            print(f"  -> Failed ({exc})")
    return None


def try_fetch_openml() -> tuple[pd.DataFrame, str] | None:
    """Fallback to scikit-learn's OpenML fetcher (dataset ID 37: diabetes)."""
    print("Attempting download from OpenML (dataset ID 37: Pima Indians Diabetes)...")
    try:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=37, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        openml_rename = {
            "preg": "Pregnancies",
            "plas": "Glucose",
            "pres": "BloodPressure",
            "skin": "SkinThickness",
            "insu": "Insulin",
            "mass": "BMI",
            "pedi": "DiabetesPedigreeFunction",
            "age": "Age",
            "class": "Outcome",
        }
        df = df.rename(columns=openml_rename)
        df["Outcome"] = df["Outcome"].map({"tested_positive": 1, "tested_negative": 0})
        df = df[EXPECTED_COLUMNS]
        df = validate_dataframe(df)
        return df, "OpenML Dataset ID 37 (NIDDK / UCI Pima Indians Diabetes)"
    except Exception as exc:
        print(f"  -> OpenML fetch failed ({exc})")
        return None


def main() -> int:
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_csv(OUTPUT_PATH)
            validate_dataframe(existing)
            sha256 = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
            print(f"Verified existing authentic dataset at: {OUTPUT_PATH}")
            print(f"  Rows: {len(existing)} | Positive: 268 | Negative: 500 | SHA-256: {sha256[:16]}...")
            return 0
        except Exception as exc:
            print(f"Existing file at {OUTPUT_PATH} failed validation ({exc}). Re-downloading...")

    result = try_download_from_mirrors() or try_fetch_openml()
    if result is None:
        print(
            "\nERROR: Could not automatically download the Pima Indians Diabetes dataset.\n"
            "Please download it manually following dataset/DATASET_INSTRUCTIONS.md:\n"
            "  1. Visit https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
            "  2. Download 'diabetes.csv'\n"
            f"  3. Save it to: {OUTPUT_PATH}\n"
            "This project will NOT substitute synthetic data."
        )
        return 1

    df, source_name = result
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    sha256 = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()

    print("\nDataset downloaded and verified successfully!")
    print(f"  Source  : {source_name}")
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"  Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Outcome : 0 (Negative) = 500, 1 (Positive) = 268")
    print(f"  SHA-256 : {sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
