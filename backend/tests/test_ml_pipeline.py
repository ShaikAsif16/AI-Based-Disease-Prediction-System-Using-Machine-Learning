"""Automated tests for the ML preprocessing pipeline and saved model artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.preprocessing import (
    FEATURE_ORDER,
    InvalidZeroToNaNTransformer,
    ZERO_AS_MISSING_COLUMNS,
    validate_diabetes_input,
)
from backend.ml.predictor import (
    DEFAULT_METADATA_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_PIPELINE_PATH,
)


def test_zero_to_nan_transformer_only_replaces_clinical_zero_columns():
    df = pd.DataFrame(
        [
            {
                "Pregnancies": 0,
                "Glucose": 0,
                "BloodPressure": 70,
                "SkinThickness": 0,
                "Insulin": 85,
                "BMI": 0.0,
                "DiabetesPedigreeFunction": 0.4,
                "Age": 30,
            }
        ]
    )
    transformer = InvalidZeroToNaNTransformer(columns=list(ZERO_AS_MISSING_COLUMNS))
    transformed = transformer.fit_transform(df)

    # Pregnancies=0 MUST remain 0 (valid value)
    assert transformed["Pregnancies"].iloc[0] == 0
    # Physiologically invalid zeros MUST become NaN
    assert np.isnan(transformed["Glucose"].iloc[0])
    assert np.isnan(transformed["SkinThickness"].iloc[0])
    assert np.isnan(transformed["BMI"].iloc[0])
    # Non-zero measurements remain intact
    assert transformed["BloodPressure"].iloc[0] == 70.0
    assert transformed["Insulin"].iloc[0] == 85.0


def test_input_validation_rejects_missing_and_invalid_zero_values():
    missing_payload = {"Pregnancies": 2, "Glucose": 110}
    df, errors = validate_diabetes_input(missing_payload)
    assert df is None
    assert "BloodPressure" in errors
    assert "BMI" in errors

    invalid_zero_payload = {
        "Pregnancies": 0,  # Valid
        "Glucose": 0,      # Invalid
        "BloodPressure": 72,
        "SkinThickness": 22,
        "Insulin": 90,
        "BMI": 0,          # Invalid
        "DiabetesPedigreeFunction": 0.35,
        "Age": 28,
    }
    df2, errors2 = validate_diabetes_input(invalid_zero_payload)
    assert df2 is None
    assert "Pregnancies" not in errors2
    assert "Glucose" in errors2
    assert "BMI" in errors2


def test_saved_pipeline_and_artifacts_exist_and_predict_deterministically():
    assert DEFAULT_PIPELINE_PATH.exists(), "Trained joblib pipeline must exist"
    assert DEFAULT_METADATA_PATH.exists(), "Model metadata JSON must exist"
    assert DEFAULT_METRICS_PATH.exists(), "Evaluation metrics JSON must exist"

    pipeline = joblib.load(DEFAULT_PIPELINE_PATH)
    metadata = json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))

    assert metadata["selected_model_key"] in metrics["models"]
    assert set(metrics["models"].keys()) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "svm",
    }

    low_risk = pd.DataFrame([[1, 90, 68, 20, 65, 22.5, 0.21, 24]], columns=FEATURE_ORDER)
    high_risk = pd.DataFrame([[6, 182, 86, 36, 220, 36.2, 0.85, 54]], columns=FEATURE_ORDER)

    prob_low_1 = float(pipeline.predict_proba(low_risk)[0][1])
    prob_low_2 = float(pipeline.predict_proba(low_risk)[0][1])
    prob_high = float(pipeline.predict_proba(high_risk)[0][1])

    # Predictions must be deterministic (within floating-point ULP) and clinically directional
    assert abs(prob_low_1 - prob_low_2) < 1e-12
    assert 0.0 <= prob_low_1 <= 1.0
    assert 0.0 <= prob_high <= 1.0
    assert prob_high > prob_low_1
