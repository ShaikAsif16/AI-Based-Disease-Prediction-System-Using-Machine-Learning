"""
Preprocessing, Feature Schema, and Input Validation for Diabetes Prediction
===========================================================================
Provides:
1. Canonical feature definitions (units, clinical guidance, validation bounds).
2. `InvalidZeroToNaNTransformer`: A scikit-learn compatible transformer that
   replaces physiologically impossible zero values in clinical measurements
   with NaN *inside* a scikit-learn Pipeline, ensuring imputation statistics
   are fitted strictly on training folds to prevent data leakage.
3. Strict user-input validation for inference that NEVER silently fabricates
   missing clinical measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER: list[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Columns in the Pima Indians Diabetes Dataset where 0 denotes a missing measurement
ZERO_AS_MISSING_COLUMNS: list[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    unit: str
    description: str
    clinical_reference: str
    min_value: float
    max_value: float
    step: float
    allow_zero: bool
    integer_only: bool
    placeholder: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DIABETES_FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        name="Pregnancies",
        label="Number of Pregnancies",
        unit="count",
        description="Total number of times pregnant (enter 0 if none).",
        clinical_reference="Typical range: 0–17",
        min_value=0,
        max_value=25,
        step=1,
        allow_zero=True,
        integer_only=True,
        placeholder="e.g., 2",
    ),
    FeatureSpec(
        name="Glucose",
        label="Plasma Glucose Concentration",
        unit="mg/dL",
        description="Plasma glucose measured 2 hours after an oral glucose tolerance test (OGTT).",
        clinical_reference="Normal 2-hr OGTT: < 140 mg/dL | Prediabetes: 140–199 mg/dL",
        min_value=40,
        max_value=400,
        step=1,
        allow_zero=False,
        integer_only=False,
        placeholder="e.g., 118",
    ),
    FeatureSpec(
        name="BloodPressure",
        label="Diastolic Blood Pressure",
        unit="mm Hg",
        description="Resting diastolic blood pressure (the bottom number in a blood pressure reading).",
        clinical_reference="Normal diastolic: 60–80 mm Hg",
        min_value=24,
        max_value=200,
        step=1,
        allow_zero=False,
        integer_only=False,
        placeholder="e.g., 72",
    ),
    FeatureSpec(
        name="SkinThickness",
        label="Triceps Skinfold Thickness",
        unit="mm",
        description="Triceps skinfold thickness measured with skinfold calipers to estimate subcutaneous fat.",
        clinical_reference="Typical adult range: 10–50 mm",
        min_value=5,
        max_value=110,
        step=0.5,
        allow_zero=False,
        integer_only=False,
        placeholder="e.g., 26",
    ),
    FeatureSpec(
        name="Insulin",
        label="2-Hour Serum Insulin",
        unit="µU/mL",
        description="2-hour post-load serum insulin level from a laboratory glucose tolerance blood panel.",
        clinical_reference="Typical 2-hr range: 16–166 µU/mL",
        min_value=10,
        max_value=900,
        step=1,
        allow_zero=False,
        integer_only=False,
        placeholder="e.g., 94",
    ),
    FeatureSpec(
        name="BMI",
        label="Body Mass Index (BMI)",
        unit="kg/m²",
        description="Weight in kilograms divided by height in meters squared (kg/m²).",
        clinical_reference="Normal: 18.5–24.9 | Overweight: 25.0–29.9 | Obese: ≥ 30.0",
        min_value=10.0,
        max_value=75.0,
        step=0.1,
        allow_zero=False,
        integer_only=False,
        placeholder="e.g., 26.4",
    ),
    FeatureSpec(
        name="DiabetesPedigreeFunction",
        label="Diabetes Pedigree Function",
        unit="score",
        description="Quantitative score reflecting genetic and family history predisposition to diabetes.",
        clinical_reference="Dataset range: 0.078–2.420 (higher indicates stronger family history)",
        min_value=0.05,
        max_value=3.0,
        step=0.001,
        allow_zero=False,
        integer_only=False,
        placeholder="e.g., 0.351",
    ),
    FeatureSpec(
        name="Age",
        label="Age",
        unit="years",
        description="Patient age in completed years. Note: training cohort consists of adults aged 21+.",
        clinical_reference="Training dataset cohort: 21–81 years",
        min_value=18,
        max_value=120,
        step=1,
        allow_zero=False,
        integer_only=True,
        placeholder="e.g., 34",
    ),
]

FEATURE_SPEC_BY_NAME: dict[str, FeatureSpec] = {
    spec.name: spec for spec in DIABETES_FEATURE_SPECS
}


class InvalidZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    """
    Replaces physiologically impossible zero values with NaN in specified columns.

    Works on both pandas DataFrames and NumPy arrays (using column indices
    inferred during fit). Designed to be placed as the first step in a
    scikit-learn Pipeline before SimpleImputer so all imputation statistics
    are computed strictly on training folds without data leakage.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "InvalidZeroToNaNTransformer":
        target_cols = self.columns if self.columns is not None else ZERO_AS_MISSING_COLUMNS
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.zero_col_indices_ = [
                int(X.columns.get_loc(col)) for col in target_cols if col in X.columns
            ]
        else:
            X_arr = np.asarray(X)
            self.n_features_in_ = X_arr.shape[1]
            self.zero_col_indices_ = [
                FEATURE_ORDER.index(col)
                for col in target_cols
                if col in FEATURE_ORDER and FEATURE_ORDER.index(col) < X_arr.shape[1]
            ]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            target_cols = self.columns if self.columns is not None else ZERO_AS_MISSING_COLUMNS
            for col in target_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float).replace(0.0, np.nan)
            return df

        arr = np.asarray(X, dtype=float).copy()
        for idx in self.zero_col_indices_:
            arr[arr[:, idx] == 0.0, idx] = np.nan
        return arr


def build_preprocessing_steps(scale_features: bool = True) -> list[tuple[str, Any]]:
    """
    Build leakage-free preprocessing steps to prepend to any scikit-learn estimator.

    Steps:
        1. InvalidZeroToNaNTransformer: Converts impossible 0s to NaN.
        2. SimpleImputer(strategy='median'): Fits column medians ONLY on training data.
        3. StandardScaler (optional): Standardizes features ONLY on training data.
    """
    steps: list[tuple[str, Any]] = [
        ("invalid_zero_to_nan", InvalidZeroToNaNTransformer(columns=list(ZERO_AS_MISSING_COLUMNS))),
        ("median_imputer", SimpleImputer(strategy="median")),
    ]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    return steps


def build_model_pipeline(estimator: BaseEstimator, scale_features: bool = True) -> Pipeline:
    """Create a complete end-to-end scikit-learn Pipeline."""
    steps = build_preprocessing_steps(scale_features=scale_features)
    steps.append(("classifier", estimator))
    return Pipeline(steps=steps)


def validate_diabetes_input(payload: Any) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """
    Strictly validate user-submitted clinical inputs for inference.

    Returns:
        (validated_dataframe, errors_dict)

    Design policy:
    - Never silently fill missing measurements with dataset medians or synthetic values.
    - Reject missing, non-numeric, NaN/Inf, out-of-range, or physiologically invalid zero values.
    """
    if not isinstance(payload, dict):
        return None, {"_payload": "Request body must be a valid JSON object containing all 8 required clinical features."}

    errors: dict[str, str] = {}
    cleaned: dict[str, float] = {}

    for spec in DIABETES_FEATURE_SPECS:
        raw_val = payload.get(spec.name)

        if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
            errors[spec.name] = (
                f"{spec.label} ({spec.unit}) is required. Missing clinical measurements "
                "cannot be fabricated for risk assessment."
            )
            continue

        if isinstance(raw_val, bool):
            errors[spec.name] = f"{spec.label} must be a numeric value, not a boolean."
            continue

        try:
            num_val = float(raw_val)
        except (TypeError, ValueError):
            errors[spec.name] = f"{spec.label} must be a valid number."
            continue

        if not np.isfinite(num_val):
            errors[spec.name] = f"{spec.label} must be a finite numeric value."
            continue

        if not spec.allow_zero and num_val == 0:
            errors[spec.name] = (
                f"{spec.label} cannot be 0 {spec.unit}. A zero value is physiologically "
                "invalid; please provide a real measurement."
            )
            continue

        if spec.integer_only and not float(num_val).is_integer():
            errors[spec.name] = f"{spec.label} must be a whole number."
            continue

        if num_val < spec.min_value or num_val > spec.max_value:
            errors[spec.name] = (
                f"{spec.label} must be between {spec.min_value} and {spec.max_value} {spec.unit}."
            )
            continue

        cleaned[spec.name] = int(num_val) if spec.integer_only else round(num_val, 4)

    if errors:
        return None, errors

    df = pd.DataFrame([[cleaned[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return df, {}
