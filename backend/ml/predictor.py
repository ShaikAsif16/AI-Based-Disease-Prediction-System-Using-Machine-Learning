"""
Model Registry & Real-Time Inference Service
============================================
Provides an extensible multi-disease predictor registry initialized with the
trained Diabetes Risk Screening pipeline (`models/diabetes_pipeline.joblib`).

Key Guarantees:
- Every prediction is generated strictly by the real fitted scikit-learn pipeline.
- Missing or invalid clinical inputs are rejected (never fabricated).
- Submitted user health metrics are processed purely in-memory and never persisted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from backend.ml.preprocessing import (
    DIABETES_FEATURE_SPECS,
    FEATURE_ORDER,
    validate_diabetes_input,
)

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_PIPELINE_PATH = MODELS_DIR / "diabetes_pipeline.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "model_metadata.json"
DEFAULT_METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"


class InputValidationError(ValueError):
    """Raised when user clinical inputs fail strict schema or range validation."""

    def __init__(self, errors: dict[str, str], message: str = "Invalid clinical input values."):
        super().__init__(message)
        self.errors = errors
        self.message = message


class ModelNotLoadedError(RuntimeError):
    """Raised when a trained model artifact is missing and must be generated first."""


def _evaluate_clinical_context(inputs: dict[str, float | int]) -> list[dict[str, Any]]:
    """
    Compare submitted inputs against widely cited general clinical reference ranges
    for educational context only (does not alter ML inference).
    """
    glucose = float(inputs["Glucose"])
    bp = float(inputs["BloodPressure"])
    bmi = float(inputs["BMI"])
    insulin = float(inputs["Insulin"])
    age = float(inputs["Age"])
    dpf = float(inputs["DiabetesPedigreeFunction"])
    skin = float(inputs["SkinThickness"])
    preg = int(inputs["Pregnancies"])

    def status_glucose(v: float) -> tuple[str, str]:
        if v < 140:
            return "within_reference", "Within typical 2-hour OGTT reference (< 140 mg/dL)"
        if v < 200:
            return "borderline", "Within 2-hour OGTT prediabetes reference band (140–199 mg/dL)"
        return "elevated", "Above standard 2-hour OGTT reference threshold (≥ 200 mg/dL)"

    def status_bp(v: float) -> tuple[str, str]:
        if 60 <= v < 80:
            return "within_reference", "Within normal diastolic reference range (60–79 mm Hg)"
        if 80 <= v < 90:
            return "borderline", "Mildly elevated diastolic blood pressure (80–89 mm Hg)"
        if v < 60:
            return "borderline", "Below typical resting diastolic range (< 60 mm Hg)"
        return "elevated", "Elevated diastolic blood pressure (≥ 90 mm Hg)"

    def status_bmi(v: float) -> tuple[str, str]:
        if 18.5 <= v < 25.0:
            return "within_reference", "Within standard healthy BMI range (18.5–24.9 kg/m²)"
        if 25.0 <= v < 30.0:
            return "borderline", "Within overweight BMI reference range (25.0–29.9 kg/m²)"
        if v < 18.5:
            return "borderline", "Below standard BMI range (< 18.5 kg/m²)"
        return "elevated", "Within obese BMI reference range (≥ 30.0 kg/m²)"

    def status_insulin(v: float) -> tuple[str, str]:
        if 16 <= v <= 166:
            return "within_reference", "Within typical 2-hour post-load serum insulin range (16–166 µU/mL)"
        return "borderline", "Outside typical 2-hour post-load serum insulin band (16–166 µU/mL)"

    def status_dpf(v: float) -> tuple[str, str]:
        if v < 0.5:
            return "within_reference", "Lower hereditary predisposition score relative to cohort"
        if v < 1.0:
            return "borderline", "Moderate hereditary predisposition score"
        return "elevated", "Higher hereditary predisposition score relative to cohort"

    g_status, g_note = status_glucose(glucose)
    bp_status, bp_note = status_bp(bp)
    bmi_status, bmi_note = status_bmi(bmi)
    ins_status, ins_note = status_insulin(insulin)
    dpf_status, dpf_note = status_dpf(dpf)

    return [
        {"feature": "Glucose", "label": "Plasma Glucose", "value": glucose, "unit": "mg/dL", "status": g_status, "note": g_note},
        {"feature": "BMI", "label": "Body Mass Index", "value": bmi, "unit": "kg/m²", "status": bmi_status, "note": bmi_note},
        {"feature": "BloodPressure", "label": "Diastolic Blood Pressure", "value": bp, "unit": "mm Hg", "status": bp_status, "note": bp_note},
        {"feature": "Insulin", "label": "2-Hour Serum Insulin", "value": insulin, "unit": "µU/mL", "status": ins_status, "note": ins_note},
        {"feature": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Function", "value": dpf, "unit": "score", "status": dpf_status, "note": dpf_note},
        {"feature": "Age", "label": "Age", "value": age, "unit": "years", "status": "informational", "note": "Age is a known non-modifiable metabolic risk factor"},
        {"feature": "SkinThickness", "label": "Triceps Skinfold Thickness", "value": skin, "unit": "mm", "status": "informational", "note": "Indirect measure of subcutaneous adiposity"},
        {"feature": "Pregnancies", "label": "Number of Pregnancies", "value": preg, "unit": "count", "status": "informational", "note": "Included because the model was trained on the Pima female cohort"},
    ]


def _classify_risk_band(probability: float | None, predicted_class: int) -> dict[str, Any]:
    """Translate model probability & predicted class into cautious clinical screening copy."""
    if probability is None:
        if predicted_class == 1:
            return {
                "category": "Elevated Estimated Risk",
                "level": "elevated",
                "headline": "Higher Estimated Diabetes Risk Pattern Detected",
                "summary": (
                    "Based on the submitted measurements, the trained machine learning model "
                    "classified this profile into the higher-risk screening category. This is a "
                    "statistical screening indicator, not a medical diagnosis."
                ),
            }
        return {
            "category": "Lower Estimated Risk",
            "level": "low",
            "headline": "Lower Estimated Diabetes Risk Pattern Detected",
            "summary": (
                "Based on the submitted measurements, the trained machine learning model "
                "classified this profile into the lower-risk screening category. Routine "
                "preventive health checkups remain important."
            ),
        }

    if probability >= 0.65:
        return {
            "category": "Elevated Estimated Risk",
            "level": "elevated",
            "headline": "Elevated Statistical Risk Pattern Identified",
            "summary": (
                f"The trained model estimated a {probability * 100:.1f}% statistical likelihood of "
                "matching patterns associated with diabetes in the training cohort. Because this tool "
                "is an educational screening model and not a diagnostic test, please discuss these "
                "measurements with a qualified healthcare professional for proper clinical evaluation "
                "(such as HbA1c or fasting plasma glucose testing)."
            ),
        }
    if probability >= 0.35:
        return {
            "category": "Moderate Estimated Risk",
            "level": "moderate",
            "headline": "Moderate / Borderline Statistical Risk Pattern Identified",
            "summary": (
                f"The trained model estimated a {probability * 100:.1f}% statistical likelihood of "
                "matching positive diabetes patterns in the training cohort. Borderline profiles warrant "
                "attentive lifestyle habits and discussion with a physician during routine screening."
            ),
        }
    return {
        "category": "Low Estimated Risk",
        "level": "low",
        "headline": "Lower Statistical Risk Pattern Identified",
        "summary": (
            f"The trained model estimated a {probability * 100:.1f}% statistical likelihood of "
            "matching positive diabetes patterns in the training cohort. A lower estimated risk score "
            "does not guarantee the absence of diabetes; regular clinical checkups remain advisable."
        ),
    }


class DiabetesPredictor:
    """Loads and serves the trained scikit-learn diabetes risk pipeline."""

    def __init__(
        self,
        pipeline_path: Path = DEFAULT_PIPELINE_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        metrics_path: Path = DEFAULT_METRICS_PATH,
    ) -> None:
        self.pipeline_path = pipeline_path
        self.metadata_path = metadata_path
        self.metrics_path = metrics_path
        self._pipeline: Any = None
        self._metadata: dict[str, Any] | None = None
        self._metrics: dict[str, Any] | None = None

    def load(self) -> None:
        """Load pipeline and evaluation metadata from disk."""
        if not self.pipeline_path.exists():
            raise ModelNotLoadedError(
                f"Trained pipeline not found at '{self.pipeline_path}'. "
                "Run `python backend/ml/train.py` to train and save the model first."
            )
        self._pipeline = joblib.load(self.pipeline_path)

        if self.metadata_path.exists():
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metrics_path.exists():
            self._metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def get_metadata(self) -> dict[str, Any]:
        self.ensure_loaded()
        if self._metadata is None:
            raise ModelNotLoadedError("Model metadata JSON not found. Re-run `python backend/ml/train.py`.")
        return self._metadata

    def get_metrics(self) -> dict[str, Any]:
        self.ensure_loaded()
        if self._metrics is None:
            raise ModelNotLoadedError("Model evaluation metrics JSON not found. Re-run `python backend/ml/train.py`.")
        return self._metrics

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate clinical features and execute real inference on the loaded ML pipeline."""
        self.ensure_loaded()

        df_input, errors = validate_diabetes_input(payload)
        if errors or df_input is None:
            raise InputValidationError(
                errors=errors,
                message="Please correct the missing or invalid clinical measurements before predicting.",
            )

        predicted_class = int(self._pipeline.predict(df_input)[0])
        probability: float | None = None
        class_probabilities: dict[str, float] | None = None

        if hasattr(self._pipeline, "predict_proba"):
            proba_arr = self._pipeline.predict_proba(df_input)[0]
            neg_prob = round(float(proba_arr[0]), 4)
            pos_prob = round(float(proba_arr[1]), 4)
            probability = pos_prob
            class_probabilities = {
                "negative": neg_prob,
                "positive": pos_prob,
            }

        risk_info = _classify_risk_band(probability=probability, predicted_class=predicted_class)
        submitted_values = {
            col: (int(df_input[col].iloc[0]) if col in ("Pregnancies", "Age") else float(df_input[col].iloc[0]))
            for col in FEATURE_ORDER
        }
        clinical_context = _evaluate_clinical_context(submitted_values)

        metadata = self.get_metadata()

        return {
            "disease": "diabetes",
            "model_used": metadata.get("selected_model_name", "Trained ML Pipeline"),
            "model_key": metadata.get("selected_model_key", "unknown"),
            "prediction": predicted_class,
            "prediction_label": (
                "Elevated Diabetes Risk Indicated"
                if predicted_class == 1
                else "Lower Diabetes Risk Indicated"
            ),
            "probability": probability,
            "probability_percentage": round(probability * 100.0, 2) if probability is not None else None,
            "class_probabilities": class_probabilities,
            "supports_probability": probability is not None,
            "risk_category": risk_info["category"],
            "risk_level": risk_info["level"],
            "headline": risk_info["headline"],
            "explanation": risk_info["summary"],
            "submitted_inputs": submitted_values,
            "clinical_context": clinical_context,
            "limitations": [
                "Population scope: The model was trained on the Pima Indians Diabetes Database (768 female patients of Pima Indian heritage aged 21 and older) and may not generalize equally across all demographic groups or sexes.",
                "Screening vs. diagnosis: Machine learning risk estimates cannot replace standardized diagnostic laboratory tests such as HbA1c, fasting plasma glucose, or physician-supervised oral glucose tolerance testing.",
                "Feature constraints: The model considers only 8 clinical attributes and does not incorporate diet, physical activity, medications, symptoms, lipid profiles, or longitudinal trends.",
                "Statistical uncertainty: Even well-validated models produce both false positives and false negatives; probability scores reflect historical cohort patterns rather than clinical certainty.",
            ],
            "general_health_information": [
                "Maintain regular physical activity (e.g., 150 minutes/week of moderate aerobic exercise as recommended by major health organizations, subject to medical clearance).",
                "Prioritize balanced nutrition rich in fiber, whole foods, lean proteins, and limited refined sugars and ultra-processed carbohydrates.",
                "Monitor key metabolic indicators periodically—including fasting blood glucose, HbA1c, blood pressure, and body weight—with your primary care provider.",
                "Ensure consistent sleep quality and stress management, both of which influence insulin sensitivity.",
            ],
            "professional_recommendation": (
                "Always consult a qualified physician or licensed healthcare provider to interpret "
                "laboratory measurements and determine whether formal diagnostic testing or medical "
                "management is appropriate for you."
            ),
            "disclaimer": (
                "Educational Screening Demonstration Only — Not a Medical Diagnosis. "
                "This AI-Based Disease Prediction System is developed for academic demonstration "
                "(AIDL Project) and does not provide medical advice, diagnosis, or treatment. "
                "No submitted health information is stored by the server."
            ),
            "data_stored": False,
        }


class DiseaseModelRegistry:
    """
    Registry managing disease prediction models.
    Starts with `diabetes` and provides a clean extension interface for additional
    disease models (e.g., cardiovascular disease, chronic kidney disease).
    """

    def __init__(self) -> None:
        self._predictors: dict[str, DiabetesPredictor] = {
            "diabetes": DiabetesPredictor(),
        }

    def get(self, disease_id: str = "diabetes") -> DiabetesPredictor:
        key = (disease_id or "diabetes").strip().lower()
        if key not in self._predictors:
            supported = ", ".join(sorted(self._predictors.keys()))
            raise KeyError(f"Unsupported disease model '{disease_id}'. Currently active: {supported}.")
        return self._predictors[key]

    def load_all(self) -> dict[str, bool]:
        status: dict[str, bool] = {}
        for disease_id, predictor in self._predictors.items():
            try:
                predictor.load()
                status[disease_id] = True
            except Exception:
                status[disease_id] = False
        return status

    def active_diseases(self) -> list[str]:
        return list(self._predictors.keys())


# Sample clinical profiles for interactive demonstration (always evaluated live by the ML model)
SAMPLE_PROFILES: list[dict[str, Any]] = [
    {
        "id": "low_risk_demo",
        "title": "Lower-Risk Reference Profile",
        "description": "Young adult with normal 2-hour glucose, healthy BMI, normal diastolic pressure, and low family pedigree score.",
        "inputs": {
            "Pregnancies": 1,
            "Glucose": 92,
            "BloodPressure": 68,
            "SkinThickness": 21,
            "Insulin": 64,
            "BMI": 22.8,
            "DiabetesPedigreeFunction": 0.215,
            "Age": 25,
        },
    },
    {
        "id": "borderline_demo",
        "title": "Borderline Metabolic Profile",
        "description": "Middle-aged adult with mildly elevated post-load glucose, overweight BMI, and moderate family history score.",
        "inputs": {
            "Pregnancies": 3,
            "Glucose": 134,
            "BloodPressure": 78,
            "SkinThickness": 30,
            "Insulin": 130,
            "BMI": 29.6,
            "DiabetesPedigreeFunction": 0.485,
            "Age": 39,
        },
    },
    {
        "id": "elevated_risk_demo",
        "title": "Elevated-Risk Clinical Profile",
        "description": "Older adult with elevated 2-hour plasma glucose, elevated BMI, higher insulin, and stronger hereditary pedigree score.",
        "inputs": {
            "Pregnancies": 6,
            "Glucose": 176,
            "BloodPressure": 86,
            "SkinThickness": 36,
            "Insulin": 215,
            "BMI": 35.4,
            "DiabetesPedigreeFunction": 0.782,
            "Age": 52,
        },
    },
]
