"""
Flask REST API Endpoints for AI-Based Disease Prediction System
===============================================================
Endpoints:
- GET  /api/health
- POST /api/predict
- GET  /api/model-info
- GET  /api/model-metrics
- GET  /api/sample-inputs
- GET  /api/figures/<filename>

Privacy Note:
Submitted patient/user health metrics are validated and scored strictly in
memory and are never written to disk, databases, or log files.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from backend.ml.predictor import (
    DiseaseModelRegistry,
    InputValidationError,
    ModelNotLoadedError,
    SAMPLE_PROFILES,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_FIGURES = {
    "eda_class_and_zeros.png",
    "eda_feature_distributions.png",
    "eda_correlation_heatmap.png",
    "model_comparison.png",
    "confusion_matrices.png",
    "roc_curves.png",
}


def _get_registry() -> DiseaseModelRegistry:
    return current_app.config["MODEL_REGISTRY"]


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Return service health and loaded model status."""
    registry = _get_registry()
    diabetes_predictor = registry.get("diabetes")
    model_loaded = diabetes_predictor.is_loaded
    model_name = None

    if model_loaded:
        try:
            model_name = diabetes_predictor.get_metadata().get("selected_model_name")
        except Exception:
            model_name = None

    status_code = 200 if model_loaded else 503
    return (
        jsonify(
            {
                "status": "healthy" if model_loaded else "degraded",
                "service": "AI-Based Disease Prediction API",
                "model_loaded": model_loaded,
                "active_model": model_name,
                "active_diseases": registry.active_diseases(),
                "data_persistence": "disabled (in-memory inference only)",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        ),
        status_code,
    )


@api_bp.route("/model-info", methods=["GET"])
def get_model_info():
    """Return metadata, input feature schema, and configuration of the active trained model."""
    disease_id = request.args.get("disease", "diabetes")
    registry = _get_registry()
    try:
        predictor = registry.get(disease_id)
        metadata = predictor.get_metadata()
        return jsonify(metadata), 200
    except KeyError as exc:
        return jsonify({"error": "unsupported_disease", "message": str(exc)}), 404
    except ModelNotLoadedError as exc:
        return jsonify({"error": "model_unavailable", "message": str(exc)}), 503


@api_bp.route("/model-metrics", methods=["GET"])
def get_model_metrics():
    """Return genuine training, cross-validation, and hold-out test evaluation metrics."""
    disease_id = request.args.get("disease", "diabetes")
    registry = _get_registry()
    try:
        predictor = registry.get(disease_id)
        metrics = predictor.get_metrics()
        return jsonify(metrics), 200
    except KeyError as exc:
        return jsonify({"error": "unsupported_disease", "message": str(exc)}), 404
    except ModelNotLoadedError as exc:
        return jsonify({"error": "model_unavailable", "message": str(exc)}), 503


@api_bp.route("/sample-inputs", methods=["GET"])
def get_sample_inputs():
    """Return sample demo profiles to populate the UI form for live ML inference."""
    return (
        jsonify(
            {
                "note": (
                    "These sample clinical profiles are provided strictly to demonstrate live "
                    "machine learning inference. No prediction outputs are precomputed or hardcoded."
                ),
                "profiles": SAMPLE_PROFILES,
            }
        ),
        200,
    )


@api_bp.route("/predict", methods=["POST"])
def predict_disease():
    """Validate clinical measurements and run real inference through the trained ML pipeline."""
    if not request.is_json:
        return (
            jsonify(
                {
                    "error": "invalid_content_type",
                    "message": "Request must use Content-Type: application/json with a valid JSON body.",
                }
            ),
            400,
        )

    payload = request.get_json(silent=True)
    if payload is None:
        return (
            jsonify(
                {
                    "error": "malformed_json",
                    "message": "Could not parse JSON payload. Please check your request syntax.",
                }
            ),
            400,
        )

    disease_id = "diabetes"
    if isinstance(payload, dict) and "disease" in payload:
        disease_id = str(payload.get("disease", "diabetes"))

    registry = _get_registry()
    try:
        predictor = registry.get(disease_id)
        result = predictor.predict(payload)
        return jsonify(result), 200
    except KeyError as exc:
        return jsonify({"error": "unsupported_disease", "message": str(exc)}), 404
    except InputValidationError as exc:
        return (
            jsonify(
                {
                    "error": "validation_error",
                    "message": exc.message,
                    "field_errors": exc.errors,
                }
            ),
            400,
        )
    except ModelNotLoadedError as exc:
        return jsonify({"error": "model_unavailable", "message": str(exc)}), 503
    except Exception:
        return (
            jsonify(
                {
                    "error": "inference_error",
                    "message": "An unexpected internal error occurred during model inference.",
                }
            ),
            500,
        )


@api_bp.route("/figures/<path:filename>", methods=["GET"])
def get_report_figure(filename: str):
    """Serve generated Matplotlib/Seaborn evaluation plots safely."""
    if filename not in ALLOWED_FIGURES:
        return jsonify({"error": "not_found", "message": "Requested report figure does not exist."}), 404

    figures_dir: Path = current_app.config["REPORTS_FIGURES_DIR"]
    if not (figures_dir / filename).exists():
        return jsonify({"error": "not_found", "message": "Figure has not been generated yet."}), 404

    return send_from_directory(figures_dir, filename, mimetype="image/png")
