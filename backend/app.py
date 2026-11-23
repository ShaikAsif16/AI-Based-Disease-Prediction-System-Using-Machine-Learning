"""
Flask Application Entrypoint — AI-Based Disease Prediction System
=================================================================
Student: Shaik Asif (2403A52220) | Subject: AIDL

Runs the REST API serving the genuinely trained scikit-learn diabetes risk
screening pipeline.

Usage:
    python backend/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify
from flask_cors import CORS

from backend.ml.predictor import DiseaseModelRegistry
from backend.routes.api import api_bp


def create_app(load_models: bool = True) -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["REPORTS_FIGURES_DIR"] = PROJECT_ROOT / "reports" / "figures"

    # Enable CORS for local React development and deployment origins
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": [
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                    "http://localhost:4173",
                    "http://127.0.0.1:4173",
                ]
            }
        },
    )

    registry = DiseaseModelRegistry()
    if load_models:
        registry.load_all()
    app.config["MODEL_REGISTRY"] = registry

    app.register_blueprint(api_bp)

    @app.route("/", methods=["GET"])
    def index():
        return (
            jsonify(
                {
                    "name": "AI-Based Disease Prediction System API",
                    "student": "Shaik Asif",
                    "hall_ticket": "2403A52220",
                    "subject": "Artificial Intelligence & Deep Learning (AIDL)",
                    "endpoints": [
                        "GET  /api/health",
                        "GET  /api/model-info",
                        "GET  /api/model-metrics",
                        "GET  /api/sample-inputs",
                        "POST /api/predict",
                    ],
                }
            ),
            200,
        )

    @app.errorhandler(404)
    def handle_404(_error):
        return jsonify({"error": "not_found", "message": "The requested API endpoint does not exist."}), 404

    @app.errorhandler(405)
    def handle_405(_error):
        return jsonify({"error": "method_not_allowed", "message": "HTTP method not allowed for this endpoint."}), 405

    @app.errorhandler(500)
    def handle_500(_error):
        return jsonify({"error": "internal_server_error", "message": "An unexpected server error occurred."}), 500

    return app


app = create_app(load_models=True)


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"Starting AI-Based Disease Prediction API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
