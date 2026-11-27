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

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from backend.ml.predictor import DiseaseModelRegistry
from backend.routes.api import api_bp


def create_app(load_models: bool = True) -> Flask:
    """Create and configure the Flask application instance."""
    frontend_dist_dir = PROJECT_ROOT / "frontend" / "dist"
    app = Flask(__name__, static_folder=None)
    app.config["JSON_SORT_KEYS"] = False
    app.config["REPORTS_FIGURES_DIR"] = PROJECT_ROOT / "reports" / "figures"
    app.config["FRONTEND_DIST_DIR"] = frontend_dist_dir

    # Enable CORS for local React development and optional external origins
    default_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]
    extra_origins = [
        o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
    ]
    CORS(
        app,
        resources={r"/api/*": {"origins": default_origins + extra_origins}},
    )

    registry = DiseaseModelRegistry()
    if load_models:
        registry.load_all()
    app.config["MODEL_REGISTRY"] = registry

    app.register_blueprint(api_bp)

    @app.route("/", methods=["GET"])
    def index():
        dist_dir: Path = app.config["FRONTEND_DIST_DIR"]
        if (dist_dir / "index.html").is_file():
            return send_from_directory(dist_dir, "index.html")
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

    @app.route("/<path:asset_path>", methods=["GET"])
    def serve_spa_or_asset(asset_path: str):
        if asset_path.startswith("api/") or asset_path == "api":
            return jsonify({"error": "not_found", "message": "The requested API endpoint does not exist."}), 404
        dist_dir: Path = app.config["FRONTEND_DIST_DIR"]
        if not (dist_dir / "index.html").is_file():
            return jsonify({"error": "not_found", "message": "The requested endpoint does not exist."}), 404
        candidate = (dist_dir / asset_path).resolve()
        if candidate.is_file() and dist_dir.resolve() in candidate.parents:
            return send_from_directory(dist_dir, asset_path)
        return send_from_directory(dist_dir, "index.html")

    @app.errorhandler(404)
    def handle_404(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not_found", "message": "The requested API endpoint does not exist."}), 404
        dist_dir: Path = app.config["FRONTEND_DIST_DIR"]
        if (dist_dir / "index.html").is_file():
            return send_from_directory(dist_dir, "index.html")
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
    default_host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    host = os.environ.get("FLASK_HOST", default_host)
    port = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", "5000")))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"Starting AI-Based Disease Prediction API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
