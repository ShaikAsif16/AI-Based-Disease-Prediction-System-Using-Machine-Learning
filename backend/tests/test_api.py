"""Automated Flask API integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app(load_models=True)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "diabetes" in data["active_diseases"]


def test_model_info_endpoint(client):
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["disease_id"] == "diabetes"
    assert len(data["features"]) == 8
    assert data["selected_model_name"] is not None


def test_model_metrics_endpoint(client):
    resp = client.get("/api/model-metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["dataset_summary"]["total_samples"] == 768
    assert len(data["models"]) == 4
    selected_key = data["selected_model_key"]
    cm = data["models"][selected_key]["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 154


def test_predict_valid_low_and_high_risk_profiles(client):
    low_payload = {
        "Pregnancies": 1,
        "Glucose": 92,
        "BloodPressure": 68,
        "SkinThickness": 21,
        "Insulin": 64,
        "BMI": 22.8,
        "DiabetesPedigreeFunction": 0.215,
        "Age": 25,
    }
    high_payload = {
        "Pregnancies": 6,
        "Glucose": 176,
        "BloodPressure": 86,
        "SkinThickness": 36,
        "Insulin": 215,
        "BMI": 35.4,
        "DiabetesPedigreeFunction": 0.782,
        "Age": 52,
    }

    r_low = client.post("/api/predict", json=low_payload)
    r_high = client.post("/api/predict", json=high_payload)

    assert r_low.status_code == 200
    assert r_high.status_code == 200

    d_low = r_low.get_json()
    d_high = r_high.get_json()

    assert d_low["data_stored"] is False
    assert d_low["prediction"] in (0, 1)
    assert d_high["prediction"] in (0, 1)
    assert d_low["supports_probability"] is True
    assert d_high["probability"] > d_low["probability"]
    assert "Educational Screening Demonstration" in d_low["disclaimer"]


def test_predict_rejects_missing_and_invalid_inputs(client):
    # Missing required fields
    r1 = client.post("/api/predict", json={"Pregnancies": 2, "Glucose": 115})
    assert r1.status_code == 400
    d1 = r1.get_json()
    assert d1["error"] == "validation_error"
    assert "BMI" in d1["field_errors"]

    # Physiologically impossible zero value
    r2 = client.post(
        "/api/predict",
        json={
            "Pregnancies": 0,
            "Glucose": 0,
            "BloodPressure": 70,
            "SkinThickness": 22,
            "Insulin": 85,
            "BMI": 24.5,
            "DiabetesPedigreeFunction": 0.32,
            "Age": 29,
        },
    )
    assert r2.status_code == 400
    d2 = r2.get_json()
    assert "Glucose" in d2["field_errors"]

    # Non-JSON request
    r3 = client.post("/api/predict", data="not json", content_type="text/plain")
    assert r3.status_code == 400


def test_sample_inputs_endpoint(client):
    resp = client.get("/api/sample-inputs")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["profiles"]) == 3


def test_spa_routes_and_api_404(client):
    # Built frontend/dist/index.html is served on root and SPA client routes
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert b'id="root"' in r_root.data

    r_spa = client.get("/analytics")
    assert r_spa.status_code == 200
    assert b'id="root"' in r_spa.data

    # Unknown /api/* endpoints still return structured JSON 404
    r_api_404 = client.get("/api/nonexistent")
    assert r_api_404.status_code == 404
    assert r_api_404.get_json()["error"] == "not_found"
