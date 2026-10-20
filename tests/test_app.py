"""
Unit and Integration Test Suite for SmartHousePricePredictor
Tests Flask endpoints, input validation boundaries, model inference,
conformal prediction intervals, Explainable AI telemetry, and error handling.
"""

import json
import pytest
from app import app
from src.validator import FEATURE_NAMES, validate_input_payload
from src.predictor import HousePricePredictor


@pytest.fixture
def client():
    """Create Flask test client fixture."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    """Test that home route renders properly with expected branding and without CDN scripts."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "SmartHousePricePredictor" in html
    assert "California House Price Predictor" in html
    assert "predictorForm" in html
    # Ensure Tailwind CDN warning script is not present
    assert "cdn.tailwindcss.com" not in html
    # Ensure production compiled CSS is linked
    assert "tailwind.min.css" in html


def test_health_check(client):
    """Test /health service readiness endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["features"] == FEATURE_NAMES


def test_api_importance(client):
    """Test /api/importance returns feature weights."""
    response = client.get("/api/importance")
    assert response.status_code == 200
    data = response.get_json()
    assert "MedInc" in data
    assert "Latitude" in data
    assert "Longitude" in data
    assert data["MedInc"] > 0


def test_api_metrics(client):
    """Test /api/metrics returns real held-out evaluation metrics."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "xgboost_model" in data
    assert data["xgboost_model"]["r2_score"] == 0.8393
    assert data["xgboost_model"]["mae_usd"] == 30245.42
    assert data["xgboost_model"]["rmse_usd"] == 45894.34
    assert data["xgboost_model"]["mape_percent"] == 17.52
    assert "baselines" in data
    assert data["baselines"]["linear_regression"]["r2_score"] == 0.5758


def test_api_uncertainty(client):
    """Test /api/uncertainty returns calibrated conformal prediction parameters."""
    response = client.get("/api/uncertainty")
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction_interval_90" in data
    assert data["prediction_interval_90"]["empirical_coverage_pct"] >= 90.0
    assert "prediction_interval_80" in data
    assert data["prediction_interval_80"]["empirical_coverage_pct"] >= 80.0


def test_predict_valid_payload(client):
    """Test successful property valuation calculation with conformal prediction intervals."""
    payload = {
        "MedInc": 8.50,
        "HouseAge": 32.0,
        "AveRooms": 6.2,
        "AveBedrms": 1.05,
        "Population": 1200.0,
        "AveOccup": 2.5,
        "Latitude": 37.77,
        "Longitude": -122.42
    }
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "prediction" in data
    assert data["prediction"] > 50000.0
    assert "range" in data
    assert "lower" in data["range"]
    assert "upper" in data["range"]
    assert data["range"]["lower"] < data["prediction"] < data["range"]["upper"]
    assert "formatted_prediction" in data
    assert "benchmark" in data
    assert data["benchmark"]["dataset_median_usd"] == 206855.0


def test_predict_sacramento_preset(client):
    """Test valuation with Sacramento valley parameters."""
    payload = {
        "MedInc": 3.40,
        "HouseAge": 22.0,
        "AveRooms": 5.2,
        "AveBedrms": 1.05,
        "Population": 1750.0,
        "AveOccup": 3.1,
        "Latitude": 38.58,
        "Longitude": -121.49
    }
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["prediction"] > 50000.0


def test_predict_missing_feature(client):
    """Test rejection when a required feature is absent."""
    payload = {
        "MedInc": 5.0,
        "HouseAge": 25.0
    }
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Missing required fields" in data["error"]


def test_predict_invalid_data_type(client):
    """Test rejection when a feature contains non-numeric data."""
    payload = {
        "MedInc": "not-a-number",
        "HouseAge": 32.0,
        "AveRooms": 6.2,
        "AveBedrms": 1.05,
        "Population": 1200.0,
        "AveOccup": 2.5,
        "Latitude": 37.77,
        "Longitude": -122.42
    }
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "must be a valid numeric" in data["error"]


def test_predict_non_finite_values(client):
    """Test rejection when values are Infinity or NaN."""
    payload = {
        "MedInc": float("inf"),
        "HouseAge": 32.0,
        "AveRooms": 6.2,
        "AveBedrms": 1.05,
        "Population": 1200.0,
        "AveOccup": 2.5,
        "Latitude": 37.77,
        "Longitude": -122.42
    }
    # Direct validator check since JSON does not allow standard Infinity
    parsed, err = validate_input_payload(payload)
    assert parsed is None
    assert "must be a finite number" in err


def test_predict_out_of_bounds_geography(client):
    """Test rejection when coordinates are outside California."""
    payload = {
        "MedInc": 5.0,
        "HouseAge": 30.0,
        "AveRooms": 5.5,
        "AveBedrms": 1.1,
        "Population": 1500.0,
        "AveOccup": 3.0,
        "Latitude": 48.8566,  # Paris latitude
        "Longitude": 2.3522    # Paris longitude
    }
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "out of realistic bounds" in data["error"]


def test_predict_bedrooms_exceeding_rooms(client):
    """Test sanity check rejecting bedrooms > total rooms."""
    payload = {
        "MedInc": 5.0,
        "HouseAge": 30.0,
        "AveRooms": 3.0,
        "AveBedrms": 5.0,  # 5 bedrooms in a 3 room house is invalid
        "Population": 1500.0,
        "AveOccup": 3.0,
        "Latitude": 35.5,
        "Longitude": -119.5
    }
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "cannot exceed Average Total Rooms" in data["error"]


def test_predict_malformed_json(client):
    """Test rejection when request body is not valid JSON."""
    response = client.post(
        "/predict",
        data="this is plain text",
        content_type="text/plain"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "must be valid JSON" in data["error"]


def test_predictor_direct_module():
    """Test direct invocation of HousePricePredictor class."""
    pred = HousePricePredictor()
    assert pred.is_ready() is True
    res = pred.predict([8.5, 32.0, 6.2, 1.05, 1200.0, 2.5, 37.77, -122.42])
    assert "prediction_usd" in res
    assert "prediction_interval_90" in res
    assert res["prediction_interval_90"]["margin_usd"] > 0


def test_favicon_route(client):
    """Test /favicon.ico route serves valid icon without 404."""
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert "image" in response.headers.get("Content-Type", "")


def test_docker_and_ci_config_present():
    """Verify Docker, Compose, and CI/CD workflow configuration files exist and are configured."""
    import os
    assert os.path.exists("Dockerfile")
    assert os.path.exists(".dockerignore")
    assert os.path.exists("docker-compose.yml")
    assert os.path.exists(".env.example")
    assert os.path.exists(".github/workflows/ci.yml")
    with open("Dockerfile", "r", encoding="utf-8") as f:
        content = f.read()
        assert "gunicorn" in content
        assert "HEALTHCHECK" in content
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        compose = f.read()
        assert "5000:5000" in compose
        assert "healthcheck" in compose
    with open(".github/workflows/ci.yml", "r", encoding="utf-8") as f:
        ci = f.read()
        assert "test:" in ci
        assert "docker:" in ci
        assert "HEALTHCHECK" not in ci or "docker build" in ci


def test_cors_headers_and_options(client):
    """Test that CORS headers and OPTIONS preflight requests are properly handled."""
    # OPTIONS preflight
    options_resp = client.open("/predict", method="OPTIONS")
    assert options_resp.status_code == 204
    assert options_resp.headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in options_resp.headers.get("Access-Control-Allow-Methods", "")

    # GET health with CORS
    health_resp = client.get("/health")
    assert health_resp.headers.get("Access-Control-Allow-Origin") == "*"


def test_predict_form_encoded(client):
    """Test that /predict accepts form-encoded data as fallback."""
    payload = {
        "MedInc": "8.50",
        "HouseAge": "32.0",
        "AveRooms": "6.2",
        "AveBedrms": "1.05",
        "Population": "1200.0",
        "AveOccup": "2.5",
        "Latitude": "37.77",
        "Longitude": "-122.42"
    }
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["prediction"] > 50000.0

