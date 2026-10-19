"""
SmartHousePricePredictor — Flask Application Entry Point
Production-ready web service providing property valuation predictions,
Explainable AI (XAI) feature importances, real model evaluation telemetry,
and strict server-side schema validation.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.validator import FEATURE_NAMES, validate_input_payload
from src.predictor import HousePricePredictor

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SmartHousePredictor")

app = Flask(__name__)

# Initialize Predictor instance
predictor = HousePricePredictor()


@app.after_request
def add_cors_headers(response):
    """Enable Cross-Origin Resource Sharing (CORS) for external frontends and local dev servers."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/favicon.ico")
def favicon():
    """Serves the application favicon."""
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.svg",
        mimetype="image/svg+xml"
    )


@app.route("/")
def home():
    """Serves the primary user interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    """Service health and readiness check for Render, Docker, and monitoring."""
    if request.method == "OPTIONS":
        return "", 204
    is_ready = predictor.is_ready()
    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "features": FEATURE_NAMES
    }), (200 if is_ready else 503)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Processes valuation requests via JSON payload (or form data).
    Validates input schema, performs inference via XGBoost, converts target
    variable to USD ($100k scale), and returns estimate with Split Conformal Prediction Intervals.
    """
    if request.method == "OPTIONS":
        return "", 204

    if not predictor.is_ready():
        return jsonify({
            "success": False,
            "error": "Model artifact is currently unavailable on the server. Please ensure models/model.pkl is trained."
        }), 503

    try:
        data = request.get_json(silent=True)
        if data is None and request.form:
            data = request.form.to_dict()

        if data is None:
            return jsonify({
                "success": False,
                "error": "Request body must be valid JSON with 'Content-Type: application/json'."
            }), 400

        features, validation_error = validate_input_payload(data)
        if validation_error:
            return jsonify({
                "success": False,
                "error": validation_error
            }), 400

        result = predictor.predict(features)

        # Backwards compatible response schema for existing client controllers
        response_payload = {
            "success": True,
            "prediction": result["prediction_usd"],
            "formatted_prediction": result["formatted_prediction"],
            "prediction_interval": result["prediction_interval_90"],
            "range": {
                "lower": result["prediction_interval_90"]["lower_usd"],
                "upper": result["prediction_interval_90"]["upper_usd"],
                "lower_formatted": result["prediction_interval_90"]["lower_formatted"],
                "upper_formatted": result["prediction_interval_90"]["upper_formatted"],
                "confidence_interval": "90% Conformal Prediction Interval (±$68,169)",
                "margin_usd": result["prediction_interval_90"]["margin_usd"]
            },
            "benchmark": result["benchmark_comparison"],
            "features_used": result["features_used"]
        }

        return jsonify(response_payload), 200

    except Exception as exc:
        logger.exception("Prediction calculation error: %s", exc)
        return jsonify({
            "success": False,
            "error": "An internal error occurred while calculating the property valuation."
        }), 500


@app.route("/api/importance", methods=["GET", "OPTIONS"])
def importance():
    """Returns model feature importance weights for Explainable AI (XAI) telemetry."""
    if request.method == "OPTIONS":
        return "", 204
    return jsonify(predictor.get_importance()), 200


@app.route("/api/metrics", methods=["GET", "OPTIONS"])
def metrics():
    """Returns verified model evaluation metrics computed on held-out test data."""
    if request.method == "OPTIONS":
        return "", 204
    metrics_data = predictor.get_metrics()
    if metrics_data:
        return jsonify(metrics_data), 200

    return jsonify({
        "error": "Metrics file not found. Run train_model.py to generate."
    }), 404


@app.route("/api/uncertainty", methods=["GET", "OPTIONS"])
def uncertainty():
    """Returns conformal prediction interval parameters and empirical coverage."""
    if request.method == "OPTIONS":
        return "", 204
    return jsonify(predictor.uncertainty_data), 200


if __name__ == "__main__":
    # Render binds dynamically to the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting SmartHousePricePredictor on port %d (debug=%s)", port, debug_mode)
    app.run(host="0.0.0.0", port=port, debug=debug_mode)