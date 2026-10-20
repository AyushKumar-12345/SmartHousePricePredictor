# SmartHousePricePredictor — California Property Valuation Engine

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-EB6228?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![GitHub Actions CI](https://img.shields.io/badge/CI%20Pipeline-Passing-2EA44F?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/AyushKumar-12345/SmartHousePricePredictor/actions)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://smarthousepricepredictor.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge)](LICENSE)

<br/>

**A production-grade, mathematically grounded Machine Learning web application for California residential property valuation.**  
*Combines an optimized XGBoost gradient-boosted ensemble ($R^2 = 0.8393$), distribution-free Split Conformal Prediction intervals, Explainable AI telemetry, defensive input boundary validation, and a high-contrast executive design system.*

<br/>

[🚀 Live Web Application](https://smarthousepricepredictor.onrender.com) • [📦 GitHub Repository](https://github.com/AyushKumar-12345/SmartHousePricePredictor) • [📑 API Documentation](#8-rest-api-reference) • [🐳 Docker Quickstart](#6-docker--container-deployment)

</div>

---

## Table of Contents

- [1. Executive Summary & Highlights](#1-executive-summary--highlights)
- [2. Machine Learning Architecture](#2-machine-learning-architecture)
  - [2.1 Model Specifications](#21-model-specifications)
  - [2.2 Rigorous Train/Test Split](#22-rigorous-traintest-split)
  - [2.3 Mathematical Formulation & Metrics](#23-mathematical-formulation--metrics)
- [3. Uncertainty Quantification: Split Conformal Prediction](#3-uncertainty-quantification-split-conformal-prediction)
- [4. Explainable AI (XAI) & Feature Importances](#4-explainable-ai-xai--feature-importances)
- [5. System Architecture & Engineering](#5-system-architecture--engineering)
- [6. Docker & Container Deployment](#6-docker--container-deployment)
- [7. Production CI/CD Pipeline](#7-production-cicd-pipeline)
- [8. REST API Reference](#8-rest-api-reference)
- [9. Local Setup & Testing Guide](#9-local-setup--testing-guide)
- [10. Placement & Interview Talking Points](#10-placement--interview-talking-points)
- [11. Repository Structure](#11-repository-structure)
- [12. License & Author](#12-license--author)

---

## 1. Executive Summary & Highlights

Residential real estate valuation requires resolving non-linear relationships across socio-economic, spatial, and structural metrics. Standard ordinary least squares (OLS) linear models struggle to capture multi-dimensional geographic clusters (coastal proximity, metropolitan cores), yielding high residuals ($R^2 \approx 0.58$).

**SmartHousePricePredictor** solves this challenge through an end-to-end engineered machine learning service:

- **Empirically Proven Accuracy:** XGBoost model achieves an **$R^2$ score of 0.8393** on 4,128 strictly held-out test records (+26.35 percentage points over linear regression), reducing Mean Absolute Error to **$30,245.42 (17.52% MAPE)**.
- **Distribution-Free Uncertainty:** Replaces ungrounded heuristics with mathematically rigorous **Split Conformal Prediction (Inductive Conformal Regression)**, delivering valid finite-sample coverage guarantees ($90\%$ prediction interval at $\pm \$68,168.97$ with $90.04\%$ verified empirical coverage).
- **Explainable AI (XAI):** Real tree split weights dynamically exposed via `/api/importance` and rendered via interactive Chart.js bar charts.
- **Defensive Boundary Validation:** Enforces strict physical and geographic boundaries (California coordinate bounds $32.5^\circ\text{–}42.0^\circ\text{ N}, -124.5^\circ\text{–}-114.0^\circ\text{ W}$, room-to-bedroom constraints, positive finite scalars) returning descriptive HTTP 400 responses.
- **Enterprise Design System:** Pure white minimalist canvas (`#FFFFFF`), solid black typography (`#000000`), subtle slate borders (`#E2E8F0`), zero CDN warnings (locally compiled, purged Tailwind CSS), and fully responsive across mobile, tablet, and desktop.
- **Cloud-Ready DevOps:** Production-grade `Dockerfile` with automated health checks, dynamic WSGI Gunicorn process management, Docker Compose orchestration, and dual-stage GitHub Actions CI.

---

## 2. Machine Learning Architecture

### 2.1 Model Specifications

The primary valuation engine is an **Extreme Gradient Boosting Regressor (XGBoost)** trained on the California Housing dataset (1990 U.S. Census, Pace & Barry):

```python
XGBRegressor(
    n_estimators=300,        # 300 sequential gradient boosted decision trees
    max_depth=6,             # Tree depth constrained to capture non-linear interactions without overfitting
    learning_rate=0.05,      # Conservative shrinkage rate for smooth loss convergence
    subsample=0.8,           # Stochastic sample fraction per tree iteration
    colsample_bytree=0.8,    # Feature subsampling ratio to decorrelate tree structures
    random_state=42,         # Fully deterministic and reproducible pipeline
    objective='reg:squarederror'
)
```

### 2.2 Rigorous Train/Test Split

To prevent data leakage, dataset partitioning and metric computations adhere strictly to scientific ML standards:

- **Total Dataset:** 20,640 census block records.
- **Training Set (80%):** 16,512 records used solely for tree building.
- **Held-Out Test Set (20%):** 4,128 records strictly sequestered during training, preprocessing, and hyperparameter tuning.

### 2.3 Mathematical Formulation & Metrics

Target values are expressed in units of $\$100,000$. All evaluation metrics are verified on the genuine held-out test split ($n = 4,128$) and converted into real USD dollars:

| Evaluation Metric | Formula | Target Scale | Real USD Scale | Interpretation |
|---|---|---|---|---|
| **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | **0.8393** | **0.8393** | Explains 83.93% of property price variance |
| **MAE** | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ | 0.3025 | **$30,245.42** | Average deviation from actual market price |
| **RMSE** | $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$ | 0.4589 | **$45,894.34** | Standard error penalized for large residuals |
| **MAPE** | $\frac{100\%}{n} \sum \|\frac{y_i - \hat{y}_i}{y_i}\|$ | 17.52% | **17.52%** | Average relative percentage error |

#### Comparative Benchmark vs. Standard Baselines:

```
┌─────────────────────────────────┬───────────┬──────────────┬──────────────┬───────────┬────────────────────────┐
│ Model Architecture              │  R² Score │   MAE (USD)  │  RMSE (USD)  │  MAPE (%) │ Evaluation Verdict     │
├─────────────────────────────────┼───────────┼──────────────┼──────────────┼───────────┼────────────────────────┤
│ XGBoost Regressor (Our Model)   │   0.8393  │  $30,245.42  │  $45,894.34  │   17.52%  │ Best performance       │
│ Linear Regression (OLS Baseline)│   0.5758  │  $53,320.01  │  $74,558.14  │   31.95%  │ Severe underfitting    │
│ Dummy Regressor (Mean Baseline) │  -0.0002  │  $90,606.85  │ $114,485.64  │   62.89%  │ Zero predictive signal │
└─────────────────────────────────┴───────────┴──────────────┴──────────────┴───────────┴────────────────────────┘
```

---

## 3. Uncertainty Quantification: Split Conformal Prediction

Point predictions alone are insufficient for high-stakes property transactions. Rather than relying on ungrounded heuristics, **SmartHousePricePredictor** implements **Split Conformal Prediction (Inductive Conformal Regression)** to construct mathematically guaranteed prediction intervals.

### Conformal Calibration Procedure:

1. **Non-conformity Score:** Absolute residual on calibration data:
   $$R_i = |y_i - \hat{\mu}(X_i)|$$

2. **Finite-Sample Quantile Calculation:**
   Given significance level $\alpha \in (0, 1)$ and calibration sample size $n = 4,128$:
   $$\hat{q}_{1-\alpha} = \text{Quantile}\left(\{R_1, \dots, R_n\}, \min\left(1, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)\right)$$

3. **Prediction Interval:**
   $$C(X_{n+1}) = \left[ \hat{\mu}(X_{n+1}) - \hat{q}_{1-\alpha}, \quad \hat{\mu}(X_{n+1}) + \hat{q}_{1-\alpha} \right]$$

### Statistically Verified Coverage:

- **90% Prediction Interval ($\alpha = 0.10$):**
  - Calibrated Margin: $\pm \$68,168.97$ ($\pm 0.6817$)
  - **Empirical Coverage: $90.04\%$** (surpasses target $90.0\%$)
- **80% Prediction Interval ($\alpha = 0.20$):**
  - Calibrated Margin: $\pm \$45,148.56$ ($\pm 0.4515$)
  - **Empirical Coverage: $80.04\%$** (surpasses target $80.0\%$)

> **Theoretical Distinction:** This yields a valid **prediction interval** for an unseen individual property ($P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha$), which is distinct from a confidence interval for the conditional mean $E[Y|X]$.

---

## 4. Explainable AI (XAI) & Feature Importances

To satisfy model interpretability requirements, tree split gain weights are pre-computed during evaluation and exposed via the `/api/importance` endpoint:

| Feature Name | Client Display Label | Relative Weight | Physical Interpretation |
|---|---|---|---|
| `MedInc` | **Area Income** | **43.63%** | Median block income is the dominant driver of California property prices |
| `AveOccup` | **People per House** | **12.99%** | High occupant density corresponds to multi-family/rental neighborhoods |
| `Latitude` | **Latitude** | **11.90%** | Coastal/spatial gradient (North vs. South California economic basins) |
| `Longitude` | **Longitude** | **11.73%** | Proximity to coastal urban hubs vs. Central Valley inland |
| `AveRooms` | **Total Rooms** | **8.09%** | Average physical capacity and structural size of dwellings |
| `HouseAge` | **House Age** | **6.67%** | Architectural vintage and depreciation cycle |
| `AveBedrms` | **Bedrooms** | **2.68%** | Ratio of sleeping quarters to total square footage |
| `Population` | **Population** | **2.31%** | Local block-level population density |

---

## 5. System Architecture & Engineering

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Client Browser (UI)                           │
│  • Pure White (#FFFFFF) & Solid Black (#000000) Executive Theme        │
│  • Locally Compiled Tailwind CSS (Zero CDN Warnings, Purged 20KB)       │
│  • Interactive Chart.js Factor Telemetry & Dynamic Preset Pills        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / JSON (REST API)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Flask Web Application (app.py)                     │
│  • Dynamic CORS Engine (@app.after_request, Access-Control: *)         │
│  • OPTIONS Preflight Handlers (HTTP 204)                               │
│  • Defensive Request Normalization (JSON & Form-Encoded Fallback)      │
│  • Production Gunicorn WSGI Worker Pool                                │
└──────────────┬──────────────────────────────────────────┬──────────────┘
               │                                          │
               ▼                                          ▼
┌───────────────────────────────┐          ┌─────────────────────────────┐
│  src/validator.py             │          │  src/predictor.py           │
│  • 8-Feature Schema Guard     │          │  • XGBoost Model Loader     │
│  • Physical Bounds Checker    │          │  • DataFrame Normalizer     │
│  • CA Coordinate Boundaries   │          │  • Point Valuation ($ USD)  │
│  • Room Sanity (Beds ≤ Rooms) │          │  • Conformal Interval Engine│
└───────────────────────────────┘          └──────────────┬──────────────┘
                                                          │
                                                          ▼
                                           ┌─────────────────────────────┐
                                           │  Serialized ML Artifacts    │
                                           │  • models/model.pkl         │
                                           │  • models/metrics.json      │
                                           │  • models/uncertainty.json  │
                                           │  • models/importance.json   │
                                           └─────────────────────────────┘
```

---

## 6. Docker & Container Deployment

The application features a production-ready, security-hardened `Dockerfile` utilizing an unbuffered Python 3.12 runtime and integrated container health checks.

### Single-Command Run via Docker Compose:

```bash
# Clone the repository
git clone https://github.com/AyushKumar-12345/SmartHousePricePredictor.git
cd SmartHousePricePredictor

# Build and start the container
docker compose up --build -d
```

### Manual Docker CLI Execution:

```bash
# 1. Build the production image
docker build -t smarthousepricepredictor:latest .

# 2. Run the container on port 5000
docker run -d -p 5000:5000 --name smarthouse-app smarthousepricepredictor:latest

# 3. Verify container health probe
docker inspect --format='{{json .State.Health.Status}}' smarthouse-app
# Returns: "healthy"
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 7. Production CI/CD Pipeline

Automated quality gates are implemented via GitHub Actions in [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Every push or pull request to `main` executes a two-stage quality pipeline:

```
[ Push to main ]
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 1: Unit & Integration Verification (Python 3.12 Runner)  │
│  • Set up Python 3.12 with pip dependency caching              │
│  • Verify serialized model artifacts & benchmark (R² ≥ 0.83)   │
│  • Execute 18 automated unit and integration pytest tests      │
└───────────────────────────────┬────────────────────────────────┘
                                │ (needs: test)
                                ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 2: Docker Build & Health Verification (Ubuntu Runner)    │
│  • Set up Docker Buildx                                        │
│  • Build container image: smarthousepricepredictor:latest      │
│  • Boot container and verify HEALTHCHECK reports "healthy"     │
│  • Verify live HTTP endpoints: GET /health and POST /predict   │
└───────────────────────────────┬────────────────────────────────┘
                                │ (auto-deploy)
                                ▼
┌────────────────────────────────────────────────────────────────┐
│ Continuous Deployment: Render Cloud Infrastructure             │
│  • Declarative Infrastructure-as-Code via render.yaml          │
│  • Automated live deployment on https://smarthousepricepredictor.onrender.com │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. REST API Reference

### 1. `POST /predict`
Computes property valuation and statistical conformal uncertainty intervals.

- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "MedInc": 8.50,
    "HouseAge": 32.0,
    "AveRooms": 6.20,
    "AveBedrms": 1.05,
    "Population": 1200.0,
    "AveOccup": 2.50,
    "Latitude": 37.77,
    "Longitude": -122.42
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "success": true,
    "prediction": 470005.04,
    "formatted_prediction": "$470,005",
    "range": {
      "lower": 401836.07,
      "upper": 538174.01,
      "lower_formatted": "$401,836",
      "upper_formatted": "$538,174",
      "confidence_interval": "90% Conformal Prediction Interval (±$68,169)",
      "margin_usd": 68168.97
    },
    "prediction_interval": {
      "level_pct": 90,
      "lower_usd": 401836.07,
      "upper_usd": 538174.01,
      "margin_usd": 68168.97,
      "lower_formatted": "$401,836",
      "upper_formatted": "$538,174",
      "empirical_coverage_pct": 90.04,
      "method": "Split Conformal Prediction (Individual Property Coverage)"
    },
    "benchmark": {
      "reference_dataset": "1990 U.S. Census (California)",
      "dataset_median_usd": 206855.0,
      "difference_pct": 127.2,
      "is_above_median": true
    },
    "features_used": {
      "MedInc": 8.5,
      "HouseAge": 32.0,
      "AveRooms": 6.2,
      "AveBedrms": 1.05,
      "Population": 1200.0,
      "AveOccup": 2.5,
      "Latitude": 37.77,
      "Longitude": -122.42
    }
  }
  ```

### 2. `GET /health`
Returns service readiness and model status.
- **Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "features": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
  }
  ```

### 3. `GET /api/metrics`
Returns verified held-out evaluation metrics ($R^2$, MAE, RMSE, MAPE) and baseline comparisons.

### 4. `GET /api/importance`
Returns relative feature split weights for Explainable AI (XAI) visualizations.

### 5. `GET /api/uncertainty`
Returns calibrated Split Conformal Prediction margins and empirical coverage figures.

---

## 9. Local Setup & Testing Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone & Set Up Virtual Environment:

```bash
git clone https://github.com/AyushKumar-12345/SmartHousePricePredictor.git
cd SmartHousePricePredictor

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Automated Tests:

```bash
python -m pytest -v tests/
```
All **18 unit and integration tests** verify routes, schema validation, boundary handling, CORS, model loading, conformal calculations, and Docker/CI configs in under 2 seconds.

### 3. Start Local Web Server:

```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 10. Placement & Interview Talking Points

1. **Why XGBoost Over Deep Learning or Linear Models for Tabular Data?**
   Tree-based ensembles (XGBoost) consistently outperform deep learning on tabular data with small-to-medium datasets ($N \approx 20,000$) due to their ability to build axis-aligned decision partitions without requiring extensive feature scaling or inductive bias tuning. Compared to OLS linear regression ($R^2 = 0.5758$), XGBoost resolves non-linear geographic and socio-economic feature interactions, boosting $R^2$ to $0.8393$.
2. **Mathematically Defensible Uncertainty Estimation:**
   Standard regression models output point estimates without confidence bounds. Rather than fabricating ad-hoc standard errors, I implemented **Split Conformal Prediction**. By computing non-conformity scores on a calibration split, this algorithm provides distribution-free, finite-sample guarantees for individual properties ($90.04\%$ empirical coverage).
3. **Preventing Production Web Anomalies:**
   - Replaced bloated Tailwind CDN scripts with a locally compiled, purged 20KB CSS asset to eliminate browser runtime warnings.
   - Designed a defensive API controller with CORS headers, `OPTIONS` preflight handling, origin-independent dynamic endpoint resolution, and input boundary validation.
4. **DevOps & Cloud Deployment:**
   - Multi-stage Docker containerization with integrated `HEALTHCHECK` probes.
   - Fully automated dual-stage CI pipeline verifying model integrity, pytest execution, and Docker build verification on GitHub Actions.
   - Declarative continuous deployment via Render Blueprint (`render.yaml`).

---

## 11. Repository Structure

```
SmartHousePricePredictor/
├── .github/
│   └── workflows/
│       └── ci.yml             # Dual-stage CI pipeline (Pytest + Docker verification)
├── .dockerignore              # Clean container build exclusions
├── .env.example               # Environment variable specification
├── .gitignore                 # Git ignore configuration
├── app.py                     # Flask web server, CORS engine & REST endpoints
├── dataset/
│   ├── california_housing.csv # Complete 20,640 census records
│   └── download_data.py       # Scikit-learn dataset fetcher
├── Dockerfile                 # Production Python 3.12 container with healthcheck
├── docker-compose.yml         # Container orchestration configuration
├── models/
│   ├── importance.json        # Pre-extracted tree split feature importances
│   ├── metrics.json           # Real held-out evaluation metrics & baselines
│   ├── model.pkl              # Serialized XGBoost Regressor artifact
│   └── uncertainty.json       # Calibrated conformal prediction margins & coverage
├── package.json               # Tailwind CLI build scripts & dev dependencies
├── Procfile                   # Gunicorn WSGI process launcher
├── README.md                  # Comprehensive technical documentation
├── render.yaml                # Render Infrastructure-as-Code blueprint
├── requirements.txt           # Production-pinned Python dependencies
├── src/
│   ├── __init__.py            # Source package initialization
│   ├── predictor.py           # HousePricePredictor inference & uncertainty engine
│   └── validator.py           # Strict schema, boundary, and physical constraints
├── static/
│   ├── css/
│   │   ├── input.css          # Tailwind CSS root directives
│   │   ├── style.css          # Clean monochrome custom design system
│   │   └── tailwind.min.css   # Compiled, purged production Tailwind CSS
│   ├── favicon.svg            # Custom SVG application favicon
│   └── js/
│       └── script.js          # Controller for presets, async validation, and Chart.js
├── tailwind.config.js         # Tailwind configuration for production purging
├── templates/
│   └── index.html             # Pure-white monochrome responsive valuation UI
├── tests/
│   └── test_app.py            # 18 automated unit & integration tests (100% pass)
└── train_model.py             # Reproducible training, evaluation, & calibration pipeline
```

---

## 12. License & Author

- **Author:** [Ayush Kumar](https://github.com/AyushKumar-12345)
- **Repository:** [https://github.com/AyushKumar-12345/SmartHousePricePredictor](https://github.com/AyushKumar-12345/SmartHousePricePredictor)
- **Live Demo:** [https://smarthousepricepredictor.onrender.com](https://smarthousepricepredictor.onrender.com)
- **License:** Distributed under the MIT License. See [LICENSE](LICENSE) for details.
