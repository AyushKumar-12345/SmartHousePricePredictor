# SmartHousePricePredictor — California Property Valuation Engine

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-green.svg)](https://scikit-learn.org/)
[![CI](https://github.com/AyushKumar-12345/SmartHousePricePredictor/actions/workflows/ci.yml/badge.svg)](https://github.com/AyushKumar-12345/SmartHousePricePredictor/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg)](https://smarthousepricepredictor.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

A production-ready Machine Learning web application that predicts California residential property valuations. Powered by an optimized **XGBoost Regressor** trained on 20,640 census block records, featuring **Split Conformal Prediction Intervals**, **Explainable AI (XAI)** telemetry, strict server-side validation boundaries, and an executive high-contrast monochrome interface.

- **Live Application:** [https://smarthousepricepredictor.onrender.com](https://smarthousepricepredictor.onrender.com)
- **Repository:** [https://github.com/AyushKumar-12345/SmartHousePricePredictor](https://github.com/AyushKumar-12345/SmartHousePricePredictor)

---

## 1. Project Overview

Real estate valuation requires synthesizing multi-dimensional spatial, structural, and demographic variables. Standard linear regression models struggle to capture non-linear geographic interactions, leading to severe underfitting.

**SmartHousePricePredictor** solves this by leveraging an ensemble **Gradient Boosted Decision Tree (GBDT)** architecture via XGBoost. Evaluated on a strictly held-out test split (4,128 records), the model achieves an **R² score of 0.8393**, outperforming the standard linear regression baseline (R² = 0.5758) by **+26.35 percentage points**, reducing prediction error to an average residual of **$30,245 (17.52% MAPE)**, and delivering distribution-free **Split Conformal Prediction Intervals** with verified 90.04% empirical coverage.

---

## 2. Key Features

- **High-Accuracy XGBoost Core:** Ensemble gradient boosting configured with 300 estimators, tree depth of 6, and subsampling (0.8) to prevent overfitting.
- **Split Conformal Prediction Intervals:** Mathematically defensible individual-property uncertainty estimation providing finite-sample coverage guarantees ($90\%$ interval at $\pm \$68,169$ with $90.04\%$ verified empirical coverage).
- **Explainable AI (XAI) Telemetry:** Interactive Chart.js horizontal bar charts dynamically displaying relative tree split feature weights from `/api/importance`.
- **Strict Server-Side Validation:** Validates physical and geographic bounds (e.g. California coordinates $32.5^\circ\text{–}42.0^\circ\text{ N}, -124.5^\circ\text{–}-114.0^\circ\text{ W}$, positive room counts, and logical constraints such as bedrooms $\le$ total rooms) returning clean, descriptive HTTP 400 diagnostics.
- **Scenario Presets:** One-click presets for realistic California real estate markets (*San Francisco Peninsula*, *Los Angeles Coastal Basin*, *Sacramento Valley*, and *San Diego Urban*).
- **Executive Monochrome UI:** Built on a pure white background (`#FFFFFF`), high-contrast black typography (`#000000`), subtle gray borders (`#E2E8F0`), zero CDN warnings (locally compiled Tailwind CSS), and clean responsive cards.
- **Containerized & CI/CD Ready:** Production Dockerfile with health checks, GitHub Actions CI workflow, and Render continuous deployment.

---

## 3. Technology Stack

| Domain | Technologies |
|---|---|
| **Backend & API** | Python 3.12, Flask 3.1, Werkzeug 3.1, Gunicorn 26.2 (WSGI) |
| **Machine Learning** | XGBoost 3.4 (`XGBRegressor`), Scikit-learn 1.9, Joblib 1.6 |
| **Data Engineering** | Pandas 3.0, NumPy 2.5 |
| **Frontend & UI/UX** | HTML5, Tailwind CSS (locally compiled), JavaScript (ES6+), Chart.js 4.4 |
| **Uncertainty Estimation** | Split Conformal Prediction (Inductive Conformal Regression) |
| **DevOps & Containers**| Docker, GitHub Actions CI/CD, Render Cloud Platform |
| **Testing** | Pytest 9.1 (14 automated unit & integration tests) |

---

## 4. System Architecture

```
[ California Census Dataset (20,640 records) ]
                     │
                     ▼
       [ 80/20 Deterministic Split ]
         ├── Train Split (16,512) ──► [ train_model.py: XGBRegressor (300 Trees) ]
         │                                       │
         │                                       ▼
         │                            [ Serialized Artifacts ]
         │                            ├── models/model.pkl
         │                            ├── models/importance.json
         │                            ├── models/metrics.json
         │                            └── models/uncertainty.json
         │
         └── Test Split (4,128) ──────► [ Evaluation & Conformal Calibration ]
                                                 │
                                                 ▼
[ Client Browser / UI ] ◄── HTTP/JSON ──► [ Flask REST API (app.py) ]
  • Pure White Monochrome Design            • Input Validation (src/validator.py)
  • Locally Compiled Tailwind CSS           • Inference Engine (src/predictor.py)
  • Real-Time Chart.js XAI Bar              • Conformal Interval Computation
  • Interactive Scenario Presets            • Endpoints: /health, /api/metrics, /predict
```

---

## 5. Dataset Provenance & Features

The model is trained on the **California Housing Dataset**, derived from the 1990 U.S. Census (Pace & Barry, 1997) and distributed via Scikit-learn:

- **Total Records:** 20,640 census block groups across California.
- **Target Variable:** `Price` — Median house value expressed in units of $100,000 (transformed to USD in application logic).
- **Dataset Median Value:** $206,855 (1990 U.S. Census baseline).

### Feature Definitions:

| Feature Name | Description | Acceptable Range | Unit / Scale |
|---|---|---|---|
| `MedInc` | Median household income in block | 0.1 – 25.0 | $10,000s (e.g. 8.5 = $85,000) |
| `HouseAge` | Median age of houses in block | 1 – 100 | Years |
| `AveRooms` | Average total rooms per household | 1.0 – 50.0 | Rooms |
| `AveBedrms` | Average bedrooms per household | 0.5 – 20.0 | Bedrooms |
| `Population` | Total population in block group | 1 – 50,000 | People |
| `AveOccup` | Average number of household members | 0.5 – 50.0 | Persons |
| `Latitude` | Geographic coordinate (North) | 32.5° – 42.0° | Decimal Degrees |
| `Longitude` | Geographic coordinate (West) | -124.5° – -114.0° | Decimal Degrees |

---

## 6. Model Evaluation & Benchmark Results

Model evaluation was executed on a strictly held-out test dataset (**4,128 records, 20% split, random_state=42**). To ensure complete integrity, preprocessing and hyperparameters were fit solely on the training partition.

### Verified Test-Set Metrics:

| Model Architecture | R² Score | MAE (Target Scale) | MAE (USD) | RMSE (Target Scale) | RMSE (USD) | MAPE (%) |
|---|---|---|---|---|---|---|
| **XGBoost Regressor (Trained)** | **0.8393** | **0.3025** | **$30,245.42** | **0.4589** | **$45,894.34** | **17.52%** |
| Linear Regression Baseline | 0.5758 | 0.5332 | $53,320.01 | 0.7456 | $74,558.14 | 31.95% |
| Dummy Regressor (Mean Baseline) | -0.0002 | 0.9061 | $90,606.85 | 1.1449 | $114,485.64 | 62.89% |

### Statistically Validated Uncertainty Estimation (Conformal Prediction):
Rather than relying on ungrounded heuristics, uncertainty is quantified via **Split Conformal Prediction (Inductive Conformal Regression)** calibrated on held-out absolute residuals $R_i = |y_i - \hat{y}_i|$:

$$\hat{q}_{1-\alpha} = \text{Quantile}\left(R, \min\left(1.0, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)\right)$$

- **90% Conformal Prediction Interval:** $\hat{y} \pm \$68,168.97$ ($\pm 0.6817$) $\rightarrow$ **$90.04\%$ Verified Empirical Coverage**.
- **80% Conformal Prediction Interval:** $\hat{y} \pm \$45,148.56$ ($\pm 0.4515$) $\rightarrow$ **$80.04\%$ Verified Empirical Coverage**.
- **Individual Property Distinction:** This constructs a true **prediction interval** for an unseen individual house, which is distinct from a confidence interval for the conditional mean.

---

## 7. Project Structure

```
SmartHousePricePredictor/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions automated test & verification pipeline
├── .dockerignore              # Docker build exclusions
├── .gitignore                 # Git repository exclusions
├── app.py                     # Flask web server and REST endpoints
├── dataset/
│   ├── california_housing.csv # Complete 20,640 census records
│   └── download_data.py       # Script to fetch dataset from scikit-learn
├── Dockerfile                 # Multi-stage production container definition
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
│   ├── __init__.py            # Source package init
│   ├── predictor.py           # HousePricePredictor inference & uncertainty engine
│   └── validator.py           # Strict schema, boundary, and physical constraints
├── static/
│   ├── css/
│   │   ├── input.css          # Tailwind CSS root directives
│   │   ├── style.css          # Clean monochrome custom design system
│   │   └── tailwind.min.css   # Compiled, purged production Tailwind CSS
│   └── js/
│       └── script.js          # Controller for presets, async validation, and Chart.js
├── tailwind.config.js         # Tailwind configuration for production purging
├── templates/
│   └── index.html             # Pure-white monochrome responsive valuation UI
├── tests/
│   └── test_app.py            # 14 automated unit & integration tests (100% pass)
└── train_model.py             # Reproducible training, evaluation, & calibration pipeline
```

---

## 8. Local Installation & Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AyushKumar-12345/SmartHousePricePredictor.git
   cd SmartHousePricePredictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install production dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the automated test suite:**
   ```bash
   python -m pytest -v tests/test_app.py
   ```

5. **Launch the application locally:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 9. Docker Deployment

The application includes a production-ready `Dockerfile` with health checks and unprivileged process execution.

### Build and Run with Docker:

```bash
# Build the Docker image
docker build -t smarthouse-predictor:latest .

# Run container on port 5000
docker run -d -p 5000:5000 --name smarthouse-app smarthouse-predictor:latest

# Check health check status
docker inspect --format='{{json .State.Health.Status}}' smarthouse-app
```
Access the application at `http://localhost:5000`.

---

## 10. REST API Reference

### 1. `POST /predict`
Calculates property valuation given input features.
- **Request Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "MedInc": 8.50,
    "HouseAge": 32.0,
    "AveRooms": 6.2,
    "AveBedrms": 1.05,
    "Population": 1200.0,
    "AveOccup": 2.5,
    "Latitude": 37.77,
    "Longitude": -122.42
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "success": true,
    "prediction": 482650.12,
    "formatted_prediction": "$482,650",
    "prediction_interval": {
      "level_pct": 90,
      "lower_usd": 414481.15,
      "upper_usd": 550819.09,
      "margin_usd": 68168.97,
      "empirical_coverage_pct": 90.04,
      "method": "Split Conformal Prediction (Individual Property Coverage)"
    },
    "benchmark": {
      "reference_dataset": "1990 U.S. Census (California)",
      "dataset_median_usd": 206855.0,
      "difference_pct": 133.3,
      "is_above_median": true
    }
  }
  ```

### 2. `GET /api/uncertainty`
Returns calibrated Split Conformal Prediction margins and verified empirical coverage percentages.

### 3. `GET /api/metrics`
Returns verified held-out evaluation metrics ($R^2$, MAE, RMSE, MAPE) and baseline comparisons.

### 4. `GET /api/importance`
Returns tree split weights across all 8 features for Explainable AI (XAI).

### 5. `GET /health`
Returns service health status (`healthy`) and model readiness.

---

## 11. Placement & Interview Talking Points

1. **Why XGBoost over Ordinary Least Squares (OLS)?** California real estate exhibits severe non-linear spatial dependencies (proximity to coastal high-income clusters and employment centers) that linear models cannot resolve without manual interaction terms. XGBoost's ensemble trees partition multi-dimensional feature space adaptively, improving $R^2$ from $0.5758$ to $0.8393$ and reducing MAPE from $31.95\%$ to $17.52\%$.
2. **Defensible Uncertainty (Conformal Prediction):** Rather than outputting arbitrary heuristics, I implemented **Split Conformal Prediction**. Calibrated on held-out residuals, it guarantees valid, distribution-free prediction intervals for individual properties with empirically verified $90.04\%$ coverage.
3. **Modular Software Architecture:** The application separates concerns into `src/validator.py` (defensive schema and physical bounds checking) and `src/predictor.py` (inference and uncertainty quantification), covered by 14 automated pytest integration tests.
4. **Production Web Engineering:** Built with a locally compiled, purged Tailwind CSS build (eliminating browser console CDN warnings), Gunicorn WSGI server, dynamic port binding, automated health checks, Docker containerization, and GitHub Actions CI.

---

## 12. License & Author

- **Author:** [Ayush Kumar](https://github.com/AyushKumar-12345)
- **License:** MIT License