# SmartHouse AI Predictor v2.0

A premium, full-stack machine learning application that utilizes an optimized XGBoost Regressor model trained on spatial and demographic variables to estimate dynamic property valuations. 

## System Architecture

The core architecture operates as a data pipeline processing geographic multidimensional regressions:

[Raw Records] -> [download_data.py] -> [train_model.py] -> [model.pkl] -> [app.py Engine] -> [Tailwind UI]

## Features

- **XGBoost Regressor Core:** High-accuracy machine learning configuration utilizing ensemble gradient boosting.
- **Explainable AI (XAI) Telemetry:** Interactive Chart.js interface visualizing live model feature weights.
- **Premium Glassmorphic UI:** Smooth, high-fidelity dark gradient user workspace styled with Tailwind CSS.
- **Asynchronous Execution:** Form values compile and process vector data dynamically using asynchronous JavaScript payloads.

## Technology Stack

- **Frontend:** HTML5, Tailwind CSS, JavaScript (ES6+), Chart.js
- **Backend Framework:** Flask 3.1.3, Werkzeug 3.1.8
- **Data Engineering:** Python, Pandas 3.0.3, NumPy 2.5.1
- **Machine Learning Platform:** Scikit-Learn 1.9.0, XGBoost 3.3.0, Joblib 1.5.3

## Execution Pipeline

### 1. Environment Setup
Initialize your virtual environment and install the production dependencies:
```bash
pip install -r requirements.txt