# RoadRisk AI

RoadRisk AI is a machine learning-based web application for road accident severity prediction, analytics, risk analysis, hotspot detection, and explainable AI insights.

## Modules

- **Severity Prediction** – Predict the severity of road accidents using machine learning models.
- **Analytics** – Interactive dashboards and visual analytics of accident data.
- **Risk Analysis** – Identify and assess risk factors contributing to accidents.
- **Hotspot Analysis** – Detect high-risk accident locations and patterns.
- **Explainable AI** – Interpret model predictions using SHAP for transparency.

## Tech Stack

- **Flask** – Web framework
- **Python** – Core programming language
- **XGBoost** – Gradient boosting for predictions
- **SHAP** – Model explainability
- **Plotly** – Interactive visualizations
- **Pandas** – Data manipulation
- **Scikit-learn** – Machine learning utilities

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open `http://127.0.0.1:5000` in your browser.

## Project Structure

```
RoadRisk_AI/
├── analytics/          # Analytics module
├── dataset/            # Dataset files
├── explainability/     # Explainable AI module
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks
├── preprocessing/      # Data preprocessing
├── reports/            # Generated reports
├── routes/             # Flask routes
├── services/           # Business logic
├── static/             # Static assets
├── templates/          # HTML templates
├── tests/              # Unit tests
├── training/           # Model training
├── utils/              # Utilities (constants, logger, helpers)
├── visualizations/     # Visualization module
├── app.py              # Flask application entry point
├── config.py           # Application configuration
└── requirements.txt    # Python dependencies