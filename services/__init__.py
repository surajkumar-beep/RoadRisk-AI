"""
RoadRiskAI service layer.

Each module encapsulates one business capability used by the web routes:

- data.py:               raw/processed dataset loading + shared cleaning
- database.py:           SQLite connection + schema helpers
- auth.py:               registration / login / session guard
- prediction_service.py: model artifacts, form options, record validation,
                         severity prediction (moved verbatim from app.py)
- dashboard_service.py:  STEP 9 home-dashboard KPIs + charts (real data)
- analytics_service.py:  STEP 10 filterable analytics (bar/pie/line/treemap)
- risk_analysis_service.py STEP 11 risk scoring + factor & correlation views
- hotspot_service.py:    STEP 12 hotspot / area & temporal-disease mapping
- explainability_service.py STEP 13 SHAP-based local + global explanations

Services are pure function modules: they never import the Flask ``app`` and
only depend on ``flask`` helpers where a route guard (auth) requires them.
"""