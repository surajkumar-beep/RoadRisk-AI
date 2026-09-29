"""
Route blueprints for RoadRiskAI (rewritten STEP 8-13 app).

All blueprints are defined in their own module and registered through the
single :func:`register_blueprints` helper so app.py stays a thin factory.
"""
from routes.admin import bp as admin_bp
from routes.analytics import bp as analytics_bp
from routes.auth import bp as auth_bp
from routes.dashboard import bp as dashboard_bp
from routes.hotspot import bp as hotspot_bp
from routes.predict import bp as predict_bp
from routes.risk import bp as risk_bp
from routes.xai import bp as xai_bp


def register_blueprints(app):
    """Attach every blueprint to the Flask app (idempotent per app object)."""
    for bp in (auth_bp, predict_bp, dashboard_bp, analytics_bp, risk_bp,
               hotspot_bp, xai_bp, admin_bp):
        app.register_blueprint(bp)