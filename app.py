"""
RoadRisk AI - Flask application factory.

Covers STEP 7 (prediction), STEP 8 (session authentication), STEP 9
(dashboard), STEP 10 (analytics), STEP 11 (risk analysis), STEP 12 (hotspot
mapping) and STEP 13 (SHAP explainability) behind one process.

Contract (kept from the original STEP 7 deployment):
  * the four saved Step 5 artifacts under models/*.pkl are loaded ONCE at
    startup via services.prediction_service (joblib, never re-fitted);
  * GET  / renders the prediction form (PUBLIC, same as the old app);
  * POST / reuses the EXACT Step 6 preprocessing chain
    (testing/test_model.preprocess_new_record + predict_with_proba);
  * class labels come from the saved target mapping, never hardcoded;
  * missing/corrupt artifacts never crash the app - pages show a friendly
    maintenance notice instead;
  * every analytical page (dashboard, analytics, risk, hotspots, XAI) is
    behind the session-auth ``login_required`` guard.

Run:  python app.py            (http://127.0.0.1:5000)
"""
import os

from flask import Flask

from routes import register_blueprints
from services.database import init_db
from services.prediction_service import initialize as init_prediction_service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_PLOTLY = os.path.join(BASE_DIR, "static", "vendor", "plotly.min.js")


def _ensure_plotly_vendor():
    """Write the Plotly JS bundle once to static/vendor so pages stay light."""
    from utils.plots import PLOTLY_JS

    if os.path.exists(_VENDOR_PLOTLY):
        return
    os.makedirs(os.path.dirname(_VENDOR_PLOTLY), exist_ok=True)
    with open(_VENDOR_PLOTLY, "w", encoding="utf-8") as fh:
        fh.write(PLOTLY_JS)


def create_app(test_config=None):
    """Build and configure the application (also used by tests)."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "roadrisk_ai_secret_key")
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 8  # 8h

    if test_config:
        app.config.update(test_config)

    # Persistence + model artifacts (idempotent, only startup costs).
    init_db()
    init_prediction_service()
    _ensure_plotly_vendor()

    register_blueprints(app)

    # Friendly error pages (no raw tracebacks served to users).
    @app.errorhandler(404)
    def not_found(error):  # noqa: ANN001
        return (
            "<h1>404 - Page not found</h1>"
            "<p>The page you are looking for does not exist.</p>"
            '<p><a href="/">Back to the prediction form</a></p>',
            404,
        )

    @app.errorhandler(500)
    def server_error(error):  # noqa: ANN001
        return (
            "<h1>500 - Something went wrong</h1>"
            "<p>The server hit an unexpected problem. Please try again.</p>",
            500,
        )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    # debug=False keeps friendly error pages and avoids the reloader double-start.
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=port, debug=False)