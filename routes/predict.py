"""
STEP 7 / 7.2 - Prediction routes.

  * GET  /    -> redirects: anonymous -> /login, authenticated -> /home
  * GET/POST /predict (PROTECTED)
               -> renders the accident form; on success shows probabilities,
                  confidence, a severity-coloured indicator, the TOP 3-5 SHAP
                  contributing factors and the composite risk score (both
                  computed with the real model/XAI pipeline, degrading to a
                  friendly notice if either service fails - never a 500), and
                  saves the record so the XAI / Risk pages can reuse it; on
                  invalid input re-renders with friendly errors.

When the model artifacts cannot be loaded the page still boots and displays a
clear maintenance-style notice instead of crashing.
"""
from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from services.prediction_service import (
    form_view,
    get_load_error,
    get_options,
    is_ready,
    make_prediction,
    validate_and_build_record,
)
from services.auth import current_user, login_required

bp = Blueprint("predict", __name__)

# How many SHAP factors to show inline on the prediction result page.
RESULT_TOP_FACTORS = 5


def _attach_top_factors(result, record):
    """Best-effort SHAP top factors for the result page (never raises)."""
    try:
        from services.explainability_service import top_factors

        result["top_factors"] = top_factors(record, k=RESULT_TOP_FACTORS)
        result["top_factors_error"] = None
    except RuntimeError as exc:  # safe SHAP/model-unavailable message
        result["top_factors"] = None
        result["top_factors_error"] = str(exc)
    except Exception:  # noqa: BLE001 - degrade, never 500
        current_app.logger.exception("Top SHAP factors failed for prediction result")
        result["top_factors"] = None
        result["top_factors_error"] = (
            "The top contributing factors could not be computed for this "
            "result. The error has been logged; predictions are unaffected."
        )


def _attach_risk_score(result, record):
    """Best-effort composite risk score for the result page (never raises)."""
    try:
        from services.risk_service import composite_risk_score

        # Reuse the probabilities already computed by make_prediction so the
        # score's model component is identical to the displayed probabilities.
        result["risk"] = composite_risk_score(
            record, probabilities=result["probabilities"]
        )
        result["risk_error"] = None
    except (ValueError, RuntimeError) as exc:
        result["risk"] = None
        result["risk_error"] = str(exc)
    except Exception:  # noqa: BLE001 - degrade, never 500
        current_app.logger.exception("Composite risk score failed for prediction result")
        result["risk"] = None
        result["risk_error"] = (
            "The composite risk score could not be computed for this result. "
            "The error has been logged; predictions are unaffected."
        )


@bp.route("/")
def root():
    """Global entry point: anonymous -> login, authenticated -> home."""
    if current_user() is None:
        return redirect(url_for("auth.login"))
    return redirect(url_for("dashboard.home"))


@bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    load_error = get_load_error()
    options = get_options()
    user = current_user()

    result = None
    errors = []
    submitted = {}

    if request.method == "POST":
        submitted = request.form.to_dict()
        if load_error:
            errors = ["Model artifacts are unavailable on the server. Please try again later."]
        else:
            record, errors = validate_and_build_record(submitted, options)
            if not errors:
                try:
                    result = make_prediction(record)
                    # Keep the valid record so the XAI page can deep-link to it.
                    session["last_record"] = record
                except RuntimeError as exc:
                    errors = [str(exc)]
                if result is not None:
                    # Enrichments: same real model/SHAP pipeline, best-effort.
                    _attach_top_factors(result, record)
                    _attach_risk_score(result, record)

    return render_template(
        "predict.html",
        form=form_view(),
        submitted=submitted,
        result=result,
        errors=errors,
        load_error=load_error,
        ready=is_ready(),
        user=user,
        active_page="predict",
    )