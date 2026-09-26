"""
STEP 7 / 7.2 - Prediction routes.

  * GET  /    -> redirects: anonymous -> /login, authenticated -> /home
  * GET/POST /predict (PROTECTED)
               -> renders the accident form; on success shows probabilities,
                  confidence and a severity-coloured indicator, and saves the
                  record so the XAI page can explain it; on invalid input
                  re-renders with friendly errors (never a 500 / traceback).

When the model artifacts cannot be loaded the page still boots and displays a
clear maintenance-style notice instead of crashing.
"""
from flask import Blueprint, redirect, render_template, request, session, url_for

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