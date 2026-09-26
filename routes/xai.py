"""
STEP 13 - SHAP explainability routes (login required).

Three ways to reach a local explanation:
  * POST /explain   - custom accident record typed into the form
  * GET  /explain   - re-explains the last valid prediction from / (session)
  * GET  /explain?sample=1 - explains a structurally valid random record

The global SHAP importance view is always computed underneath and cached.

Explainability failures NEVER return HTTP 500: SHAP/model problems are logged
with the real exception and surfaced as a clear on-page message.
"""
from flask import Blueprint, current_app, render_template, request, session

from services.auth import current_user, login_required
from services.explainability_service import explain_record, global_summary
from services.prediction_service import (
    form_view,
    get_load_error,
    is_ready,
    sample_record,
    validate_and_build_record,
)

bp = Blueprint("xai", __name__)


@bp.route("/explain", methods=["GET", "POST"])
@login_required
def explain():
    load_error = get_load_error()
    explanation = None
    explainability_error = None
    errors = []
    submitted = {}
    record = None
    via = ""

    if load_error:
        errors = ["Model artifacts are unavailable on the server. Please try again later."]
    elif request.method == "POST":
        submitted = request.form.to_dict()
        # `validate_and_build_record` resolves the saved option lists itself.
        record, errors = validate_and_build_record(submitted)
        via = "custom form"
    elif request.args.get("sample"):
        record = sample_record()
        submitted = record
        via = "random sample"
    else:
        last = session.get("last_record")
        if last:
            record = last
            submitted = last
            via = "last prediction"

    if record is not None and not errors:
        try:
            explanation = explain_record(record)
        except RuntimeError as exc:              # SHAP/model unavailable (safe text)
            explainability_error = str(exc)
        except Exception:                        # unexpected - log the real error
            current_app.logger.exception("Local SHAP explanation failed")
            explainability_error = (
                "The local explanation could not be generated for this record. "
                "The error has been logged."
            )

    global_view = None
    global_error = None
    try:
        global_view = global_summary()
    except RuntimeError as exc:
        global_error = str(exc)
    except Exception:
        current_app.logger.exception("Global SHAP summary failed")
        global_error = (
            "The global SHAP summary could not be generated. The error has been logged."
        )

    return render_template(
        "xai.html",
        form=form_view(),
        submitted=submitted,
        explanation=explanation,
        explainability_error=explainability_error,
        global_view=global_view,
        global_error=global_error,
        errors=errors,
        load_error=load_error,
        ready=is_ready(),
        via=via,
        user=current_user(),
        active_page="xai",
    )