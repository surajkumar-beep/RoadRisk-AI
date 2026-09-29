"""
STEP 11 / 7.4 - Risk analysis routes (login required).

The main page receives two control-panel parameters:
  scope -> all | severe | fatal  (applies to KPIs, factor separation, combos)
  factor -> which factor to compare on the factor page

It also shows the COMPOSITE RISK SCORE for the session's most recent
prediction (session['last_record']) together with its component factors.
Score failures degrade to an on-page notice - never HTTP 500.
"""
from flask import Blueprint, current_app, render_template, request, session

from services.auth import current_user, login_required
from services.prediction_service import is_ready
from services.risk_service import (
    FACTOR_COLUMNS,
    SCOPES,
    build_risk_overview,
    composite_risk_score,
    severity_by_factor,
)

bp = Blueprint("risk", __name__)

_FACTOR_CHOICES = {column: label for column, label in FACTOR_COLUMNS}


def _clean_scope():
    scope = request.args.get("scope", "all")
    return scope if scope in SCOPES else "all"


def _score_for_last_record():
    """Composite score of the session's last prediction (graceful on error)."""
    record = session.get("last_record")
    if not record:
        return None, None
    try:
        return composite_risk_score(record), None
    except (ValueError, RuntimeError) as exc:
        return None, str(exc)
    except Exception:  # noqa: BLE001 - never 500 on the risk page
        current_app.logger.exception("Composite risk score failed on /risk")
        return None, (
            "The composite risk score could not be computed for your last "
            "prediction. The error has been logged."
        )


@bp.route("/risk")
@login_required
def risk():
    scope = _clean_scope()
    data = build_risk_overview(scope)
    score, score_error = _score_for_last_record()
    return render_template(
        "risk.html",
        data=data,
        factors=_FACTOR_CHOICES,
        score=score,
        score_error=score_error,
        has_record=bool(session.get("last_record")),
        user=current_user(),
        ready=is_ready(),
        active_page="risk",
    )


@bp.route("/risk/factor")
@login_required
def factor():
    factor_column = request.args.get("factor", "")
    if factor_column not in _FACTOR_CHOICES:
        factor_column = FACTOR_COLUMNS[0][0]
    data = severity_by_factor(factor_column)
    return render_template(
        "risk_factor.html",
        data=data,
        factors=_FACTOR_CHOICES,
        active_factor=factor_column,
        user=current_user(),
        ready=is_ready(),
        active_page="risk",
    )