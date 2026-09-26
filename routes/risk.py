"""
STEP 11 / 7.4 - Risk analysis routes (login required).

The main page receives two control-panel parameters:
  scope -> all | severe | fatal  (applies to KPIs, factor separation, combos)
  factor -> which factor to compare on the factor page
"""
from flask import Blueprint, render_template, request

from services.auth import current_user, login_required
from services.prediction_service import is_ready
from services.risk_service import (
    FACTOR_COLUMNS,
    SCOPES,
    build_risk_overview,
    severity_by_factor,
)

bp = Blueprint("risk", __name__)

_FACTOR_CHOICES = {column: label for column, label in FACTOR_COLUMNS}


def _clean_scope():
    scope = request.args.get("scope", "all")
    return scope if scope in SCOPES else "all"


@bp.route("/risk")
@login_required
def risk():
    scope = _clean_scope()
    data = build_risk_overview(scope)
    return render_template(
        "risk.html",
        data=data,
        factors=_FACTOR_CHOICES,
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