"""
STEP 10 / 7.3 - Analytics routes (login required, filterable charts).
"""
from flask import Blueprint, render_template, request

from services.analytics_service import FILTERS, build_analytics, filter_options
from services.auth import current_user, login_required
from services.prediction_service import is_ready

bp = Blueprint("analytics", __name__)


@bp.route("/analytics")
@login_required
def analytics():
    data = build_analytics(request.args)
    filters = filter_options()
    filter_bar = [
        {"key": key, "label": label, "options": filters[key]}
        for key, label, _column in FILTERS
    ]
    return render_template(
        "analytics.html",
        data=data,
        filter_bar=filter_bar,
        applied=data["summary"]["filters_applied"],
        user=current_user(),
        ready=is_ready(),
        active_page="analytics",
    )