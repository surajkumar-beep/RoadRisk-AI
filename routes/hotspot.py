"""
STEP 12 - Hotspot mapping routes (login required).

Query params (all optional, validated against the dataset):
  day      -> weekday filter (Day_of_week)
  severity -> Accident_severity filter
  top      -> number of ranked areas (5 / 10 / 15)
"""
from flask import Blueprint, render_template, request

from services.auth import current_user, login_required
from services.hotspot_service import build_hotspots
from services.prediction_service import is_ready

bp = Blueprint("hotspot", __name__)


@bp.route("/hotspots")
@login_required
def hotspots():
    data = build_hotspots(request.args)
    return render_template(
        "hotspots.html",
        data=data,
        user=current_user(),
        ready=is_ready(),
        active_page="hotspots",
    )