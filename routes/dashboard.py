"""
STEP 9 - Home dashboard routes (login required).
"""
from flask import Blueprint, render_template

from services.auth import current_user, login_required
from services.dashboard_service import build_dashboard
from services.prediction_service import is_ready

bp = Blueprint("dashboard", __name__)


@bp.route("/home")
@login_required
def home():
    data = build_dashboard()
    return render_template(
        "dashboard.html",
        data=data,
        user=current_user(),
        ready=is_ready(),
        active_page="dashboard",
    )