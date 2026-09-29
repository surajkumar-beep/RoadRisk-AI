"""
Admin routes (STEP 8b - RBAC).

``/admin`` lists every registered account (name, email, role, joined date) so
administrators can see who holds which role.  Guarded by ``admin_required``:
anonymous visitors are sent to login, authenticated non-admins get HTTP 403.
"""
from flask import Blueprint, render_template

from services.auth import current_user, admin_required, list_users, login_required
from services.prediction_service import is_ready

bp = Blueprint("admin", __name__)


@bp.route("/admin")
@login_required
@admin_required
def admin():
    rows = list_users()
    users = [
        {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "role": row["role"] or "user",
            "created_at": (row["created_at"] or "")[:10],
        }
        for row in rows
    ]
    admin_count = sum(1 for u in users if u["role"] == "admin")
    return render_template(
        "admin.html",
        users=users,
        total=len(users),
        admin_count=admin_count,
        user_count=len(users) - admin_count,
        user=current_user(),
        ready=is_ready(),
        active_page="admin",
    )