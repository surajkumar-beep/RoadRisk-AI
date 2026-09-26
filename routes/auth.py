"""
STEP 8 - Authentication routes (login / register / logout).

Uses the built-in session authentication from services.auth
(no flask-login dependency).  Every page that requires auth is decorated with
``@login_required`` which redirects to ``auth.login`` preserving the original
target in the ``next`` query parameter.
"""
from flask import Blueprint, flash, redirect, render_template, request, url_for

from services.auth import (
    authenticate_user,
    current_user,
    login_user,
    logout_user,
    register_user,
)

bp = Blueprint("auth", __name__)


def _safe_next(default_view):
    """Only allow in-app redirect targets (no open redirect)."""
    target = request.args.get("next")
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for(default_view)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user() is not None:
        return redirect(url_for("dashboard.home"))

    error = None
    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        user = authenticate_user(email, password)
        if user is None:
            error = "Invalid email or password."
        else:
            login_user(user)
            flash(f"Welcome back, {user['name'].split()[0]}!", "success")
            return redirect(_safe_next("dashboard.home"))

    return render_template("auth/login.html", error=error, next_url=request.args.get("next", ""))


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user() is not None:
        return redirect(url_for("dashboard.home"))

    error = None
    form = {"name": "", "email": ""}
    if request.method == "POST":
        form["name"] = request.form.get("name", "")
        form["email"] = request.form.get("email", "")
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        user, error = register_user(form["name"], form["email"], password, confirm)
        if user is not None:
            login_user(user)
            flash("Account created - you are now signed in.", "success")
            return redirect(url_for("dashboard.home"))

    return render_template("auth/register.html", error=error, form=form)


@bp.route("/logout")
def logout():
    if current_user() is not None:
        logout_user()
        flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))