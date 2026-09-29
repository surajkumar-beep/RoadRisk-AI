"""
Authentication helpers for RoadRiskAI (STEP 8 + RBAC).

Uses Flask's session cookie plus ``werkzeug.security`` password hashing
(no external auth library required). Provides:

- :func:`register_user`   -- validates + inserts a new account (always role 'user')
- :func:`authenticate_user` -- email/password check
- :func:`login_user`      -- stores the session marker
- :func:`logout_user`     -- clears the session marker
- :func:`current_user`    -- the logged-in user row (or None)
- :func:`login_required`  -- route decorator guarding every dashboard route
- :func:`admin_required`  -- RBAC decorator guarding admin-only routes
- :func:`promote_admins`  -- promotes ROADRISK_ADMIN_EMAILS to 'admin'
- :func:`list_users`      -- admin page helper (read-only, parameterised)

Passwords are never stored in plain text (PBKDF2-SHA256 by werkzeug) and all
database statements are parameterised (SQL injection safe).

RBAC model
----------
``users.role`` is ``'user'`` (default) or ``'admin'``.  Self-registration can
ONLY create ``'user'`` accounts - the role is never taken from user input.
Admins are granted by listing their email in the ``ROADRISK_ADMIN_EMAILS``
environment variable (comma separated); :func:`promote_admins` applies that
list idempotently at startup.
"""
import os
import re
from datetime import datetime, timezone
from functools import wraps

from flask import abort, g, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from services.database import run_query, run_write

# --- roles (RBAC) ----------------------------------------------------------
ROLE_USER = "user"
ROLE_ADMIN = "admin"
VALID_ROLES = (ROLE_USER, ROLE_ADMIN)

# --- validation rules ------------------------------------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 6
MAX_FIELDS_LEN = 120


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def user_role(user):
    """Return the safe role for a user row ('user' when absent/invalid)."""
    if user is None:
        return ROLE_USER
    try:
        role = user["role"]
    except (KeyError, IndexError, TypeError):
        role = None
    return role if role in VALID_ROLES else ROLE_USER


def is_admin(user):
    """True when the user row carries the admin role."""
    return user_role(user) == ROLE_ADMIN


def register_user(name, email, password, confirm_password=None):
    """Create a user account with the default 'user' role.

    Returns ``(user_row, None)`` on success or ``(None, error_message)`` when
    validation fails. Validation is duplicated client-side but enforced here
    (server side) as the source of truth.  The role is always ``'user'``:
    registration input can never grant admin.
    """
    name = (name or "").strip()
    email = (email or "").strip().lower()
    password = password or ""

    if not name or len(name) > MAX_FIELDS_LEN:
        return None, "Please enter your full name."
    if not EMAIL_RE.match(email):
        return None, "Please enter a valid email address."
    if password != confirm_password:
        return None, "Passwords do not match."
    if len(password) < MIN_PASSWORD_LEN:
        return None, f"Password must be at least {MIN_PASSWORD_LEN} characters long."

    existing = run_query("SELECT id FROM users WHERE email = ?", (email,))
    if existing:
        return None, "An account with this email already exists."

    password_hash = generate_password_hash(password)
    user_id = run_write(
        "INSERT INTO users (name, email, password_hash, created_at, role) "
        "VALUES (?, ?, ?, ?, ?)",
        (name, email, password_hash, _now_iso(), ROLE_USER),
    )
    row = run_query("SELECT * FROM users WHERE id = ?", (user_id,))
    return (row[0], None) if row else (None, "Account could not be created.")


def authenticate_user(email, password):
    """Return the user row when credentials are correct, else None."""
    email = (email or "").strip().lower()
    if not email or not password:
        return None
    rows = run_query("SELECT * FROM users WHERE email = ?", (email,))
    if not rows:
        return None
    user = rows[0]
    if not check_password_hash(user["password_hash"], password or ""):
        return None
    return user


def login_user(user):
    """Persist the authenticated user in the session."""
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = True


def logout_user():
    session.clear()


def current_user():
    """The logged-in user Row for the current request (cached on app context)."""
    if "user_id" not in session:
        return None
    cached = getattr(g, "_roadrisk_user", None)
    if cached is not None:
        return cached
    rows = run_query("SELECT * FROM users WHERE id = ?", (session["user_id"],))
    user = rows[0] if rows else None
    g._roadrisk_user = user
    if user is None:
        session.clear()
    return user


def login_required(view):
    """Route decorator: redirect anonymous visitors to the login screen."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            target = request.full_path if request.query_string else request.path
            return redirect(url_for("auth.login", next=target))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    """RBAC decorator: only authenticated ``admin`` users may enter.

    Anonymous visitors are redirected to the login screen (same behaviour as
    :func:`login_required`); authenticated non-admin users receive HTTP 403.
    """
    wrapped_login = login_required(view)

    @wraps(wrapped_login)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:  # unreachable after login_required, kept for safety
            target = request.full_path if request.query_string else request.path
            return redirect(url_for("auth.login", next=target))
        if not is_admin(user):
            abort(403)
        return wrapped_login(*args, **kwargs)

    return wrapped


def promote_admins():
    """Promote every account listed in ``ROADRISK_ADMIN_EMAILS`` to admin.

    Comma-separated email list; applied idempotently at app startup.  Returns
    the list of emails whose role changed (for logging/tests).  Unknown
    emails are ignored (the account may register later - re-run the app or
    update the env var then).
    """
    raw = os.getenv("ROADRISK_ADMIN_EMAILS", "")
    emails = {e.strip().lower() for e in raw.split(",") if e.strip()}
    promoted = []
    for email in sorted(emails):
        rows = run_query("SELECT id, role FROM users WHERE email = ?", (email,))
        if rows and user_role(rows[0]) != ROLE_ADMIN:
            run_write("UPDATE users SET role = ? WHERE id = ?", (ROLE_ADMIN, rows[0]["id"]))
            promoted.append(email)
    return promoted


def list_users():
    """All user accounts ordered by id (admin page, read-only)."""
    return run_query(
        "SELECT id, name, email, role, created_at FROM users ORDER BY id"
    )