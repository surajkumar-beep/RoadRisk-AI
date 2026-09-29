"""
SQLite persistence for RoadRiskAI user accounts.

RoadRisk had no database before STEP 8; a small SQLite store is created at
``roadrisk.db`` in the project root (override with ``ROADRISK_DB_PATH`` env
var so tests / other environments can use an isolated file).

Schema is created idempotently via :func:`init_db` (called once on app start).

RBAC (role-based access control): every user row carries a ``role`` column
(``'user'`` by default, ``'admin'`` for administrators).  Databases created
before RBAC existed are migrated in place by :func:`_migrate` - the column is
added with a DEFAULT so existing users are preserved and keep working as
regular users.

Only the users table is persisted today; it exists solely so the built-in
session authentication keeps working without extra dependencies such as
flask-login.  All statements are parameterised (SQL injection safe).
"""
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.getenv("ROADRISK_DB_PATH", os.path.join(BASE_DIR, "roadrisk.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'user'
);
"""


def get_connection():
    """Open a fresh SQLite connection (thread-safe: one per call)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate(conn):
    """Bring an existing database up to the current schema.

    * Pre-RBAC databases (no ``role`` column) get ``ADD COLUMN role TEXT NOT
      NULL DEFAULT 'user'`` - existing rows keep their accounts and are
      treated as regular users.  Safe to run repeatedly.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    if columns and "role" not in columns:
        conn.execute(
            "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'"
        )


def init_db():
    """Create (if missing) all tables and apply pending migrations.

    Safe to call on every startup.
    """
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def run_query(sql, params=()):
    """Execute a SELECT and return a list of sqlite3.Row."""
    conn = get_connection()
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def run_write(sql, params=()):
    """Execute an INSERT/UPDATE/DELETE and return the last row id."""
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute(sql, params)
            return cursor.lastrowid
    finally:
        conn.close()