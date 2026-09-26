"""
SQLite persistence for RoadRiskAI user accounts.

RoadRisk had no database before STEP 8; a small SQLite store is created at
``roadrisk.db`` in the project root (override with ``ROADRISK_DB_PATH`` env
var so tests / other environments can use an isolated file).

Schema is created idempotently via :func:`init_db` (called once on app start).

Only the users table is persisted today; it exists solely so the built-in
session authentication keeps working without extra dependencies such as
flask-login.
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
    created_at    TEXT    NOT NULL
);
"""


def get_connection():
    """Open a fresh SQLite connection (thread-safe: one per call)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create (if missing) all tables. Safe to call on every startup."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
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