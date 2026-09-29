"""
Central application configuration for RoadRisk AI.

Single source of truth for app configuration, loaded by ``app.py`` via
``config.load_config()``.  (This module was previously an orphaned stub that
was never imported and pointed at a non-existent dataset file.)

Environment handling
--------------------
* ``.env`` is loaded with ``python-dotenv`` when present (real environment
  variables always take precedence over ``.env`` values).
* ``SECRET_KEY``:
    - MUST be provided via the environment (or ``.env``) in production
      (``ROADRISK_ENV=production``) - startup fails otherwise, so a weak
      hardcoded fallback can never reach production.
    - In development, if unset, an ephemeral random key is generated for the
      process (sessions reset on restart - convenient AND safe).
* Runtime profile: ``ROADRISK_ENV`` (preferred), ``FLASK_ENV`` or ``APP_ENV``
  -> ``production`` | ``development`` (default).

Never commit real secrets: ``.env`` is gitignored.
"""
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ``.env`` once at import time (override=False: real env vars win).
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except ImportError:  # pragma: no cover - python-dotenv is in requirements.txt
    pass

# --------------------------------------------------------------------------- #
# Paths (corrected: the old stub pointed at dataset/road_accident_dataset.csv,
# which does not exist; the project uses dataset/RTA Dataset.csv)
# --------------------------------------------------------------------------- #
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "RTA Dataset.csv")
PROCESSED_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "processed_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")
REPORT_PATH = os.path.join(BASE_DIR, "reports")


def runtime_env():
    """Active runtime profile: ``'production'`` or ``'development'``."""
    value = (
        os.getenv("ROADRISK_ENV")
        or os.getenv("FLASK_ENV")
        or os.getenv("APP_ENV")
        or "development"
    )
    return value.strip().lower() or "development"


def is_production():
    return runtime_env() == "production"


def resolve_secret_key():
    """Return the session secret key.

    * Environment / ``.env`` value wins when present.
    * Production without ``SECRET_KEY`` -> ``RuntimeError`` (never a weak
      hardcoded default).
    * Development without ``SECRET_KEY`` -> fresh random key for this process
      (documented trade-off: sessions reset when the dev server restarts).
    """
    key = (os.getenv("SECRET_KEY") or "").strip()
    if key:
        return key
    if is_production():
        raise RuntimeError(
            "SECRET_KEY environment variable is required when ROADRISK_ENV=production. "
            'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
        )
    return secrets.token_hex(32)


class Config:
    """Base configuration applied by ``app.create_app()``."""

    SECRET_KEY = None            # resolved per-call by load_config()
    DEBUG = False
    TESTING = False

    # Session hardening (kept from the original deployment contract).
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 8  # 8 hours

    # Set SESSION_COOKIE_SECURE=1 behind TLS (recommended for production).
    SESSION_COOKIE_SECURE = False

    # Project paths.
    BASE_DIR = BASE_DIR
    DATASET_PATH = DATASET_PATH
    PROCESSED_DATASET_PATH = PROCESSED_DATASET_PATH
    MODEL_PATH = MODEL_PATH
    REPORT_PATH = REPORT_PATH


class ProductionConfig(Config):
    """Production profile: strict secret handling, no debug."""

    DEBUG = False


def load_config():
    """Return a configured ``Config`` instance for the active runtime profile.

    Re-reads the environment on every call so tests can switch profiles by
    patching ``os.environ``.
    """
    config = ProductionConfig() if is_production() else Config()
    config.SECRET_KEY = resolve_secret_key()
    config.DEBUG = (
        os.getenv("ROADRISK_DEBUG", "").strip().lower() in ("1", "true", "yes")
        and not is_production()
    )
    cookie_secure = os.getenv("SESSION_COOKIE_SECURE", "").strip().lower()
    config.SESSION_COOKIE_SECURE = cookie_secure in ("1", "true", "yes")
    return config