"""
Production WSGI entry point.

Used by Gunicorn (see gunicorn.conf.py / Dockerfile):

    gunicorn -c gunicorn.conf.py wsgi:app

``app`` is the module-level Flask application created by ``app.create_app()``
(config, database migration, RBAC promotion and model artifacts are applied
at import time, exactly like ``python app.py``).
"""
from app import app  # noqa: F401

if __name__ == "__main__":
    # Convenience fallback for platforms that run `python wsgi.py` directly.
    app.run(host="0.0.0.0", port=8000)