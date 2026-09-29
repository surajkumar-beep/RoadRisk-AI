"""
Gunicorn production configuration.

    gunicorn -c gunicorn.conf.py wsgi:app

Environment variables (all optional except SECRET_KEY in production):
    PORT               - listen port (default 8000)
    WEB_CONCURRENCY    - worker processes (default 2)
    GUNICORN_TIMEOUT   - worker timeout seconds (default 120)
"""
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = int(os.getenv('WEB_CONCURRENCY', '2'))
timeout = int(os.getenv('GUNICORN_TIMEOUT', '120'))
graceful_timeout = 30

# Logging to stdout (container-friendly).
accesslog = '-'
errorlog = '-'
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')