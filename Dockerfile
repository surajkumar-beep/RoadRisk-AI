# RoadRisk AI - production container image.
#
# IMPORTANT: models/*.pkl are gitignored generated artifacts. Build them
# BEFORE `docker build` (they are copied into the image below):
#     python preprocessing/preprocessing_pipeline.py
#     python training/train_model.py
# See README "Generating the model artifacts".
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ROADRISK_ENV=production \
    PORT=8000

WORKDIR /app

# Dependencies first (layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (models/ must exist at this point - see header note).
COPY . .

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# SECRET_KEY must be provided at runtime (production refuses weak fallbacks):
#   docker run -e SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))") ...
CMD ["sh", "-c", "gunicorn -c gunicorn.conf.py wsgi:app"]