# RoadRisk AI

RoadRisk AI is a machine-learning web application for road-accident severity
prediction, analytics, risk analysis, hotspot analysis, and explainable-AI
insights, built on the real RTA (Road Traffic Accident) dataset.

## Modules

- **Severity Prediction** – Predict accident severity (Slight / Serious / Fatal) with a trained XGBoost model and the exact preprocessing pipeline it was trained with.
- **Home dashboard** – Factual KPIs and charts computed from the cleaned dataset.
- **Analytics** – Filterable dashboards (area, severity, weather, road surface, road type, day). The dataset has **no date column**, so no date-range/year filter exists.
- **Risk Analysis** – Descriptive factor statistics plus a documented **composite risk score** (see below).
- **Hotspot Analysis** – Area-based **spatial proxy** (the dataset has **no coordinates** – no fabricated maps).
- **Explainable AI (SHAP)** – Local/global SHAP explanations from the saved XGBoost model; the prediction result also shows the top contributing factors inline.
- **Admin (RBAC)** – Role-based access control: `user` (default) and `admin` roles; `/admin` is admin-only.

## Tech Stack

- **Flask** – Web framework (session auth, no external auth dependency)
- **Python 3.12** – Core programming language
- **XGBoost** – Gradient boosting for predictions
- **SHAP** – Model explainability (with a compatibility fallback for SHAP/XGBoost version mismatches)
- **Plotly** – Interactive server-rendered charts
- **Pandas / Scikit-learn** – Data handling and ML utilities
- **SQLite** – User accounts (parameterised queries)
- **Gunicorn** – Production WSGI server

## Requirements

- **Python 3.12** (the pinned dependencies target 3.12)
- Dependencies are listed in `requirements.txt` (Flask, pandas, numpy,
  scikit-learn, xgboost, joblib, shap, plotly, matplotlib, python-dotenv,
  gunicorn, pytest).

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Environment variables

Configuration is loaded by `config.py`, which reads the real environment
first and falls back to a local `.env` file (via python-dotenv; `.env` is
gitignored – **never commit secrets**).

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `SECRET_KEY` | **Yes in production** | dev: random per-process key | Session signing key. Production (`ROADRISK_ENV=production`) refuses to start without it – there is no hardcoded fallback. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ROADRISK_ENV` | No | `development` | Runtime profile: `development` or `production` (also honours `FLASK_ENV`/`APP_ENV`) |
| `ROADRISK_ADMIN_EMAILS` | No | – | Comma-separated emails promoted to the `admin` role at startup (RBAC) |
| `ROADRISK_DB_PATH` | No | `./roadrisk.db` | SQLite database location (used by tests for isolation) |
| `PORT` / `HOST` | No | `5000` / `127.0.0.1` | Dev server bind (`python app.py`) / gunicorn port |
| `SESSION_COOKIE_SECURE` | No | off | Set to `1` when serving over HTTPS |
| `ROADRISK_DEBUG` | No | off | Dev-only debug toggle (never enabled in production) |

### Database initialization & migration

The SQLite database is created **automatically on startup** (`init_db()` in
`services/database.py`) – no manual step is needed for a fresh checkout:

- New databases get the full schema including `users.role` (`'user'` default).
- **Existing pre-RBAC databases are migrated in place**: `ALTER TABLE users
  ADD COLUMN role TEXT NOT NULL DEFAULT 'user'` is applied only when the
  column is missing, so existing accounts are preserved as regular users.
  The migration is idempotent (safe on every startup).

To grant admin rights, either set `ROADRISK_ADMIN_EMAILS` to an existing
account's email and restart, or promote in SQL:
`UPDATE users SET role='admin' WHERE email='...';`

### Generating the model artifacts (`models/*.pkl`)

`models/*.pkl` are **generated artifacts and intentionally gitignored** –
they must exist before running the app (the app degrades to a friendly
"artifacts unavailable" notice if they are missing). Regenerate them from the
committed raw dataset with the same ML pipeline used for development:

```bash
# Step 4: cleaning, feature engineering, encoding, scaling
#   -> dataset/processed_dataset.csv, models/label_encoders.pkl,
#      models/scaler.pkl, models/preprocessing_pipeline.pkl
python preprocessing/preprocessing_pipeline.py

# Step 5: baselines + XGBoost training/tuning + persistence
#   -> models/accident_severity_model.pkl, models/feature_columns.pkl,
#      reports/ (metrics, confusion matrix, classification report)
python training/train_model.py
```

The Jupyter notebooks under `notebooks/` document the same stages
(interactive form of Steps 1–6). The raw dataset
(`dataset/RTA Dataset.csv`) is never modified.

## Running locally

```bash
python app.py                 # http://127.0.0.1:5000
```

Register an account, then use the sidebar: Prediction → Analytics → Risk
Analysis → Hotspots → Explainability. Grant yourself admin by setting
`ROADRISK_ADMIN_EMAILS` and restarting to unlock the **Admin** page.

### Tests

```bash
pytest tests/ -v               # or: python -m unittest discover -s tests -v
```

The suite uses an isolated temporary SQLite file and the real saved model
artifacts, so it doubles as a live smoke test of every service.

## Composite risk score (Risk Analysis)

```
score = 100 × (0.6 × p_severe + 0.4 × f_severe)
```

- **p_severe** – saved XGBoost `predict_proba`: P(Serious Injury) + P(Fatal
  injury) for the record (model evidence).
- **f_severe** – mean *observed* serious/fatal share in the real dataset for
  each of the record's factor categories (weather, light, road surface,
  driver age band, driving experience, vehicle type, weekday, cause, area;
  categories with n ≥ 30, `Unknown` excluded; full-dataset share is the
  documented baseline when nothing qualifies – dataset evidence).
- **Range** 0–100 (1 decimal). Bands: Low < 25, Moderate 25–49.9, High
  50–74.9, Severe ≥ 75 (interpretation labels only).
- **Not** a probability, **not** a SHAP value, **not** the predicted class.
  The three stay distinct in the UI: class probabilities, composite score,
  and SHAP/gain attribution.
- Limitations: fixed (not fitted) weights; factor component covers the 9
  form-visible factor columns; reproducible for identical inputs.

## RBAC

- `users.role`: `'user'` (default) or `'admin'`.
- Self-registration always creates `'user'` – the role is never taken from
  user input. Passwords are hashed with werkzeug (PBKDF2-SHA256); all SQL is
  parameterised.
- `admin_required` guards `/admin` (anonymous → login redirect,
  non-admin → HTTP 403). Admins are granted via `ROADRISK_ADMIN_EMAILS`.

## Deployment

### Docker

```bash
# 1. Generate artifacts first (models/*.pkl are gitignored)
python preprocessing/preprocessing_pipeline.py
python training/train_model.py

# 2. Build & run (SECRET_KEY is mandatory in production)
docker build -t roadrisk-ai .
docker run -p 8000:8000 \
  -e SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  -e ROADRISK_ADMIN_EMAILS="you@example.com" \
  roadrisk-ai
```

### Gunicorn (bare Linux)

```bash
export ROADRISK_ENV=production
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
gunicorn -c gunicorn.conf.py wsgi:app      # binds 0.0.0.0:${PORT:-8000}
```

Behind TLS, also set `SESSION_COOKIE_SECURE=1`. Persist `roadrisk.db`
(e.g. a mounted volume) if you need accounts to survive redeployments.

### CI

`.github/workflows/tests.yml` installs dependencies, regenerates the
gitignored `models/*.pkl` artifacts with the documented pipeline (reduced
tuning iterations for runtime), and runs the full test suite.

## Project Structure

```
RoadRisk/
├── dataset/            # RTA Dataset.csv (raw, never modified) + processed_dataset.csv
├── models/             # Generated *.pkl artifacts (gitignored – see setup)
├── notebooks/          # Step 1-6 Jupyter notebooks + runners
├── preprocessing/      # Step 4 pipeline (cleaning, features, encoding, scaling)
├── reports/            # Generated evaluation reports (gitignored)
├── routes/             # Flask blueprints (auth, predict, dashboard, analytics,
│                       #   risk, hotspot, xai, admin)
├── services/           # Business logic (auth+RBAC, database, prediction,
│                       #   explainability, risk, hotspot, analytics, dashboard)
├── static/             # CSS / JS / vendored Plotly
├── templates/          # Jinja templates
├── tests/              # Automated test suite
├── training/           # Step 5 model training/evaluation/persistence
├── testing/            # Step 6 new-record preprocessing helpers (reused by app)
├── utils/              # Constants, logger, helpers, Plotly chart builders
├── app.py              # Flask application factory
├── wsgi.py             # Production WSGI entry point
├── gunicorn.conf.py    # Gunicorn configuration
├── config.py           # Environment-driven configuration (python-dotenv)
├── Dockerfile          # Production container
└── requirements.txt    # Pinned Python dependencies
```

## Known limitations (by design, not bugs)

- The dataset has **no geographic coordinates** – hotspots are an honest
  area-based spatial proxy, never a GPS map.
- The dataset has **no date column** – only time-of-day and day-of-week are
  used; there is no date-range/year filter.
- SHAP/XGBoost version mismatches degrade to a clear on-page notice (never
  HTTP 500) via the sanitised-booster fallback in
  `services/explainability_service.py`.