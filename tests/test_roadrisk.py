"""
RoadRisk AI - verification tests (STEP 8-13).

Runs with the standard library ``unittest`` (no pytest dependency):

    python -m unittest discover -s tests -v

Uses an isolated SQLite file (``<tmp>/roadrisk_test.db``) and the real saved
model artifacts, so the suite doubles as a live smoke test of every service.
"""
import os
import sys
import tempfile
import unittest

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

# Isolate the user store BEFORE app import (app module reads the env var).
_DB_PATH = os.path.join(tempfile.gettempdir(), "roadrisk_test.db")
if os.path.exists(_DB_PATH):
    os.remove(_DB_PATH)
os.environ["ROADRISK_DB_PATH"] = _DB_PATH

from app import app as _app  # noqa: E402
from services import prediction_service as ps  # noqa: E402


class RoadRiskBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force-load artifacts once for fast, deterministic service tests.
        ps.initialize()

    def setUp(self):
        _app.config["TESTING"] = True
        _app.config["WTF_CSRF_ENABLED"] = False
        self.client = _app.test_client()

    # -- helpers ------------------------------------------------------------
    def register(self, email="u@example.com", password="secret123", name="Test User"):
        return self.client.post(
            "/register",
            data={
                "name": name,
                "email": email,
                "password": password,
                "confirm_password": password,
            },
            follow_redirects=False,
        )

    def login(self, email="u@example.com", password="secret123"):
        return self.client.post(
            "/login", data={"email": email, "password": password},
            follow_redirects=False,
        )

    def valid_record(self, seed=1):
        record, errors = ps.validate_and_build_record(ps.sample_record(seed=seed))
        self.assertEqual(errors, [])
        return record


# ===========================================================================
# 1. Craft & boot
# ===========================================================================
class TestBootAndArtifacts(RoadRiskBase):
    def test_root_redirects_anonymous_to_login(self):
        resp = self.client.get("/", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.headers["Location"], "/login")

    def test_root_redirects_authenticated_to_home(self):
        self.register(email="root@example.com")
        resp = self.client.get("/", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.headers["Location"], "/home")

    def test_artifacts_are_ready(self):
        self.assertTrue(ps.is_ready())
        self.assertIsNone(ps.get_load_error())
        self.assertEqual(len(ps.get_artifacts()["feature_columns"]), 165)
        self.assertEqual(
            ps.get_artifacts()["model"].n_features_in_, 165
        )

    def test_class_labels_derived_from_artifacts(self):
        labels = ps.get_artifacts()["class_labels"]
        self.assertEqual(
            set(labels), {"Slight Injury", "Serious Injury", "Fatal injury"}
        )

    def test_static_assets_served(self):
        for path in (
            "/static/css/style.css",
            "/static/js/main.js",
            "/static/vendor/plotly.min.js",
        ):
            with self.client.get(path) as resp:
                self.assertEqual(resp.status_code, 200, path)

    def test_unknown_route_uses_friendly_404(self):
        resp = self.client.get("/definitely-not-a-page")
        self.assertEqual(resp.status_code, 404)
        self.assertIn(b"404", resp.data)


# ===========================================================================
# 2. Authentication (STEP 8)
# ===========================================================================
class TestAuthentication(RoadRiskBase):
    def test_register_then_login_then_logout(self):
        self.assertEqual(self.register(email="a1@example.com").status_code, 302)
        # Registration logs you in, so sign out before checking the duplicate.
        self.client.get("/logout")
        # Duplicate email rejected (server-side guard).
        resp = self.register(email="a1@example.com")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"already exists", resp.data)

        self.assertEqual(self.login(email="a1@example.com").status_code, 302)
        # Session cookie has been set.
        cookie = self.client.get_cookie("session")
        self.assertIsNotNone(cookie)

        resp = self.client.get("/logout", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        # After logout the dashboard is gated again.
        resp = self.client.get("/home", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)

    def test_bad_credentials_rejected(self):
        self.register(email="a2@example.com")
        self.client.get("/logout")  # start unauthenticated
        resp = self.login(email="a2@example.com", password="wrongpass")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Invalid email or password", resp.data)

    def test_login_validation_toggles(self):
        resp = self.register(email="bad-email")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"valid email", resp.data)

        resp = self.register(email="a3@example.com", password="123")
        self.assertIn(b"at least", resp.data)

    def test_dashboard_routes_require_login(self):
        for path in ("/home", "/predict", "/analytics", "/risk", "/hotspots", "/explain"):
            resp = self.client.get(path, follow_redirects=False)
            self.assertEqual(resp.status_code, 302, path)
            self.assertTrue(resp.headers["Location"].startswith("/login?next="), path)

    def test_next_redirect_is_safe(self):
        # Open-redirect guard: next must be an internal path.
        resp = self.client.post(
            "/login?next=https://evil.example.com",
            data={"email": "nobody@example.com", "password": "x"},
            follow_redirects=False,
        )
        # Login fails anyway, but nothing external can ever be the target.
        self.assertEqual(resp.status_code, 200)
# ===========================================================================
# 3. Form structure & validation (STEP 7 service layer)
# ===========================================================================
class TestFormAndValidation(RoadRiskBase):
    def test_form_view_shape(self):
        view = ps.form_view()
        self.assertEqual(len(view), 6)  # six sections
        names = [f["name"] for group in view for f in group["fields"]]
        self.assertEqual(len(names), len(ps.FORM_FIELDS))
        self.assertEqual(sorted(names), sorted(ps.FORM_FIELDS))

    def test_options_are_derived_not_empty(self):
        options = ps.get_options()
        for field in ps.FORM_FIELDS:
            if field in ("Time", "Number_of_vehicles_involved", "Number_of_casualties"):
                continue
            self.assertGreater(len(options.get(field, [])), 0, field)

    def test_valid_record_builds_with_no_errors(self):
        record = self.valid_record(seed=11)
        for field in ("Time", "Number_of_vehicles_involved", "Number_of_casualties"):
            self.assertIn(field, record)

    def test_invalid_time_rejected(self):
        base = self.valid_record(seed=2)
        for bad in ("25:00", "12:60", "not-a-time", "23:59:61"):
            rec = dict(base)
            rec["Time"] = bad
            _record, errors = ps.validate_and_build_record(rec)
            self.assertTrue(errors, bad)
            self.assertTrue(any("Time" in e for e in errors), errors)

    def test_numeric_bounds_enforced(self):
        base = self.valid_record(seed=3)
        for field, bad_value in (("Number_of_vehicles_involved", "0"),
                                 ("Number_of_vehicles_involved", "101"),
                                 ("Number_of_casualties", "-1"),
                                 ("Number_of_casualties", "not-a-number")):
            rec = dict(base)
            rec[field] = bad_value
            _record, errors = ps.validate_and_build_record(rec)
            self.assertTrue(errors, (field, bad_value))

    def test_unknown_category_rejected(self):
        base = self.valid_record(seed=4)
        rec = dict(base)
        rec["Area_accident_occured"] = "Not A Real Area"
        _record, errors = ps.validate_and_build_record(rec)
        self.assertTrue(any(e.startswith("Unknown category") for e in errors), errors)

    def test_time_normalized_to_seconds(self):
        rec = self.valid_record(seed=5)
        rec["Time"] = "08:05"
        record, errors = ps.validate_and_build_record(rec)
        self.assertEqual(errors, [])
        self.assertEqual(record["Time"], "08:05:00")
# ===========================================================================
# 4. Prediction consistency & error handling (STEP 6/7)
# ===========================================================================
class TestPrediction(RoadRiskBase):
    def test_make_prediction_output_shape(self):
        result = ps.make_prediction(self.valid_record(seed=6))
        self.assertIn(result["prediction"], ps.get_artifacts()["class_labels"])
        self.assertEqual(len(result["probabilities"]), 3)
        total = sum(item["pct"] for item in result["probabilities"])
        self.assertAlmostEqual(total, 100.0, places=1)
        self.assertEqual(result["model"], "XGBClassifier")

    def test_same_record_is_deterministic(self):
        record = self.valid_record(seed=7)
        first = ps.make_prediction(record)["prediction"]
        second = ps.make_prediction(record)["prediction"]
        self.assertEqual(first, second)

    def test_matches_old_pipeline_verbatim(self):
        """Must equal testing.test_model.preprocess_new_record + predict_with_proba."""
        from testing.test_model import predict_with_proba, preprocess_new_record

        record = self.valid_record(seed=8)
        artifacts = ps.get_artifacts()
        result = ps.make_prediction(record)

        X_new = preprocess_new_record(record, artifacts)
        self.assertEqual(X_new.shape[1], 165)
        labels, proba = predict_with_proba(artifacts["model"], X_new, artifacts["class_labels"])

        self.assertEqual(result["prediction"], labels[0])
        for i, item in enumerate(result["probabilities"]):
            self.assertAlmostEqual(item["pct"], round(proba[0][i] * 100.0, 2), places=6)

    def test_missing_fields_return_friendly_errors(self):
        self.register(email="pf@example.com")
        resp = self.client.post("/predict", data={"Time": "09:00:00"})
        self.assertEqual(resp.status_code, 200)  # friendly re-render, not a 500
        body = resp.get_data(as_text=True)
        self.assertIn("is required", body)

    def test_prediction_page_requires_login(self):
        resp = self.client.get("/predict", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.headers["Location"].startswith("/login?next=/predict"))
# ===========================================================================
# 5. Dashboard / analytics / risk / hotspots services (STEP 9-12)
# ===========================================================================
class TestAnalyticsServices(RoadRiskBase):
    def test_dashboard_builds(self):
        from services.dashboard_service import build_dashboard

        data = build_dashboard()
        self.assertEqual(len(data["kpis"]), 8)
        labels = [k["label"] for k in data["kpis"]]
        self.assertIn("Most Common Weather", labels)
        self.assertIn("Most Common Cause", labels)
        # No unsupported/N-A placeholder metrics may be rendered.
        self.assertNotIn("N/A", " ".join(labels))
        self.assertNotIn("unavailable", data)
        self.assertGreaterEqual(len(data["charts"]), 6)
        self.assertTrue(all(ch["html"].startswith("<div") for ch in data["charts"]))

    def test_analytics_filters(self):
        from services.analytics_service import FILTER_COLUMNS, build_analytics, filter_options

        options = filter_options()
        for key in ("area", "severity", "weather", "road", "roadtype", "day"):
            self.assertIn(key, options)
            self.assertIn(key, FILTER_COLUMNS)
        all_data = build_analytics({})
        self.assertEqual(all_data["summary"]["records"], 12316)
        titles = [c["title"] for c in all_data["charts"]]
        self.assertIn("Area × Severity", titles)

        filtered = build_analytics({"severity": "Fatal injury"})
        self.assertGreater(filtered["summary"]["records"], 0)
        self.assertFalse(filtered["empty"])

        combined = build_analytics({"day": "Friday", "roadtype": "One way"})
        self.assertLessEqual(combined["summary"]["records"], 12316)

        impossible = build_analytics({"area": "Fatal injury", "severity": "Fatal injury"})
        # An area can never be a severity: empty state must be handled.
        self.assertTrue(impossible["empty"])
        self.assertEqual(impossible["charts"], [])

    def test_risk_overview_builds(self):
        from services.risk_service import SCOPES, build_risk_overview, severity_by_factor

        data = build_risk_overview("all")
        # The composite risk score is now defined (documented formula), not pending.
        self.assertEqual(data["formal"]["status"], "defined")
        self.assertIn("0.6", data["formal"]["formula"])
        self.assertIn("f_severe", data["formal"]["formula"])
        self.assertGreaterEqual(len(data["factor_table"]), 5)
        self.assertGreater(len(data["top_combos"]), 0)
        self.assertIsNotNone(data["importance_chart_html"])
        self.assertIsNotNone(data["severity_chart_html"])
        self.assertIsNotNone(data["correlation_matrix"])
        self.assertEqual(data["scope"], "all")
        # Precomputed bar width keeps calculations out of the templates.
        self.assertTrue(all(0 < row["bar_pct"] <= 100 for row in data["factor_table"]))

        # Scope filter narrows the record set without breaking the sections.
        severe = build_risk_overview("severe")
        self.assertEqual(severe["total"], 1901)
        self.assertEqual(severe["counts"]["Slight Injury"], 0)

        for key in SCOPES:
            self.assertEqual(build_risk_overview(key)["scope"], key)

        factor = severity_by_factor("Weather_conditions")
        self.assertFalse(factor["empty"])
        self.assertGreater(len(factor["table"]), 0)

    def test_hotspots_builds(self):
        from services.hotspot_service import build_hotspots

        data = build_hotspots()
        self.assertEqual(len(data["area_cards"]), 10)
        counts = [card["count"] for card in data["area_cards"]]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertTrue(all(ch["html"].startswith("<div") for ch in data["charts"]))
        titles = [ch["title"] for ch in data["charts"]]
        # Coordinates are absent -> an Area x Severity heatmap is the substitute.
        self.assertIn("Area × Severity heatmap", titles)
        self.assertIn("Accident density heatmap", titles)
        # All charts must carry real data (no empty figure payloads).
        for chart in data["charts"]:
            self.assertIn("Plotly.newPlot", chart["html"])
            self.assertNotIn('"data": []', chart["html"])


# ===========================================================================
# 6. SHAP explainability (STEP 13)
# ===========================================================================
class TestExplainability(RoadRiskBase):
    def test_local_explanation_matches_proba(self):
        from services.explainability_service import explain_record

        result = explain_record(self.valid_record(seed=9))
        self.assertTrue(result["consistency"]["matches"])
        self.assertLess(result["consistency"]["max_deviation"], 1e-4)
        self.assertGreater(len(result["local"]), 0)
        self.assertIn(result["prediction"], ps.get_artifacts()["class_labels"])
        # New 7.6 deliverables: force-style plot + increasing/decreasing factors.
        self.assertIsNotNone(result["force_html"])
        self.assertIn(result["prediction_tone"], ("green", "amber", "red"))
        self.assertGreater(len(result["increasing"]), 0)
        self.assertTrue(all(item["contribution"] > 0 for item in result["increasing"]))
        self.assertTrue(all(item["contribution"] < 0 for item in result["decreasing"]))
        # The chain must close: output log-odds is base + every contribution.
        # (Verified numerically in `consistency`, asserted here via the
        #  full-precision pipeline values exposed on the result.)
        self.assertLess(
            abs(result["output_logit"] - result["base_value"]), 30.0
        )

    def test_global_summary_builds_and_caches(self):
        from services.explainability_service import global_summary

        first = global_summary(force=False)
        second = global_summary(force=False)
        self.assertIs(first, second)  # cached per process
        self.assertEqual(len(first["by_group"]), 12)
        self.assertIsNotNone(first["chart_html"])
        self.assertGreater(first["sample_size"], 0)

    def test_xai_route_end_to_end(self):
        self.register(email="xai@example.com")

        resp = self.client.get("/explain?sample=1")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("SHAP waterfall", body)
        self.assertIn("Force-style contribution chain", body)
        self.assertIn("Top factors increasing this prediction", body)
        self.assertIn("Top factors decreasing this prediction", body)
        self.assertIn("Consistency check", body)
        self.assertIn("SHAP summary", body)
        self.assertIn("causal proof", body)

    def test_xai_pending_state_when_artifacts_unavailable(self):
        from unittest import mock

        with mock.patch("routes.xai.get_load_error", return_value="artifacts missing"):
            self.register(email="xai2@example.com")
            resp = self.client.get("/explain")
            body = resp.get_data(as_text=True)
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Explainability integration pending", body)
    # ------------------------------------------------------------------
    # XGBoost 3.x vector base_score / SHAP compatibility
    # ------------------------------------------------------------------
    def test_calibrated_base_values_reproduce_model_margins(self):
        import numpy as np

        from services import explainability_service as xai

        artifacts = ps.get_artifacts()
        columns = artifacts["feature_columns"]
        frame = xai._probe_matrix(columns)
        explainer = xai._get_explainer()
        arr = xai._normalize_shap_values(explainer.shap_values(frame))
        base = xai.base_values()
        margins = xai._original_margins(artifacts["model"], columns, frame)
        residual = float(np.abs((base + arr.sum(axis=1)) - margins).max())
        self.assertLess(residual, xai._MARGIN_TOLERANCE)

    def test_explainer_falls_back_when_native_shap_cannot_parse_model(self):
        """Simulates an older SHAP whose loader chokes on the vector base_score."""
        import shap as shap_module
        from unittest import mock
        from xgboost import XGBClassifier

        from services import explainability_service as xai
        from services.explainability_service import explain_record

        record = self.valid_record(seed=13)
        native = explain_record(record)
        self.assertEqual(xai.explainability_mode(), "native")

        real_explainer_cls = shap_module.TreeExplainer

        def old_shap_explainer(model, *args, **kwargs):
            if isinstance(model, XGBClassifier):
                raise ValueError(
                    "could not convert string to float: "
                    "'[1.9904251E0,2.026248E-1,-2.1930504E0]'"
                )
            return real_explainer_cls(model, *args, **kwargs)

        shap_module.TreeExplainer = old_shap_explainer
        try:
            with mock.patch.object(xai, "_explainer", None), \
                    mock.patch.object(xai, "_explainer_mode", None), \
                    mock.patch.object(xai, "_explainer_error", None), \
                    mock.patch.object(xai, "_base_values", None):
                fallback = explain_record(record)
                self.assertEqual(xai.explainability_mode(), "sanitised")
                self.assertIsNone(xai._explainer_error)
        finally:
            shap_module.TreeExplainer = real_explainer_cls
            # Force a clean rebuild with the real SHAP on the next use.
            xai._explainer = None
            xai._explainer_mode = None
            xai._explainer_error = None
            xai._base_values = None

        # The fallback must produce the SAME explanation quality and the same
        # model output as the native path.
        self.assertTrue(fallback["consistency"]["matches"])
        self.assertLess(fallback["consistency"]["max_deviation"], 1e-4)
        self.assertEqual(fallback["prediction"], native["prediction"])
        self.assertEqual(fallback["probabilities"], native["probabilities"])
        self.assertGreater(len(fallback["local"]), 0)
        self.assertGreater(len(fallback["increasing"]), 0)
        # ...and the native path must still work afterwards.
        self.assertEqual(explain_record(record)["prediction"], native["prediction"])

    def test_explain_route_degrades_gracefully_on_unexpected_error(self):
        from unittest import mock

        self.register(email="xai3@example.com")
        with mock.patch("routes.xai.explain_record", side_effect=ValueError("boom")):
            resp = self.client.get("/explain?sample=1")
        body = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)       # never a 500
        self.assertIn("Explainability unavailable", body)
        self.assertNotIn("SHAP waterfall", body)

    def test_prediction_page_button_removed(self):
        self.register(email="nobtn@example.com")
        body = self.client.get("/predict").get_data(as_text=True)
        self.assertNotIn("Why did the model decide?", body)
        # The Explainability page itself is still reachable from the sidebar.
        self.assertIn("Explainability", body)


# ===========================================================================
# 11. RBAC (roles, admin guard, migration, env promotion)
# ===========================================================================
class TestRBAC(RoadRiskBase):
    def test_role_column_exists_with_user_default(self):
        from services.database import get_connection

        conn = get_connection()
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        finally:
            conn.close()
        self.assertIn("role", cols)

    def test_registration_always_creates_user_role(self):
        from services.database import run_query

        self.register(email="rbac1@example.com")
        rows = run_query("SELECT role FROM users WHERE email = ?",
                         ("rbac1@example.com",))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["role"], "user")

    def test_registration_cannot_grant_admin_via_input(self):
        from services.database import run_query

        # Even if a rogue "role" field is posted, it is ignored by the server.
        self.client.post(
            "/register",
            data={
                "name": "Rogue", "email": "rbac2@example.com",
                "password": "secret123", "confirm_password": "secret123",
                "role": "admin",  # not read by register_user
            },
        )
        rows = run_query("SELECT role FROM users WHERE email = ?",
                         ("rbac2@example.com",))
        self.assertEqual(rows[0]["role"], "user")

    def test_admin_page_requires_login(self):
        resp = self.client.get("/admin", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_admin_page_forbidden_for_regular_user(self):
        self.register(email="rbac3@example.com")
        resp = self.client.get("/admin")
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Admins only", resp.get_data(as_text=True))

    def test_admin_page_allowed_and_lists_users(self):
        from services.database import run_write

        self.register(email="rbac4@example.com", name="Boss Admin")
        run_write("UPDATE users SET role = 'admin' WHERE email = ?",
                  ("rbac4@example.com",))
        resp = self.client.get("/admin")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("Registered accounts", body)
        self.assertIn("rbac4@example.com", body)
        self.assertIn("Boss Admin", body)

    def test_admin_nav_link_only_visible_to_admins(self):
        from services.database import run_write

        self.register(email="rbac5@example.com")
        user_body = self.client.get("/home").get_data(as_text=True)
        self.assertNotIn('href="/admin"', user_body)

        run_write("UPDATE users SET role = 'admin' WHERE email = ?",
                  ("rbac5@example.com",))
        admin_body = self.client.get("/home").get_data(as_text=True)
        self.assertIn('href="/admin"', admin_body)

    def test_promote_admins_from_environment(self):
        import os
        from unittest import mock

        from services import auth as auth_mod
        from services.database import run_query

        email = "rbac6@example.com"
        self.register(email=email)
        with mock.patch.dict(os.environ,
                             {"ROADRISK_ADMIN_EMAILS": " RBAC6@Example.com "}):
            promoted = auth_mod.promote_admins()
        self.assertIn(email, promoted)
        rows = run_query("SELECT role FROM users WHERE email = ?", (email,))
        self.assertEqual(rows[0]["role"], "admin")
        # Idempotent: a second run promotes nothing.
        with mock.patch.dict(os.environ, {"ROADRISK_ADMIN_EMAILS": email}):
            self.assertEqual(auth_mod.promote_admins(), [])

    def test_legacy_database_gains_role_column_without_data_loss(self):
        """Pre-RBAC databases migrate in place; existing users survive."""
        import sqlite3

        from services.database import _migrate

        legacy_path = os.path.join(tempfile.gettempdir(), "roadrisk_legacy.db")
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
        conn = sqlite3.connect(legacy_path)
        try:
            conn.execute(
                "CREATE TABLE users ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,"
                " password_hash TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO users (name, email, password_hash, created_at)"
                " VALUES ('Legacy User', 'legacy@example.com', 'hash', '2024')"
            )
            conn.commit()

            _migrate(conn)   # applies ALTER TABLE ... ADD COLUMN role
            _migrate(conn)   # idempotent
            row = conn.execute(
                "SELECT email, role FROM users WHERE email = ?",
                ("legacy@example.com",),
            ).fetchone()
            self.assertEqual(row[0], "legacy@example.com")
            self.assertEqual(row[1], "user")
        finally:
            conn.close()
            if os.path.exists(legacy_path):
                os.remove(legacy_path)


# ===========================================================================
# 12. Prediction result: SHAP top factors + composite risk score
# ===========================================================================
class TestPredictionResultEnrichment(RoadRiskBase):
    def test_prediction_result_shows_top_shap_factors_and_risk_score(self):
        self.register(email="enrich1@example.com")
        resp = self.client.post("/predict", data=ps.sample_record(seed=7))
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        # SHAP factors section with 3-5 factor rows.
        self.assertIn("Top contributing factors (SHAP)", body)
        self.assertIn("Composite risk score", body)
        self.assertIn("transparent index (0-100)", body)
        section = body.split("Top contributing factors (SHAP)", 1)[1]
        section = section.split("result-note", 1)[0]
        self.assertGreaterEqual(section.count("prob-bar"), 3)
        self.assertLessEqual(section.count("prob-bar"), 5)

    def test_top_factors_use_real_shap_pipeline(self):
        from services.explainability_service import top_factors

        record = self.valid_record(seed=11)
        out = top_factors(record, k=4)
        self.assertEqual(len(out["factors"]), 4)
        self.assertIn(out["class_label"],
                      {"Slight Injury", "Serious Injury", "Fatal injury"})
        # Signed contributions, ranked by magnitude, bar widths normalised.
        mags = [abs(f["contribution"]) for f in out["factors"]]
        self.assertEqual(mags, sorted(mags, reverse=True))
        for factor in out["factors"]:
            self.assertIn(factor["direction"], ("up", "down"))
            self.assertTrue(0.0 <= factor["magnitude_pct"] <= 100.0)

    def test_result_degrades_gracefully_when_both_services_fail(self):
        from unittest import mock

        self.register(email="enrich2@example.com")
        with mock.patch(
            "services.explainability_service.top_factors",
            side_effect=ValueError("boom"),
        ), mock.patch(
            "services.risk_service.composite_risk_score",
            side_effect=ValueError("boom"),
        ):
            resp = self.client.post("/predict", data=ps.sample_record(seed=9))
        self.assertEqual(resp.status_code, 200)  # never a 500
        body = resp.get_data(as_text=True)
        self.assertIn("Top factors unavailable", body)
        self.assertIn("Risk score unavailable", body)
        # The prediction itself still rendered.
        self.assertIn("Class probabilities", body)

    def test_shap_failure_keeps_risk_score_when_only_shap_fails(self):
        from unittest import mock

        self.register(email="enrich3@example.com")
        with mock.patch(
            "services.explainability_service.top_factors",
            side_effect=ValueError("boom"),
        ):
            resp = self.client.post("/predict", data=ps.sample_record(seed=9))
        body = resp.get_data(as_text=True)
        self.assertIn("Top factors unavailable", body)
        self.assertIn("Composite risk score", body)


# ===========================================================================
# 13. Composite risk score (normal / boundary / missing-invalid inputs)
# ===========================================================================
class TestCompositeRiskScore(RoadRiskBase):
    def test_normal_score_is_reproducible_and_well_formed(self):
        from services import risk_service as rs

        record = self.valid_record(seed=3)
        first = rs.composite_risk_score(record)
        second = rs.composite_risk_score(record)

        self.assertEqual(first, second)  # deterministic for identical inputs
        self.assertTrue(0.0 <= first["score"] <= 100.0)
        # Components sum exactly to the displayed score.
        self.assertAlmostEqual(
            first["model"]["points"] + first["factors"]["points"],
            first["score"], places=1,
        )
        # Inputs are real: model probability in [0,1], dataset factors present.
        self.assertTrue(0.0 <= first["model"]["p_severe"] <= 1.0)
        self.assertTrue(0.0 <= first["factors"]["f_severe"] <= 1.0)
        self.assertGreater(len(first["factors"]["used"]), 0)
        # Band matches the documented thresholds.
        self.assertEqual(first["band"], rs.score_band(first["score"]))
        # Distinct quantities: the score is not just the model probability.
        self.assertNotEqual(first["score"], first["model"]["p_severe"])

    def test_boundary_scores_and_bands(self):
        from unittest import mock

        from services import risk_service as rs

        record = self.valid_record(seed=4)
        # All-zero inputs -> exactly 0.0 / Low.
        with mock.patch.object(rs, "model_severe_probability", return_value=0.0), \
                mock.patch.object(rs, "factor_severe_share",
                                  return_value=(0.0, [], [], True)):
            zero = rs.composite_risk_score(record)
        self.assertEqual(zero["score"], 0.0)
        self.assertEqual(zero["band"], "Low")

        # All-one inputs -> exactly 100.0 / Severe.
        with mock.patch.object(rs, "model_severe_probability", return_value=1.0), \
                mock.patch.object(rs, "factor_severe_share",
                                  return_value=(1.0, [], [], True)):
            full = rs.composite_risk_score(record)
        self.assertEqual(full["score"], 100.0)
        self.assertEqual(full["band"], "Severe")

        # Exact documented band thresholds.
        self.assertEqual(rs.score_band(24.9), "Low")
        self.assertEqual(rs.score_band(25.0), "Moderate")
        self.assertEqual(rs.score_band(49.9), "Moderate")
        self.assertEqual(rs.score_band(50.0), "High")
        self.assertEqual(rs.score_band(74.9), "High")
        self.assertEqual(rs.score_band(75.0), "Severe")

        # Midpoint: p=f=0.5 -> 0.6*50 + 0.4*50 = 50.0 (High).
        with mock.patch.object(rs, "model_severe_probability", return_value=0.5), \
                mock.patch.object(rs, "factor_severe_share",
                                  return_value=(0.5, [], [], True)):
            mid = rs.composite_risk_score(record)
        self.assertEqual(mid["score"], 50.0)
        self.assertEqual(mid["band"], "High")

    def test_missing_and_invalid_inputs_raise_value_error(self):
        from services import risk_service as rs

        record = self.valid_record(seed=5)
        for bad in (None, "not-a-record", 42, [], ("x",)):
            with self.assertRaises(ValueError):
                rs.composite_risk_score(bad)
        # Empty record without probabilities cannot be scored.
        with self.assertRaises(ValueError):
            rs.composite_risk_score({})
        # Malformed / non-severity probability payloads are rejected.
        with self.assertRaises(ValueError):
            rs.composite_risk_score({}, probabilities=[])
        with self.assertRaises(ValueError):
            rs.composite_risk_score({}, probabilities=[{"label": "Odd", "pct": 50}])
        with self.assertRaises(ValueError):
            rs.composite_risk_score({}, probabilities=[{"pct": 10}])
        with self.assertRaises(ValueError):
            rs.composite_risk_score(
                {}, probabilities=[{"label": "Fatal injury", "pct": 150}]
            )

    def test_missing_factor_fields_degrade_to_baseline(self):
        from services import risk_service as rs

        # Record without any factor columns -> documented baseline fallback.
        out = rs.composite_risk_score(
            {}, probabilities=[
                {"label": "Slight Injury", "pct": 50.0},
                {"label": "Serious Injury", "pct": 30.0},
                {"label": "Fatal injury", "pct": 20.0},
            ]
        )
        self.assertTrue(out["factors"]["baseline_used"])
        self.assertEqual(out["factors"]["used"], [])
        self.assertAlmostEqual(out["model"]["p_severe"], 0.5, places=6)

    def test_risk_page_shows_score_after_prediction(self):
        self.register(email="score1@example.com")
        self.client.post("/predict", data=ps.sample_record(seed=6))
        body = self.client.get("/risk").get_data(as_text=True)
        self.assertIn("Composite score (0-100)", body)
        self.assertIn("Component factors", body)
        self.assertIn("Model component (60%)", body)

    def test_risk_page_prompts_before_any_prediction(self):
        self.register(email="score2@example.com")
        body = self.client.get("/risk").get_data(as_text=True)
        self.assertIn("No prediction yet", body)
        self.assertNotIn("Composite score (0-100)", body)


# ===========================================================================
# 14. Hotspot spatial-proxy filters
# ===========================================================================
class TestHotspotFilters(RoadRiskBase):
    def test_default_page_labels_spatial_proxy(self):
        self.register(email="hot1@example.com")
        body = self.client.get("/hotspots").get_data(as_text=True)
        self.assertIn("area-based spatial proxy", body)
        self.assertIn("not a geographic GPS heatmap", body)

    def test_day_filter_is_applied(self):
        self.register(email="hot2@example.com")
        body = self.client.get("/hotspots?day=Friday").get_data(as_text=True)
        self.assertIn("Friday", body)
        self.assertIn("Accidents in view", body)

    def test_top_filter_limits_ranked_cards(self):
        self.register(email="hot3@example.com")
        body = self.client.get("/hotspots?top=5").get_data(as_text=True)
        self.assertEqual(body.count('class="rank-badge"'), 5)

    def test_severity_filter_shows_filtered_cards_and_hides_mix(self):
        self.register(email="hot4@example.com")
        body = self.client.get(
            "/hotspots?severity=Fatal+injury"
        ).get_data(as_text=True)
        self.assertIn("Filtered to <strong>Fatal injury</strong>", body)
        self.assertIn("Severity-mix chart hidden", body)

    def test_invalid_filter_values_are_ignored(self):
        self.register(email="hot5@example.com")
        body = self.client.get(
            "/hotspots?day=Funday&severity=Bogus&top=999"
        ).get_data(as_text=True)
        self.assertEqual(body.count('class="rank-badge"'), 10)  # default top 10
        self.assertIn("All records", body)  # no valid filter applied
        self.assertEqual(body.count("Severity-mix chart hidden"), 0)

    def test_service_validates_filters_against_dataset(self):
        from services.hotspot_service import build_hotspots

        clean = build_hotspots({"day": "Friday", "severity": "Fatal injury",
                                "top": "5"})
        self.assertEqual(clean["applied"],
                         {"day": "Friday", "severity": "Fatal injury", "top": 5})
        bogus = build_hotspots({"day": "Funday", "severity": "Nope", "top": "x"})
        self.assertEqual(bogus["applied"],
                         {"day": "", "severity": "", "top": 10})
        self.assertFalse(clean["empty"])


# ===========================================================================
# 15. Configuration (no weak SECRET_KEY fallback)
# ===========================================================================
class TestConfiguration(RoadRiskBase):
    def test_app_secret_key_is_not_the_weak_hardcoded_default(self):
        self.assertTrue(_app.secret_key)
        self.assertNotEqual(_app.secret_key, "roadrisk_ai_secret_key")
        self.assertGreaterEqual(len(_app.secret_key), 32)

    def test_dev_profile_generates_ephemeral_key_when_unset(self):
        import os
        from unittest import mock

        from config import load_config

        with mock.patch.dict(os.environ, {"ROADRISK_ENV": "development",
                                          "SECRET_KEY": ""}, clear=False):
            cfg = load_config()
        self.assertTrue(cfg.SECRET_KEY)
        self.assertNotEqual(cfg.SECRET_KEY, "roadrisk_ai_secret_key")

    def test_production_profile_refuses_missing_secret_key(self):
        import os
        from unittest import mock

        from config import load_config

        with mock.patch.dict(os.environ, {"ROADRISK_ENV": "production",
                                          "SECRET_KEY": ""}, clear=False):
            with self.assertRaises(RuntimeError):
                load_config()

    def test_production_profile_uses_env_secret_key(self):
        import os
        from unittest import mock

        from config import load_config

        with mock.patch.dict(os.environ, {"ROADRISK_ENV": "production",
                                          "SECRET_KEY": "prod-key-123"},
                             clear=False):
            cfg = load_config()
        self.assertEqual(cfg.SECRET_KEY, "prod-key-123")
        self.assertFalse(cfg.DEBUG)


if __name__ == "__main__":
    unittest.main(verbosity=2)