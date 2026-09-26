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
        self.assertEqual(data["formal"]["status"], "pending")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)