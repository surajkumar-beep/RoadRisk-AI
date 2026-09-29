"""
STEP 13 - Explainability service (genuine SHAP explanations).

Uses ``shap.TreeExplainer`` on the SAVED XGBoost model to produce:

  * LOCAL  - per-feature contributions for a single accident record,
             shown as a waterfall in log-odds space.  The returned
             ``consistency`` dict verifies shap_values round-trips into the
             exact same probabilities as ``model.predict_proba`` (a real
             numerical check, not an assertion of an ideal model).
  * GLOBAL - mean |SHAP| per feature over a fixed random sample of the real
             processed dataset, grouped by original column for readability.

All values are raw SHAP outputs; nothing is invented or hand-adjusted.

XGBoost 3.x / SHAP compatibility
--------------------------------
XGBoost >= 3.1 stores a *vector* ``base_score`` for multiclass boosters, e.g.
``"[1.9904251E0,2.026248E-1,-2.1930504E0]"``.  Recent SHAP releases parse that
form, but older releases call ``float()`` on the raw string inside
``XGBTreeModelLoader`` and die with::

    ValueError: could not convert string to float:
    '[1.9904251E0,2.026248E-1,-2.1930504E0]'

To stay compatible with BOTH SHAP generations this module:

  1. tries the native ``shap.TreeExplainer(saved_model)`` first;
  2. if that (or its additive-consistency probe) fails, rebuilds the *same*
     trees as an ``xgboost.Booster`` whose JSON only differs in a scalar
     ``base_score`` - the trees are byte-identical, so per-feature
     attributions are unchanged - and explains that booster instead;
  3. restores the TRUE per-class intercepts by calibrating against the
     original model's margins, so ``base + sum(shap_values)`` still equals the
     original model's output exactly.

The saved model is never modified, retrained or re-serialised on disk, and
predictions keep flowing through the original estimator untouched.
"""
import json
import logging

import numpy as np
import shap
import xgboost

from services.data import load_processed_dataset
from services.prediction_service import (
    TONE_BY_LABEL,
    get_artifacts,
    preprocess_new_record,
)
from utils import plots

logger = logging.getLogger("RoadRiskAI.explainability")

# Deterministic sample for the global view (fast: ~0.4 s measured on this model).
GLOBAL_SAMPLE_N = 300
GLOBAL_RANDOM_STATE = 42
TOP_LOCAL_FEATURES = 12
TOP_GLOBAL_FEATURES = 20
TOP_GLOBAL_GROUPS = 12

# Probe rows used to validate/calibrate the explainer against the real model.
_PROBE_ROWS = 4
# Maximum tolerated |(base + sum(shap)) - model margin| in log-odds space.
_MARGIN_TOLERANCE = 1e-3
# Scalar placeholder written into the sanitised booster (real intercepts are
# restored during calibration; this value never influences the UI output).
_SANITISED_BASE_SCORE = "5E-1"

_explainer = None
_explainer_mode = None
_explainer_error = None
_base_values = None
_global_cache = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def pretty_feature(name):
    """Readable human label for a one-hot / numeric model feature name."""
    if "_" not in name:
        return name.replace("_", " ").title()
    prefix, value = name.split("_", 1)
    return f"{value.replace('_', ' ').title()} ({prefix.replace('_', ' ')})"


def _group_of(name):
    """Group one-hot columns by their original raw column."""
    return name.split("_", 1)[0] if "_" in name else name


# ---------------------------------------------------------------------------
# explainer construction (native SHAP, with a version-compatible fallback)
# ---------------------------------------------------------------------------


def _probe_matrix(columns):
    """Small deterministic slice of the real scaled matrix for calibration."""
    frame = load_processed_dataset()
    frame = frame.reindex(columns=list(columns))
    frame = frame.dropna()
    if frame.empty:
        raise RuntimeError(
            "The processed dataset does not match the model's feature columns; "
            "cannot calibrate the explainer."
        )
    return frame.head(_PROBE_ROWS)


def _original_margins(model, columns, frame):
    """Raw (log-odds) model output for the probe rows, from the SAVED model."""
    booster = model.get_booster()
    dmat = xgboost.DMatrix(frame.values, feature_names=list(booster.feature_names) or None)
    margins = booster.predict(dmat, output_margin=True)
    return np.atleast_2d(margins)


def _read_base_score_vector(booster):
    """Parse the booster's base_score, accepting the XGBoost >= 3 vector form."""
    config = json.loads(booster.save_config())
    raw = str(config["learner"]["learner_model_param"]["base_score"]).strip()
    if raw.startswith("["):
        return [float(value) for value in json.loads(raw)]
    return [float(raw)]


def _build_sanitised_booster(booster):
    """Rebuild the same trees with a scalar base_score for older SHAP loaders.

    Only the constant intercept changes; the tree structure and every leaf are
    copied verbatim, so per-feature attributions are unaffected. The true
    intercepts are restored afterwards by :func:`_calibrate`.
    """
    model_json = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    model_json["learner"]["learner_model_param"]["base_score"] = _SANITISED_BASE_SCORE
    sanitised = xgboost.Booster()
    sanitised.load_model(bytearray(json.dumps(model_json), "utf-8"))
    return sanitised


def _calibrate(explainer, model, columns, frame):
    """Derive the per-class base value from the ORIGINAL model's margins.

    ``shap`` recomputes (and overwrites) ``explainer.expected_value`` on every
    ``shap_values()`` call, so the calibrated value is returned to the caller
    instead of being stored on the explainer object.

    Returns ``(base_values, residual)`` where ``residual`` is the worst
    ``|(base + sum(shap)) - model margin|`` across the probe rows (log-odds
    space) - proof that the additive decomposition holds for the saved model.
    """
    arr = _normalize_shap_values(explainer.shap_values(frame))
    sums = arr.sum(axis=1)                       # (rows, classes)
    margins = _original_margins(model, columns, frame)[: sums.shape[0]]

    base = margins[0] - sums[0]                  # constant intercept per class
    residual = float(np.nanmax(np.abs((base + sums) - margins)))
    return np.asarray(base, dtype=float), residual


def _get_explainer():
    """Return the cached SHAP explainer (building it at most once per process).

    Raises :class:`RuntimeError` with a user-safe message when SHAP genuinely
    cannot initialise; the real exception is logged for debugging.
    """
    global _explainer, _explainer_mode, _explainer_error, _base_values
    if _explainer is not None:
        return _explainer
    if _explainer_error is not None:
        raise RuntimeError(_explainer_error)

    artifacts = get_artifacts()
    if artifacts is None:
        raise RuntimeError("Model artifacts are unavailable.")
    model = artifacts["model"]
    columns = artifacts["feature_columns"]
    frame = _probe_matrix(columns)

    # 1) native explainer (works with SHAP releases that parse vector base_score)
    try:
        native = shap.TreeExplainer(model)
        base, residual = _calibrate(native, model, columns, frame)
        if residual > _MARGIN_TOLERANCE:
            raise ValueError(
                "native TreeExplainer failed the additive-consistency probe "
                f"(residual {residual:.3e})"
            )
        _explainer, _explainer_mode, _base_values = native, "native", base
        logger.info("SHAP TreeExplainer ready (native, residual %.2e).", residual)
        return _explainer
    except Exception as exc:  # noqa: BLE001 - older SHAP cannot parse the model
        logger.warning(
            "Native SHAP TreeExplainer unusable (%s: %s); falling back to the "
            "sanitised-booster compatibility path.",
            type(exc).__name__, exc,
        )

    # 2) compatibility path: identical trees, scalar base_score, real intercepts
    try:
        sanitised = _build_sanitised_booster(model.get_booster())
        compat = shap.TreeExplainer(sanitised)
        base, residual = _calibrate(compat, model, columns, frame)
        if residual > _MARGIN_TOLERANCE:
            raise ValueError(
                "sanitised booster failed the additive-consistency probe "
                f"(residual {residual:.3e})"
            )
        _explainer, _explainer_mode, _base_values = compat, "sanitised", base
        logger.info(
            "SHAP TreeExplainer ready (sanitised-compat, residual %.2e); "
            "original base_score=%s.",
            residual, _read_base_score_vector(model.get_booster()),
        )
        return _explainer
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never 500
        logger.exception("SHAP explainer initialisation failed entirely.")
        _explainer_error = (
            "Explainability is unavailable because SHAP could not initialise for "
            "the installed SHAP/XGBoost combination. The real error has been logged."
        )
        raise RuntimeError(_explainer_error) from exc


def base_values():
    """Calibrated per-class intercepts, measured against the saved model.

    Held outside the explainer because ``shap_values()`` overwrites
    ``explainer.expected_value``.
    """
    _get_explainer()
    return np.asarray(_base_values, dtype=float)


def explainability_mode():
    """'native' | 'sanitised' | None - how the current explainer was built."""
    return _explainer_mode


def top_factors(record, k=5):
    """Top-k signed SHAP factors for this record's predicted class.

    This is the SAME explanation pipeline the Explainability page uses
    (:func:`explain_record` -> ``shap_values`` on the saved XGBoost booster);
    only the presentation is trimmed so the prediction result page can show
    3-5 factors inline.

    Returns::

        {
          "factors": [{"label", "contribution", "direction", "magnitude_pct"}],
          "class_label": predicted severity label,
          "base_value": calibrated base log-odds for that class,
          "output_logit": base + sum(shap),
          "k": requested number of factors,
          "mode": explainer mode ("native" / "sanitised"),
        }

    ``contribution`` is in log-odds space (same units as the Explainability
    waterfall); ``direction`` is ``"up"`` (pushes towards the predicted
    class) or ``"down"``.  Raises the same exceptions as
    :func:`explain_record` so callers can degrade gracefully instead of
    returning HTTP 500.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    explanation = explain_record(record)

    # increasing: positive contributions (desc); decreasing: negative (asc).
    pool = {}
    for factor in explanation["increasing"] + explanation["decreasing"]:
        pool[factor["label"]] = factor
    ranked = sorted(
        pool.values(), key=lambda f: abs(f["contribution"]), reverse=True
    )[:k]

    max_abs = max((abs(f["contribution"]) for f in ranked), default=0.0)
    factors = []
    for factor in ranked:
        magnitude = abs(factor["contribution"])
        factors.append({
            "label": factor["label"],
            "contribution": factor["contribution"],
            "direction": "up" if factor["contribution"] > 0 else "down",
            "magnitude_pct": (
                round(100.0 * magnitude / max_abs, 1) if max_abs else 0.0
            ),
        })

    return {
        "factors": factors,
        "class_label": explanation["prediction"],
        "base_value": explanation["base_value"],
        "output_logit": explanation["output_logit"],
        "k": k,
        "mode": _explainer_mode,
    }


def _normalize_shap_values(raw):
    """Return an ndarray standing for (n_samples, n_features, n_classes)."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        return np.expand_dims(arr, axis=-1)
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")


def _class_logit_deltas(expected_value, arr, class_idx):
    """(base_value, contributions array) for one class."""
    base = float(np.asarray(expected_value, dtype=float)[class_idx])
    return base, arr[0, :, class_idx]


def _roundtrip_check(artifacts, explainer, X_row, arr, expected_value):
    """Verify shap_values + expected_value reproduce predict_proba logits."""
    class_logits = []
    for c in range(arr.shape[-1]):
        class_logits.append(float(expected_value[c]) + float(arr[0, :, c].sum()))
    class_logits = np.array(class_logits)
    shap_probs = np.exp(class_logits - class_logits.max())
    shap_probs = shap_probs / shap_probs.sum()
    model_probs = artifacts["model"].predict_proba(X_row)[0]
    max_dev = float(np.abs(shap_probs - model_probs).max())
    return {
        "checked": True,
        "max_deviation": max_dev,
        "matches": max_dev < 1e-4,
        "note": (
            "SHAP log-odds contributions were recombined and softmaxed; the result "
            "matches model.predict_proba to within "
            + (f"{max_dev:.2e}" if max_dev else "0")
            + "."
        ),
    }
def explain_record(record):
    """Full local explanation for a validated raw record dict.

    Returns a dict ready for the XAI page (you may also pass in the output of
    ``prediction_service.validate_and_build_record``; validation errors stay
    with the prediction route).
    """
    artifacts = get_artifacts()
    if artifacts is None:
        raise RuntimeError("Model artifacts are unavailable.")
    X_new = preprocess_new_record(record, artifacts)
    if X_new.shape[1] != len(artifacts["feature_columns"]):
        raise RuntimeError("Feature mismatch while building the explanation matrix.")

    explainer = _get_explainer()
    arr = _normalize_shap_values(explainer.shap_values(X_new))
    # Use the calibrated per-class intercepts: shap's own
    # `explainer.expected_value` is recomputed on every shap_values() call and
    # would not match the saved model's margins.
    expected_value = base_values()

    proba = artifacts["model"].predict_proba(X_new)[0]
    class_idx = int(np.argmax(proba))
    class_labels = artifacts["class_labels"]
    prediction = class_labels[class_idx]

    base_value, contributions = _class_logit_deltas(
        expected_value, arr, class_idx)
    consistency = _roundtrip_check(
        artifacts, explainer, X_new, arr, expected_value)
    columns = artifacts["feature_columns"]

    pairs = sorted(
        (columns[i], float(contributions[i]))
        for i in range(len(columns))
        if float(contributions[i]) != 0.0
    )
    pairs = sorted(pairs, key=lambda p: abs(p[1]), reverse=True)[:TOP_LOCAL_FEATURES]

    local = [
        {
            "feature": name,
            "label": pretty_feature(name),
            "contribution": round(value, 5),
            "direction": "pushes toward" if value > 0 else "pushes away from",
        }
        for name, value in pairs
    ]

    waterfall_html = plots.render_div(
        plots.waterfall_chart(
            [pretty_feature(n) for n, _ in pairs],
            [v for _, v in pairs],
            base_value,
            f"Local contributions for: {prediction}",
        )
    )

    # Force-style chain: base log-odds -> contributions -> output log-odds.
    all_contribs = np.asarray(contributions, dtype=float)
    output_logit = float(base_value + all_contribs.sum())
    force_html = plots.render_div(
        plots.force_style_chart(
            [pretty_feature(c) for c in columns],
            all_contribs.tolist(),
            base_value,
            output_logit,
            f"Force-style contribution chain for: {prediction}",
        )
    )

    # Explicit increasing / decreasing factor lists (signed contributions).
    signed = sorted(
        ((columns[i], float(all_contribs[i]))
         for i in range(len(columns)) if float(all_contribs[i]) != 0.0),
        key=lambda p: p[1],
    )
    increasing = [
        {"label": pretty_feature(n), "contribution": round(v, 5)}
        for n, v in reversed(signed) if v > 0
    ][:6]
    decreasing = [
        {"label": pretty_feature(n), "contribution": round(v, 5)}
        for n, v in signed if v < 0
    ][:6]

    probabilities = [
        {"label": label, "pct": round(float(p) * 100.0, 2)}
        for label, p in zip(class_labels, proba)
    ]

    return {
        "prediction": prediction,
        "prediction_tone": TONE_BY_LABEL.get(prediction, "green"),
        "class_idx": class_idx,
        "probabilities": probabilities,
        "confidence_pct": max(item["pct"] for item in probabilities),
        "base_value": round(base_value, 5),
        "output_logit": round(output_logit, 5),
        "local": local,
        "waterfall_html": waterfall_html,
        "force_html": force_html,
        "increasing": increasing,
        "decreasing": decreasing,
        "consistency": consistency,
    }
def _global_shap_data():
    """Mean |SHAP| per feature over a fixed sample (+ grouped-by-column view).

    Cached per process; recomputation is cheap (~0.4 s) and deterministic
    (fixed random_state).
    """
    global _global_cache
    if _global_cache is not None:
        return _global_cache

    artifacts = get_artifacts()
    if artifacts is None:
        raise RuntimeError("Model artifacts are unavailable.")
    columns = artifacts["feature_columns"]

    X_full = load_processed_dataset()
    X_sample = X_full.sample(n=min(GLOBAL_SAMPLE_N, len(X_full)),
                             random_state=GLOBAL_RANDOM_STATE)
    X = X_sample.values

    explainer = _get_explainer()
    arr = _normalize_shap_values(explainer.shap_values(X))
    mean_abs = np.abs(arr).mean(axis=(0, 2))  # (n_features,)

    by_feature = sorted(
        zip(columns, mean_abs), key=lambda p: p[1], reverse=True
    )[:TOP_GLOBAL_FEATURES]

    # Group one-hot columns under their original raw column.
    grouped = {}
    for name, value in zip(columns, mean_abs):
        grouped[_group_of(name)] = grouped.get(_group_of(name), 0.0) + float(value)
    by_group = sorted(grouped.items(), key=lambda p: p[1], reverse=True)[
        :TOP_GLOBAL_GROUPS
    ]

    chart_html = plots.render_div(
        plots.ranked_bar(
            [pretty_feature(n) for n, _ in by_group],
            [float(v) for _, v in by_group],
            "Mean |SHAP| by original column (sample of "
            f"{len(X_sample):,} records)",
            color=plots.ACCENT,
        )
    )

    _global_cache = {
        "by_feature": [
            {"feature": name, "label": pretty_feature(name),
             "mean_abs": round(float(v), 5)}
            for name, v in by_feature
        ],
        "by_group": [
            {"feature": name, "label": pretty_feature(name),
             "mean_abs": round(float(v), 5)}
            for name, v in by_group
        ],
        "sample_size": len(X_sample),
        "chart_html": chart_html,
    }
    return _global_cache


def global_summary(force=False):
    """Public accessor: global SHAP importance (cached once per process)."""
    if force:
        global _global_cache
        _global_cache = None
    return _global_shap_data()