"""
STEP 11 / 7.4 - Risk analysis service.

Documented calculations (every number is descriptive, from the real dataset):

  1. FACTOR SEPARATION - for each factor column, consider categories with
     n >= 30 records (excluding "Unknown"); for each category compute
         serious_fatal_share = (count[Serious] + count[Fatal]) / n
     The factor's "risk spread" is (max share - min share) in percentage
     points.  This measures how strongly a factor SEPARATES outcomes. It is
     NOT a probability of accident and NOT a composite risk score.

  2. SEVERE-OUTCOME COMBINATIONS - same share per single factor-value
     combination, ranked descending (n >= 30).

  3. CORRELATION MATRIX - Pearson correlations among the model's numeric /
     ordinal-encoded features + Hour + encoded severity (0/1/2), computed on
     the scaled processed matrix that the model actually trained on.

  4. MODEL IMPORTANCE - XGBoost gain (model.feature_importances_), real.

  5. COMPOSITE RISK INDICATOR - NOT defined anywhere in this project; the UI
     reports it as "pending definition" instead of inventing one. Class
     probabilities from the model are never relabelled as "risk %".

Control panel scope filter (applies to 1 + 2 + KPI cards):
  all    -> every record
  severe -> Serious Injury + Fatal injury records only
  fatal  -> Fatal injury records only
"""
import pandas as pd

from utils import plots
from services.data import load_processed_dataset, load_raw_dataset, severity_counts
from services.prediction_service import get_artifacts

# Factor columns offered in the "severity by factor" comparison, in UI order.
FACTOR_COLUMNS = [
    ("Weather_conditions", "Weather conditions"),
    ("Light_conditions", "Light conditions"),
    ("Road_surface_conditions", "Road surface conditions"),
    ("Age_band_of_driver", "Driver age band"),
    ("Driving_experience", "Driving experience"),
    ("Type_of_vehicle", "Type of vehicle"),
    ("Day_of_week", "Day of week"),
    ("Cause_of_accident", "Cause of accident"),
    ("Area_accident_occured", "Accident area"),
]

# Numeric features present in the dataset with a meaningful severity link.
_NUMERIC_FOR_CORR = ["Hour", "Number_of_vehicles_involved", "Number_of_casualties"]

# Features shown in the correlation matrix (all numeric in the processed set).
CORR_FEATURES = [
    "Hour", "Number_of_vehicles_involved", "Number_of_casualties",
    "Casualty_severity", "Age_band_of_casualty", "Age_band_of_driver",
    "Driving_experience", "Service_year_of_vehicle", "Road_surface_conditions",
    "Light_conditions", "Educational_level",
]

_SEVERITY_CODE = {"Slight Injury": 0, "Serious Injury": 1, "Fatal injury": 2}
_MIN_COMBOS_CUTOFF = 30  # ignore tiny categories when scoring combinations

# Scope options offered by the control panel: key -> (label, severity subset).
SCOPES = {
    "all": ("All records", None),
    "severe": ("Serious + Fatal only", ["Serious Injury", "Fatal injury"]),
    "fatal": ("Fatal only", ["Fatal injury"]),
}

FORMAL_SCORE_STATE = {
    "status": "pending",
    "title": "Composite risk indicator",
    "explanation": (
        "No composite risk score is defined for this project: there is no "
        "documented formula that collapses all 30 accident fields into one "
        "number, so inventing one would be misleading. This page therefore "
        "reports descriptive statistics only - factor separation (percentage "
        "points between the highest and lowest serious/fatal share), "
        "correlations from the model's own feature matrix, and the model's "
        "gain-based feature importance. Model class probabilities on the "
        "prediction page are class probabilities, NOT 'risk percentages'."
    ),
}


def _apply_scope(df, scope):
    """Filter the dataset to the control-panel scope (or all records)."""
    _label, subset = SCOPES.get(scope, SCOPES["all"])
    if subset is None:
        return df
    return df[df["Accident_severity"].isin(subset)]


def _factor_separation(df):
    factor_table = []
    for column, label in FACTOR_COLUMNS:
        rows = []
        grouped = df.groupby(column)["Accident_severity"]
        for value, grp in grouped:
            n = int(len(grp))
            if n < _MIN_COMBOS_CUTOFF or value == "Unknown":
                continue
            share_sf = float((grp != "Slight Injury").mean())
            share_fatal = float((grp == "Fatal injury").mean())
            rows.append((str(value), n, share_sf, share_fatal))
        if not rows:
            continue
        rows.sort(key=lambda r: r[2], reverse=True)
        spread = max(r[2] for r in rows) - min(r[2] for r in rows)
        spread_pct = round(spread * 100.0, 1)
        factor_table.append({
            "label": label,
            "column": column,
            "categories": len(rows),
            "spread_pct": spread_pct,
            # Precomputed bar width (display-only) so the template does no math.
            "bar_pct": min(100, max(4, round(spread_pct / 60 * 100))),
            "worst": rows[0][0],
            "worst_share_pct": round(rows[0][2] * 100.0, 1),
        })
    factor_table.sort(key=lambda r: r["spread_pct"], reverse=True)
    return factor_table


def _top_combos(df):
    combos = []
    for column, _label in FACTOR_COLUMNS:
        grouped = df.groupby(column)["Accident_severity"]
        for value, grp in grouped:
            n = int(len(grp))
            if n < _MIN_COMBOS_CUTOFF or value == "Unknown":
                continue
            combos.append({
                "factor": column,
                "category": str(value),
                "n": n,
                "serious_fatal_pct": round(float((grp != "Slight Injury").mean()) * 100.0, 1),
                "fatal_pct": round(float((grp == "Fatal injury").mean()) * 100.0, 1),
            })
    combos.sort(key=lambda r: r["serious_fatal_pct"], reverse=True)
    return combos[:8]


def _numeric_correlations(df):
    """Spearman rows: numeric features vs encoded severity (descriptive)."""
    coded = df["Accident_severity"].map(_SEVERITY_CODE)
    rows = []
    for column in _NUMERIC_FOR_CORR:
        valid = pd.DataFrame({column: df[column], "_severity": coded}).dropna()
        if len(valid) > 20:
            rho = valid[[column, "_severity"]].corr(method="spearman").iloc[0, 1]
            rows.append({
                "feature": column,
                "spearman": round(float(rho), 3),
                "n": int(len(valid)),
                "signed": round(float(rho), 3),  # template-friendly
            })
    return rows


def _correlation_matrix():
    """Pearson matrix over the model's numeric/ordinal features + severity.

    Computed on the SCALED processed matrix (what the model trained on) with
    the encoded severity column (0/1/2) appended positionally. Returns
    dict(row_labels, col_labels, z, html).
    """
    processed = load_processed_dataset()
    raw = load_raw_dataset()
    available = [c for c in CORR_FEATURES if c in processed.columns]
    if not available:
        return None
    matrix_df = processed[available].copy()
    if len(matrix_df) == len(raw):
        matrix_df["Severity (encoded)"] = raw["Accident_severity"].map(_SEVERITY_CODE)
    corr = matrix_df.corr(method="pearson").round(2)
    labels = [c.replace("_", " ") for c in corr.columns]
    z = corr.values.tolist()
    html = plots.render_div(
        plots.matrix_heatmap(labels, labels, z,
                             "Feature correlation matrix (Pearson, model matrix)",
                             colorscale="RdYlGn", texttemplate="{z:.2f}",
                             colorbar_title="r")
    )
    return {"labels": labels, "z": z, "html": html}


def build_risk_overview(scope="all"):
    """Compute the risk summary sections for the control-panel scope."""
    df_full = load_raw_dataset()
    df = _apply_scope(df_full, scope)
    scope_label = SCOPES.get(scope, SCOPES["all"])[0]
    total_full = int(len(df_full))
    total = int(len(df))

    counts_full = severity_counts(df_full)
    counts_scope = severity_counts(df)
    if df.empty:
        counts_scope = {k: 0 for k in counts_full}

    serious_fatal_share = (
        (counts_scope["Serious Injury"] + counts_scope["Fatal injury"]) / total
        if total else 0.0
    )

    factor_table = _factor_separation(df) if total >= _MIN_COMBOS_CUTOFF else []
    top_combos = _top_combos(df) if total >= _MIN_COMBOS_CUTOFF else []
    correlations = _numeric_correlations(df) if total > 20 else []
    corr_matrix = _correlation_matrix()

    # Severity comparison donut (full dataset - scope-independent reference).
    severity_chart = plots.render_div(
        plots.severity_donut(counts_full, "Severity comparison (full dataset)")
    )

    importance = _model_feature_importance()
    importance_html = (
        plots.render_div(importance["fig"]) if importance["fig"] is not None else None
    )

    return {
        "formal": FORMAL_SCORE_STATE,
        "scope": scope,
        "scope_label": scope_label,
        "scopes": [(key, label) for key, (label, _sub) in SCOPES.items()],
        "total": total,
        "total_full": total_full,
        "counts": counts_scope,
        "counts_full": counts_full,
        "serious_fatal_share_pct": round(serious_fatal_share * 100.0, 1),
        "factor_table": factor_table,
        "top_combos": top_combos,
        "correlations": correlations,
        "correlation_matrix": corr_matrix,
        "severity_chart_html": severity_chart,
        "importance_chart_html": importance_html,
        "importance_labels": importance["labels"],
        "importance_values": importance["values"],
    }


def _model_feature_importance():
    """Top-20 gain-based importance straight from the saved model."""
    artifacts = get_artifacts()
    if artifacts is None:
        return {"fig": None, "labels": [], "values": []}
    model = artifacts["model"]
    columns = artifacts["feature_columns"]
    gains = getattr(model, "feature_importances_", None)
    if gains is None:
        return {"fig": None, "labels": [], "values": []}
    pairs = sorted(zip(columns, gains), key=lambda p: p[1], reverse=True)[:20]
    labels, values = [p[0] for p in pairs], [round(float(p[1]), 6) for p in pairs]
    fig = plots.ranked_bar([pretty_feature(l) for l in labels], values,
                           "Top model features by gain", color=plots.ACCENT)
    return {"fig": fig, "labels": labels, "values": values}
def severity_by_factor(factor_column):
    """100%-stacked severity comparison for ONE factor (categories real)."""
    df = load_raw_dataset()
    grouped = df.groupby(factor_column)["Accident_severity"]
    rows = []
    for value, grp in grouped:
        n = int(len(grp))
        if n < _MIN_COMBOS_CUTOFF or value == "Unknown":
            continue
        rows.append({"value": str(value), "n": n, "group": grp})
    if not rows:
        return {"chart": None, "table": [], "factor_label": factor_column, "empty": True}
    rows.sort(key=lambda r: r["n"], reverse=True)
    rows = rows[:12]

    factor_labels = [r["value"] for r in rows]
    matrix = {severity: [0.0] * len(rows) for severity in plots.SEVERITY_ORDER}
    for i, row in enumerate(rows):
        counts = row["group"].value_counts()
        for severity in plots.SEVERITY_ORDER:
            matrix[severity][i] = float(counts.get(severity, 0)) / row["n"]

    chart = plots.render_div(
        plots.stacked_severity(factor_labels, matrix,
                               f"Severity split by {factor_column.replace('_', ' ').lower()}")
    )
    table = [{
        "value": r["value"], "n": r["n"],
        "slight_pct": round(matrix["Slight Injury"][i] * 100.0, 1),
        "serious_pct": round(matrix["Serious Injury"][i] * 100.0, 1),
        "fatal_pct": round(matrix["Fatal injury"][i] * 100.0, 1),
    } for i, r in enumerate(rows)]
    return {"chart": chart, "table": table, "factor_label": factor_column,
            "empty": False}


def pretty_feature(name):
    """Readable label for a model feature column name."""
    if "_" not in name:
        return name.replace("_", " ").title()
    prefix, value = name.split("_", 1)
    return f"{value.replace('_', ' ').title()} ({prefix.replace('_', ' ')})"