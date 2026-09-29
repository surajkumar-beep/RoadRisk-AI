"""
STEP 12 - Hotspot mapping service (area-based spatial proxy).

IMPORTANT, honest limitation: the RTA dataset has NO latitude / longitude
columns and NO street addresses.  A geospatial map is therefore impossible
without fabricating coordinates, so this page maps hotspots by the real
administrative AREA field instead (Area_accident_occured) plus junction,
road-alignment and weekday/hour density analyses.

Everything rendered here is a true count from the cleaned dataset.  The page
is explicitly labelled an "area-based spatial proxy" - NOT a geographic GPS
heatmap.

Available filters (validated against the dataset, applied server-side):
  day      -> Day_of_week    (real temporal field; the dataset has NO date)
  severity -> Accident_severity
  top      -> number of ranked areas shown (5 / 10 / 15)

Invalid filter values are ignored (never fabricated), so a mistyped URL can
never produce a misleading chart.
"""
from utils import plots
from services.data import day_counts, hour_counts, load_raw_dataset, top_categories

NOT_IN_DATASET = (
    "Latitude/longitude coordinates are not present in the dataset. "
    "Hotspots are therefore ranked by accident area (Area_accident_occured) - "
    "this is an area-based spatial proxy, not a geographic GPS heatmap."
)
MAX_AREAS = 10
_AREA_COLUMN = "Area_accident_occured"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TOP_CHOICES = (5, 10, 15)


def filter_options(df=None):
    """Allowed filter values derived from the real dataset (day order fixed)."""
    if df is None:
        df = load_raw_dataset()
    days_present = set(df["Day_of_week"].dropna().astype(str).unique())
    return {
        "days": [d for d in DAY_ORDER if d in days_present],
        "severities": [s for s in plots.SEVERITY_ORDER
                       if s in set(df["Accident_severity"].dropna().astype(str).unique())],
        "tops": list(TOP_CHOICES),
    }


def _clean_filters(params, df):
    """Validate raw request params against the dataset (ignore invalid)."""
    options = filter_options(df)
    day = str((params.get("day") if params else "") or "").strip()
    if day not in options["days"]:
        day = ""
    severity = str((params.get("severity") if params else "") or "").strip()
    if severity not in options["severities"]:
        severity = ""
    top = str((params.get("top") if params else "") or "").strip()
    top_n = int(top) if top in {str(t) for t in TOP_CHOICES} else MAX_AREAS
    return {"day": day, "severity": severity, "top": top_n}


def _severity_shares(df, column, limit):
    """Return (labels, matrix) for a 100% stacked severity chart."""
    names = [name for name, _ in top_categories(df, column, limit=limit)]
    labels = list(names)
    matrix = {sev: [] for sev in plots.SEVERITY_ORDER}
    for name in names:
        grp = df[df[column] == name]["Accident_severity"]
        n = len(grp)
        for sev in plots.SEVERITY_ORDER:
            matrix[sev].append(float((grp == sev).sum() / n) if n else 0.0)
    return labels, matrix


def build_hotspots(params=None):
    """Compute every hotspot section: area cards, density charts, junction views.

    ``params`` is any mapping-like object (e.g. ``request.args``) carrying the
    optional ``day``, ``severity`` and ``top`` filters.  Invalid values are
    ignored; the returned dict always includes the validated ``applied``
    filters, the ``options`` for the filter form and an ``empty`` flag when
    the filtered subset has no records.
    """
    df_full = load_raw_dataset()
    total_full = int(len(df_full))
    applied = _clean_filters(params, df_full)
    options = filter_options(df_full)

    # Apply the validated filters (all real dataset columns).
    df = df_full
    if applied["day"]:
        df = df[df["Day_of_week"].astype(str) == applied["day"]]
    if applied["severity"]:
        df = df[df["Accident_severity"].astype(str) == applied["severity"]]
    total = int(len(df))

    applied_label = " · ".join(
        [v for v in (applied["day"], applied["severity"]) if v]
    ) or "All records"

    if df.empty:
        return {
            "empty": True,
            "not_in_dataset": NOT_IN_DATASET,
            "total": 0,
            "total_full": total_full,
            "max_areas": applied["top"],
            "area_cards": [],
            "charts": [],
            "applied": applied,
            "applied_label": applied_label,
            "options": options,
            "severity_mix_skipped": bool(applied["severity"]),
        }

    # -- top areas (within the current filter) --------------------------------
    top_n = applied["top"]
    areas = top_categories(df, _AREA_COLUMN, limit=top_n)
    area_cards = []
    for rank, (name, count) in enumerate(areas, start=1):
        grp = df[df[_AREA_COLUMN] == name]["Accident_severity"]
        n = int(len(grp))
        area_cards.append({
            "rank": rank,
            "name": name,
            "count": count,
            "share_pct": round(100.0 * count / total, 1) if total else 0.0,
            "fatal_pct": round(100.0 * (grp == "Fatal injury").sum() / n, 1),
            "serious_pct": round(100.0 * (grp == "Serious Injury").sum() / n, 1),
            "sight_pct": round(100.0 * (grp == "Slight Injury").sum() / n, 1),
        })

    # -- charts -------------------------------------------------------------
    charts = []

    # Area ranking by severity - the interactive spatial-proxy centrepiece
    # (Plotly: hover, zoom, pan; ranked by real counts, no coordinates exist).
    rank_areas = [a for a, _ in top_categories(df, _AREA_COLUMN, limit=top_n)]
    sev_counts = {
        sev: [int((df[df[_AREA_COLUMN] == a]["Accident_severity"] == sev).sum())
              for a in rank_areas]
        for sev in plots.SEVERITY_ORDER
    }
    charts.append({
        "title": "Area ranking by severity (spatial proxy)",
        "subtitle": (
            f"Interactive: top {len(rank_areas)} areas by accident count, stacked "
            "by severity — hover for exact counts (no coordinates in the dataset)"
        ),
        "html": plots.render_div(
            plots.stacked_counts(rank_areas, sev_counts,
                                 "Area ranking (area-based spatial proxy)")),
        "span": 7,
    })

    # Area x Severity count heatmap (coordinates do not exist; this is the
    # honest spatial substitute explicitly requested for 7.5).
    heat_areas = [a for a, _ in top_categories(df, _AREA_COLUMN, limit=min(top_n, 12))]
    heat_rows = []
    heat_counts = []
    for area in heat_areas:
        grp = df[df[_AREA_COLUMN] == area]["Accident_severity"]
        heat_rows.append(area)
        heat_counts.append([int((grp == sev).sum()) for sev in plots.SEVERITY_ORDER])
    charts.append({
        "title": "Area × Severity heatmap",
        "subtitle": "Accident counts per severity for the busiest areas in the current filter (no coordinates in the dataset)",
        "html": plots.render_div(
            plots.matrix_heatmap(heat_rows, plots.SEVERITY_ORDER, heat_counts,
                                 "Area × Severity accident counts",
                                 colorscale="YlOrRd", texttemplate="{z}",
                                 colorbar_title="n")),
        "span": 5,
    })

    # Weekday x hour density heatmap (real counts).
    weekday_hours = {}
    for day, grp in df.groupby("Day_of_week"):
        weekday_hours[day] = hour_counts(grp)
    charts.append({
        "title": "Accident density heatmap",
        "subtitle": "Weekday × hour of day (real counts)",
        "html": plots.render_div(plots.weekday_hour_heatmap(weekday_hours, "Density: weekday × hour")),
        "span": 5,
    })

    # 100% stacked severity mix - only meaningful when the severity filter is
    # off (with a severity filter every bar would trivially be 100% one color).
    severity_mix_skipped = bool(applied["severity"])
    if not severity_mix_skipped:
        area_labels, area_matrix = _severity_shares(df, _AREA_COLUMN, top_n)
        charts.append({
            "title": "Severity mix inside the busiest areas",
            "subtitle": "100% stacked share per area",
            "html": plots.render_div(
                plots.stacked_severity(area_labels, area_matrix, "Area severity mix")),
            "span": 6,
        })

    # Peak hours.
    hours = hour_counts(df)
    peak = sorted(hours.items(), key=lambda kv: kv[1], reverse=True)[:8]
    peak_labels = [f"{h:02d}:00" for h, _ in peak]
    peak_values = [v for _, v in peak]
    charts.append({
        "title": "Peak accident hours",
        "subtitle": "Top 8 hours by number of accidents",
        "html": plots.render_div(
            plots.ranked_bar(peak_labels, peak_values, "Peak accident hours")),
        "span": 6,
    })

    # Junction type.
    junction = top_categories(df, "Types_of_Junction", limit=8)
    charts.append({
        "title": "Junction type at accident sites",
        "subtitle": "How the junction was controlled",
        "html": plots.render_div(
            plots.ranked_bar([l for l, _ in junction], [v for _, v in junction],
                             "Junction type")),
        "span": 6,
    })

    # Road alignment.
    alignment = top_categories(df, "Road_allignment", limit=6)
    charts.append({
        "title": "Road alignment",
        "subtitle": "Geometry of the road at the accident site",
        "html": plots.render_div(
            plots.ranked_bar([l for l, _ in alignment], [v for _, v in alignment],
                             "Road alignment")),
        "span": 6,
    })

    # Day-of-week totals (context for the heatmap).
    days = day_counts(df)
    day_chart = plots.bar_chart(
        list(days.keys()), list(days.values()), "Accidents per weekday")
    charts.append({
        "title": "Accidents per weekday",
        "subtitle": "Monday – Sunday totals",
        "html": plots.render_div(day_chart),
        "span": 6,
    })

    return {
        "empty": False,
        "not_in_dataset": NOT_IN_DATASET,
        "total": total,
        "total_full": total_full,
        "max_areas": top_n,
        "area_cards": area_cards,
        "charts": charts,
        "applied": applied,
        "applied_label": applied_label,
        "options": options,
        "severity_mix_skipped": severity_mix_skipped,
    }