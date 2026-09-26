"""
STEP 12 - Hotspot mapping service.

IMPORTANT, honest limitation: the RTA dataset has NO latitude / longitude
columns and NO street addresses.  A geospatial map is therefore impossible
without fabricating coordinates, so this page maps hotspots by the real
administrative AREA field instead (Area_accident_occured) plus junction,
road-alignment and weekday/hour density analyses.

Everything rendered here is a true count from the cleaned dataset.
"""
from utils import plots
from services.data import day_counts, hour_counts, load_raw_dataset, top_categories

NOT_IN_DATASET = (
    "Latitude/longitude coordinates are not present in the dataset. "
    "Hotspots are therefore ranked by accident area (Area_accident_occured)."
)
MAX_AREAS = 10
_AREA_COLUMN = "Area_accident_occured"


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


def build_hotspots():
    """Compute every hotspot section: area cards, density charts, junction views."""
    df = load_raw_dataset()
    total = int(len(df))

    # -- top areas ----------------------------------------------------------
    areas = top_categories(df, _AREA_COLUMN, limit=MAX_AREAS)
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

    # Area x Severity count heatmap (coordinates do not exist; this is the
    # honest spatial substitute explicitly requested for 7.5).
    heat_areas = [a for a, _ in top_categories(df, _AREA_COLUMN, limit=12)]
    heat_rows = []
    heat_counts = []
    for area in heat_areas:
        grp = df[df[_AREA_COLUMN] == area]["Accident_severity"]
        heat_rows.append(area)
        heat_counts.append([int((grp == sev).sum()) for sev in plots.SEVERITY_ORDER])
    charts.append({
        "title": "Area × Severity heatmap",
        "subtitle": "Accident counts per severity for the 12 busiest areas (no coordinates in the dataset)",
        "html": plots.render_div(
            plots.matrix_heatmap(heat_rows, plots.SEVERITY_ORDER, heat_counts,
                                 "Area × Severity accident counts",
                                 colorscale="YlOrRd", texttemplate="{z}",
                                 colorbar_title="n")),
        "span": 7,
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

    area_labels, area_matrix = _severity_shares(df, _AREA_COLUMN, MAX_AREAS)
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
        "not_in_dataset": NOT_IN_DATASET,
        "total": total,
        "max_areas": MAX_AREAS,
        "area_cards": area_cards,
        "charts": charts,
    }