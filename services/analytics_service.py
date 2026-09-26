"""
STEP 10 / 7.3 - Analytics service.

Builds the filterable analytics dashboard from the real, cleaned dataset.

Filters (URL query params, all optional):
  area      -> Area_accident_occured
  severity  -> Accident_severity
  weather   -> Weather_conditions
  road      -> Road_surface_type
  roadtype  -> Lanes_or_Medians  (closest real "road type" column)
  day       -> Day_of_week
  Year is NOT offered: the dataset has clock times only, no date column.

Every chart and KPI is computed from the FILTERED subset and rendered
server-side with the shared Plotly theme.  When a filter combination matches
no records the page shows an empty state instead of crashing.
"""
import pandas as pd

from utils import plots
from services.data import (
    day_counts,
    hour_counts,
    load_raw_dataset,
    severity_counts,
    top_categories,
)

# Filter bar shown in the UI, in display order: (key, label, column).
FILTERS = [
    ("area", "Area", "Area_accident_occured"),
    ("severity", "Severity", "Accident_severity"),
    ("weather", "Weather", "Weather_conditions"),
    ("road", "Road surface", "Road_surface_type"),
    ("roadtype", "Road type", "Lanes_or_Medians"),
    ("day", "Day", "Day_of_week"),
]

FILTER_COLUMNS = {key: column for key, _label, column in FILTERS}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _no_data(message="No records match the current filters."):
    return (
        f'<div class="empty-chart"><span class="empty-icon">&#128269;</span>'
        f"<p>{message}</p></div>"
    )


def filter_options():
    """Allowed values for every filter, derived from the full dataset.

    Day-of-week is returned in canonical Monday..Sunday order.
    """
    df = load_raw_dataset()
    options = {}
    for key, column in FILTER_COLUMNS.items():
        values = sorted(
            df[column].dropna().astype(str).unique().tolist(),
            key=lambda v: (v == "Unknown", v),
        )
        options[key] = values
    if "day" in options:
        ordered = [d for d in DAY_ORDER if d in options["day"]]
        ordered += [d for d in options["day"] if d not in DAY_ORDER]
        options["day"] = ordered
    return options


def apply_filters(params):
    """Return (filtered_df, applied_filters_dict)."""
    df = load_raw_dataset()
    applied = {}
    for key, column in FILTER_COLUMNS.items():
        value = (params.get(key) or "").strip()
        if value:
            df = df[df[column].astype(str) == value]
            applied[key] = value
    return df, applied


def _summarize(df, applied):
    total_all = len(load_raw_dataset())
    if df.empty:
        return {
            "records": 0, "share_pct": 0.0, "top_area": "—", "busiest_day": "—", "peak_hour": "—",
            "filters_applied": applied,
        }
    hours = hour_counts(df)
    peak = max(hours, key=hours.get)
    days = day_counts(df)
    busiest = max(days, key=days.get)
    area_counts = df["Area_accident_occured"].value_counts()
    top_area = area_counts.index[0] if len(area_counts) else "—"
    return {
        "records": int(len(df)),
        "share_pct": round(100.0 * len(df) / total_all, 1) if total_all else 0.0,
        "top_area": str(top_area),
        "busiest_day": busiest,
        "peak_hour": f"{peak:02d}:00",
        "filters_applied": applied,
    }
def build_analytics(params):
    """Compute summary cards + chart divs for the analytics page."""
    df, applied = apply_filters(params)
    summary = _summarize(df, applied)

    if df.empty:
        return {"summary": summary, "charts": [], "empty": True}

    charts = []

    # 1. Severity donut
    charts.append({
        "title": "Severity distribution", "subtitle": "Outcomes within the filtered records",
        "html": plots.render_div(plots.severity_donut(severity_counts(df))), "span": 5,
    })

    # 2. Hour-of-day trend
    hour = hour_counts(df)
    hsl = [f"{h:02d}:00" for h in range(24)]
    charts.append({
        "title": "Hour-of-day trend", "subtitle": "Clock time of accidents",
        "html": plots.render_div(plots.trend_area(hsl, [hour[h] for h in range(24)], "Hour of day")),
        "span": 7,
    })

    # 3. Top areas
    areas = top_categories(df, "Area_accident_occured", limit=10)
    charts.append({
        "title": "Top areas", "subtitle": "Ranked by number of records",
        "html": plots.render_div(plots.ranked_bar([l for l, _ in areas], [v for _, v in areas],
                                                  "Top areas")),
        "span": 6,
    })

    # 4. Weekday totals
    days = day_counts(df)
    charts.append({
        "title": "Accidents per weekday", "subtitle": "Monday – Sunday",
        "html": plots.render_div(plots.bar_chart(DAY_ORDER, [days[d] for d in DAY_ORDER],
                                                 "Per weekday")),
        "span": 6,
    })
# 5. Road surface type
    road = top_categories(df, "Road_surface_type", limit=8)
    charts.append({
        "title": "Road surface type", "subtitle": "Recorded at the accident site",
        "html": plots.render_div(plots.ranked_bar([l for l, _ in road], [v for _, v in road],
                                                  "Road surface type")),
        "span": 4,
    })

    # 6. Light conditions
    light = top_categories(df, "Light_conditions", limit=6)
    charts.append({
        "title": "Light conditions", "subtitle": "Visibility at the moment of impact",
        "html": plots.render_div(plots.ranked_bar([l for l, _ in light], [v for _, v in light],
                                                  "Light conditions")),
        "span": 4,
    })

    # 7. Weather
    weather = top_categories(df, "Weather_conditions", limit=7)
    charts.append({
        "title": "Weather conditions", "subtitle": "At the time of the accident",
        "html": plots.render_div(plots.ranked_bar([l for l, _ in weather], [v for _, v in weather],
                                                  "Weather conditions")),
        "span": 4,
    })

    # 8. Causes
    causes = top_categories(df, "Cause_of_accident", limit=10)
    charts.append({
        "title": "Accident causes", "subtitle": "Ranked by number of records",
        "html": plots.render_div(plots.ranked_bar([l for l, _ in causes], [v for _, v in causes],
                                                  "Accident causes")),
        "span": 6,
    })

    # 9. Area x Severity stacked bar (real counts per severity).
    top_areas = [a for a, _ in top_categories(df, "Area_accident_occured", limit=8)]
    sub = df[df["Area_accident_occured"].isin(top_areas)]
    sev_values = {
        sev: [int(((sub["Area_accident_occured"] == a) & (sub["Accident_severity"] == sev)).sum())
              for a in top_areas]
        for sev in plots.SEVERITY_ORDER
    }
    charts.append({
        "title": "Area × Severity",
        "subtitle": "Accident counts per severity, stacked by area (top 8 areas)",
        "html": plots.render_div(
            plots.stacked_counts(top_areas, sev_values, "Area × Severity counts")),
        "span": 7,
    })

    # 10. Area -> severity sunburst (hierarchy view; meaningful drill-down).
    top5 = [a for a, _ in top_categories(df, "Area_accident_occured", limit=5)]
    sub5 = df[df["Area_accident_occured"].isin(top5)]
    labels, parents, values = [], [], []
    for area in top5:
        area_mask = sub5["Area_accident_occured"] == area
        labels.append(str(area))
        parents.append("")
        values.append(int(area_mask.sum()))
        for severity in plots.SEVERITY_ORDER:
            count = int((area_mask & (sub5["Accident_severity"] == severity)).sum())
            if count:
                labels.append(f"{area} · {severity}")
                parents.append(str(area))
                values.append(count)
    charts.append({
        "title": "Area → severity hierarchy",
        "subtitle": "Sunburst drill-down for the five busiest areas",
        "html": plots.render_div(plots.sunburst(labels, parents, values, "Area → severity")),
        "span": 5,
    })

    return {"summary": summary, "charts": charts, "empty": False}