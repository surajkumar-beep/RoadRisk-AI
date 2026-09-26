"""
STEP 9 / 7.1 - Home dashboard service.

Aggregates ONLY the data that actually exists in the RTA dataset:

  * KPI cards - total accidents, fatal / serious / slight counts, areas
                covered, average casualties, most common weather and most
                common accident cause.  Metrics with no backing column
                (medical cost, response time, countries, ...) are never
                rendered, not even as "N/A" cards.
  * Charts    - accidents by hour of day, severity distribution (count + %),
                top areas, top causes, accidents per weekday and weather
                conditions (Plotly, shared theme, severity palette).

Nothing is invented: every number below is a real count over the 12,316 row
clean dataset (same cleaning rules the model saw).
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


def _kpi(label, value, sub, icon, tone="accent"):
    return {"label": label, "value": value, "sub": sub, "icon": icon, "tone": tone}


def _chart(title, subtitle, html, span=6):
    return {"title": title, "subtitle": subtitle, "html": html, "span": span}


def build_dashboard():
    """Compute every KPI and chart needed by the home dashboard."""
    df = load_raw_dataset()
    total = int(len(df))
    counts = severity_counts(df)

    # -- KPI cards (all backed by real columns) ----------------------------
    fatal_pct = counts["Fatal injury"] / total if total else 0
    serious_pct = counts["Serious Injury"] / total if total else 0
    slight_pct = counts["Slight Injury"] / total if total else 0

    areas_series = df["Area_accident_occured"]
    areas_known = int(areas_series[areas_series != "Unknown"].nunique())
    avg_casualties = round(float(df["Number_of_casualties"].mean()), 2)

    weather_top = top_categories(df, "Weather_conditions", limit=1)
    weather_label, weather_count = weather_top[0] if weather_top else ("-", 0)
    cause_top = top_categories(df, "Cause_of_accident", limit=1)
    cause_label, cause_count = cause_top[0] if cause_top else ("-", 0)

    kpis = [
        _kpi("Total Accidents", f"{total:,}",
             "records in the cleaned RTA dataset", "▤", "accent"),
        _kpi("Fatal", f"{counts['Fatal injury']:,}",
             f"{fatal_pct:.1%} of all accidents", "▲", "red"),
        _kpi("Serious", f"{counts['Serious Injury']:,}",
             f"{serious_pct:.1%} of all accidents", "!", "amber"),
        _kpi("Slight / Minor", f"{counts['Slight Injury']:,}",
             f"{slight_pct:.1%} of all accidents", "✓", ""),
        _kpi("Accident Areas Covered", f"{areas_known}",
             "distinct areas (excluding Unknown)", "⌖", ""),
        _kpi("Average Casualties", f"{avg_casualties}",
             "casualties per accident event", "⊕", ""),
        _kpi("Most Common Weather", str(weather_label),
             (f"{weather_count:,} accidents ({weather_count / total:.1%})"
              if total else "-"),
             "☀", ""),
        _kpi("Most Common Cause", str(cause_label),
             (f"{cause_count:,} accidents ({cause_count / total:.1%})"
              if total else "-"),
             "◇", ""),
    ]

    # -- Charts ------------------------------------------------------------
    hour = hour_counts(df)
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    hour_values = [hour[h] for h in range(24)]
    counts_line = " · ".join(
        f"{label.split()[0]} {counts[label]:,}" for label in plots.SEVERITY_ORDER
    )
    charts = [
        _chart(
            "Accidents by hour of day",
            "Time analysis in 24h buckets (clock time only; no calendar dates exist in the dataset)",
            plots.render_div(plots.trend_area(hour_labels, hour_values,
                                              "Accidents by hour of day")),
            span=7,
        ),
        _chart(
            "Severity distribution",
            f"Count + percentage: {counts_line}",
            plots.render_div(plots.severity_donut(counts, "Severity distribution")),
            span=5,
        ),
    ]

    areas = top_categories(df, "Area_accident_occured", limit=10)
    charts.append(_chart(
        "Top accident areas",
        "Ranked by number of recorded accidents",
        plots.render_div(plots.ranked_bar([l for l, _ in areas], [v for _, v in areas],
                                          "Accident-prone areas")),
        span=6,
    ))

    causes = top_categories(df, "Cause_of_accident", limit=12)
    charts.append(_chart(
        "Accident causes",
        "Ranked by number of records",
        plots.render_div(plots.ranked_bar([l for l, _ in causes], [v for _, v in causes],
                                          "Cause of accident", color=plots.ACCENT)),
        span=6,
    ))

    days = day_counts(df)
    charts.append(_chart(
        "Accidents by weekday",
        "Monday to Sunday totals",
        plots.render_div(plots.bar_chart(list(days.keys()), list(days.values()),
                                         "Accidents by day of week")),
        span=5,
    ))

    weather = top_categories(df, "Weather_conditions", limit=8)
    charts.append(_chart(
        "Weather conditions",
        "Conditions recorded at accident time",
        plots.render_div(plots.ranked_bar([l for l, _ in weather], [v for _, v in weather],
                                          "Weather conditions")),
        span=7,
    ))

    return {"kpis": kpis, "charts": charts}