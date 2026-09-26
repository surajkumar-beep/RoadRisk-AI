"""
Shared Plotly chartering helpers for the RoadRisk AI dashboard pages.

Centralises the look of every chart (RoadRisk SaaS theme):
  - transparent/white backgrounds
  - minimal gridlines (#E7E8E5, subtle)
  - compact typography
  - semantic severity palette
  - responsive sizing + useful hover tooltips

The Plotly JavaScript bundle is served ONCE from here (offline, no CDN) and
injected by the base layout; every chart is rendered as a <div> + <script>
pair via :func:`render_div`.
"""
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

# Inline Plotly JS bundle (served once in base.html, offline-friendly).
PLOTLY_JS = get_plotlyjs()

# Semantic, stable palette -------------------------------------------------
SEVERITY_ORDER = ["Slight Injury", "Serious Injury", "Fatal injury"]
SEVERITY_COLORS = {
    "Slight Injury": "#16A34A",
    "Serious Injury": "#D97706",
    "Fatal injury": "#DC2626",
}
GRIDLINE_COLOR = "#E7E8E5"
ACCENT = "#16A34A"
MUTED = "#606A74"
INK = "#16181C"

_FONT = "'Segoe UI', 'Inter', Roboto, Helvetica, Arial, sans-serif"
def apply_theme(fig, height=320, title=None, show_legend=True, hovertemplate=None):
    """Apply the RoadRisk chart theme to a Plotly figure (in place)."""
    fig.update_layout(
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, size=12, color=INK),
        margin=dict(l=8, r=8, t=38 if title else 12, b=8),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=GRIDLINE_COLOR,
            font=dict(family=_FONT, size=12, color=INK),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=MUTED),
            bgcolor="rgba(0,0,0,0)",
        )
        if show_legend
        else dict(),
        height=height,
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=13, color=INK),
                x=0.02,
                xanchor="left",
            )
        )
    # Strip heavy gridlines / zero lines on every axis.
    for axis in ("xaxis", "yaxis", "xaxis2", "yaxis2"):
        if axis in fig.layout:
            getattr(fig.layout, axis).update(
                gridcolor=GRIDLINE_COLOR,
                zerolinecolor=GRIDLINE_COLOR,
                showline=False,
            )
    if hovertemplate:
        fig.update_traces(hovertemplate=hovertemplate)
    return fig


def render_div(fig):
    """Render a figure as a standalone (embedded) HTML div block, no JS bundle."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
        default_width="100%",
    )


# ---------------------------------------------------------------------------
# Reusable chart builders (called by the services; always fed real values)
# ---------------------------------------------------------------------------


def severity_donut(counts, title="Severity distribution"):
    """Donut of accident severity shares (counts: dict label -> count).

    Displays label + percentage on the chart; raw counts are always visible in
    the hover tooltip (count + % requirement).
    """
    labels = [s for s in SEVERITY_ORDER if counts.get(s)]
    values = [counts[s] for s in labels]
    colors = [SEVERITY_COLORS[s] for s in labels]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent",
            textposition="outside",
            insidetextorientation="horizontal",
            hovertemplate="%{label}: %{value:,} accidents (%{percent})<extra></extra>",
            sort=False,
        )
    )
    apply_theme(fig, title=title, show_legend=True, height=340)
    fig.update_layout(margin=dict(l=8, r=8, t=44, b=8))
    return fig


def ranked_bar(labels, values, title, color=ACCENT, limit=None, xlabel="", ylabel=""):
    """Horizontal ranked bar chart (assumes labels/values already sorted desc)."""
    if limit:
        labels, values = labels[:limit], values[:limit]
    fig = go.Figure(
        go.Bar(
            x=values[::-1],
            y=labels[::-1],
            orientation="h",
            marker=dict(color=color),
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        )
    )
    fig.update_xaxes(title=dict(text=xlabel, font=dict(size=11)), showgrid=True, gridcolor=GRIDLINE_COLOR)
    fig.update_yaxes(title=dict(text=ylabel, font=dict(size=11)))
    apply_theme(fig, title=title, show_legend=False)
    return fig
def trend_area(labels, values, title, color=ACCENT):
    """Line/area trend chart (positive direction)."""
    fig = go.Figure(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            line=dict(color=color, width=2.2, shape="spline"),
            marker=dict(size=5, color=color),
            fill="tozeroy",
            fillcolor="rgba(22,163,74,0.10)",
            hovertemplate="%{x}: %{y:,} accidents<extra></extra>",
        )
    )
    fig.update_yaxes(gridcolor=GRIDLINE_COLOR)
    apply_theme(fig, title=title, show_legend=False)
    return fig


def bar_chart(labels, values, title, color=ACCENT, xlabel="", ylabel=""):
    """Vertical bar chart."""
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=color),
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
    )
    fig.update_xaxes(tickfont=dict(size=10), title=dict(text=xlabel, font=dict(size=11)))
    fig.update_yaxes(title=dict(text=ylabel, font=dict(size=11)), gridcolor=GRIDLINE_COLOR)
    apply_theme(fig, title=title, show_legend=False)
    return fig


def stacked_severity(factor_labels, severity_matrix, title):
    """
    100% stacked severity bars. severity_matrix: dict severity -> list(share 0-1),
    factor_labels: parallel list of x labels.
    """
    fig = go.Figure()
    for severity in SEVERITY_ORDER:
        if severity not in severity_matrix:
            continue
        fig.add_trace(
            go.Bar(
                x=factor_labels,
                y=severity_matrix[severity],
                name=severity,
                marker=dict(color=SEVERITY_COLORS[severity]),
                hovertemplate="%{x}<br>%{fullData.name}: %{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickformat=".0%", gridcolor=GRIDLINE_COLOR)
    apply_theme(fig, title=title, show_legend=True, height=360)
    return fig


def weekday_hour_heatmap(matrix, title):
    """Density heatmap weekday x hour (real counts; no fabricated coordinates)."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hours = list(range(24))
    z = [[matrix.get(day, {}).get(h, 0) for h in hours] for day in days]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"{h:02d}:00" for h in hours],
            y=days,
            colorscale=[
                [0.0, "#F6F6F4"],
                [0.25, "#DCEFDC"],
                [0.5, "#A6D9B4"],
                [0.75, "#4CAF70"],
                [1.0, "#16A34A"],
            ],
            colorbar=dict(thickness=12, title=dict(text="accidents", font=dict(size=10))),
            hovertemplate="%{y} %{x}: %{z} accidents<extra></extra>",
        )
    )
    apply_theme(fig, title=title, show_legend=False, height=380)
    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def waterfall_chart(labels, contributions, base_value, title):
    """Horizontal contribution chart for a SHAP-style local explanation.

    contributions are real model-derived values (log-odds space). Positive
    values push towards the predicted class, negative away from it.
    """
    pairs = sorted(zip(labels, contributions), key=lambda p: abs(p[1]), reverse=True)
    top = [p for p in pairs if p[1] != 0][:15]
    top_labels = [p[0] for p in top]
    top_values = [p[1] for p in top]
    colors = ["#16A34A" if v > 0 else "#D1D5DB" for v in top_values]
    fig = go.Figure(
        go.Bar(
            x=top_values,
            y=top_labels,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}: %{x:+.4f} (log-odds)<extra></extra>",
        )
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=1.08,
        text=f"Base value (predicted class log-odds): {base_value:+.3f}",
        showarrow=False, font=dict(size=11, color=MUTED),
    )
    fig.update_xaxes(gridcolor=GRIDLINE_COLOR, title=dict(text="contribution (log-odds)", font=dict(size=11)))
    apply_theme(fig, title=title, show_legend=False, height=360)
    return fig


def sunburst(labels, parents, values, title):
    """Sunburst for hierarchical counts (e.g. area -> severity)."""
    colors = [SEVERITY_COLORS.get(label, MUTED) if parent else ACCENT
              for label, parent in zip(labels, parents)]
    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="white", width=1)),
            hovertemplate="%{label}: %{value:,} accidents<extra></extra>",
        )
    )
    apply_theme(fig, title=title, show_legend=False, height=400)
    return fig


def stacked_counts(labels, severity_values, title):
    """Absolute stacked bars: one bar per category, segments per severity.

    labels: category labels (already ordered); severity_values: dict
    severity -> list of raw counts (parallel to labels). Used for the
    Area x Severity view (real counts, severity palette).
    """
    fig = go.Figure()
    for severity in SEVERITY_ORDER:
        if severity not in severity_values:
            continue
        fig.add_trace(
            go.Bar(
                x=labels,
                y=severity_values[severity],
                name=severity,
                marker=dict(color=SEVERITY_COLORS[severity]),
                hovertemplate="%{x}<br>%{fullData.name}: %{y:,} accidents<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(gridcolor=GRIDLINE_COLOR, title=dict(text="accidents", font=dict(size=11)))
    fig.update_xaxes(tickfont=dict(size=10))
    apply_theme(fig, title=title, show_legend=True, height=360)
    return fig


def matrix_heatmap(row_labels, col_labels, z, title, colorscale="Greens",
                   texttemplate="{z:.2f}", colorbar_title="ρ"):
    """Generic labelled heatmap (correlation matrices, Area x Severity, ...).

    row_labels/col_labels are shown verbatim; z is a 2D list of numbers.
    """
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=col_labels,
            y=row_labels,
            colorscale=colorscale,
            zmid=0 if min(min(r) for r in z) < 0 else None,
            colorbar=dict(thickness=12, title=dict(text=colorbar_title, font=dict(size=10))),
            texttemplate=texttemplate if texttemplate else None,
            hovertemplate="%{y} · %{x}: %{z:,}<extra></extra>"
            if "%{z:.2f}" not in (texttemplate or "")
            else "%{y} · %{x}: %{z:.2f}<extra></extra>",
        )
    )
    apply_theme(fig, title=title, show_legend=False, height=360)
    fig.update_xaxes(tickfont=dict(size=10), side="bottom")
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def force_style_chart(labels, contributions, base_value, output_value, title,
                       top_n=10):
    """Force-plot style contribution chain rendered with Plotly.

    SHAP's JS force plot cannot be embedded offline without its bundled JS, so
    the same real information (base log-odds -> per-feature contributions ->
    output log-odds) is rendered as a horizontal segment chain: green segments
    push toward the predicted class, red segments push away. Values are genuine
    SHAP contributions in log-odds space.
    """
    pairs = sorted(
        ((l, float(v)) for l, v in zip(labels, contributions) if float(v) != 0.0),
        key=lambda p: abs(p[1]),
        reverse=True,
    )[:top_n]
    # Keep chronological order along the chain (top contributions first is fine
    # for readability; cumulative positions still sum to the output delta).
    running = float(base_value)
    xs, ys, colors, texts = [], [], [], []
    for name, value in pairs:
        start, end = running, running + value
        xs.append([start, end])
        ys.append(name)
        colors.append("#16A34A" if value > 0 else "#DC2626")
        texts.append(f"{value:+.3f}")
        running = end
    # Residual (features beyond top_n) keeps the chain honest.
    residual = float(output_value) - running
    if abs(residual) > 1e-6:
        xs.append([running, running + residual])
        ys.append("Other features (combined)")
        colors.append("#16A34A" if residual > 0 else "#DC2626")
        texts.append(f"{residual:+.3f}")

    fig = go.Figure()
    for i, (seg, name) in enumerate(zip(xs, ys)):
        fig.add_trace(
            go.Scatter(
                x=seg,
                y=[name, name],
                mode="lines",
                line=dict(color=colors[i], width=16),
                hoverinfo="text",
                text=f"{name}: {texts[i]} log-odds",
                showlegend=False,
            )
        )
    fig.add_vline(x=float(base_value), line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(
        x=float(base_value), y=len(ys) - 0.35, text=" base", showarrow=False,
        font=dict(size=10, color=MUTED), xanchor="left",
    )
    fig.add_vline(x=float(output_value), line=dict(color=INK, width=1.5, dash="dash"))
    fig.add_annotation(
        x=float(output_value), y=len(ys) - 0.35, text=" output ", showarrow=False,
        font=dict(size=10, color=INK), xanchor="right",
    )
    apply_theme(fig, title=title, show_legend=False, height=330)
    fig.update_xaxes(title=dict(text="log-odds (predicted class)", font=dict(size=11)))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig