"""Plotly figures for the A/B test analysis.

Every function returns a figure and shows nothing: call fig.show() in a script
or st.plotly_chart(fig) in Streamlit.
"""

from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from abtest.data import CONTROL, EXPERIMENTAL, OUTLIER_PERCENTILE, add_log_order_value

Theme = Literal["light", "dark"]

# Categorical slots 1 and 2 of the reference palette, stepped for each theme.
# A group keeps its color in every chart.
GROUP_COLORS: dict[Theme, dict[str, str]] = {
    "light": {CONTROL: "#2a78d6", EXPERIMENTAL: "#eb6834"},
    "dark": {CONTROL: "#3987e5", EXPERIMENTAL: "#d95926"},
}

# Two steps of the same blue ramp, not two distinct hues: the A/A samples are
# meant to be indistinguishable draws from one population, not two identities,
# so giving them different colors like Control/Experimental would visually
# imply a difference that should not be there.
AA_SAMPLE_SHADES: dict[Theme, tuple[str, str]] = {
    "light": ("#6da7ec", "#184f95"),
    "dark": ("#86b6ef", "#1c5cab"),
}

INK: dict[Theme, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

GROUP_ORDER = {"group": [CONTROL, EXPERIMENTAL]}
FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _label(column: str) -> str:
    return column.replace("_", " ").capitalize()


def _style(fig: go.Figure, title: str, theme: Theme) -> go.Figure:
    """Apply the shared look: transparent surface, quiet grid, left-aligned title."""
    ink = INK[theme]
    fig.update_layout(
        template="none",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=ink["primary"])),
        font=dict(family=FONT_FAMILY, size=13, color=ink["secondary"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=ink["surface"], font_color=ink["primary"]),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""
        ),
        margin=dict(t=90, r=24, b=56, l=64),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=ink["axis"],
        ticks="outside", tickcolor=ink["axis"],
    )
    fig.update_yaxes(gridcolor=ink["grid"], gridwidth=1, zeroline=False, showline=False)
    return fig


def sessions_by_day(df: pd.DataFrame, theme: Theme = "light") -> go.Figure:
    """Line chart of daily session counts per group, to check for seasonality."""
    daily = (
        df.assign(date=pd.to_datetime(df["date"]))
        .groupby(["date", "group"])
        .size()
        .reset_index(name="sessions")
    )

    fig = px.line(
        daily,
        x="date",
        y="sessions",
        color="group",
        markers=True,
        color_discrete_map=GROUP_COLORS[theme],
        category_orders=GROUP_ORDER,
        labels={"date": "Date", "sessions": "Sessions", "group": ""},
    )
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_layout(hovermode="x unified")
    return _style(fig, "Number of sessions per day by group", theme)


def histogram_by_group(
    df: pd.DataFrame,
    column: str,
    title: str,
    x_label: str | None = None,
    nbins: int | None = None,
    theme: Theme = "light",
) -> go.Figure:
    """One histogram panel per group, sharing axes so the panels compare directly."""
    fig = px.histogram(
        df,
        x=column,
        color="group",
        facet_col="group",
        nbins=nbins,
        color_discrete_map=GROUP_COLORS[theme],
        category_orders=GROUP_ORDER,
        labels={column: x_label or _label(column)},
    )
    fig.update_traces(marker_line_color=INK[theme]["surface"], marker_line_width=2)
    fig.update_yaxes(title_text="")
    fig.update_yaxes(title_text="Frequency", col=1)
    # Panel titles name the group, so a separate legend would repeat them.
    fig.for_each_annotation(lambda note: note.update(text=note.text.split("=")[-1]))
    fig.update_layout(showlegend=False)
    return _style(fig, title, theme)


def aa_samples_comparison(aa_test: pd.DataFrame, theme: Theme = "light") -> go.Figure:
    """Box plot of the two A/A samples, with every point shown: both the raw
    values and the visual comparison in one chart.
    """
    shade_1, shade_2 = AA_SAMPLE_SHADES[theme]

    fig = go.Figure()
    for column, color in zip(["Sample 1", "Sample 2"], [shade_1, shade_2]):
        fig.add_trace(
            go.Box(
                y=aa_test[column],
                name=column,
                marker_color=color,
                line_color=color,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker_size=5,
            )
        )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Order value")
    return _style(fig, "A/A samples: order value, every session shown", theme)


def session_scatter(
    ab_test: pd.DataFrame,
    percentile: float = OUTLIER_PERCENTILE,
    theme: Theme = "light",
) -> go.Figure:
    """Every session as one point: order value against session duration,
    colored by group. Dashed lines mark the percentile cutoff used to trim
    outliers; points beyond either line are the ones set aside.
    """
    ink = INK[theme]
    thresholds = ab_test[["order_value", "session_duration"]].quantile(percentile / 100)

    fig = px.scatter(
        ab_test,
        x="session_duration",
        y="order_value",
        color="group",
        color_discrete_map=GROUP_COLORS[theme],
        category_orders=GROUP_ORDER,
        opacity=0.6,
        labels={"session_duration": "Session duration", "order_value": "Order value", "group": ""},
    )
    fig.update_traces(marker=dict(size=6, line_width=0))
    fig.add_hline(y=thresholds["order_value"], line=dict(color=ink["axis"], width=1, dash="dash"))
    fig.add_vline(x=thresholds["session_duration"], line=dict(color=ink["axis"], width=1, dash="dash"))
    # A handful of extreme order values would otherwise squash the rest of
    # the points into an unreadable cluster near zero; a log axis (the same
    # fix compare_aov's log transform applies) keeps every point visible.
    fig.update_yaxes(type="log", dtick=1)  # major ticks only (1, 10, 100, ...)
    return _style(fig, "Every session: order value vs. duration", theme)


def log_order_value_histogram(df: pd.DataFrame, theme: Theme = "light") -> go.Figure:
    """Histogram of log-transformed order values, the input to the parametric test."""
    return histogram_by_group(
        add_log_order_value(df),
        column="log_order_value",
        title="Distribution of log-transformed order value by group",
        x_label="ln(Order value)",
        nbins=30,
        theme=theme,
    )
