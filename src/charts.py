"""Plotly charts. Every figure is built from a dataframe or the saved test results."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    EDUCATION_LABELS,
    FEATURE_LABELS,
    STUDY_TIME_LABELS,
    TARGET,
    category_for,
)

PALETTE = ["#312E81", "#0F766E", "#C2410C", "#0369A1"]
CATEGORY_COLORS = {
    "Excellent": "#047857",
    "Good": "#1D4ED8",
    "Average": "#B45309",
    "Needs Improvement": "#BE123C",
}
CATEGORY_ORDER = ["Excellent", "Good", "Average", "Needs Improvement"]


def apply_layout(fig, title: str, height: int = 440):
    fig.update_layout(
        template="simple_white",
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=18, color="#0F172A", family="Source Sans 3, sans-serif"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, sans-serif", color="#334155", size=13),
        height=height,
        margin=dict(l=16, r=16, t=64, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        colorway=PALETTE,
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0")
    return fig


def grade_distribution(frame: pd.DataFrame):
    plot = frame.copy()
    plot["Course"] = plot["subject"]
    fig = px.histogram(
        plot,
        x=TARGET,
        color="Course",
        nbins=21,
        barmode="overlay",
        opacity=0.82,
        labels={TARGET: "Final grade (0–20)"},
        color_discrete_sequence=PALETTE,
    )
    fig.add_vline(x=10, line_dash="dash", line_color="#0F172A")
    return apply_layout(fig, "Student performance distribution")


def category_donut(frame: pd.DataFrame):
    counts = (
        frame[TARGET]
        .map(category_for)
        .value_counts()
        .reindex(CATEGORY_ORDER, fill_value=0)
    )
    fig = go.Figure(
        go.Pie(
            labels=CATEGORY_ORDER,
            values=counts.to_numpy(),
            hole=0.62,
            sort=False,
            marker=dict(colors=[CATEGORY_COLORS[label] for label in CATEGORY_ORDER]),
            textinfo="percent",
            hovertemplate="%{label}<br>%{value} records<extra></extra>",
        )
    )
    fig.update_layout(showlegend=True)
    return apply_layout(fig, "Historical grades by category")


def study_time_chart(frame: pd.DataFrame):
    plot = frame.copy()
    plot["Weekly study time"] = plot["studytime"].map(STUDY_TIME_LABELS)
    plot["Course"] = plot["subject"]
    fig = px.box(
        plot,
        x="Weekly study time",
        y=TARGET,
        color="Course",
        category_orders={"Weekly study time": list(STUDY_TIME_LABELS.values())},
        labels={TARGET: "Final grade"},
        color_discrete_sequence=PALETTE,
        points=False,
    )
    return apply_layout(fig, "Study time versus performance")


def absences_chart(frame: pd.DataFrame):
    plot = frame.copy()
    plot["Course"] = plot["subject"]
    fig = px.scatter(
        plot,
        x="absences",
        y=TARGET,
        color="Course",
        opacity=0.45,
        labels={"absences": "Absences", TARGET: "Final grade"},
        color_discrete_sequence=PALETTE,
    )
    plot["band"] = pd.cut(
        plot["absences"],
        bins=[-0.01, 0, 2, 5, 10, 20, 80],
        include_lowest=True,
    )
    means = (
        plot.groupby(["Course", "band"], observed=True)
        .agg(absences=("absences", "mean"), grade=(TARGET, "mean"), rows=(TARGET, "size"))
        .reset_index()
    )
    for course, color in (("Math", PALETTE[0]), ("Portuguese", PALETTE[1])):
        part = means[means["Course"] == course].sort_values("absences")
        fig.add_scatter(
            x=part["absences"],
            y=part["grade"],
            mode="lines+markers",
            name=f"{course} band average",
            line=dict(color=color, width=3),
            marker=dict(size=8),
        )
    return apply_layout(fig, "Attendance versus performance")


def education_chart(frame: pd.DataFrame, column: str, title: str):
    plot = frame.copy()
    plot["Education"] = plot[column].map(EDUCATION_LABELS)
    plot["Course"] = plot["subject"]
    fig = px.box(
        plot,
        x="Education",
        y=TARGET,
        color="Course",
        category_orders={"Education": [EDUCATION_LABELS[level] for level in range(5)]},
        labels={TARGET: "Final grade"},
        color_discrete_sequence=PALETTE,
        points=False,
    )
    fig.update_xaxes(tickangle=-25)
    return apply_layout(fig, title)


def prediction_distribution(records: list[dict]):
    """Distribution of scores the user has saved. Empty input is not plotted."""
    plot = pd.DataFrame(records)
    fig = px.histogram(
        plot,
        x="predicted_score",
        color="category",
        nbins=21,
        category_orders={"category": CATEGORY_ORDER},
        color_discrete_map=CATEGORY_COLORS,
        labels={"predicted_score": "Predicted score (0–20)", "category": "Category"},
    )
    fig.add_vline(x=10, line_dash="dash", line_color="#0F172A")
    return apply_layout(fig, "Performance distribution")


def saved_study_chart(records: list[dict]):
    """Study time versus the scores the user saved. Not the training file."""
    plot = pd.DataFrame(records)
    plot["Weekly study time"] = plot["studytime"].map(
        lambda value: STUDY_TIME_LABELS.get(int(value), str(value))
    )
    fig = px.scatter(
        plot,
        x="Weekly study time",
        y="predicted_score",
        color="category",
        hover_name="student_name",
        category_orders={
            "Weekly study time": list(STUDY_TIME_LABELS.values()),
            "category": CATEGORY_ORDER,
        },
        color_discrete_map=CATEGORY_COLORS,
        labels={"predicted_score": "Predicted score (0–20)", "category": "Category"},
    )
    fig.add_hline(y=10, line_dash="dash", line_color="#64748B")
    return apply_layout(fig, "Study time versus predicted score")


def saved_absences_chart(records: list[dict]):
    """Absences versus the scores the user saved. Not the training file."""
    plot = pd.DataFrame(records)
    fig = px.scatter(
        plot,
        x="absences",
        y="predicted_score",
        color="category",
        hover_name="student_name",
        category_orders={"category": CATEGORY_ORDER},
        color_discrete_map=CATEGORY_COLORS,
        labels={
            "absences": "Absences",
            "predicted_score": "Predicted score (0–20)",
            "category": "Category",
        },
    )
    fig.add_hline(y=10, line_dash="dash", line_color="#64748B")
    return apply_layout(fig, "Attendance versus predicted score")


def prediction_history(records: list[dict]):
    plot = pd.DataFrame(records)
    fig = px.scatter(
        plot,
        x="created_at",
        y="predicted_score",
        color="category",
        hover_name="student_name",
        category_orders={"category": CATEGORY_ORDER},
        color_discrete_map=CATEGORY_COLORS,
        labels={
            "created_at": "Saved",
            "predicted_score": "Predicted score",
            "category": "Category",
        },
    )
    fig.add_hline(y=10, line_dash="dash", line_color="#64748B")
    return apply_layout(fig, "Saved predictions")


def model_comparison_chart(artifact: dict, metric_key: str, metric_label: str):
    rows = []
    labels = {
        "with_priors": "With previous grades",
        "without_priors": "Without previous grades",
    }
    for key, label in labels.items():
        for candidate in artifact[key]["candidates"]:
            rows.append(
                {
                    "Model": candidate["algorithm"],
                    "Features": label,
                    metric_label: candidate[metric_key],
                }
            )
    fig = px.bar(
        pd.DataFrame(rows),
        x="Model",
        y=metric_label,
        color="Features",
        barmode="group",
        color_discrete_sequence=PALETTE,
    )
    return apply_layout(fig, f"{metric_label} by model")


def actual_vs_predicted(y_test: list[float], y_pred: list[float]):
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Actual final grade", "y": "Predicted final grade"},
        opacity=0.75,
        color_discrete_sequence=["#312E81"],
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=20,
        y1=20,
        line=dict(color="#0F172A", dash="dash"),
    )
    fig.update_xaxes(range=[-0.4, 20.4])
    fig.update_yaxes(range=[-0.4, 20.4])
    return apply_layout(fig, "Actual versus predicted on the test set")


def importance_chart(rows: list[dict]):
    ordered = sorted(rows, key=lambda item: item["importance"])
    plot = pd.DataFrame(
        {
            "Feature": [
                FEATURE_LABELS.get(item["feature"], item["feature"]) for item in ordered
            ],
            "Importance": [item["importance"] for item in ordered],
        }
    )
    fig = px.bar(
        plot,
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#312E81"],
    )
    fig.update_xaxes(tickformat=".0%")
    height = max(420, 34 * max(len(ordered), 1) + 90)
    return apply_layout(fig, "Which inputs the selected model used", height=height)


def score_gauge(score: float, category: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 40, "color": "#0F172A"}},
            gauge={
                "axis": {"range": [0, 20], "tickcolor": "#94A3B8"},
                "bar": {"color": CATEGORY_COLORS[category], "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 10], "color": "#FEE2E2"},
                    {"range": [10, 14], "color": "#FEF3C7"},
                    {"range": [14, 16], "color": "#DBEAFE"},
                    {"range": [16, 20], "color": "#D1FAE5"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=18, r=18, t=24, b=8),
        font={"family": "Source Sans 3, sans-serif", "color": "#334155"},
    )
    return fig
