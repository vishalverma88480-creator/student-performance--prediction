"""Landing page for predictions submitted through the form."""

import streamlit as st

from src.charts import prediction_distribution
from src.database import list_predictions, summarize_predictions
from src.ui import ensure_app_ready, kpi_cards, page_header, predictions_display_frame, show_chart, show_table


def _score(value) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


page_header(
    "Dashboard",
    "Counts and charts on this page use only predictions you submit.",
    show_back=False,
)
ensure_app_ready()
summary = summarize_predictions()
count = summary["count"]
has_scores = count > 0

kpi_cards(
    [
        {
            "label": "Total predictions",
            "value": f"{count:,}",
            "note": "Students scored in this app",
        },
        {
            "label": "Average predicted score",
            "value": _score(summary["average"]),
            "unit": "/ 20" if has_scores else "",
            "note": "Saved predictions" if has_scores else "No predictions yet",
        },
        {
            "label": "Highest predicted score",
            "value": _score(summary["highest"]),
            "unit": "/ 20" if has_scores else "",
            "note": "Saved predictions" if has_scores else "No predictions yet",
        },
    ]
)

recent = list_predictions()
if not recent:
    st.info("No predictions yet. The distribution and the recent list stay empty until you submit a student.")
    st.page_link(
        "pages/2_Predict_Performance.py",
        label="Predict performance",
        icon=":material/arrow_forward:",
    )
else:
    show_chart(prediction_distribution(recent), key="dash-distribution")
    st.caption("Each bar is a saved prediction. The dashed line is 10 out of 20.")
    st.subheader("Recent predictions")
    show_table(predictions_display_frame(recent[:8]))
