"""Charts of predictions saved from the form. The training file is not plotted."""

import streamlit as st

from src.charts import prediction_distribution, saved_absences_chart, saved_study_chart
from src.database import list_predictions
from src.ui import ensure_app_ready, page_header, show_chart

page_header(
    "Analytics",
    "These charts use only predictions saved from the form.",
)
ensure_app_ready()
records = list_predictions()

if not records:
    st.info(
        "No saved predictions yet. Submit a student on Predict performance, then return here."
    )
    st.stop()

count_label = "saved prediction" if len(records) == 1 else "saved predictions"
st.caption(f"{len(records):,} {count_label}. The training file is not included.")

left, right = st.columns(2)
with left:
    show_chart(prediction_distribution(records), key="analytics-distribution")
    st.caption("Scores from saved predictions. The line marks 10.")
with right:
    show_chart(saved_study_chart(records), key="analytics-study")
    st.caption("Each point is one saved prediction.")

show_chart(saved_absences_chart(records), key="analytics-absences")
st.caption(
    "Each point is one saved prediction. Actual final grades are not stored with a "
    "submission, so this page does not compare actual and predicted scores."
)
