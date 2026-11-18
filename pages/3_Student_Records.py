"""Search, filter, inspect, and delete saved predictions."""

import streamlit as st

from src.config import SEX_LABELS, SUBJECTS
from src.database import delete_prediction, get_prediction, list_predictions, summarize_predictions
from src.ui import (
    display_feature_value,
    ensure_app_ready,
    format_timestamp,
    page_header,
    predictions_display_frame,
    section_label,
    show_table,
)

page_header(
    "Student records",
    "Every prediction saved on this computer. Search, filter, open, or delete a record.",
)
_frame, _artifact = ensure_app_ready()
st.session_state.setdefault("delete_nonce", 0)

total = summarize_predictions()["count"]
filters = st.columns([1.4, 1, 1, 1])
search = filters[0].text_input("Search by name", placeholder="Type part of a name")
category = filters[1].selectbox(
    "Category",
    ["All", "Excellent", "Good", "Average", "Needs Improvement"],
)
sex_label = filters[2].selectbox("Sex", ["All", "Female", "Male"])
subject = filters[3].selectbox("Course", ["All", *SUBJECTS])

sex_code = None if sex_label == "All" else {label: code for code, label in SEX_LABELS.items()}[sex_label]
rows = list_predictions(
    search=search,
    category=None if category == "All" else category,
    sex=sex_code,
    subject=None if subject == "All" else subject,
)
st.caption(f"Showing {len(rows)} of {total} saved predictions.")

if total == 0:
    st.info("No predictions have been saved yet. Use Predict performance to create the first one.")
    st.page_link(
        "pages/2_Predict_Performance.py",
        label="Predict performance",
        icon=":material/arrow_forward:",
    )
    st.stop()

if not rows:
    st.warning("No records match this search or filter.")
    st.stop()

show_table(predictions_display_frame(rows))

options = {
    f"#{row['id']} · {row['student_name']} · {row['predicted_score']:.2f}": row["id"]
    for row in rows
}
selected_label = st.selectbox("Open a record", list(options))
record = get_prediction(options[selected_label])
if record is None:
    st.error("That record is no longer in the database.")
    st.stop()

section_label("Record detail")
detail = st.columns(4)
detail[0].metric("Predicted score", f"{record['predicted_score']:.2f}")
detail[1].metric("Category", record["category"])
detail[2].metric("Model", record["model_name"])
detail[3].metric("Saved", format_timestamp(record["created_at"]))

grades = "Not used"
if record["used_previous_grades"] and record["g1"] is not None:
    grades = f"G1 {record['g1']:.0f} · G2 {record['g2']:.0f}"
st.caption(
    f"Course: {record['subject']}. Previous grades: {grades}. "
    f"Study time code: {record['studytime']}. Absences: {record['absences']}."
)

left, right = st.columns(2)
with left:
    st.markdown("**Recommendations saved with this record**")
    for item in record["recommendations"]:
        st.write(f"{item['priority']} · {item['title']}")
        st.write(item["detail"])
with right:
    st.markdown("**Study plan saved with this record**")
    for tip in record["study_suggestions"]:
        st.write(f"• {tip}")

with st.expander("Fields stored for this student"):
    stored = {
        "sex": record["sex"],
        "age": record["age"],
        "address": record["address"],
        "famsize": record["famsize"],
        "Medu": record["medu"],
        "Fedu": record["fedu"],
        "studytime": record["studytime"],
        "failures": record["failures"],
        "absences": record["absences"],
        "internet": record["internet"],
        "schoolsup": record["schoolsup"],
        "famsup": record["famsup"],
        "activities": record["activities"],
        "higher": record["higher"],
        "subject": record["subject"],
    }
    for feature, value in stored.items():
        st.write(f"{feature}: {display_feature_value(feature, value)}")

st.checkbox(
    "Delete this record permanently",
    key=f"confirm_delete_{st.session_state.delete_nonce}",
)
if st.button(
    "Delete record",
    type="primary",
    disabled=not st.session_state.get(f"confirm_delete_{st.session_state.delete_nonce}", False),
):
    removed = delete_prediction(record["id"])
    st.session_state.delete_nonce += 1
    if removed:
        st.success(f"Deleted record #{record['id']}.")
        st.rerun()
    st.error("That record was already gone.")
