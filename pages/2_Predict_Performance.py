"""Form, validation, prediction, and the saved record."""

import streamlit as st

from src.config import (
    ADDRESS_LABELS,
    EDUCATION_LABELS,
    FAMSIZE_LABELS,
    SAMPLE_FORM,
    SEX_LABELS,
    STUDY_TIME_LABELS,
    SUBJECTS,
    YES_NO_LABELS,
)
from src.data_loader import reference_stats
from src.database import save_prediction
from src.predict import PredictionInputError, predict_and_explain
from src.ui import ensure_app_ready, page_header, render_result, section_label

FORM_DEFAULTS = {
    "form_student_name": "",
    "form_age": 17,
    "form_sex": "F",
    "form_address": "U",
    "form_famsize": "GT3",
    "form_subject": "Math",
    "form_medu": 2,
    "form_fedu": 2,
    "form_studytime": 2,
    "form_failures": 0,
    "form_absences": 4,
    "form_internet": "yes",
    "form_schoolsup": "no",
    "form_famsup": "yes",
    "form_activities": "no",
    "form_higher": "yes",
    "form_use_priors": False,
    "form_g1": 12,
    "form_g2": 12,
}


def _apply_mapping(values: dict) -> None:
    st.session_state.form_student_name = values["student_name"]
    st.session_state.form_age = int(values["age"])
    st.session_state.form_sex = values["sex"]
    st.session_state.form_address = values["address"]
    st.session_state.form_famsize = values["famsize"]
    st.session_state.form_subject = values["subject"]
    st.session_state.form_medu = int(values["Medu"])
    st.session_state.form_fedu = int(values["Fedu"])
    st.session_state.form_studytime = int(values["studytime"])
    st.session_state.form_failures = int(values["failures"])
    st.session_state.form_absences = int(values["absences"])
    st.session_state.form_internet = values["internet"]
    st.session_state.form_schoolsup = values["schoolsup"]
    st.session_state.form_famsup = values["famsup"]
    st.session_state.form_activities = values["activities"]
    st.session_state.form_higher = values["higher"]
    st.session_state.form_use_priors = bool(values["use_priors"])
    st.session_state.form_g1 = int(values["G1"])
    st.session_state.form_g2 = int(values["G2"])
    st.session_state.form_errors = []


def _apply_sample() -> None:
    _apply_mapping(SAMPLE_FORM)


def _clear_form() -> None:
    for key, value in FORM_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.form_errors = []
    st.session_state.pop("latest_result", None)
    st.session_state.pop("latest_record_id", None)


def _form_from_state() -> dict:
    return {
        "student_name": st.session_state.form_student_name,
        "age": st.session_state.form_age,
        "sex": st.session_state.form_sex,
        "address": st.session_state.form_address,
        "famsize": st.session_state.form_famsize,
        "subject": st.session_state.form_subject,
        "Medu": st.session_state.form_medu,
        "Fedu": st.session_state.form_fedu,
        "studytime": st.session_state.form_studytime,
        "failures": st.session_state.form_failures,
        "absences": st.session_state.form_absences,
        "internet": st.session_state.form_internet,
        "schoolsup": st.session_state.form_schoolsup,
        "famsup": st.session_state.form_famsup,
        "activities": st.session_state.form_activities,
        "higher": st.session_state.form_higher,
        "use_priors": st.session_state.form_use_priors,
        "G1": st.session_state.form_g1,
        "G2": st.session_state.form_g2,
    }


page_header(
    "Predict performance",
    "Enter the academic details the model was trained on. The score is a final course grade from 0 to 20.",
)
frame, _artifact = ensure_app_ready()

for key, value in FORM_DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("form_errors", [])

actions, _space = st.columns([2, 2])
with actions:
    sample_col, clear_col = st.columns(2)
    sample_col.button("Use sample student", on_click=_apply_sample, width="stretch")
    clear_col.button("Clear form", on_click=_clear_form, width="stretch")

with st.container(border=True):
    section_label("Student information")
    row_one = st.columns(3)
    row_one[0].text_input("Student name", key="form_student_name", placeholder="Student name")
    row_one[1].number_input(
        "Age (15–22)",
        step=1,
        key="form_age",
        help="The training data includes ages 15 through 22.",
    )
    row_one[2].selectbox(
        "Sex",
        options=list(SEX_LABELS),
        format_func=lambda code: SEX_LABELS[code],
        key="form_sex",
        help="The dataset records sex as female or male.",
    )
    row_two = st.columns(3)
    row_two[0].selectbox(
        "Address type",
        options=list(ADDRESS_LABELS),
        format_func=lambda code: ADDRESS_LABELS[code],
        key="form_address",
    )
    row_two[1].selectbox(
        "Family size",
        options=list(FAMSIZE_LABELS),
        format_func=lambda code: FAMSIZE_LABELS[code],
        key="form_famsize",
    )

with st.container(border=True):
    section_label("Academic information")
    academic = st.columns(3)
    academic[0].selectbox("Course", options=SUBJECTS, key="form_subject")
    academic[1].selectbox(
        "Mother's education",
        options=list(EDUCATION_LABELS),
        format_func=lambda code: EDUCATION_LABELS[code],
        key="form_medu",
    )
    academic[2].selectbox(
        "Father's education",
        options=list(EDUCATION_LABELS),
        format_func=lambda code: EDUCATION_LABELS[code],
        key="form_fedu",
    )
    support = st.columns(3)
    support[0].number_input(
        "Past class failures (0–3)",
        step=1,
        key="form_failures",
        help="Enter 3 if there were three or more. Every training row is 0, 1, 2, or 3.",
    )
    support[1].selectbox(
        "School support",
        options=list(YES_NO_LABELS),
        format_func=lambda code: YES_NO_LABELS[code],
        key="form_schoolsup",
        help="Extra educational support from the school.",
    )
    support[2].selectbox(
        "Family educational support",
        options=list(YES_NO_LABELS),
        format_func=lambda code: YES_NO_LABELS[code],
        key="form_famsup",
    )
    higher_row = st.columns(3)
    higher_row[0].selectbox(
        "Plans higher education",
        options=list(YES_NO_LABELS),
        format_func=lambda code: YES_NO_LABELS[code],
        key="form_higher",
    )

with st.container(border=True):
    section_label("Study habits")
    habits = st.columns(3)
    habits[0].selectbox(
        "Weekly study time",
        options=list(STUDY_TIME_LABELS),
        format_func=lambda code: STUDY_TIME_LABELS[code],
        key="form_studytime",
        help="These are the four bands in the training file, not a free-typed hour count.",
    )
    habits[1].selectbox(
        "Extra activities",
        options=list(YES_NO_LABELS),
        format_func=lambda code: YES_NO_LABELS[code],
        key="form_activities",
    )
    habits[2].selectbox(
        "Internet at home",
        options=list(YES_NO_LABELS),
        format_func=lambda code: YES_NO_LABELS[code],
        key="form_internet",
    )

with st.container(border=True):
    section_label("Attendance")
    absence_row = st.columns(3)
    absence_row[0].number_input(
        "Absences (0–75)",
        step=1,
        key="form_absences",
        help="75 is the highest absence count in the training file.",
    )

with st.container(border=True):
    section_label("Previous academic performance")
    st.checkbox(
        "I have both period grades for this course (G1 and G2)",
        key="form_use_priors",
        help="Turn this on only when both earlier grades in the same course are known.",
    )
    if st.session_state.form_use_priors:
        st.caption(
            "G1 and G2 are earlier grades in this course, not the final grade. "
            "The app will use the model that was trained with them."
        )
        grade_cols = st.columns(3)
        grade_cols[0].number_input("Period grade G1 (0–20)", step=1, key="form_g1")
        grade_cols[1].number_input("Period grade G2 (0–20)", step=1, key="form_g2")
    else:
        st.caption(
            "Leave this off when those grades are not known. "
            "The app then uses the model trained without them, and it does not fill in a guessed grade."
        )

st.caption("A successful prediction is saved on this computer and appears under Student records.")
predict_clicked = st.button("Predict performance", type="primary")

if predict_clicked:
    try:
        with st.spinner("Scoring this student..."):
            result = predict_and_explain(_form_from_state(), stats=reference_stats(frame))
    except PredictionInputError as exc:
        st.session_state.form_errors = exc.errors
    except Exception as exc:
        st.session_state.form_errors = [f"The prediction could not be completed: {exc}"]
    else:
        st.session_state.form_errors = []
        try:
            st.session_state.latest_record_id = save_prediction(result)
        except Exception as exc:
            st.session_state.latest_record_id = None
            st.error(f"The score was calculated, but it could not be saved: {exc}")
        st.session_state.latest_result = result

for message in st.session_state.form_errors:
    st.error(message)

if st.session_state.get("latest_result"):
    render_result(st.session_state.latest_result, st.session_state.get("latest_record_id"))
