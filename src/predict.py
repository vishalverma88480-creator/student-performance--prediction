"""Validate a form, load the saved pipeline, and score one student.

The model file is produced by src.train_model. This module does not fit anything.
"""

import math
import re

import joblib
import pandas as pd

from src.config import (
    ABSENCE_MAX,
    ABSENCE_MIN,
    ADDRESS_LABELS,
    AGE_MAX,
    AGE_MIN,
    EDU_MAX,
    EDU_MIN,
    FAILURES_MAX,
    FAILURES_MIN,
    FAMSIZE_LABELS,
    GRADE_MAX,
    GRADE_MIN,
    MODEL_PATH,
    NUMERIC_FEATURES,
    SAMPLE_FORM,
    SEX_LABELS,
    SHORT_LIMITATION,
    STUDY_MAX,
    STUDY_MIN,
    SUBJECTS,
    TARGET,
    YES_NO_LABELS,
    category_for,
)
from src.data_loader import load_student_data, reference_stats
from src.recommendations import build_recommendations, study_suggestions


class ModelNotFoundError(FileNotFoundError):
    """Raised when models/best_model.joblib is missing or incomplete."""


class PredictionInputError(ValueError):
    """Raised when the form cannot be sent to the model. `errors` is a list of messages."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(" ".join(errors))


def load_artifact() -> dict:
    """Load the trained pipelines and the metrics saved beside them."""
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            "No trained model was found at models/best_model.joblib. "
            "Train it with: python -m src.train_model"
        )
    artifact = joblib.load(MODEL_PATH)
    for key in ("with_priors", "without_priors", "dataset"):
        if key not in artifact:
            raise ModelNotFoundError(
                "The saved model file is incomplete. Train it again with: "
                "python -m src.train_model"
            )
    return artifact


def validate_form(form: dict) -> dict:
    """Check types and ranges. Return a cleaned copy, or raise PredictionInputError."""
    errors: list[str] = []
    cleaned: dict = {}

    name = str(form.get("student_name", "")).strip()
    if not 2 <= len(name) <= 60:
        errors.append("Enter a student name between 2 and 60 characters.")
    elif not re.fullmatch(r"[\w .'\-]+", name, flags=re.UNICODE):
        errors.append("Use letters, numbers, spaces, apostrophes, or hyphens in the name.")
    elif not re.search(r"[^\W\d_]", name, flags=re.UNICODE):
        errors.append("Include at least one letter in the student name.")
    else:
        cleaned["student_name"] = name

    cleaned["sex"] = _choice(form.get("sex"), set(SEX_LABELS), "sex", errors)
    cleaned["address"] = _choice(form.get("address"), set(ADDRESS_LABELS), "address type", errors)
    cleaned["famsize"] = _choice(form.get("famsize"), set(FAMSIZE_LABELS), "family size", errors)
    cleaned["subject"] = _choice(form.get("subject"), set(SUBJECTS), "course", errors)
    for field, label in (
        ("schoolsup", "school support"),
        ("famsup", "family support"),
        ("activities", "extra activities"),
        ("higher", "higher-education plan"),
        ("internet", "internet access"),
    ):
        cleaned[field] = _choice(form.get(field), set(YES_NO_LABELS), label, errors)

    cleaned["age"] = _whole_number(form.get("age"), "Age", AGE_MIN, AGE_MAX, errors)
    cleaned["Medu"] = _whole_number(
        form.get("Medu"), "Mother's education", EDU_MIN, EDU_MAX, errors
    )
    cleaned["Fedu"] = _whole_number(
        form.get("Fedu"), "Father's education", EDU_MIN, EDU_MAX, errors
    )
    cleaned["studytime"] = _whole_number(
        form.get("studytime"), "Weekly study time", STUDY_MIN, STUDY_MAX, errors
    )
    cleaned["failures"] = _whole_number(
        form.get("failures"), "Past class failures", FAILURES_MIN, FAILURES_MAX, errors
    )
    cleaned["absences"] = _whole_number(
        form.get("absences"), "Absences", ABSENCE_MIN, ABSENCE_MAX, errors
    )

    use_priors = bool(form.get("use_priors"))
    cleaned["use_priors"] = use_priors
    if use_priors:
        cleaned["G1"] = _whole_number(
            form.get("G1"), "Period grade G1", GRADE_MIN, GRADE_MAX, errors
        )
        cleaned["G2"] = _whole_number(
            form.get("G2"), "Period grade G2", GRADE_MIN, GRADE_MAX, errors
        )
    else:
        cleaned["G1"] = None
        cleaned["G2"] = None

    # Drop messages that were appended twice for a missing period grade.
    unique_errors = list(dict.fromkeys(errors))
    if unique_errors:
        raise PredictionInputError(unique_errors)
    return cleaned


def model_frame(cleaned: dict, features: list[str]) -> pd.DataFrame:
    """Build the one-row frame the pipeline expects. G3 is not copied in."""
    row = {}
    for feature in features:
        value = cleaned[feature]
        if feature in NUMERIC_FEATURES:
            row[feature] = float(value)
        else:
            row[feature] = str(value)
    frame = pd.DataFrame([row], columns=features)
    if TARGET in frame.columns:
        raise RuntimeError("The final grade was included in the model frame.")
    return frame


def predict_and_explain(form: dict, stats: dict | None = None) -> dict:
    """Validate, choose the pipeline, and return the score plus advice."""
    cleaned = validate_form(form)
    artifact = load_artifact()
    feature_set = "with_priors" if cleaned["use_priors"] else "without_priors"
    bundle = artifact[feature_set]
    frame = model_frame(cleaned, bundle["features"])
    raw_score = float(bundle["pipeline"].predict(frame)[0])
    clipped = min(float(GRADE_MAX), max(float(GRADE_MIN), raw_score))
    score = round(clipped, 2)
    category = category_for(score)

    if stats is None:
        stats = reference_stats(load_student_data())

    return {
        "student_name": cleaned["student_name"],
        "predicted_score": score,
        "raw_score": round(raw_score, 4),
        "was_clipped": abs(raw_score - clipped) > 1e-9,
        "predicted_percent": round(score / GRADE_MAX * 100, 1),
        "category": category,
        "model_name": bundle["algorithm"],
        "feature_set": feature_set,
        "features_used": list(bundle["features"]),
        "used_previous_grades": cleaned["use_priors"],
        "recommendations": build_recommendations(cleaned, score, category, stats),
        "study_suggestions": study_suggestions(cleaned, score),
        "inputs": cleaned,
        "limitation": SHORT_LIMITATION,
    }


def sample_form(**overrides) -> dict:
    """Copy the demo student and apply any test overrides."""
    form = dict(SAMPLE_FORM)
    form.update(overrides)
    return form


def _choice(value, allowed: set, label: str, errors: list[str]):
    if value not in allowed:
        errors.append(f"Choose a valid {label}.")
        return None
    return value


def _whole_number(value, label: str, low: int, high: int, errors: list[str]):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        errors.append(f"{label} must be a whole number from {low} to {high}.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a whole number from {low} to {high}.")
        return None
    if math.isnan(number) or not number.is_integer():
        errors.append(f"{label} must be a whole number from {low} to {high}.")
        return None
    whole = int(number)
    if whole < low or whole > high:
        errors.append(f"{label} must be a whole number from {low} to {high}.")
        return None
    return whole
