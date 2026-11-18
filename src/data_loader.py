"""Load the UCI Student Performance files and check that they still match the study.

The two course files are semicolon-separated. This module combines them into
data/student_data.csv and refuses to train if a required column or an unexpected
category appears.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ABSENCE_MAX,
    ADDRESS_LABELS,
    AGE_MAX,
    AGE_MIN,
    CATEGORICAL_FEATURES,
    CITATION,
    DATA_CSV,
    DATA_DIR,
    EDU_MAX,
    EDU_MIN,
    FAILURES_MAX,
    FAILURES_MIN,
    FAMSIZE_LABELS,
    FEATURES_NO_PRIORS,
    FEATURES_WITH_PRIORS,
    GRADE_MAX,
    GRADE_MIN,
    NUMERIC_FEATURES,
    OVERLAP_NOTE,
    RAW_MAT,
    RAW_POR,
    SEX_LABELS,
    SOURCE_URL,
    STUDY_MAX,
    STUDY_MIN,
    SUBJECTS,
    TARGET,
    YES_NO_LABELS,
)

RAW_FEATURE_COLUMNS = [column for column in FEATURES_WITH_PRIORS if column != "subject"]
PREPARED_COLUMNS = FEATURES_WITH_PRIORS + [TARGET]

ALLOWED_VALUES = {
    "sex": set(SEX_LABELS),
    "address": set(ADDRESS_LABELS),
    "famsize": set(FAMSIZE_LABELS),
    "schoolsup": set(YES_NO_LABELS),
    "famsup": set(YES_NO_LABELS),
    "activities": set(YES_NO_LABELS),
    "higher": set(YES_NO_LABELS),
    "internet": set(YES_NO_LABELS),
    "subject": set(SUBJECTS),
}

NUMERIC_RANGES = {
    "age": (AGE_MIN, AGE_MAX),
    "Medu": (EDU_MIN, EDU_MAX),
    "Fedu": (EDU_MIN, EDU_MAX),
    "studytime": (STUDY_MIN, STUDY_MAX),
    "failures": (FAILURES_MIN, FAILURES_MAX),
    "absences": (0, 93),
    "G1": (GRADE_MIN, GRADE_MAX),
    "G2": (GRADE_MIN, GRADE_MAX),
    "G3": (GRADE_MIN, GRADE_MAX),
}


class DataError(Exception):
    """Raised when the student file is missing or does not match the expected schema."""


def build_student_data() -> pd.DataFrame:
    """Rebuild data/student_data.csv from the two original UCI course files."""
    frame = _combine_raw_files()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_CSV, index=False)
    return _read_prepared(DATA_CSV)


def load_student_data() -> pd.DataFrame:
    """Return the prepared course table, building it from the UCI files if needed."""
    if not DATA_CSV.exists():
        return build_student_data()
    return _read_prepared(DATA_CSV)


def reference_stats(frame: pd.DataFrame) -> dict:
    """Summaries used to phrase recommendations against the real dataset."""
    return {
        "absences_median": float(frame["absences"].median()),
        "absences_p75": float(frame["absences"].quantile(0.75)),
        "grade_mean": float(frame[TARGET].mean()),
    }


def profile_dataset(frame: pd.DataFrame) -> dict:
    """Describe the table with values computed from the rows, not typed by hand."""
    return {
        "rows": int(len(frame)),
        "math_rows": int((frame["subject"] == "Math").sum()),
        "portuguese_rows": int((frame["subject"] == "Portuguese").sum()),
        "schools": _source_schools(),
        "missing_cells": int(frame.isna().sum().sum()),
        "target_mean": float(frame[TARGET].mean()),
        "target_min": int(frame[TARGET].min()),
        "target_max": int(frame[TARGET].max()),
        "zero_grades": int((frame[TARGET] == 0).sum()),
        "pass_rate": float((frame[TARGET] >= 10).mean()),
        "age_min": int(frame["age"].min()),
        "age_max": int(frame["age"].max()),
        "absence_max": int(frame["absences"].max()),
        "source_url": SOURCE_URL,
        "citation": CITATION,
        "overlap_note": OVERLAP_NOTE,
        "input_features_with_priors": len(FEATURES_WITH_PRIORS),
        "input_features_without_priors": len(FEATURES_NO_PRIORS),
    }


def _source_schools() -> list[str]:
    """School codes in the original files. School is not a model input."""
    found = set()
    for path in (RAW_MAT, RAW_POR):
        if not path.exists():
            continue
        raw = pd.read_csv(path, sep=";", usecols=["school"])
        found.update(raw["school"].astype(str).str.strip().tolist())
    return sorted(found)


def _combine_raw_files() -> pd.DataFrame:
    if not RAW_MAT.exists() or not RAW_POR.exists():
        raise DataError(
            "The UCI source files are missing. Expected "
            f"{RAW_MAT} and {RAW_POR}. Download the Student Performance dataset "
            f"from {SOURCE_URL}, unzip it, and place student-mat.csv and "
            "student-por.csv in data/raw/. Then run: python -m src.train_model"
        )

    mathematics = _read_course(RAW_MAT, "Math")
    portuguese = _read_course(RAW_POR, "Portuguese")
    frame = pd.concat([mathematics, portuguese], ignore_index=True)

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").str.strip()

    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[TARGET].isna().any():
        raise DataError("Some rows have no final grade (G3), so the file was not saved.")

    if frame[NUMERIC_FEATURES].isna().any().any():
        raise DataError("Some model inputs could not be read as numbers.")

    _check_ranges(frame)
    _check_categories(frame)

    if int(frame["absences"].max()) > ABSENCE_MAX:
        raise DataError(
            "The absences column goes above the form maximum of "
            f"{ABSENCE_MAX}. Review the file before raising that limit."
        )

    for column in NUMERIC_FEATURES + [TARGET]:
        values = frame[column].to_numpy(dtype=float)
        if not np.allclose(values, np.round(values)):
            raise DataError(f"{column} contains a value that is not a whole number.")
        frame[column] = np.round(values).astype(int)

    return frame[PREPARED_COLUMNS].reset_index(drop=True)


def _read_course(path: Path, subject: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    required = RAW_FEATURE_COLUMNS + [TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataError(f"{path.name} is missing columns: {', '.join(missing)}")
    kept = frame[required].copy()
    kept["subject"] = subject
    return kept


def _read_prepared(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in PREPARED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(
            f"{path.name} is missing columns: {', '.join(missing)}. "
            "Rebuild it with: python -m src.train_model"
        )
    frame = frame[PREPARED_COLUMNS].copy()
    if frame.empty:
        raise DataError(f"{path.name} has no rows.")
    if TARGET not in frame.columns:
        raise DataError("The prepared file has no final grade column.")
    _check_ranges(frame)
    _check_categories(frame)
    if frame.isna().any().any():
        raise DataError(f"{path.name} contains missing values.")
    return frame.reset_index(drop=True)


def _check_ranges(frame: pd.DataFrame) -> None:
    for column, (low, high) in NUMERIC_RANGES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.between(low, high).all():
            raise DataError(
                f"{column} has a value outside the expected range {low}–{high}."
            )


def _check_categories(frame: pd.DataFrame) -> None:
    for column, allowed in ALLOWED_VALUES.items():
        observed = set(frame[column].astype(str).str.strip().unique())
        unexpected = observed - allowed
        if unexpected:
            raise DataError(
                f"{column} has values the model is not built for: {sorted(unexpected)}"
            )
