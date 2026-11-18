"""Paths, feature lists, and labels shared by training and the app.

The final grade column is G3. It is the prediction target and is intentionally
absent from every feature list below.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

DATA_CSV = DATA_DIR / "student_data.csv"
RAW_MAT = RAW_DIR / "student-mat.csv"
RAW_POR = RAW_DIR / "student-por.csv"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
REPORT_PATH = MODEL_DIR / "training_report.json"
COMPARISON_CHART = ASSETS_DIR / "training_comparison.png"
DATABASE_PATH = DATA_DIR / "predictions.db"

TARGET = "G3"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = 1

# Ordered codes are treated as numbers so that "more study time" stays ordered.
# G1 and G2 are earlier grades in the same course. G3 must not be added here.
NUMERIC_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2",
]
CATEGORICAL_FEATURES = [
    "sex",
    "address",
    "famsize",
    "schoolsup",
    "famsup",
    "activities",
    "higher",
    "internet",
    "subject",
]

FEATURES_NO_PRIORS = [
    "sex",
    "age",
    "address",
    "famsize",
    "Medu",
    "Fedu",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "activities",
    "higher",
    "internet",
    "absences",
    "subject",
]
FEATURES_WITH_PRIORS = FEATURES_NO_PRIORS + ["G1", "G2"]

GRADE_MIN = 0
GRADE_MAX = 20
PASS_MARK = 10

# Ranges checked against the published UCI file and the form.
AGE_MIN, AGE_MAX = 15, 22
EDU_MIN, EDU_MAX = 0, 4
STUDY_MIN, STUDY_MAX = 1, 4
FAILURES_MIN, FAILURES_MAX = 0, 3
# 75 is the largest absences value in the shipped mathematics file.
ABSENCE_MIN, ABSENCE_MAX = 0, 75

SOURCE_URL = "https://archive.ics.uci.edu/dataset/320/student+performance"
CITATION = (
    "P. Cortez and A. Silva. Using Data Mining to Predict Secondary School "
    "Student Performance. In A. Brito and J. Teixeira (Eds.), Proceedings of "
    "the 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, "
    "Porto, Portugal, April 2008."
)
OVERLAP_NOTE = (
    "The dataset authors report that 382 students appear in both course files. "
    "This app keeps both rows. Each prediction is a course grade, identified "
    "by the course field, not a single combined GPA."
)

SEX_LABELS = {"F": "Female", "M": "Male"}
ADDRESS_LABELS = {"U": "Urban", "R": "Rural"}
FAMSIZE_LABELS = {"LE3": "3 or fewer", "GT3": "More than 3"}
SCHOOL_LABELS = {
    "GP": "Gabriel Pereira (GP)",
    "MS": "Mousinho da Silveira (MS)",
}
YES_NO_LABELS = {"yes": "Yes", "no": "No"}
STUDY_TIME_LABELS = {
    1: "Under 2 hours",
    2: "2 to 5 hours",
    3: "5 to 10 hours",
    4: "More than 10 hours",
}
EDUCATION_LABELS = {
    0: "None",
    1: "Primary (4th grade)",
    2: "5th to 9th grade",
    3: "Secondary",
    4: "Higher education",
}
SUBJECTS = ["Math", "Portuguese"]

FEATURE_LABELS = {
    "school": "School",
    "sex": "Sex",
    "age": "Age",
    "address": "Address type",
    "famsize": "Family size",
    "Medu": "Mother's education",
    "Fedu": "Father's education",
    "studytime": "Weekly study time",
    "failures": "Past class failures",
    "schoolsup": "School support",
    "famsup": "Family support",
    "activities": "Extra activities",
    "higher": "Plans higher education",
    "internet": "Internet at home",
    "absences": "Absences",
    "subject": "Course",
    "G1": "Period grade G1",
    "G2": "Period grade G2",
    "G3": "Final grade G3",
}

CATEGORY_RULES = (
    (16, "Excellent"),
    (14, "Good"),
    (10, "Average"),
)

TRAINING_SETUP = [
    "80/20 train-test split, stratified by course, with random_state 42.",
    "Five shuffled cross-validation folds on the training set only.",
    "The deployed model is the one with the lowest mean cross-validated RMSE.",
    "The test set is scored once after that choice. It is not used to pick the model.",
    "Numeric fields are median-imputed. Linear models also standardize them.",
    "Categorical fields are imputed with the most frequent value and one-hot encoded.",
    "Tree models are not standardized, because a split does not depend on the scale of a number.",
    "G3, the final grade, is never an input. G1 and G2 are used only by the model that expects previous grades.",
]

LIMITATIONS = [
    "The model predicts a final course grade on the 0–20 scale used by two Portuguese secondary schools in the 2005–2006 UCI study. It is not a GPA, and it is not a grade from another country or year.",
    "Training data comes only from Gabriel Pereira and Mousinho da Silveira. Use outside those schools is extrapolation.",
    "G1 and G2 are earlier grades in the same course. When they are available they usually dominate the prediction. The separate model without them is the estimate to use before those grades exist.",
    "The final grade is not an input. Preprocessing is fit inside the training pipeline only, and the model is chosen with cross-validation on the training split.",
    "Some final grades are 0. In the original study, extreme absence patterns can force a zero. Another school may not use that rule.",
    "The form accepts up to 75 absences because that is the highest value in this file. The original codebook mentions 93, so a count above 75 was left out of the form rather than guessed.",
    "Weekday and weekend alcohol use, and parent occupation, are columns in the source file. They are not collected on the form and are not model inputs.",
    "A result is a statistical estimate from about a thousand course records. It is not a judgment of ability and should not be the only basis for placement, discipline, or admissions.",
]

SHORT_LIMITATION = (
    "This estimates a 0–20 final course grade from the UCI Portuguese secondary-school data. "
    "It is not a guarantee and should not be the only basis for an academic decision."
)

ALGORITHM_NOTES = {
    "Linear Regression": (
        "Fits a weighted sum of the encoded features. It is easy to explain and "
        "cannot represent interactions unless they are added as features."
    ),
    "Ridge Regression": (
        "Linear regression with a penalty on large weights. The penalty keeps "
        "the model steadier when encoded features overlap."
    ),
    "Random Forest": (
        "Averages many decision trees. It can represent non-linear patterns and "
        "reports which input fields the trees used most often."
    ),
    "Gradient Boosting": (
        "Builds trees in sequence. Each new tree focuses on the errors left by "
        "the earlier trees. It is often accurate on tables like this one."
    ),
}

# Used by the sample button and by tests. These are inputs, not a stored score.
SAMPLE_FORM = {
    "student_name": "Maria Costa",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Medu": 4,
    "Fedu": 2,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "activities": "yes",
    "higher": "yes",
    "internet": "yes",
    "absences": 4,
    "subject": "Portuguese",
    "use_priors": True,
    "G1": 14,
    "G2": 15,
}


def category_for(score: float) -> str:
    """Map a 0–20 score to a label. 10 is the usual pass mark on this scale."""
    if score >= 16:
        return "Excellent"
    if score >= 14:
        return "Good"
    if score >= 10:
        return "Average"
    return "Needs Improvement"


def category_slug(category: str) -> str:
    return {
        "Excellent": "excellent",
        "Good": "good",
        "Average": "average",
        "Needs Improvement": "needs",
    }[category]
