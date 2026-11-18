"""Build the scikit-learn preprocessing steps used inside every model pipeline.

Nothing in this file fits the full dataset. The pipeline fits these steps on
training rows only, which is what prevents the imputer and encoder from seeing
the test set.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def column_groups(features: list[str]) -> tuple[list[str], list[str]]:
    """Split a feature list into numeric and categorical columns, in list order."""
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    unknown = [
        feature
        for feature in features
        if feature not in numeric and feature not in categorical
    ]
    if unknown:
        raise ValueError(
            "These features are not classified as numeric or categorical: "
            + ", ".join(unknown)
        )
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale: bool,
) -> ColumnTransformer:
    """Impute missing values and encode categories.

    Scaling is on for linear models so a wide column such as absences does not
    outweigh a 0–4 code just because its numbers are larger. Tree models leave
    the numbers unscaled because a split does not care about units.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", Pipeline(categorical_steps), categorical_features),
        ],
        remainder="drop",
    )


def make_pipeline(estimator, features: list[str], scale: bool) -> Pipeline:
    numeric_features, categorical_features = column_groups(features)
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(numeric_features, categorical_features, scale),
            ),
            ("model", estimator),
        ]
    )
