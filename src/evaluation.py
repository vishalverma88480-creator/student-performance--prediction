"""Metrics, feature importance, and the plain-language notes shown in the app."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_HELP = {
    "mae": "Mean absolute error. The average number of grade points the prediction is off by. Lower is better.",
    "mse": "Mean squared error. Like the absolute error, but large misses count more because the gap is squared. Lower is better.",
    "rmse": "Root mean squared error. The square root of MSE, so it is back in grade points. Lower is better.",
    "r2": "R². How much of the grade variation the model explains. 1 is a perfect fit, 0 matches predicting the average, and below 0 is worse than the average.",
}


def regression_metrics(y_true, y_pred) -> dict:
    """Return MAE, MSE, RMSE, and R² for one set of predictions.

    RMSE is computed as the square root of MSE so the two numbers always agree.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def grouped_feature_importance(pipeline, features: list[str]) -> list[dict]:
    """Collapse one-hot columns back to the original form fields.

    Importance is scaled to sum to 1. For a linear model it is the absolute
    coefficient. For a tree model it is the model's built-in importance.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    encoded_names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        return []

    if len(encoded_names) != len(values):
        return []

    totals = {feature: 0.0 for feature in features}
    # Longer names first so "schoolsup" is matched before "school".
    ordered = sorted(features, key=len, reverse=True)
    for encoded_name, value in zip(encoded_names, values):
        raw_name = encoded_name.split("__", 1)[-1]
        matched = None
        for feature in ordered:
            if raw_name == feature or raw_name.startswith(feature + "_"):
                matched = feature
                break
        if matched is not None:
            totals[matched] += float(value)

    total = sum(totals.values())
    if total <= 0:
        return []

    rows = [
        {"feature": feature, "importance": totals[feature] / total}
        for feature in features
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows


def comparison_frame(artifact: dict) -> pd.DataFrame:
    """One row per trained candidate, for the analytics and model tables."""
    feature_set_labels = {
        "with_priors": "With previous grades",
        "without_priors": "Without previous grades",
    }
    rows = []
    for key, label in feature_set_labels.items():
        for candidate in artifact[key]["candidates"]:
            rows.append(
                {
                    "Feature set": label,
                    "Model": candidate["algorithm"],
                    "Selected": "Yes" if candidate["selected"] else "No",
                    "CV RMSE": round(candidate["cv_rmse"], 3),
                    "CV MAE": round(candidate["cv_mae"], 3),
                    "CV R²": round(candidate["cv_r2"], 3),
                    "Test MAE": round(candidate["test_mae"], 3),
                    "Test MSE": round(candidate["test_mse"], 3),
                    "Test RMSE": round(candidate["test_rmse"], 3),
                    "Test R²": round(candidate["test_r2"], 3),
                    "Parameters": format_params(candidate["params"]),
                }
            )
    return pd.DataFrame(rows)


def selection_note(candidates: list[dict]) -> str:
    """Explain which score chose the model, in language a demo can use."""
    selected = next(item for item in candidates if item["selected"])
    best_test = min(item["test_rmse"] for item in candidates)
    if selected["test_rmse"] > best_test + 1e-9:
        names = [
            item["algorithm"]
            for item in candidates
            if abs(item["test_rmse"] - best_test) <= 1e-9
        ]
        return (
            f"The lowest test RMSE on this split belongs to {', '.join(names)}. "
            f"{selected['algorithm']} is still the deployed model because selection "
            "used cross-validation on the training set only."
        )
    return (
        f"{selected['algorithm']} had the lowest cross-validated RMSE "
        f"({selected['cv_rmse']:.3f}) and also the lowest test RMSE on this split "
        f"({selected['test_rmse']:.3f}). The test score was not used to choose it."
    )


def format_params(params: dict) -> str:
    if not params:
        return "defaults"
    return ", ".join(f"{key}={value}" for key, value in params.items())


def input_columns_from_pipeline(pipeline) -> list[str]:
    """Return the original columns the preprocessor was told to read."""
    columns = []
    preprocessor = pipeline.named_steps["preprocess"]
    transformers = getattr(preprocessor, "transformers_", preprocessor.transformers)
    for name, transformer, transformer_columns in transformers:
        if name == "remainder" or transformer in {"drop", "passthrough"}:
            continue
        if isinstance(transformer_columns, str):
            continue
        columns.extend(list(transformer_columns))
    return columns
