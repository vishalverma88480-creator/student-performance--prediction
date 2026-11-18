"""Train, compare, and save the grade models.

Run from the project root:

    python -m src.train_model

Two models are saved in one file. One expects the earlier period grades G1 and
G2. The other is for a student whose period grades are not known yet. G3, the
final grade, is never an input to either model.
"""

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split

from src.config import (
    ASSETS_DIR,
    COMPARISON_CHART,
    CV_FOLDS,
    FEATURES_NO_PRIORS,
    FEATURES_WITH_PRIORS,
    MODEL_DIR,
    MODEL_PATH,
    N_JOBS,
    RANDOM_STATE,
    REPORT_PATH,
    TARGET,
    TEST_SIZE,
)
from src.data_loader import build_student_data, profile_dataset
from src.evaluation import grouped_feature_importance, regression_metrics
from src.preprocessing import column_groups, make_pipeline

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2",
}

# Each estimator factory returns a new object so folds do not share one model.
CANDIDATES = [
    {
        "name": "Linear Regression",
        "estimator": LinearRegression,
        "scale": True,
        "params": None,
    },
    {
        "name": "Ridge Regression",
        "estimator": Ridge,
        "scale": True,
        "params": {"model__alpha": [0.1, 1.0, 10.0, 50.0]},
    },
    {
        "name": "Random Forest",
        "estimator": lambda: RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "scale": False,
        "params": {
            "model__max_depth": [8, 16, None],
            "model__min_samples_leaf": [1, 4],
        },
    },
    {
        "name": "Gradient Boosting",
        "estimator": lambda: GradientBoostingRegressor(
            n_estimators=150,
            random_state=RANDOM_STATE,
        ),
        "scale": False,
        "params": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    },
]


def train_and_save() -> dict:
    """Fit both feature sets, save the winner of each, and return the artifact."""
    frame = build_student_data()
    artifact = {
        "target": TARGET,
        "target_scale": "Final course grade from 0 to 20",
        "random_state": RANDOM_STATE,
        "test_size_fraction": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_rule": (
            "Lowest mean cross-validated RMSE on the training set. "
            "The test set is scored only after the model is chosen."
        ),
        "dataset": profile_dataset(frame),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "with_priors": train_feature_set(frame, FEATURES_WITH_PRIORS, "With previous grades"),
        "without_priors": train_feature_set(
            frame, FEATURES_NO_PRIORS, "Without previous grades"
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _save_model(artifact)
    _save_report(artifact)
    _save_comparison_chart(artifact)
    _print_summary(artifact)
    return artifact


def train_feature_set(frame: pd.DataFrame, features: list[str], label: str) -> dict:
    """Compare candidates on one feature list and return the chosen pipeline."""
    if TARGET in features:
        raise ValueError("Refusing to train with the final grade as an input.")

    print(f"\n{label}: {len(features)} input fields", flush=True)
    feature_frame = frame[features].copy()
    target = frame[TARGET].astype(float).copy()
    subjects = frame["subject"].copy()

    # Same random state and stratification for both feature sets, so the holdout
    # rows are the same students-courses. The test rows are not fit or searched.
    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=subjects,
    )
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fitted = [
        _fit_candidate(spec, features, x_train, y_train, cv) for spec in CANDIDATES
    ]
    winner = min(
        fitted,
        key=lambda item: (item["cv"]["rmse"], -item["cv"]["r2"], item["algorithm"]),
    )

    candidates = []
    winner_predictions = None
    winner_pipeline = None
    for item in fitted:
        # Scoring the test set does not refit the pipeline.
        predictions = item["pipeline"].predict(x_test)
        test_metrics = regression_metrics(y_test, predictions)
        selected = item["algorithm"] == winner["algorithm"]
        if selected:
            winner_predictions = predictions
            winner_pipeline = item["pipeline"]
        candidates.append(
            {
                "algorithm": item["algorithm"],
                "selected": selected,
                "params": item["params"],
                "cv_mae": item["cv"]["mae"],
                "cv_mse": item["cv"]["mse"],
                "cv_rmse": item["cv"]["rmse"],
                "cv_r2": item["cv"]["r2"],
                "test_mae": test_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
        print(
            f"  {item['algorithm']}: CV RMSE {item['cv']['rmse']:.3f} | "
            f"test RMSE {test_metrics['rmse']:.3f} | test R² {test_metrics['r2']:.3f}"
            + ("  <- selected" if selected else ""),
            flush=True,
        )

    selected_report = next(item for item in candidates if item["selected"])
    numeric_features, categorical_features = column_groups(features)
    importance = grouped_feature_importance(winner_pipeline, features)
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(winner_predictions, dtype=float)

    return {
        "pipeline": winner_pipeline,
        "algorithm": winner["algorithm"],
        "features": list(features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_params": winner["params"],
        "candidates": candidates,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "encoded_feature_count": int(
            len(winner_pipeline.named_steps["preprocess"].get_feature_names_out())
        ),
        "feature_importance": importance,
        "test_metrics": {
            "mae": selected_report["test_mae"],
            "mse": selected_report["test_mse"],
            "rmse": selected_report["test_rmse"],
            "r2": selected_report["test_r2"],
        },
        "cv_metrics": {
            "mae": selected_report["cv_mae"],
            "mse": selected_report["cv_mse"],
            "rmse": selected_report["cv_rmse"],
            "r2": selected_report["cv_r2"],
        },
        "y_test": [float(value) for value in actual.tolist()],
        "y_pred": [float(value) for value in predicted.tolist()],
        "mean_residual": float(np.mean(actual - predicted)),
    }


def _fit_candidate(spec: dict, features: list[str], x_train, y_train, cv) -> dict:
    """Fit one candidate on the training split. This function never sees X_test."""
    print(f"  fitting {spec['name']}...", flush=True)
    pipeline = make_pipeline(spec["estimator"](), features, spec["scale"])
    if spec["params"]:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            scoring=SCORING,
            refit="rmse",
            cv=cv,
            n_jobs=N_JOBS,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        index = int(search.best_index_)
        results = search.cv_results_
        cv_metrics = {
            "mae": float(-results["mean_test_mae"][index]),
            "mse": float(-results["mean_test_mse"][index]),
            "rmse": float(-results["mean_test_rmse"][index]),
            "r2": float(results["mean_test_r2"][index]),
        }
        return {
            "algorithm": spec["name"],
            "pipeline": search.best_estimator_,
            "params": _clean_params(search.best_params_),
            "cv": cv_metrics,
        }

    scores = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=N_JOBS,
        error_score="raise",
    )
    pipeline.fit(x_train, y_train)
    return {
        "algorithm": spec["name"],
        "pipeline": pipeline,
        "params": {},
        "cv": {
            "mae": float(-np.mean(scores["test_mae"])),
            "mse": float(-np.mean(scores["test_mse"])),
            "rmse": float(-np.mean(scores["test_rmse"])),
            "r2": float(np.mean(scores["test_r2"])),
        },
    }


def _clean_params(params: dict) -> dict:
    cleaned = {}
    for key, value in params.items():
        name = key.split("__", 1)[-1]
        if isinstance(value, (bool, np.bool_)):
            cleaned[name] = bool(value)
        elif value is None:
            cleaned[name] = None
        elif isinstance(value, (int, np.integer)):
            cleaned[name] = int(value)
        elif isinstance(value, (float, np.floating)):
            cleaned[name] = float(value)
        else:
            cleaned[name] = str(value)
    return cleaned


def _save_model(artifact: dict) -> None:
    import joblib

    joblib.dump(artifact, MODEL_PATH, compress=3)


def _save_report(artifact: dict) -> None:
    """Write a readable report. The pipelines themselves stay in the joblib file."""
    report = {
        "trained_at": artifact["trained_at"],
        "selection_rule": artifact["selection_rule"],
        "dataset": artifact["dataset"],
        "library_versions": artifact["library_versions"],
        "random_state": artifact["random_state"],
        "test_size_fraction": artifact["test_size_fraction"],
        "cv_folds": artifact["cv_folds"],
    }
    for key in ("with_priors", "without_priors"):
        bundle = artifact[key]
        report[key] = {
            "algorithm": bundle["algorithm"],
            "features": bundle["features"],
            "best_params": bundle["best_params"],
            "train_size": bundle["train_size"],
            "test_size": bundle["test_size"],
            "encoded_feature_count": bundle["encoded_feature_count"],
            "test_metrics": bundle["test_metrics"],
            "cv_metrics": bundle["cv_metrics"],
            "mean_residual": bundle["mean_residual"],
            "feature_importance": bundle["feature_importance"],
            "candidates": bundle["candidates"],
        }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _save_comparison_chart(artifact: dict) -> None:
    """Static chart for the model page. The app's interactive charts use Plotly."""
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ASSETS_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    rows = []
    for key, label in (
        ("with_priors", "With previous grades"),
        ("without_priors", "Without previous grades"),
    ):
        for candidate in artifact[key]["candidates"]:
            rows.append(
                {
                    "Model": candidate["algorithm"],
                    "Feature set": label,
                    "CV RMSE": candidate["cv_rmse"],
                }
            )
    plot_frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for axis, label in zip(axes, ["With previous grades", "Without previous grades"]):
        part = plot_frame[plot_frame["Feature set"] == label]
        sns.barplot(
            data=part,
            x="CV RMSE",
            y="Model",
            color="#312E81",
            ax=axis,
        )
        axis.set_title(label)
        axis.set_xlabel("Cross-validated RMSE (lower is better)")
        axis.set_ylabel("")
    fig.suptitle("Model selection scores on the training folds", fontsize=14)
    fig.tight_layout()
    fig.savefig(COMPARISON_CHART, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _print_summary(artifact: dict) -> None:
    print("\nSaved", MODEL_PATH, flush=True)
    print("Report", REPORT_PATH, flush=True)
    for key, label in (
        ("with_priors", "With previous grades"),
        ("without_priors", "Without previous grades"),
    ):
        bundle = artifact[key]
        metrics = bundle["test_metrics"]
        print(
            f"{label}: {bundle['algorithm']} | "
            f"test MAE {metrics['mae']:.3f} | RMSE {metrics['rmse']:.3f} | "
            f"R² {metrics['r2']:.3f}",
            flush=True,
        )


def main() -> None:
    train_and_save()


if __name__ == "__main__":
    main()
