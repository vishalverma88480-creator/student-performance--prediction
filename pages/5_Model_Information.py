"""What was trained, how it was scored, and what it should not be used for."""

import streamlit as st

from src.charts import actual_vs_predicted, importance_chart
from src.config import (
    ALGORITHM_NOTES,
    COMPARISON_CHART,
    FEATURE_LABELS,
    LIMITATIONS,
    SCHOOL_LABELS,
    TRAINING_SETUP,
)
from src.evaluation import METRIC_HELP, comparison_frame, selection_note
from src.ui import ensure_app_ready, format_timestamp, kpi_cards, page_header, show_chart, show_table

page_header(
    "Model information",
    "Dataset, training setup, the model that was selected, and the limits of the estimate.",
)
frame, artifact = ensure_app_ready()
profile = artifact["dataset"]

if len(frame) != profile["rows"]:
    st.warning(
        "The dataset file has changed since this model was trained. "
        "Retrain so the metrics match the current file."
    )

schools = ", ".join(SCHOOL_LABELS.get(code, code) for code in profile["schools"])
kpi_cards(
    [
        {"label": "Course records", "value": f"{profile['rows']:,}", "note": "Rows used for training"},
        {"label": "Mathematics", "value": f"{profile['math_rows']:,}", "note": "student-mat.csv"},
        {"label": "Portuguese", "value": f"{profile['portuguese_rows']:,}", "note": "student-por.csv"},
        {
            "label": "Final grades of 0",
            "value": f"{profile['zero_grades']:,}",
            "note": "Included, not dropped",
        },
    ]
)
st.caption(
    f"Schools: {schools}. Ages {profile['age_min']}–{profile['age_max']}. "
    f"Highest absence count in the file: {profile['absence_max']}. "
    f"Missing cells: {profile['missing_cells']}. "
    f"Average final grade: {profile['target_mean']:.2f}. "
    f"Trained {format_timestamp(artifact['trained_at'])}."
)
st.markdown(f"[UCI Student Performance dataset]({profile['source_url']})")
st.write(profile["citation"])
st.write(profile["overlap_note"])

st.subheader("Training setup")
for step in TRAINING_SETUP:
    st.write(f"• {step}")
versions = artifact["library_versions"]
st.caption(
    "Library versions at training time: "
    f"scikit-learn {versions['scikit-learn']}, "
    f"pandas {versions['pandas']}, numpy {versions['numpy']}."
)

feature_tabs = st.tabs(["With previous grades", "Without previous grades"])
for tab, key in zip(feature_tabs, ("with_priors", "without_priors")):
    bundle = artifact[key]
    metrics = bundle["test_metrics"]
    cv_metrics = bundle["cv_metrics"]
    with tab:
        st.markdown(f"### {bundle['algorithm']}")
        st.write(ALGORITHM_NOTES.get(bundle["algorithm"], ""))
        st.info(selection_note(bundle["candidates"]))
        kpi_cards(
            [
                {"label": "Training rows", "value": f"{bundle['train_size']:,}", "note": "Fit and cross-validated"},
                {"label": "Test rows", "value": f"{bundle['test_size']:,}", "note": "Scored once"},
                {
                    "label": "Input features",
                    "value": str(len(bundle["features"])),
                    "note": "Fields on the form",
                },
                {
                    "label": "Encoded columns",
                    "value": str(bundle["encoded_feature_count"]),
                    "note": "After one-hot encoding",
                },
            ]
        )
        st.markdown("**Test set.** Scored once, after the model was chosen.")
        kpi_cards(
            [
                {"label": "MAE", "value": f"{metrics['mae']:.3f}", "note": "Grade points"},
                {"label": "MSE", "value": f"{metrics['mse']:.3f}", "note": "Squared grade points"},
                {"label": "RMSE", "value": f"{metrics['rmse']:.3f}", "note": "Grade points"},
                {"label": "R²", "value": f"{metrics['r2']:.3f}", "note": "Variation explained"},
            ]
        )
        st.markdown("**Cross-validation.** These training-fold scores chose the model.")
        kpi_cards(
            [
                {"label": "CV MAE", "value": f"{cv_metrics['mae']:.3f}", "note": "Mean across folds"},
                {"label": "CV MSE", "value": f"{cv_metrics['mse']:.3f}", "note": "Mean across folds"},
                {"label": "CV RMSE", "value": f"{cv_metrics['rmse']:.3f}", "note": "Selection score"},
                {"label": "CV R²", "value": f"{cv_metrics['r2']:.3f}", "note": "Mean across folds"},
            ]
        )
        for code, explanation in METRIC_HELP.items():
            st.caption(explanation)
        st.caption(
            f"Mean residual on the test set (actual minus predicted): {bundle['mean_residual']:.3f}."
        )
        show_chart(
            actual_vs_predicted(bundle["y_test"], bundle["y_pred"]),
            key=f"actual-{key}",
        )
        st.caption(
            "Held-out test rows, where the actual final grade is known. "
            "The dashed line is a perfect prediction. This is not the saved-prediction list."
        )
        parameter_text = ", ".join(
            f"{name}={value}" for name, value in bundle["best_params"].items()
        ) or "defaults"
        st.write(f"Selected parameters: {parameter_text}.")
        if bundle["feature_importance"]:
            show_chart(importance_chart(bundle["feature_importance"]), key=f"importance-{key}")
            st.caption(
                "Tree models show built-in importance. Linear models show absolute standardized "
                "coefficients. One-hot columns are added back together and scaled to 100%."
            )
        else:
            st.info("This algorithm does not expose feature importance.")
        named = ", ".join(FEATURE_LABELS.get(feature, feature) for feature in bundle["features"])
        st.caption(f"Inputs: {named}.")

st.subheader("All compared models")
show_table(comparison_frame(artifact))

st.subheader("Limitations")
items = "".join(f"<li>{item}</li>" for item in LIMITATIONS)
st.markdown(f"<article class='limits-card'><ol class='limits'>{items}</ol></article>", unsafe_allow_html=True)

with st.expander("Static training chart (Matplotlib and Seaborn)"):
    if COMPARISON_CHART.exists():
        st.image(str(COMPARISON_CHART), width="stretch")
        st.caption(
            "The training script writes this figure. The interactive charts use the same results through Plotly."
        )
    else:
        st.info("Train the model to create the static comparison chart.")

st.subheader("Retrain")
st.write(
    "Retraining replaces models/best_model.joblib and the metrics on this page. "
    "It does not delete saved student records."
)
if st.button("Retrain and replace the saved model", type="primary"):
    from src.train_model import train_and_save

    with st.spinner("Training and comparing models. This usually takes one to two minutes."):
        try:
            train_and_save()
        except Exception as exc:
            st.error("Training failed.")
            st.exception(exc)
        else:
            st.success("Training finished. The new model is saved.")
            st.rerun()
