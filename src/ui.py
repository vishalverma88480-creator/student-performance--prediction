"""Shared layout for the Streamlit pages."""

import html

import pandas as pd
import streamlit as st

from src.charts import score_gauge
from src.config import (
    ADDRESS_LABELS,
    ASSETS_DIR,
    EDUCATION_LABELS,
    FAMSIZE_LABELS,
    FEATURE_LABELS,
    SCHOOL_LABELS,
    SEX_LABELS,
    STUDY_TIME_LABELS,
    YES_NO_LABELS,
    category_slug,
)
from src.data_loader import DataError, load_student_data
from src.database import list_predictions, summarize_predictions
from src.predict import ModelNotFoundError, load_artifact


def inject_css() -> None:
    styles = (ASSETS_DIR / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)


def render_sidebar() -> None:
    """Brand and model status. Navigation itself is drawn by st.navigation."""
    st.sidebar.markdown(
        """
        <div class="brand">
          <div class="brand-mark">SP</div>
          <div>
            <strong>Performance</strong>
            <em>Student predictions</em>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        saved = f"{summarize_predictions()['count']:,}"
    except Exception:
        saved = "0"
    try:
        artifact = load_artifact()
        model_line = (
            f"With grades: {artifact['with_priors']['algorithm']}<br>"
            f"Without grades: {artifact['without_priors']['algorithm']}"
        )
    except ModelNotFoundError:
        model_line = "Model not trained yet"
    st.sidebar.markdown(
        f"""
        <div class="side-card">
          <span>Saved predictions</span>
          <strong>{saved}</strong>
          <p>{model_line}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


_SIDEBAR_SCRIPT = """
<script>
  function sppOpenSidebar() {
    const side = document.querySelector('[data-testid="stSidebar"]');
    if (!side || side.getAttribute("data-collapsed") !== "true") return;
    const opener = document.querySelector(
      'button[aria-label="Expand sidebar"], button[title="Expand sidebar"], [data-testid="stSidebarCollapsedControl"] button, [data-testid="collapsedControl"] button'
    );
    if (opener) opener.click();
  }
  const params = new URLSearchParams(window.location.search);
  if (params.get("open_sidebar") === "1") {
    sppOpenSidebar();
    setTimeout(sppOpenSidebar, 150);
    setTimeout(sppOpenSidebar, 500);
    params.delete("open_sidebar");
    const clean = window.location.pathname + (params.toString() ? "?" + params.toString() : "");
    window.history.replaceState({}, "", clean);
  }
  const home = document.getElementById("spp-home");
  if (home) {
    home.addEventListener("click", function () {
      sppOpenSidebar();
      const next = new URL("/", window.location.origin);
      next.searchParams.set("open_sidebar", "1");
      window.location.assign(next.toString());
    });
  }
  const menu = document.getElementById("spp-menu");
  if (menu) menu.addEventListener("click", sppOpenSidebar);
</script>
"""


def page_header(title: str, subtitle: str, *, show_back: bool = True) -> None:
    """Page title. Home opens the sidebar and returns to the dashboard."""
    if show_back:
        action = """
        <button type="button" class="back-home" id="spp-home">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 18l-6-6 6-6"/></svg>
          Home
        </button>
        """
    else:
        action = """
        <button type="button" class="back-home" id="spp-menu">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
          Menu
        </button>
        """
    st.html(
        f"""
        <header class="page-header">
          <div class="page-header-top">
            <div>
              <p class="eyebrow">Student Performance Prediction</p>
              <h1>{html.escape(title)}</h1>
            </div>
            {action}
          </div>
          <p class="subtitle">{html.escape(subtitle)}</p>
        </header>
        {_SIDEBAR_SCRIPT}
        """,
        unsafe_allow_javascript=True,
    )


def section_label(text: str) -> None:
    st.markdown(
        f'<p class="section-label">{html.escape(text)}</p>',
        unsafe_allow_html=True,
    )


def kpi_cards(cards: list[dict]) -> None:
    columns = 3 if len(cards) == 3 else 4
    parts = [f'<div class="kpi-grid cols-{columns}">']
    for card in cards:
        unit = card.get("unit", "")
        unit_html = f" <span>{html.escape(unit)}</span>" if unit else ""
        parts.append(
            "<article class='kpi-card'>"
            f"<p class='kpi-label'>{html.escape(card['label'])}</p>"
            f"<p class='kpi-value'>{html.escape(str(card['value']))}{unit_html}</p>"
            f"<p class='kpi-note'>{html.escape(card.get('note', ''))}</p>"
            "</article>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def show_chart(fig, key: str | None = None) -> None:
    # theme=None keeps the colors set in src/charts.py.
    st.plotly_chart(fig, width="stretch", theme=None, key=key)


def show_table(frame: pd.DataFrame) -> None:
    st.dataframe(frame, width="stretch", hide_index=True)


def ensure_app_ready():
    """Load the dataset and model, or stop with a way to train."""
    inject_css()
    try:
        frame = load_student_data()
    except DataError as exc:
        st.error("The dataset could not be loaded.")
        st.write(str(exc))
        st.stop()
    try:
        artifact = load_artifact()
    except ModelNotFoundError as exc:
        st.warning(str(exc))
        if st.button("Train the model now", type="primary", key="train_missing_model"):
            from src.train_model import train_and_save

            with st.spinner("Training and comparing models. This usually takes one to two minutes."):
                try:
                    train_and_save()
                except Exception as train_error:
                    st.error("Training failed.")
                    st.exception(train_error)
                    st.stop()
            st.rerun()
        st.stop()
    return frame, artifact


def format_timestamp(value: str) -> str:
    text = str(value).replace("T", " ")
    if text.endswith("+00:00"):
        text = text[: -len("+00:00")] + " UTC"
    return text


def display_feature_value(feature: str, value) -> str:
    if value is None:
        return "Not used"
    if feature == "sex":
        return SEX_LABELS.get(value, str(value))
    if feature == "address":
        return ADDRESS_LABELS.get(value, str(value))
    if feature == "famsize":
        return FAMSIZE_LABELS.get(value, str(value))
    if feature == "school":
        return SCHOOL_LABELS.get(value, str(value))
    if feature in {"Medu", "Fedu"}:
        return EDUCATION_LABELS.get(int(value), str(value))
    if feature == "studytime":
        return STUDY_TIME_LABELS.get(int(value), str(value))
    if feature in {"schoolsup", "famsup", "activities", "higher", "internet"}:
        return YES_NO_LABELS.get(value, str(value))
    return str(value)


def predictions_display_frame(rows: list[dict] | None = None) -> pd.DataFrame:
    if rows is None:
        rows = list_predictions()
    records = []
    for row in rows:
        if row["used_previous_grades"] and row["g1"] is not None and row["g2"] is not None:
            grades = f"G1 {row['g1']:.0f} · G2 {row['g2']:.0f}"
        else:
            grades = "Not used"
        records.append(
            {
                "ID": row["id"],
                "Saved": format_timestamp(row["created_at"]),
                "Student": row["student_name"],
                "Course": row["subject"],
                "Sex": SEX_LABELS.get(row["sex"], row["sex"]),
                "Age": row["age"],
                "Study time": STUDY_TIME_LABELS.get(int(row["studytime"]), row["studytime"]),
                "Absences": row["absences"],
                "Previous grades": grades,
                "Predicted score": row["predicted_score"],
                "Category": row["category"],
                "Model": row["model_name"],
            }
        )
    return pd.DataFrame(records)


def render_result(result: dict, record_id: int | None) -> None:
    """Score, gauge, advice, and the exact fields that were sent to the model."""
    slug = category_slug(result["category"])
    if result["predicted_score"] >= 10:
        pass_line = "At or above 10, the usual pass mark on this 0–20 scale."
    else:
        pass_line = "Below 10, the usual pass mark on this 0–20 scale."
    if result["used_previous_grades"]:
        model_line = f"{result['model_name']} · previous grades included"
    else:
        model_line = f"{result['model_name']} · previous grades were not available"

    with st.container(border=True):
        section_label("Result")
        score_col, tier_col, link_col = st.columns(3)
        score_col.metric("Predicted score", f"{result['predicted_score']:.2f}", "out of 20")
        tier_col.metric("Performance tier", result["category"])
        if record_id is not None:
            link_col.page_link(
                "pages/3_Student_Records.py",
                label=f"View record #{record_id}",
                icon=":material/groups:",
            )
        else:
            link_col.caption("This score was calculated but not saved.")

    score_col, gauge_col = st.columns([1.15, 0.85])
    with score_col:
        st.markdown(
            f"""
            <article class="score-card">
              <p class="score-kicker">Predicted final grade</p>
              <p class="score-value">{result['predicted_score']:.2f}</p>
              <p class="score-unit">out of 20 · {result['predicted_percent']:.1f}%</p>
              <span class="badge {slug}">{html.escape(result['category'])}</span>
              <p class="score-pass">{html.escape(pass_line)}</p>
              <p class="score-model">{html.escape(model_line)}</p>
              <ul class="legend">
                <li><i class="excellent"></i>Excellent · 16 to 20</li>
                <li><i class="good"></i>Good · 14 to 16</li>
                <li><i class="average"></i>Average · 10 to 14</li>
                <li><i class="needs"></i>Needs improvement · below 10</li>
              </ul>
            </article>
            """,
            unsafe_allow_html=True,
        )
    with gauge_col:
        show_chart(score_gauge(result["predicted_score"], result["category"]), key="score-gauge")

    if result["was_clipped"]:
        st.warning(
            f"The raw model output was {result['raw_score']:.2f}, outside 0–20, "
            "so the reported score is clipped to the grade scale."
        )

    st.write("")
    section_label("Recommendations")
    cards = ["<div class='rec-grid'>"]
    for item in result["recommendations"]:
        priority = item["priority"].lower()
        cards.append(
            "<article class='rec'>"
            f"<span class='pill {html.escape(priority)}'>{html.escape(item['priority'])} priority</span>"
            f"<h3>{html.escape(item['title'])}</h3>"
            f"<p>{html.escape(item['detail'])}</p>"
            "</article>"
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)

    st.write("")
    section_label("Study plan")
    items = "".join(f"<li>{html.escape(tip)}</li>" for tip in result["study_suggestions"])
    st.markdown(f"<article class='plan-card'><ol class='plan'>{items}</ol></article>", unsafe_allow_html=True)

    st.write("")
    st.markdown(
        f"<div class='callout'>{html.escape(result['limitation'])}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("How this prediction was made"):
        st.write(
            "The final grade, G3, was not sent to the model. "
            "A saved scikit-learn pipeline imputed, encoded, and scored only the fields below. "
            "The pipeline was fit on the training split, not on this student."
        )
        rows = [
            {
                "Field": FEATURE_LABELS.get(feature, feature),
                "Value sent to the model": display_feature_value(feature, result["inputs"].get(feature)),
            }
            for feature in result["features_used"]
        ]
        show_table(pd.DataFrame(rows))
        st.caption("The student name is stored in the database only. It is not a model feature.")
