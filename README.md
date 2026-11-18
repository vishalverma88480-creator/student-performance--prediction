# Student Performance Prediction

An end-to-end machine learning web application that predicts a student's final course grade (0–20 scale) from the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance), generates actionable recommendations, and stores prediction history in SQLite.

**Live App:** [https://student-performance-vishal.streamlit.app](https://student-performance-vishal.streamlit.app)

---

## Key Features
* **Dual-Pipeline Prediction:** Automatically selects between a **Gradient Boosting** model (when prior period grades `G1` and `G2` are available) and a **Random Forest** model (when predicting without prior grades).
* **Interactive Analytics Dashboard:** Tracks total saved predictions, average/highest/lowest scores, grade distribution, and feature relationships using Plotly.
* **Persistent Student Records:** Full CRUD support (save, search, filter, and delete) backed by a local SQLite database.
* **Model Transparency:** Built-in model comparison page displaying cross-validation metrics, held-out test scores, and feature importance.

---

## Model Performance
Evaluated on an 80/20 stratified train-test split (835 training / 209 test records) across 1,044 combined Mathematics and Portuguese course records:

| Feature Set | Best Model | Held-Out MAE | Held-Out RMSE | Held-Out $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **With Prior Grades (`G1`, `G2`)** | Gradient Boosting | 0.812 | 1.436 | **0.845** |
| **Without Prior Grades** | Random Forest | 2.264 | 3.117 | **0.267** |

---

## Tech Stack
* **Frontend & App Framework:** Python 3.13, Streamlit
* **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
* **Visualization & Storage:** Plotly, Matplotlib, Seaborn, SQLite

---

## Local Setup

```bash
# Clone the repository
git clone [https://github.com/vishalverma88480-creator/student-performance--prediction.git](https://github.com/vishalverma88480-creator/student-performance--prediction.git)
cd student-performance--prediction

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the application
streamlit run app.py
