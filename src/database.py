"""SQLite storage for predictions.

Each call opens its own connection, so a Streamlit rerun does not keep a stale
handle. Records stay in data/predictions.db until they are deleted.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import sqlite3

from src.config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    student_name TEXT NOT NULL,
    sex TEXT NOT NULL,
    age INTEGER NOT NULL,
    address TEXT NOT NULL,
    famsize TEXT NOT NULL,
    medu INTEGER NOT NULL,
    fedu INTEGER NOT NULL,
    studytime INTEGER NOT NULL,
    failures INTEGER NOT NULL,
    schoolsup TEXT NOT NULL,
    famsup TEXT NOT NULL,
    activities TEXT NOT NULL,
    higher TEXT NOT NULL,
    internet TEXT NOT NULL,
    absences INTEGER NOT NULL,
    subject TEXT NOT NULL,
    g1 REAL,
    g2 REAL,
    used_previous_grades INTEGER NOT NULL,
    predicted_score REAL NOT NULL,
    predicted_percent REAL NOT NULL,
    category TEXT NOT NULL,
    model_name TEXT NOT NULL,
    recommendations TEXT NOT NULL,
    study_suggestions TEXT NOT NULL
)
"""


def database_path():
    override = os.environ.get("SPP_DATABASE_PATH")
    if override:
        return os.fspath(override)
    return DATABASE_PATH


@contextmanager
def get_connection():
    path = database_path()
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Create the table if needed, and drop the retired school column."""
    with get_connection() as connection:
        connection.execute(SCHEMA)
        columns = [row[1] for row in connection.execute("PRAGMA table_info(predictions)")]
        if "school" not in columns:
            return
        connection.execute("ALTER TABLE predictions RENAME TO predictions_old")
        connection.execute(SCHEMA)
        connection.execute(
            """
            INSERT INTO predictions (
                id, created_at, student_name, sex, age, address, famsize,
                medu, fedu, studytime, failures, schoolsup, famsup, activities,
                higher, internet, absences, subject, g1, g2, used_previous_grades,
                predicted_score, predicted_percent, category, model_name,
                recommendations, study_suggestions
            )
            SELECT
                id, created_at, student_name, sex, age, address, famsize,
                medu, fedu, studytime, failures, schoolsup, famsup, activities,
                higher, internet, absences, subject, g1, g2, used_previous_grades,
                predicted_score, predicted_percent, category, model_name,
                recommendations, study_suggestions
            FROM predictions_old
            """
        )
        connection.execute("DROP TABLE predictions_old")


def save_prediction(result: dict) -> int:
    """Store one explained prediction and return its new id."""
    init_db()
    inputs = result["inputs"]
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                created_at, student_name, sex, age, address, famsize,
                medu, fedu, studytime, failures, schoolsup, famsup, activities,
                higher, internet, absences, subject, g1, g2, used_previous_grades,
                predicted_score, predicted_percent, category, model_name,
                recommendations, study_suggestions
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                created_at,
                inputs["student_name"],
                inputs["sex"],
                inputs["age"],
                inputs["address"],
                inputs["famsize"],
                inputs["Medu"],
                inputs["Fedu"],
                inputs["studytime"],
                inputs["failures"],
                inputs["schoolsup"],
                inputs["famsup"],
                inputs["activities"],
                inputs["higher"],
                inputs["internet"],
                inputs["absences"],
                inputs["subject"],
                inputs["G1"],
                inputs["G2"],
                1 if result["used_previous_grades"] else 0,
                result["predicted_score"],
                result["predicted_percent"],
                result["category"],
                result["model_name"],
                json.dumps(result["recommendations"], ensure_ascii=False),
                json.dumps(result["study_suggestions"], ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid)


def list_predictions(
    search: str = "",
    category: str | None = None,
    sex: str | None = None,
    subject: str | None = None,
) -> list[dict]:
    """Return saved rows, newest first, with optional name search and filters."""
    init_db()
    query = "SELECT * FROM predictions WHERE 1=1"
    params: list = []
    if search and search.strip():
        query += " AND student_name LIKE ? ESCAPE '\\'"
        params.append(_like(search.strip()))
    if category:
        query += " AND category = ?"
        params.append(category)
    if sex:
        query += " AND sex = ?"
        params.append(sex)
    if subject:
        query += " AND subject = ?"
        params.append(subject)
    query += " ORDER BY datetime(created_at) DESC, id DESC"
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [_hydrate(row) for row in rows]


def get_prediction(record_id: int) -> dict | None:
    init_db()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (int(record_id),),
        ).fetchone()
    if row is None:
        return None
    return _hydrate(row)


def delete_prediction(record_id: int) -> bool:
    """Delete one row. Returns True when a row was removed."""
    init_db()
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM predictions WHERE id = ?",
            (int(record_id),),
        )
        return cursor.rowcount > 0


def summarize_predictions() -> dict:
    """Counts and score range for the dashboard cards."""
    init_db()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS n,
                AVG(predicted_score) AS average,
                MAX(predicted_score) AS highest,
                MIN(predicted_score) AS lowest
            FROM predictions
            """
        ).fetchone()
    return {
        "count": int(row["n"]),
        "average": None if row["average"] is None else float(row["average"]),
        "highest": None if row["highest"] is None else float(row["highest"]),
        "lowest": None if row["lowest"] is None else float(row["lowest"]),
    }


def _hydrate(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["recommendations"] = json.loads(item["recommendations"])
    item["study_suggestions"] = json.loads(item["study_suggestions"])
    item["used_previous_grades"] = bool(item["used_previous_grades"])
    return item


def _like(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"
