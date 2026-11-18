"""Train the real dataset once, then check predictions, validation, and storage."""

import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.charts import (
    absences_chart,
    actual_vs_predicted,
    category_donut,
    education_chart,
    grade_distribution,
    importance_chart,
    model_comparison_chart,
    score_gauge,
    study_time_chart,
)
from src.config import (
    COMPARISON_CHART,
    FEATURES_NO_PRIORS,
    FEATURES_WITH_PRIORS,
    MODEL_PATH,
    REPORT_PATH,
    TARGET,
)
from src.data_loader import build_student_data, load_student_data
from src.database import delete_prediction, get_prediction, list_predictions, save_prediction
from src.evaluation import input_columns_from_pipeline, selection_note
from src.predict import (
    PredictionInputError,
    load_artifact,
    model_frame,
    predict_and_explain,
    sample_form,
    validate_form,
)
from src.recommendations import build_recommendations, study_suggestions
from src.train_model import train_and_save


class TestDataset(unittest.TestCase):
    def test_builds_the_uci_course_files(self):
        frame = build_student_data()
        self.assertEqual(len(frame), 1044)
        self.assertEqual(int((frame["subject"] == "Math").sum()), 395)
        self.assertEqual(int((frame["subject"] == "Portuguese").sum()), 649)
        self.assertEqual(int(frame.isna().sum().sum()), 0)
        self.assertNotIn(TARGET, FEATURES_WITH_PRIORS)
        self.assertNotIn("G1", FEATURES_NO_PRIORS)
        self.assertGreater(float(frame[TARGET].mean()), 0)
        reloaded = load_student_data()
        self.assertEqual(len(reloaded), 1044)


class TestRecommendations(unittest.TestCase):
    def test_advice_names_the_submitted_risks(self):
        form = sample_form(
            studytime=1,
            failures=2,
            absences=20,
            use_priors=False,
            internet="no",
            schoolsup="no",
            address="R",
            Medu=1,
            Fedu=0,
            higher="yes",
        )
        stats = {"absences_median": 2, "absences_p75": 6}
        advice = build_recommendations(form, 8.0, "Needs Improvement", stats)
        text = " ".join(item["detail"] for item in advice)
        self.assertIn("20", text)
        self.assertIn("under 2 hours", text)
        self.assertGreaterEqual(len(advice), 3)
        self.assertLessEqual(len(advice), 6)
        order = {"High": 0, "Medium": 1, "Low": 2}
        priorities = [item["priority"] for item in advice]
        self.assertEqual(priorities, sorted(priorities, key=order.get))

    def test_study_plan_mentions_the_failure_count(self):
        tips = study_suggestions(sample_form(failures=2, studytime=1), 8)
        self.assertTrue(any("2" in tip and "failure" in tip for tip in tips))
        self.assertGreaterEqual(len(tips), 3)


class TestValidation(unittest.TestCase):
    def test_rejects_out_of_range_and_incomplete_inputs(self):
        with self.assertRaises(PredictionInputError) as caught:
            validate_form(sample_form(age=12))
        self.assertTrue(any("Age" in error for error in caught.exception.errors))

        with self.assertRaises(PredictionInputError) as caught:
            validate_form(sample_form(absences=90))
        self.assertTrue(any("Absences" in error for error in caught.exception.errors))

        with self.assertRaises(PredictionInputError) as caught:
            validate_form(sample_form(failures=9))
        self.assertTrue(any("failures" in error.lower() for error in caught.exception.errors))

        with self.assertRaises(PredictionInputError) as caught:
            validate_form(sample_form(student_name=""))
        self.assertTrue(any("name" in error.lower() for error in caught.exception.errors))

        with self.assertRaises(PredictionInputError):
            validate_form(sample_form(student_name="@@@"))

        with self.assertRaises(PredictionInputError) as caught:
            validate_form(sample_form(use_priors=True, G2=None))
        self.assertTrue(any("G2" in error for error in caught.exception.errors))

        with self.assertRaises(PredictionInputError) as caught:
            validate_form(sample_form(G1=25))
        self.assertTrue(any("G1" in error for error in caught.exception.errors))

        cleaned = validate_form(sample_form(student_name="José Almeida"))
        self.assertEqual(cleaned["student_name"], "José Almeida")

    def test_model_frame_never_includes_the_final_grade(self):
        cleaned = validate_form(sample_form())
        cleaned["G3"] = 20
        frame = model_frame(cleaned, FEATURES_WITH_PRIORS)
        self.assertNotIn("G3", list(frame.columns))
        self.assertEqual(list(frame.columns), FEATURES_WITH_PRIORS)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["SPP_DATABASE_PATH"] = str(Path(self.tmp.name) / "predictions.db")

    def tearDown(self):
        os.environ.pop("SPP_DATABASE_PATH", None)
        self.tmp.cleanup()

    def test_records_survive_a_new_connection_and_can_be_filtered(self):
        cleaned = validate_form(sample_form(student_name="José Almeida"))
        result = {
            "inputs": cleaned,
            "used_previous_grades": True,
            "predicted_score": 14.2,
            "predicted_percent": 71.0,
            "category": "Good",
            "model_name": "Test Model",
            "recommendations": [
                {"priority": "Low", "title": "Keep going", "detail": "Detail"}
            ],
            "study_suggestions": ["Study the missed questions."],
        }
        record_id = save_prediction(result)

        connection = sqlite3.connect(os.environ["SPP_DATABASE_PATH"])
        stored = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        connection.close()
        self.assertEqual(stored, 1)

        loaded = get_prediction(record_id)
        self.assertEqual(loaded["student_name"], "José Almeida")
        self.assertEqual(loaded["recommendations"][0]["title"], "Keep going")
        self.assertEqual(len(list_predictions(search="José")), 1)
        self.assertEqual(list_predictions(search="%"), [])
        self.assertEqual(list_predictions(search="' OR 1=1 --"), [])
        self.assertEqual(len(list_predictions(category="Good", sex="F", subject="Portuguese")), 1)
        self.assertEqual(list_predictions(category="Excellent"), [])
        self.assertEqual(list_predictions(sex="M"), [])
        self.assertTrue(delete_prediction(record_id))
        self.assertIsNone(get_prediction(record_id))
        self.assertFalse(delete_prediction(record_id))


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = train_and_save()

    def test_artifact_is_saved_and_reloads(self):
        loaded = load_artifact()
        self.assertEqual(
            loaded["with_priors"]["algorithm"],
            self.artifact["with_priors"]["algorithm"],
        )
        self.assertGreater(MODEL_PATH.stat().st_size, 10_000)
        self.assertGreater(COMPARISON_CHART.stat().st_size, 1_000)
        self.assertTrue(REPORT_PATH.exists())

    def test_split_selection_and_target_exclusion(self):
        frame = load_student_data()
        for key, features in (
            ("with_priors", FEATURES_WITH_PRIORS),
            ("without_priors", FEATURES_NO_PRIORS),
        ):
            bundle = self.artifact[key]
            columns = input_columns_from_pipeline(bundle["pipeline"])
            self.assertNotIn(TARGET, columns)
            self.assertNotIn(TARGET, bundle["features"])
            self.assertEqual(set(columns), set(features))
            self.assertEqual(bundle["train_size"] + bundle["test_size"], len(frame))
            self.assertEqual(len(bundle["y_test"]), bundle["test_size"])
            metrics = bundle["test_metrics"]
            self.assertAlmostEqual(metrics["rmse"], metrics["mse"] ** 0.5, places=6)
            self.assertGreaterEqual(metrics["mae"], 0)
            self.assertLessEqual(metrics["r2"], 1)
            selected = [item for item in bundle["candidates"] if item["selected"]]
            self.assertEqual(len(selected), 1)
            best_cv = min(item["cv_rmse"] for item in bundle["candidates"])
            self.assertAlmostEqual(selected[0]["cv_rmse"], best_cv, places=6)
            self.assertIn("cross-validat", selection_note(bundle["candidates"]).lower())
            self.assertGreater(len(bundle["feature_importance"]), 0)
        self.assertNotIn("G1", self.artifact["without_priors"]["features"])
        self.assertNotIn("G2", self.artifact["without_priors"]["features"])
        self.assertAlmostEqual(
            self.artifact["dataset"]["target_mean"],
            float(frame[TARGET].mean()),
            places=5,
        )

    def test_predictions_follow_the_inputs(self):
        low = predict_and_explain(sample_form(G1=0, G2=0, student_name="Low Grade"))
        high = predict_and_explain(sample_form(G1=20, G2=20, student_name="High Grade"))
        self.assertGreater(high["predicted_score"], low["predicted_score"])
        self.assertTrue(low["used_previous_grades"])
        self.assertGreaterEqual(low["predicted_score"], 0)
        self.assertLessEqual(high["predicted_score"], 20)
        self.assertGreaterEqual(len(low["recommendations"]), 3)
        self.assertTrue(low["study_suggestions"])
        self.assertNotIn(TARGET, low["features_used"])

        early_low = predict_and_explain(
            sample_form(use_priors=False, G1=0, G2=0, student_name="Early A")
        )
        early_high_grades = predict_and_explain(
            sample_form(use_priors=False, G1=20, G2=20, student_name="Early B")
        )
        early_absent = predict_and_explain(
            sample_form(use_priors=False, absences=40, student_name="Absent Student")
        )
        self.assertFalse(early_low["used_previous_grades"])
        self.assertEqual(early_low["predicted_score"], early_high_grades["predicted_score"])
        self.assertNotEqual(early_low["raw_score"], early_absent["raw_score"])
        self.assertIn("early-estimate" if False else "not a guarantee", early_low["limitation"])

    def test_charts_are_built_from_the_real_rows(self):
        frame = load_student_data()
        self.assertGreater(len(grade_distribution(frame).data), 0)
        self.assertGreater(len(category_donut(frame).data), 0)
        self.assertGreater(len(study_time_chart(frame).data), 0)
        self.assertGreater(len(absences_chart(frame).data), 0)
        self.assertGreater(len(education_chart(frame, "Medu", "Mother").data), 0)
        comparison = model_comparison_chart(self.artifact, "cv_rmse", "CV RMSE")
        self.assertGreater(len(comparison.data), 0)
        actual = actual_vs_predicted(
            self.artifact["with_priors"]["y_test"],
            self.artifact["with_priors"]["y_pred"],
        )
        self.assertEqual(len(list(actual.data[0].x)), self.artifact["with_priors"]["test_size"])
        importance = importance_chart(self.artifact["with_priors"]["feature_importance"])
        self.assertGreater(len(importance.data), 0)
        gauge = score_gauge(14.2, "Good")
        self.assertGreater(len(gauge.data), 0)


if __name__ == "__main__":
    unittest.main()
