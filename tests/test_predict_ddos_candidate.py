import csv
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

from src.predict_ddos_candidate import predict_csv


class PredictDdosCandidateTests(unittest.TestCase):
    def _write_model(self, path: Path) -> None:
        features = ("Avg Fwd Segment Size", "Protocol")
        training = pd.DataFrame(
            [
                [10.0, 6.0],
                [12.0, 6.0],
                [200.0, 17.0],
                [220.0, 17.0],
            ],
            columns=features,
        )
        labels = np.array([1, 1, -1, -1], dtype=np.int8)
        imputer = SimpleImputer(strategy="median").fit(training)
        estimator = ExtraTreesClassifier(n_estimators=5, random_state=42).fit(
            imputer.transform(training), labels
        )
        joblib.dump(
            {
                "model": estimator,
                "imputer": imputer,
                "features": features,
                "threshold": 0.5,
                "status": "provisional_experiment_not_production_ready",
                "source_sha256": "test-source",
            },
            path,
        )

    def test_predict_csv_supports_cicflowmeter_alias_and_evaluates_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            input_path = root / "test.csv"
            output_path = root / "predictions.csv"
            report_path = root / "report.json"
            self._write_model(model_path)

            with input_path.open("w", encoding="utf-8", newline="") as source:
                writer = csv.writer(source)
                writer.writerow(("Fwd Segment Size Avg", "Protocol", "Class"))
                writer.writerow((11, 6, "BENIGN"))
                writer.writerow((210, 17, "Attack"))

            report = predict_csv(
                input_path,
                output_path,
                report_path,
                model_path,
                label_column="Class",
                capture_id="held-out-day",
                chunk_size=1,
            )

            with output_path.open("r", encoding="utf-8", newline="") as output:
                rows = list(csv.DictReader(output))
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["scored_rows"], 2)
            self.assertEqual(report["capture_id"], "held-out-day")
            self.assertEqual(report["metrics"]["rows_evaluated"], 2)
            self.assertEqual(report["feature_aliases_used"]["Avg Fwd Segment Size"], "Fwd Segment Size Avg")
            self.assertEqual([row["PredictionStatus"] for row in rows], ["scored", "scored"])
            self.assertTrue(all(0 <= float(row["AttackScore"]) <= 1 for row in rows))
            self.assertEqual(saved_report["status"], "provisional_test_metrics_not_production_validation")

    def test_predict_csv_marks_rows_with_no_numeric_features_unscored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            input_path = root / "test.csv"
            output_path = root / "predictions.csv"
            report_path = root / "report.json"
            self._write_model(model_path)

            with input_path.open("w", encoding="utf-8", newline="") as source:
                writer = csv.writer(source)
                writer.writerow(("Avg Fwd Segment Size", "Protocol"))
                writer.writerow(("", ""))

            report = predict_csv(input_path, output_path, report_path, model_path)

            with output_path.open("r", encoding="utf-8", newline="") as output:
                row = next(csv.DictReader(output))
            self.assertEqual(report["scored_rows"], 0)
            self.assertEqual(report["rows_without_valid_features"], 1)
            self.assertEqual(row["PredictedLabel"], "NOT_SCORED")
            self.assertEqual(row["PredictionStatus"], "no_valid_features")

    def test_predict_csv_rejects_missing_model_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            input_path = root / "test.csv"
            self._write_model(model_path)
            input_path.write_text("Avg Fwd Segment Size\n10\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Protocol"):
                predict_csv(
                    input_path,
                    root / "predictions.csv",
                    root / "report.json",
                    model_path,
                )

    def test_predict_csv_rejects_unknown_evaluation_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            input_path = root / "test.csv"
            self._write_model(model_path)
            input_path.write_text(
                "Avg Fwd Segment Size,Protocol,Class\n10,6,unknown\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "label tidak dikenal"):
                predict_csv(
                    input_path,
                    root / "predictions.csv",
                    root / "report.json",
                    model_path,
                    label_column="Class",
                )


if __name__ == "__main__":
    unittest.main()
