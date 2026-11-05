import csv
import json
import tempfile
import unittest
from pathlib import Path

import joblib

from src.train_traffic_model import train_and_evaluate


class TrainTrafficModelTests(unittest.TestCase):
    def _write_dataset(
        self,
        path: Path,
        rows: list[tuple[str, float, int, float, int]],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                (
                    "source_file",
                    "flow_duration",
                    "protocol",
                    "init_win_bytes_forward",
                    "label",
                )
            )
            writer.writerows(rows)

    def test_trains_evaluates_and_drops_invalid_negative_feature_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            model_path = root / "model.joblib"
            report_path = root / "report.json"
            self._write_dataset(
                train_path,
                [
                    ("train-normal.csv", 10, 6, -1, 1),
                    ("train-normal.csv", 11, 6, 32, 1),
                    ("train-attack.csv", 100, 17, -1, -1),
                    ("train-attack.csv", 110, 17, 64, -1),
                    ("train-invalid.csv", -10, 6, 32, 1),
                ],
            )
            self._write_dataset(
                test_path,
                [
                    ("test-normal.csv", 12, 6, -1, 1),
                    ("test-attack.csv", 120, 17, 32, -1),
                    ("test-invalid.csv", -5, 6, 32, 1),
                ],
            )

            report = train_and_evaluate(
                train_path,
                test_path,
                model_path,
                report_path,
                epochs=1,
                seed=10,
            )

            self.assertTrue(model_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertEqual(report["feature_count"], 3)
            self.assertEqual(report["train"]["rows_used"], 4)
            self.assertEqual(report["train"]["negative_feature_rows_dropped"], 1)
            self.assertEqual(report["test"]["rows_used"], 2)
            self.assertEqual(report["test"]["negative_feature_rows_dropped"], 1)
            self.assertEqual(report["metrics"]["test_rows_evaluated"], 2)
            self.assertEqual(
                report["metrics"]["confusion_matrix_labels"],
                ["attack", "normal"],
            )
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))["feature_count"],
                3,
            )
            payload = joblib.load(model_path)
            self.assertEqual(
                payload["label_mapping"], {"1": "normal", "-1": "attack"}
            )
            self.assertEqual(
                payload["features"],
                ("flow_duration", "protocol", "init_win_bytes_forward"),
            )

    def test_rejects_source_file_leakage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            self._write_dataset(
                train_path,
                [
                    ("shared.csv", 10, 6, 0, 1),
                    ("shared.csv", 100, 17, 0, -1),
                ],
            )
            self._write_dataset(
                test_path,
                [
                    ("shared.csv", 11, 6, 0, 1),
                    ("test-attack.csv", 110, 17, 0, -1),
                ],
            )
            with self.assertRaisesRegex(ValueError, "source_file muncul"):
                train_and_evaluate(
                    train_path,
                    test_path,
                    root / "model.joblib",
                    root / "report.json",
                    epochs=1,
                )


if __name__ == "__main__":
    unittest.main()
