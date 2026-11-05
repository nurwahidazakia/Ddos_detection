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

from src.prepare_cicddos2019_diagnostic_test import prepare_diagnostic_test_set


class PrepareDiagnosticTestSetTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        normal_path = root / "normal.csv"
        attack_path = root / "attack.csv"
        training_path = root / "training.csv"
        model_path = root / "model.joblib"

        with normal_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("Protocol", "Label", "Class"))
            for value in range(10, 20):
                writer.writerow((value, "NeedManualLabel", ""))

        with attack_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("Protocol", "Label", "Class"))
            writer.writerow((99, "Syn", "Attack"))
            for value in range(1, 10):
                writer.writerow((value, "Syn", "Attack"))

        with training_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("Protocol", "Label"))
            writer.writerow((99, "Syn"))

        features = ("Protocol",)
        train_x = pd.DataFrame({"Protocol": [1.0, 2.0, 10.0, 11.0]})
        train_y = np.array([-1, -1, 1, 1], dtype=np.int8)
        imputer = SimpleImputer(strategy="median").fit(train_x)
        model = ExtraTreesClassifier(n_estimators=5, random_state=42).fit(
            imputer.transform(train_x), train_y
        )
        joblib.dump(
            {
                "model": model,
                "imputer": imputer,
                "features": features,
                "threshold": 0.5,
                "status": "provisional",
            },
            model_path,
        )
        return normal_path, attack_path, training_path, model_path

    def test_builds_balanced_diagnostic_and_excludes_training_attack_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal, attack, training, model = self._write_inputs(root)
            output = root / "test.csv"
            report_path = root / "report.json"

            report = prepare_diagnostic_test_set(
                normal,
                attack,
                training,
                model,
                output,
                report_path,
                rows_per_class=3,
                random_state=17,
            )

            with output.open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            attack_rows = [
                row for row in rows if row["CaptureID"] == "unknown-no-capture-day-id"
            ]
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                set(row["Label"] for row in rows if row["SourceDataset"] == "CIC-IDS2017"),
                {"BENIGN"},
            )
            self.assertEqual(set(row["Label"] for row in attack_rows), {"Syn"})
            self.assertTrue(all(row["LabelEvidence"] for row in rows))
            self.assertEqual(report["class_counts"], {"normal": 3, "attack": 3})
            self.assertEqual(
                report["sources"]["attack"]["duplicate_feature_rows_matching_training_excluded"],
                1,
            )
            self.assertNotIn("99", [row["Protocol"] for row in attack_rows])
            self.assertEqual(
                report["status"],
                "diagnostic_cross_source_not_independent_capture_test",
            )
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["rows"], 6)
            self.assertIn("not an independent capture/day test", saved["limitations"][0].lower())

    def test_sampling_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal, attack, training, model = self._write_inputs(root)
            first = prepare_diagnostic_test_set(
                normal,
                attack,
                training,
                model,
                root / "first.csv",
                root / "first.json",
                rows_per_class=3,
                random_state=23,
            )
            second = prepare_diagnostic_test_set(
                normal,
                attack,
                training,
                model,
                root / "second.csv",
                root / "second.json",
                rows_per_class=3,
                random_state=23,
            )
            with (root / "first.csv").open("rb") as source:
                first_content = source.read()
            with (root / "second.csv").open("rb") as source:
                second_content = source.read()
            self.assertEqual(first_content, second_content)
            self.assertEqual(first["class_counts"], second["class_counts"])

    def test_rejects_insufficient_class_rows_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal, attack, training, model = self._write_inputs(root)
            output = root / "test.csv"
            report_path = root / "report.json"

            with self.assertRaisesRegex(ValueError, "Baris normal eligible"):
                prepare_diagnostic_test_set(
                    normal,
                    attack,
                    training,
                    model,
                    output,
                    report_path,
                    rows_per_class=11,
                    random_state=17,
                )

            self.assertFalse(output.exists())
            self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()
