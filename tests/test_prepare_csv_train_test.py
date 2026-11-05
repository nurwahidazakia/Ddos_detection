import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.prepare_csv_train_test import prepare_csv_train_test


class PrepareCsvTrainTestTests(unittest.TestCase):
    def _write_csv(
        self,
        path: Path,
        rows: list[tuple[str, str, str]],
    ) -> None:
        path.write_text(
            "Flow ID,Timestamp,Feature A,Feature B,Label\n"
            + "".join(
                f"flow-{index},2026-01-01T00:00:00Z,{feature_a},{feature_b},{label}\n"
                for index, (feature_a, feature_b, label) in enumerate(rows)
            ),
            encoding="utf-8",
        )

    def test_group_split_balances_training_and_keeps_test_distribution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            rows = [
                ("1", "10", "Benign"),
                ("2", "20", "Benign"),
                ("3", "30", "Benign"),
                ("4", "40", "DDoS"),
            ]
            for name in ("capture_a.csv", "capture_b.csv", "capture_c.csv", "capture_d.csv"):
                self._write_csv(input_dir / name, rows)

            result = prepare_csv_train_test(input_dir, output_dir, test_size=0.25, seed=3)

            schema = result["schemas"][0]
            self.assertEqual(schema["split_strategy"], "source_file_group")
            self.assertEqual(schema["train_class_counts"], {"normal": 3, "attack": 3})
            train = pd.read_csv(schema["train_output"])
            test = pd.read_csv(schema["test_output"])
            self.assertEqual(set(train["label"]), {-1, 1})
            self.assertEqual(len(set(train["source_file"]) & set(test["source_file"])), 0)
            self.assertEqual(test["label"].value_counts().to_dict(), {1: 3, -1: 1})
            self.assertNotIn("flow_id", train.columns)
            self.assertNotIn("timestamp", train.columns)

    def test_single_source_stratifies_rows_and_drops_invalid_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            rows = [
                ("1", "10", "Benign"),
                ("2", "20", "Benign"),
                ("3", "30", "Benign"),
                ("4", "40", "Benign"),
                ("5", "50", "DDoS"),
                ("6", "60", "DDoS"),
                ("7", "70", "DDoS"),
                ("8", "80", "DDoS"),
            ]
            self._write_csv(input_dir / "combined.csv", rows)
            with (input_dir / "combined.csv").open("a", encoding="utf-8") as source:
                source.write("flow-header,time-header,Feature A,Feature B,Label\n")
                source.write("bad-flow,time,inf,99,DDoS\n")

            result = prepare_csv_train_test(input_dir, output_dir, seed=17)

            schema = result["schemas"][0]
            self.assertEqual(
                schema["split_strategy"], "stratified_rows_single_source"
            )
            self.assertEqual(schema["dropped_rows"], 2)
            train = pd.read_csv(schema["train_output"])
            test = pd.read_csv(schema["test_output"])
            self.assertEqual(train["label"].value_counts().to_dict(), {1: 3, -1: 3})
            self.assertEqual(test["label"].value_counts().to_dict(), {1: 1, -1: 1})
            self.assertEqual(set(train["source_file"]), {"combined.csv"})

    def test_different_schemas_are_written_to_separate_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            self._write_csv(
                input_dir / "first.csv",
                [
                    ("1", "10", "Benign"),
                    ("2", "20", "Benign"),
                    ("3", "30", "DDoS"),
                    ("4", "40", "DDoS"),
                ],
            )
            (input_dir / "second.csv").write_text(
                "Timestamp,Feature A,Feature B,Extra,Label\n"
                "2026-01-01,1,10,100,Benign\n"
                "2026-01-02,2,20,200,Benign\n"
                "2026-01-03,3,30,300,DDoS\n"
                "2026-01-04,4,40,400,DDoS\n",
                encoding="utf-8",
            )

            result = prepare_csv_train_test(input_dir, output_dir)

            self.assertEqual(result["schema_count"], 2)
            self.assertNotEqual(
                result["schemas"][0]["train_output"],
                result["schemas"][1]["train_output"],
            )

    def test_unknown_non_ddos_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            input_dir.mkdir()
            self._write_csv(
                input_dir / "cicids2018.csv",
                [
                    ("1", "10", "Benign"),
                    ("2", "20", "FTP-BruteForce"),
                ],
            )

            with self.assertRaisesRegex(ValueError, "label tidak dipetakan"):
                prepare_csv_train_test(input_dir, root / "output")


if __name__ == "__main__":
    unittest.main()
