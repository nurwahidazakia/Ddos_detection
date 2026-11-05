import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_balanced_cicddos2019 import prepare_balanced_dataset


class PrepareBalancedCicDdos2019Tests(unittest.TestCase):
    def _write_source(self, path: Path, *, inconsistent: bool = False) -> None:
        columns = [
            "",
            "Protocol",
            "Flow Duration",
            "Fwd Header Length",
            "Init Fwd Win Bytes",
            "CWE Flag Count",
            "Label",
            "Class",
        ]
        rows = [
            {
                "": "10",
                "Protocol": "17",
                "Flow Duration": "100",
                "Fwd Header Length": "20",
                "Init Fwd Win Bytes": "-1",
                "CWE Flag Count": "0",
                "Label": "Benign",
                "Class": "Benign",
            },
            {
                "": "11",
                "Protocol": "6",
                "Flow Duration": "200",
                "Fwd Header Length": "-2147483648",
                "Init Fwd Win Bytes": "-1",
                "CWE Flag Count": "0",
                "Label": "Benign",
                "Class": "Benign",
            },
            {
                "": "12",
                "Protocol": "6",
                "Flow Duration": "300",
                "Fwd Header Length": "40",
                "Init Fwd Win Bytes": "65535",
                "CWE Flag Count": "0",
                "Label": "Syn",
                "Class": "Attack",
            },
            {
                "": "13",
                "Protocol": "17",
                "Flow Duration": "400",
                "Fwd Header Length": "50",
                "Init Fwd Win Bytes": "-1",
                "CWE Flag Count": "0",
                "Label": "UDP",
                "Class": "Attack",
            },
            {
                "": "14",
                "Protocol": "6",
                "Flow Duration": "500",
                "Fwd Header Length": "60",
                "Init Fwd Win Bytes": "8192",
                "CWE Flag Count": "0",
                "Label": "Syn",
                "Class": "Attack",
            },
        ]
        if inconsistent:
            rows[-1]["Class"] = "Benign"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_creates_balanced_stratified_sample_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            output = root / "balanced.csv"
            report_path = root / "report.json"
            self._write_source(source)

            report = prepare_balanced_dataset(source, output, report_path, seed=7)

            self.assertEqual(report["output_rows_by_class"], {"attack": 2, "benign": 2})
            self.assertEqual(report["output_rows"], 4)
            self.assertEqual(report["feature_count"], 2)
            self.assertEqual(report["duplicated_rows_added"], 0)
            self.assertEqual(
                report["output_rows_by_subtype"]["attack:Syn"], 1
            )
            self.assertEqual(
                report["output_rows_by_subtype"]["attack:UDP"], 1
            )
            self.assertEqual(
                report["excluded_columns_and_reasons"]["Fwd Header Length"],
                "negative int32-like overflow values in source",
            )
            with output.open("r", encoding="utf-8", newline="") as source_file:
                reader = csv.DictReader(source_file)
                rows = list(reader)
            self.assertEqual(
                reader.fieldnames,
                ["Unnamed: 0", "Protocol", "Flow Duration", "Label"],
            )
            self.assertEqual(len({row["Unnamed: 0"] for row in rows}), 4)
            self.assertEqual(sum(row["Label"] == "Benign" for row in rows), 2)
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))["status"],
                "provisional_balanced_development_dataset_not_independent",
            )

    def test_sampling_is_reproducible_for_same_seed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            self._write_source(source)
            first = root / "first.csv"
            second = root / "second.csv"

            prepare_balanced_dataset(
                source, first, root / "first.json", seed=21
            )
            prepare_balanced_dataset(
                source, second, root / "second.json", seed=21
            )

            with first.open("rb") as left, second.open("rb") as right:
                self.assertEqual(left.read(), right.read())

    def test_rejects_class_and_subtype_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            self._write_source(source, inconsistent=True)

            with self.assertRaisesRegex(ValueError, "Class dan Label bertentangan"):
                prepare_balanced_dataset(
                    source, root / "balanced.csv", root / "report.json"
                )

    def test_rejects_negative_retained_features(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            self._write_source(source)
            text = source.read_text(encoding="utf-8")
            source.write_text(
                text.replace(",17,100,20,", ",17,-100,20,", 1),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Fitur negatif"):
                prepare_balanced_dataset(
                    source, root / "balanced.csv", root / "report.json"
                )

    def test_refuses_to_overwrite_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            output = root / "balanced.csv"
            report = root / "report.json"
            self._write_source(source)
            output.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_balanced_dataset(source, output, report)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
