import csv
import tempfile
import unittest
from pathlib import Path

from src.harmonize_cicflowmeter_features import (
    harmonize_cicflowmeter_features,
)


class HarmonizeCicFlowMeterFeaturesTests(unittest.TestCase):
    def _write_csv(
        self,
        path: Path,
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_aligns_aliases_drops_identifiers_and_keeps_normal_class_pending(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            normal = root / "normal.csv"
            attack = root / "attack.csv"
            output = root / "output"
            self._write_csv(
                normal,
                [
                    "Flow ID",
                    "Src IP",
                    "Timestamp",
                    "Flow Duration",
                    "Total Fwd Packet",
                    "CWR Flag Count",
                    "Label",
                ],
                [
                    {
                        "Flow ID": "private-flow-id",
                        "Src IP": "192.0.2.1",
                        "Timestamp": "01/01/2020 00:00:00",
                        "Flow Duration": "-1",
                        "Total Fwd Packet": "",
                        "CWR Flag Count": "0",
                        "Label": "NeedManualLabel",
                    }
                ],
            )
            self._write_csv(
                attack,
                [
                    "Unnamed: 0",
                    "Flow Duration",
                    "Total Fwd Packets",
                    "CWE Flag Count",
                    "Label",
                    "Class",
                ],
                [
                    {
                        "Unnamed: 0": "7",
                        "Flow Duration": "9",
                        "Total Fwd Packets": "4",
                        "CWE Flag Count": "0",
                        "Label": "UDP",
                        "Class": "Attack",
                    }
                ],
            )

            result = harmonize_cicflowmeter_features(normal, attack, output)

            self.assertEqual(
                result["features"], ["Flow Duration", "Total Fwd Packets"]
            )
            self.assertEqual(result["normal"]["negative_values"], {"Flow Duration": 1})
            self.assertEqual(
                result["normal"]["missing_values"], {"Total Fwd Packets": 1}
            )
            self.assertEqual(
                result["excluded_unverified_flag_columns"],
                ["CWE Flag Count", "CWR Flag Count"],
            )
            with (output / "normal_features_unlabeled.csv").open(
                "r", encoding="utf-8", newline=""
            ) as source:
                normal_reader = csv.DictReader(source)
                normal_row = next(normal_reader)
            with (output / "attack_features_labeled.csv").open(
                "r", encoding="utf-8", newline=""
            ) as source:
                attack_reader = csv.DictReader(source)
                attack_row = next(attack_reader)

            self.assertEqual(normal_reader.fieldnames, attack_reader.fieldnames)
            self.assertEqual(
                normal_reader.fieldnames,
                ["source_file", "Flow Duration", "Total Fwd Packets", "Label", "Class"],
            )
            self.assertEqual(normal_row["Label"], "NeedManualLabel")
            self.assertEqual(normal_row["Class"], "")
            self.assertEqual(normal_row["Flow Duration"], "-1")
            self.assertEqual(normal_row["Total Fwd Packets"], "")
            self.assertEqual(attack_row["Label"], "UDP")
            self.assertEqual(attack_row["Class"], "Attack")
            self.assertNotIn("Src IP", normal_row)
            self.assertNotIn("Unnamed: 0", attack_row)

    def test_rejects_nonmatching_feature_schemas(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            normal = root / "normal.csv"
            attack = root / "attack.csv"
            self._write_csv(
                normal,
                ["Flow Duration", "Label"],
                [{"Flow Duration": "1", "Label": "NeedManualLabel"}],
            )
            self._write_csv(
                attack,
                ["Protocol", "Label", "Class"],
                [{"Protocol": "17", "Label": "UDP", "Class": "Attack"}],
            )

            with self.assertRaisesRegex(ValueError, "Skema fitur tidak sepadan"):
                harmonize_cicflowmeter_features(normal, attack, root / "output")

    def test_preserves_and_reports_nonfinite_feature_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            normal = root / "normal.csv"
            attack = root / "attack.csv"
            self._write_csv(
                normal,
                ["Flow Duration", "Label"],
                [{"Flow Duration": "NaN", "Label": "NeedManualLabel"}],
            )
            self._write_csv(
                attack,
                ["Flow Duration", "Label", "Class"],
                [{"Flow Duration": "1", "Label": "UDP", "Class": "Attack"}],
            )

            result = harmonize_cicflowmeter_features(normal, attack, root / "output")

            self.assertEqual(
                result["normal"]["nonfinite_values"], {"Flow Duration": 1}
            )
            with (root / "output" / "normal_features_unlabeled.csv").open(
                "r", encoding="utf-8", newline=""
            ) as source:
                row = next(csv.DictReader(source))
            self.assertEqual(row["Flow Duration"], "NaN")

    def test_rejects_non_numeric_feature_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            normal = root / "normal.csv"
            attack = root / "attack.csv"
            self._write_csv(
                normal,
                ["Flow Duration", "Label"],
                [{"Flow Duration": "not-a-number", "Label": "NeedManualLabel"}],
            )
            self._write_csv(
                attack,
                ["Flow Duration", "Label", "Class"],
                [{"Flow Duration": "1", "Label": "UDP", "Class": "Attack"}],
            )

            with self.assertRaisesRegex(ValueError, "Nilai fitur bukan angka"):
                harmonize_cicflowmeter_features(normal, attack, root / "output")


if __name__ == "__main__":
    unittest.main()
