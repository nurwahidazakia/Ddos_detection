import csv
import tempfile
import unittest
from pathlib import Path

from src.label_public_benign_windows import label_benign_windows


class PublicBenignWindowLabelTests(unittest.TestCase):
    def test_labels_only_fully_covered_verified_benign_windows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_path = root / "features.csv"
            intervals_path = root / "intervals.csv"
            output_path = root / "labeled.csv"
            fields = ("source_capture", "window_start_utc", "label")
            rows = [
                {
                    "source_capture": "SAT-03-11-2018_0",
                    "window_start_utc": "2018-11-03T09:00:00Z",
                    "label": "",
                },
                {
                    "source_capture": "SAT-03-11-2018_0",
                    "window_start_utc": "2018-11-03T09:00:01Z",
                    "label": "",
                },
                {
                    "source_capture": "SAT-03-11-2018_0",
                    "window_start_utc": "2018-11-03T09:00:02Z",
                    "label": "-1",
                },
                {
                    "source_capture": "DATA1",
                    "window_start_utc": "2018-11-03T09:00:00Z",
                    "label": "",
                },
            ]
            with feature_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            with intervals_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=(
                        "source_capture",
                        "start_utc",
                        "end_utc",
                        "evidence_reference",
                        "review_status",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "source_capture": "SAT-03-11-2018_0",
                        "start_utc": "2018-11-03T09:00:00Z",
                        "end_utc": "2018-11-03T09:00:01.500Z",
                        "evidence_reference": "official benign interval",
                        "review_status": "verified",
                    }
                )

            result = label_benign_windows(
                feature_path, intervals_path, output_path
            )

            with output_path.open(encoding="utf-8-sig", newline="") as source:
                labeled_rows = list(csv.DictReader(source))

            self.assertEqual(
                [row["label"] for row in labeled_rows], ["1", "", "-1", ""]
            )
            self.assertEqual(result["benign_labeled_windows"], 1)
            self.assertEqual(result["preserved_attack_windows"], 1)
            self.assertEqual(result["remaining_unlabeled_windows"], 1)

    def test_rejects_unverified_benign_intervals(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_path = root / "features.csv"
            intervals_path = root / "intervals.csv"
            output_path = root / "labeled.csv"
            with feature_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=("source_capture", "window_start_utc", "label"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "source_capture": "SAT-03-11-2018_0",
                        "window_start_utc": "2018-11-03T09:00:00Z",
                        "label": "",
                    }
                )
            with intervals_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=(
                        "source_capture",
                        "start_utc",
                        "end_utc",
                        "evidence_reference",
                        "review_status",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "source_capture": "SAT-03-11-2018_0",
                        "start_utc": "2018-11-03T09:00:00Z",
                        "end_utc": "2018-11-03T09:00:01Z",
                        "evidence_reference": "unverified",
                        "review_status": "pending_verification",
                    }
                )

            with self.assertRaisesRegex(ValueError, "review_status harus 'verified'"):
                label_benign_windows(feature_path, intervals_path, output_path)


if __name__ == "__main__":
    unittest.main()
