import csv
import struct
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.label_public_attack_windows import (
    ATTACK_INTERVALS,
    _attack_subtype,
    label_feature_csv,
)


class PublicAttackWindowLabelTests(unittest.TestCase):
    def test_only_fully_covered_attack_windows_are_labeled(self):
        capture_start_ns = 100_250_000_000

        self.assertIsNone(_attack_subtype(2_700_000_000_000, capture_start_ns))
        self.assertEqual(
            _attack_subtype(2_701_000_000_000, capture_start_ns), "NETBIOS"
        )
        self.assertIsNone(_attack_subtype(3_169_000_000_000, capture_start_ns))

    def test_labels_attack_windows_and_leaves_other_sources_unknown(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture_path = root / "SAT-03-11-2018_0"
            feature_path = root / "features.csv"
            output_path = root / "labeled.csv"

            global_header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
            packet_header = struct.pack("<IIII", 100, 250_000, 1, 1)
            capture_path.write_bytes(global_header + packet_header + b"x")

            feature_rows = []
            for start_seconds, _end_seconds, _subtype in ATTACK_INTERVALS:
                window_start = datetime.fromtimestamp(
                    101 + start_seconds, timezone.utc
                ).isoformat().replace("+00:00", "Z")
                feature_rows.append(
                    {
                        "source_capture": "SAT-03-11-2018_0",
                        "window_start_utc": window_start,
                        "label": "",
                    }
                )
            feature_rows.append(
                {
                    "source_capture": "DATA1",
                    "window_start_utc": "1970-01-01T00:45:01Z",
                    "label": "",
                }
            )

            with feature_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=("source_capture", "window_start_utc", "label"),
                )
                writer.writeheader()
                writer.writerows(feature_rows)

            result = label_feature_csv(feature_path, output_path, capture_path)

            with output_path.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))

            self.assertEqual([row["label"] for row in rows[:5]], ["-1"] * 5)
            self.assertEqual(rows[5]["label"], "")
            self.assertEqual(result["labeled_attack_windows"], 5)
            self.assertEqual(result["subtype_counts"]["NETBIOS"], 1)


if __name__ == "__main__":
    unittest.main()
