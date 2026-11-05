import csv
import tempfile
import unittest
from pathlib import Path

from src.analyze_unlabeled_features import analyze_unlabeled_features, run
from src.prepare_training_data import FEATURE_COLUMNS


def _row(capture: str, timestamp: str, packets: int, label: str = "") -> dict[str, str]:
    row = {feature: "0" for feature in FEATURE_COLUMNS}
    row.update(
        {
            "source_capture": capture,
            "window_start_utc": timestamp,
            "packets_total": str(packets),
            "bytes_total": str(packets * 100),
            "tcp_packets": str(packets),
            "mean_packet_bytes": "100",
            "max_packet_bytes": "100",
            "unique_source_ips": "1",
            "unique_destination_ips": "1",
            "label": label,
        }
    )
    return row


class UnlabeledFeatureAnalysisTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        fieldnames: list[str] = [
            "source_capture",
            "window_start_utc",
            *FEATURE_COLUMNS,
            "label",
        ]
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_report_summarizes_unlabeled_features_by_family(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "features.csv"
            self._write_csv(
                input_path,
                [
                    _row("DATA1", "2026-01-01T00:00:00+00:00", 2),
                    _row("DATA2", "2026-01-01T00:00:01+00:00", 4),
                    _row("SAT-01", "2018-11-03T00:00:00+00:00", 8),
                ],
            )

            report = analyze_unlabeled_features(input_path)

        self.assertEqual(report["summary"]["window_count"], 3)
        self.assertEqual(report["summary"]["capture_count"], 3)
        self.assertEqual(report["summary"]["unlabeled_rows"], 3)
        self.assertEqual(report["feature_summary"]["packets_total"]["min"], 2)
        self.assertEqual(report["feature_summary"]["packets_total"]["max"], 8)
        self.assertEqual(
            report["feature_summary_by_capture_family"]["DATA"]["capture_count"], 2
        )
        self.assertEqual(
            report["feature_summary_by_capture_family"]["SAT"]["features"][
                "packets_total"
            ]["p50"],
            8,
        )
        self.assertTrue(report["limitations"])

    def test_analysis_rejects_nonempty_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "features.csv"
            self._write_csv(
                input_path,
                [_row("DATA1", "2026-01-01T00:00:00+00:00", 2, label="1")],
            )

            with self.assertRaisesRegex(ValueError, "hanya untuk fitur tanpa label"):
                analyze_unlabeled_features(input_path)

    def test_analysis_rejects_duplicate_capture_windows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "features.csv"
            row = _row("DATA1", "2026-01-01T00:00:00+00:00", 2)
            self._write_csv(input_path, [row, row])

            with self.assertRaisesRegex(ValueError, "duplikat"):
                analyze_unlabeled_features(input_path)

    def test_run_writes_report_and_protects_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "features.csv"
            output_path = root / "reports" / "baseline.json"
            self._write_csv(
                input_path,
                [_row("DATA1", "2026-01-01T00:00:00+00:00", 2)],
            )

            report = run(input_path, output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual(report["summary"]["window_count"], 1)
            with self.assertRaises(FileExistsError):
                run(input_path, output_path)


if __name__ == "__main__":
    unittest.main()
