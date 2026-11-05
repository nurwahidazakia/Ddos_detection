import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from src.prepare_labeled_csv import OUTPUT_COLUMNS, prepare_labeled_csv


MANIFEST_COLUMNS = (
    "file",
    "source_dataset",
    "sha256",
    "scope",
    "label",
    "subtype",
    "start_utc",
    "end_utc",
    "timezone",
    "evidence_type",
    "evidence_reference",
    "review_status",
    "verified_by",
    "verified_at_utc",
    "notes",
)


class PrepareLabeledCsvTests(unittest.TestCase):
    def _write_source(self, path: Path, label: int) -> str:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                ("timestamp", "src_ip", "dst_ip", "protocol", "packet_length", "tcp_flags", "label")
            )
            writer.writerow(
                ("2026-01-01T00:00:00", "192.0.2.1", "192.0.2.2", "TCP", "60", "2", label)
            )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_manifest(
        self,
        path: Path,
        entries: list[tuple[str, str, int]],
        *,
        status: str = "verified",
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for filename, digest, label in entries:
                writer.writerow(
                    {
                        "file": filename,
                        "source_dataset": "public",
                        "sha256": digest,
                        "scope": "capture",
                        "label": label,
                        "subtype": "",
                        "start_utc": "unknown",
                        "end_utc": "unknown",
                        "timezone": "unknown",
                        "evidence_type": "official_dataset_label_file",
                        "evidence_reference": "official-docs:label-mapping",
                        "review_status": status,
                        "verified_by": "dataset-reviewer",
                        "verified_at_utc": "2026-01-02T00:00:00Z",
                        "notes": "",
                    }
                )

    def test_manifest_labels_override_input_names_and_hashes_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline_data.csv"
            attack = root / "attack_data.csv"
            baseline_hash = self._write_source(baseline, label=-1)
            attack_hash = self._write_source(attack, label=1)
            manifest = root / "public_manifest.csv"
            self._write_manifest(
                manifest,
                [
                    (baseline.name, baseline_hash, -1),
                    (attack.name, attack_hash, 1),
                ],
            )
            output = root / "features.csv"

            result = prepare_labeled_csv(baseline, attack, output, labels_path=manifest)

            self.assertEqual(result["normal_packets"], 1)
            self.assertEqual(result["attack_packets"], 1)
            with output.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, list(OUTPUT_COLUMNS))
            self.assertEqual(
                {row["source_capture"]: row["label"] for row in rows},
                {"baseline_data.csv": "-1", "attack_data.csv": "1"},
            )
            self.assertEqual(
                {row["source_capture"]: row["source_sha256"] for row in rows},
                {"baseline_data.csv": baseline_hash, "attack_data.csv": attack_hash},
            )

    def test_pending_public_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline_data.csv"
            attack = root / "attack_data.csv"
            baseline_hash = self._write_source(baseline, label=1)
            attack_hash = self._write_source(attack, label=-1)
            manifest = root / "public_manifest.csv"
            self._write_manifest(
                manifest,
                [
                    (baseline.name, baseline_hash, 1),
                    (attack.name, attack_hash, -1),
                ],
                status="pending_verification",
            )

            with self.assertRaisesRegex(ValueError, "hanya baris verified"):
                prepare_labeled_csv(
                    baseline, attack, root / "features.csv", labels_path=manifest
                )

    def test_public_source_hash_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline_data.csv"
            attack = root / "attack_data.csv"
            self._write_source(baseline, label=1)
            attack_hash = self._write_source(attack, label=-1)
            manifest = root / "public_manifest.csv"
            self._write_manifest(
                manifest,
                [
                    (baseline.name, "0" * 64, 1),
                    (attack.name, attack_hash, -1),
                ],
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 tidak cocok"):
                prepare_labeled_csv(
                    baseline, attack, root / "features.csv", labels_path=manifest
                )


if __name__ == "__main__":
    unittest.main()
