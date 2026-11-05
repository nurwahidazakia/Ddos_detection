import csv
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from src.prepare_training_data import CSV_COLUMNS, prepare_training_data


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


class GroundTruthManifestTests(unittest.TestCase):
    def _write_capture(self, path: Path) -> str:
        global_header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65_535, 1)
        frame = b"\x00" * 12 + b"\x08\x06"
        timestamps = (1_767_225_600, 1_767_225_601)
        records = b"".join(
            struct.pack("<IIII", timestamp, 0, len(frame), len(frame)) + frame
            for timestamp in timestamps
        )
        capture_bytes = global_header + records
        path.write_bytes(capture_bytes)
        return hashlib.sha256(capture_bytes).hexdigest()

    def _write_manifest(
        self,
        path: Path,
        *,
        sha256: str,
        start: str = "2026-01-01T00:00:00Z",
        end: str = "2026-01-01T00:00:01Z",
        evidence_reference: str = "incident-ticket:IR-123",
        review_status: str = "verified",
        source_dataset: str = "bps",
    ) -> None:
        row = {
            "file": "capture.pcap",
            "source_dataset": source_dataset,
            "sha256": sha256,
            "scope": "capture",
            "label": "1",
            "subtype": "",
            "start_utc": start,
            "end_utc": end,
            "timezone": "UTC",
            "evidence_type": "controlled_test_record",
            "evidence_reference": evidence_reference,
            "review_status": review_status,
            "verified_by": "data-owner",
            "verified_at_utc": "2026-01-02T00:00:00Z",
            "notes": "",
        }
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerow(row)

    def test_legacy_file_label_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.pcap"
            manifest = root / "labels.csv"
            output = root / "features.csv"
            self._write_capture(capture)
            manifest.write_text("file,label\ncapture.pcap,1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Manifest ground truth harus memiliki kolom"):
                prepare_training_data(capture, output, manifest)

    def test_missing_evidence_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.pcap"
            manifest = root / "labels.csv"
            digest = self._write_capture(capture)
            self._write_manifest(manifest, sha256=digest, evidence_reference="")

            with self.assertRaisesRegex(ValueError, "evidence_type, evidence_reference"):
                prepare_training_data(capture, root / "features.csv", manifest)

    def test_label_coverage_must_include_all_capture_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.pcap"
            manifest = root / "labels.csv"
            digest = self._write_capture(capture)
            self._write_manifest(
                manifest,
                sha256=digest,
                start="2026-01-01T00:00:00Z",
                end="2026-01-01T00:00:00Z",
            )

            with self.assertRaisesRegex(ValueError, "cakupan waktu manifest tidak mencakup"):
                prepare_training_data(capture, root / "features.csv", manifest)

    def test_verified_full_capture_manifest_produces_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.pcap"
            manifest = root / "labels.csv"
            output = root / "features.csv"
            digest = self._write_capture(capture)
            self._write_manifest(manifest, sha256=digest)

            result = prepare_training_data(capture, output, manifest)

            self.assertEqual(result, {"capture_count": 1, "packets": 2, "windows": 2})
            with output.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                self.assertEqual(reader.fieldnames, list(CSV_COLUMNS))
                self.assertEqual([row["label"] for row in reader], ["1", "1"])

    def test_pending_verification_cannot_label_training_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.pcap"
            manifest = root / "labels.csv"
            digest = self._write_capture(capture)
            self._write_manifest(
                manifest,
                sha256=digest,
                review_status="pending_verification",
                evidence_reference="not_provided",
            )

            with self.assertRaisesRegex(ValueError, "hanya baris verified"):
                prepare_training_data(capture, root / "features.csv", manifest)

    def test_manifest_hash_must_match_capture_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.pcap"
            manifest = root / "labels.csv"
            self._write_capture(capture)
            self._write_manifest(manifest, sha256="0" * 64)

            with self.assertRaisesRegex(ValueError, "SHA-256 tidak cocok"):
                prepare_training_data(capture, root / "features.csv", manifest)

    def test_expected_source_dataset_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.pcap"
            manifest = root / "labels.csv"
            digest = self._write_capture(capture)
            self._write_manifest(manifest, sha256=digest, source_dataset="public")

            with self.assertRaisesRegex(ValueError, "diharapkan bps"):
                prepare_training_data(
                    capture,
                    root / "features.csv",
                    manifest,
                    expected_source_dataset="bps",
                )


if __name__ == "__main__":
    unittest.main()
