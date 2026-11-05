import csv
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from src.prepare_training_data import FEATURE_COLUMNS
from src.prepare_training_dataset import OUTPUT_COLUMNS, prepare_training_dataset


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


def _write_manifest(
    path: Path,
    *,
    filename: str,
    source_dataset: str,
    digest: str,
    label: str,
    timestamps_known: bool,
) -> None:
    row = {
        "file": filename,
        "source_dataset": source_dataset,
        "sha256": digest,
        "scope": "capture",
        "label": label,
        "subtype": "",
        "start_utc": "2026-01-01T00:00:00Z" if timestamps_known else "unknown",
        "end_utc": "2026-01-01T00:00:01Z" if timestamps_known else "unknown",
        "timezone": "UTC" if timestamps_known else "unknown",
        "evidence_type": "official_dataset_label_file",
        "evidence_reference": "dataset-documentation:reference",
        "review_status": "verified",
        "verified_by": "data-owner",
        "verified_at_utc": "2026-01-02T00:00:00Z",
        "notes": "",
    }
    with path.open("w", encoding="utf-8", newline="") as output:
        manifest_columns: list[str] = list(MANIFEST_COLUMNS)
        writer = csv.DictWriter(output, fieldnames=manifest_columns)
        writer.writeheader()
        writer.writerow(row)


class GroundTruthDatasetMergeTests(unittest.TestCase):
    def _write_pcap(self, path: Path) -> str:
        global_header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65_535, 1)
        frame = b"\x00" * 12 + b"\x08\x06"
        records = b"".join(
            struct.pack("<IIII", timestamp, 0, len(frame), len(frame)) + frame
            for timestamp in (1_767_225_600, 1_767_225_601)
        )
        capture = global_header + records
        path.write_bytes(capture)
        return hashlib.sha256(capture).hexdigest()

    def _write_public_features(
        self,
        path: Path,
        source_sha256: str = "a" * 64,
        label: str = "1",
    ) -> None:
        fields: list[str] = [
            "source_capture",
            "source_sha256",
            "window_start",
            *FEATURE_COLUMNS,
            "label",
        ]
        row = {feature: "0" for feature in FEATURE_COLUMNS}
        row.update(
            {
                "source_capture": "baseline.csv",
                "source_sha256": source_sha256,
                "window_start": "2026-01-01T00:00:00",
                "packets_total": "1",
                "bytes_total": "14",
                "other_transport_packets": "1",
                "mean_packet_bytes": "14",
                "max_packet_bytes": "14",
                "label": label,
            }
        )
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    def test_public_and_bps_ground_truth_are_verified_separately(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bps_capture = root / "bps_capture.pcap"
            bps_digest = self._write_pcap(bps_capture)
            bps_manifest = root / "bps_manifest.csv"
            _write_manifest(
                bps_manifest,
                filename=bps_capture.name,
                source_dataset="bps",
                digest=bps_digest,
                label="1",
                timestamps_known=True,
            )
            public_manifest = root / "public_manifest.csv"
            _write_manifest(
                public_manifest,
                filename="baseline.csv",
                source_dataset="public",
                digest="a" * 64,
                label="1",
                timestamps_known=False,
            )
            public_features = root / "public_features.csv"
            self._write_public_features(public_features)
            output = root / "combined.csv"

            result = prepare_training_dataset(
                public_features,
                bps_capture,
                bps_manifest,
                output,
                public_labels_path=public_manifest,
            )

            self.assertEqual(result["public_windows"], 1)
            self.assertEqual(result["bps_windows"], 2)
            with output.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, list(OUTPUT_COLUMNS))
            self.assertEqual(
                {row["source_dataset"] for row in rows}, {"public", "bps"}
            )
            self.assertEqual(
                {row["label"] for row in rows}, {"1"}
            )

    def test_public_feature_labels_must_match_verified_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bps_capture = root / "bps_capture.pcap"
            bps_digest = self._write_pcap(bps_capture)
            bps_manifest = root / "bps_manifest.csv"
            _write_manifest(
                bps_manifest,
                filename=bps_capture.name,
                source_dataset="bps",
                digest=bps_digest,
                label="1",
                timestamps_known=True,
            )
            public_manifest = root / "public_manifest.csv"
            _write_manifest(
                public_manifest,
                filename="baseline.csv",
                source_dataset="public",
                digest="a" * 64,
                label="-1",
                timestamps_known=False,
            )
            public_features = root / "public_features.csv"
            self._write_public_features(public_features, label="1")

            with self.assertRaisesRegex(ValueError, "tidak cocok dengan manifest"):
                prepare_training_dataset(
                    public_features,
                    bps_capture,
                    bps_manifest,
                    root / "combined.csv",
                    public_labels_path=public_manifest,
                )

            self._write_public_features(
                public_features,
                source_sha256="b" * 64,
                label="-1",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 fitur publik"):
                prepare_training_dataset(
                    public_features,
                    bps_capture,
                    bps_manifest,
                    root / "combined.csv",
                    public_labels_path=public_manifest,
                )


if __name__ == "__main__":
    unittest.main()
