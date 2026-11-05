import csv
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.data_profiling import profile_csv
from src.inspect_data import run as inspect_data
from src.map_features import build_mapping
from src.pcap_profiling import profile_pcap


class InitialInspectionTests(unittest.TestCase):
    def test_csv_profile_reports_duplicates_infinity_and_candidate_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "sample.csv"
            csv_path.write_text(
                "timestamp,protocol,label,value\n"
                "2026-01-01T00:00:00Z,TCP,Normal,1\n"
                "2026-01-01T00:00:01Z,UDP,DDoS,inf\n"
                "2026-01-01T00:00:01Z,UDP,DDoS,inf\n"
                "2026-01-01T00:00:02Z,ICMP,Normal,\n",
                encoding="utf-8",
            )

            profile = profile_csv(csv_path, chunk_size=2)

        self.assertEqual(profile["row_count"], 4)
        self.assertEqual(profile["exact_duplicate_rows"], 1)
        self.assertEqual(profile["label_candidates"], ["label"])
        self.assertEqual(
            profile["label_distribution"]["label"]["value_counts"],
            {"Normal": 2, "DDoS": 2},
        )
        value_column = next(column for column in profile["columns"] if column["name"] == "value")
        self.assertEqual(value_column["infinity_count"], 2)
        self.assertEqual(value_column["null_count"], 1)

    def test_mapping_requires_both_sources_to_be_confirmed_ethernet_pcaps(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "inspection.csv"
            with report_path.open("w", encoding="utf-8", newline="") as report_file:
                writer = csv.DictWriter(
                    report_file,
                    fieldnames=("source_dataset", "detected_format", "linktype"),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "source_dataset": "BPS",
                            "detected_format": "classic PCAP",
                            "linktype": "1",
                        },
                        {
                            "source_dataset": "PUBLIC",
                            "detected_format": "classic PCAP",
                            "linktype": "1",
                        },
                    )
                )

            mapping = build_mapping(report_path)

        self.assertIn("kompatibel bersyarat", mapping[0]["Status kesesuaian"])
        self.assertEqual(
            mapping[0]["Nama fitur pada dataset publik"],
            mapping[0]["Nama fitur pada dataset BPS"],
        )
        self.assertEqual(
            mapping[-1]["Status kesesuaian"],
            "ground truth tidak dapat ditentukan dari fitur PCAP",
        )

    def test_pcap_inspection_decodes_sampled_protocols_and_packet_lengths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            pcap_path = Path(temporary_directory) / "sample.pcap"
            global_header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65_535, 1)
            ipv4_header = bytearray(20)
            ipv4_header[0] = 0x45
            ipv4_header[2:4] = (20).to_bytes(2, "big")
            ipv4_header[9] = 17
            udp_frame = b"\x00" * 12 + b"\x08\x00" + ipv4_header
            other_frame = b"\x00" * 12 + b"\x08\x06"
            records = b"".join(
                struct.pack("<IIII", 1, 0, len(frame), len(frame)) + frame
                for frame in (udp_frame, other_frame)
            )
            pcap_path.write_bytes(global_header + records)

            profile = profile_pcap(pcap_path, decode_packets=True)

        self.assertEqual(profile["packets_scanned"], 2)
        self.assertEqual(profile["protocol_distribution"]["UDP"], 1)
        self.assertEqual(profile["protocol_distribution"]["IPv4"], 1)
        self.assertEqual(profile["protocol_distribution"]["other_or_truncated"], 1)
        self.assertEqual(profile["captured_packet_bytes_scanned"], 48)
        self.assertEqual(profile["packet_length_bytes"]["min"], 14)
        self.assertEqual(profile["packet_length_bytes"]["max"], 34)

    def test_zip_pcap_is_profiled_without_extraction(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pcap_bytes = (
                struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65_535, 1)
                + struct.pack("<IIII", 1, 0, 14, 14)
                + b"\x00" * 12
                + b"\x08\x06"
                + struct.pack("<IIII", 2, 0, 14, 14)
                + b"\x00" * 12
                + b"\x08\x06"
            )
            archive_path = root / "captures.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("capture.pcap", pcap_bytes)

            csv_path, text_path, rows = inspect_data(
                {"PUBLIC": archive_path},
                root / "reports",
                max_packets=1,
            )
            self.assertTrue(csv_path.is_file())
            self.assertTrue(text_path.is_file())
            self.assertEqual(rows[0]["status"], "partial_profile")
            self.assertEqual(rows[0]["packets_scanned"], 1)
            self.assertEqual(rows[0]["protocol_distribution"]["other_or_truncated"], 1)
            self.assertEqual(rows[0]["member_name"], "capture.pcap")
            self.assertIn(
                "dibatasi maksimal 1 paket per file",
                text_path.read_text(encoding="utf-8"),
            )

            _, full_scan_text_path, full_scan_rows = inspect_data(
                {"PUBLIC": archive_path},
                root / "full-scan-reports",
                max_packets=None,
            )
            self.assertEqual(full_scan_rows[0]["status"], "profiled")
            self.assertIn(
                "Semua paket classic PCAP dipindai",
                full_scan_text_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
