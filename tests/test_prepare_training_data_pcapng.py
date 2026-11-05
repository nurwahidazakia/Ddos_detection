import csv
import struct
import tempfile
import unittest
from pathlib import Path

from src.prepare_training_data import CSV_COLUMNS, prepare_training_data


def _block(block_type: int, body: bytes) -> bytes:
    block_length = len(body) + 12
    return (
        struct.pack("<II", block_type, block_length)
        + body
        + struct.pack("<I", block_length)
    )


def _write_pcapng(path: Path, *, corrupt_trailer: bool = False) -> None:
    section = _block(
        0x0A0D0D0A,
        struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1),
    )
    interface_options = struct.pack("<HHB3xHH", 9, 1, 6, 0, 0)
    interface = _block(1, struct.pack("<HHI", 1, 0, 65_535) + interface_options)
    frame = b"\x00" * 12 + b"\x08\x06"
    first_timestamp = 1_767_225_600 * 1_000_000
    packet_blocks = []
    for timestamp in (first_timestamp, first_timestamp + 1_000_000):
        body = struct.pack(
            "<IIIII",
            0,
            timestamp >> 32,
            timestamp & 0xFFFFFFFF,
            len(frame),
            len(frame),
        ) + frame
        packet_blocks.append(_block(6, body + b"\x00\x00"))
    capture = section + interface + b"".join(packet_blocks)
    if corrupt_trailer:
        capture = capture[:-1] + b"\x01"
    path.write_bytes(capture)


class PrepareTrainingDataPcapngTests(unittest.TestCase):
    def test_pcapng_extracts_one_second_windows_without_inventing_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.pcapng"
            output = root / "features.csv"
            _write_pcapng(capture)

            result = prepare_training_data(capture, output)

            self.assertEqual(result, {"capture_count": 1, "packets": 2, "windows": 2})
            with output.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, list(CSV_COLUMNS))
            self.assertEqual([row["label"] for row in rows], ["", ""])
            self.assertEqual([int(row["packets_total"]) for row in rows], [1, 1])
            self.assertEqual([int(row["other_transport_packets"]) for row in rows], [1, 1])

    def test_pcapng_invalid_block_trailer_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.pcapng"
            output = root / "features.csv"
            _write_pcapng(capture, corrupt_trailer=True)

            with self.assertRaisesRegex(ValueError, "panjang blok PCAPNG tidak cocok"):
                prepare_training_data(capture, output)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
