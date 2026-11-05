"""Label documented CIC-DDoS2019 attack windows without guessing benign labels."""

from __future__ import annotations

import argparse
import csv
import struct
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .pcap_profiling import PCAP_MAGICS


CAPTURE_PREFIX = "SAT-03-11-2018_"
ATTACK_INTERVALS = (
    (2_600, 3_069, "NETBIOS"),
    (3_827, 4_297, "LDAP"),
    (4_560, 5_082, "MSSQL"),
    (5_739, 6_252, "UDP"),
    (7_871, 8_341, "SYN"),
)
WINDOW_NS = 1_000_000_000
REQUIRED_COLUMNS = {"source_capture", "window_start_utc", "label"}


def _read_first_packet_timestamp(path: Path) -> int:
    file_size = path.stat().st_size
    with path.open("rb") as source:
        global_header = source.read(24)
        if len(global_header) != 24:
            raise ValueError(f"Header PCAP global tidak lengkap: {path}")

        metadata = PCAP_MAGICS.get(global_header[:4])
        if metadata is None:
            raise ValueError(f"Format PCAP tidak didukung: {path}")
        byte_order, ticks_per_second, _precision = metadata
        major, minor, _offset, _accuracy, snaplen, _linktype = struct.unpack(
            f"{byte_order}HHiIII", global_header[4:]
        )
        if (major, minor) != (2, 4):
            raise ValueError(f"Versi PCAP tidak didukung: {major}.{minor} ({path})")

        record_header = source.read(16)
        if len(record_header) != 16:
            raise ValueError(f"Record paket pertama tidak lengkap: {path}")
        seconds, fraction, captured_length, original_length = struct.unpack(
            f"{byte_order}IIII", record_header
        )
        if fraction >= ticks_per_second:
            raise ValueError(f"Timestamp paket pertama tidak valid: {path}")
        if captured_length > original_length or (snaplen and captured_length > snaplen):
            raise ValueError(f"Panjang paket pertama tidak valid: {path}")
        if captured_length > file_size - source.tell():
            raise ValueError(f"Payload paket pertama terpotong: {path}")

    return seconds * 1_000_000_000 + fraction * (
        1_000_000_000 // ticks_per_second
    )


def _parse_utc_window(value: str, line_number: int) -> int:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"Timestamp window tidak valid pada baris {line_number}: {value}"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError(
            f"Timestamp window harus menyertakan zona waktu UTC pada baris {line_number}."
        )
    utc_timestamp = timestamp.astimezone(timezone.utc)
    epoch_seconds = int(utc_timestamp.timestamp())
    return epoch_seconds * 1_000_000_000 + utc_timestamp.microsecond * 1_000


def _attack_subtype(window_start_ns: int, capture_start_ns: int) -> str | None:
    for start_seconds, end_seconds, subtype in ATTACK_INTERVALS:
        attack_start_ns = capture_start_ns + start_seconds * WINDOW_NS
        attack_end_ns = capture_start_ns + end_seconds * WINDOW_NS
        if (
            window_start_ns >= attack_start_ns
            and window_start_ns + WINDOW_NS <= attack_end_ns
        ):
            return subtype
    return None


def label_feature_csv(
    input_path: Path,
    output_path: Path,
    first_capture_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"CSV fitur tidak ditemukan: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("File output harus berbeda dari file input.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
        )

    capture_start_ns = _read_first_packet_timestamp(first_capture_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    labeled_windows = 0
    public_windows = 0
    subtype_counts = {subtype: 0 for _, _, subtype in ATTACK_INTERVALS}
    seen_public_capture = False

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames or []
            missing_columns = REQUIRED_COLUMNS - set(fieldnames)
            if missing_columns:
                raise ValueError(
                    "Kolom CSV tidak lengkap: " + ", ".join(sorted(missing_columns))
                )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                dir=output_path.parent,
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
                writer.writeheader()
                for line_number, row in enumerate(reader, start=2):
                    if (row.get("label") or "").strip():
                        raise ValueError(
                            f"Input harus belum berlabel; nilai label ditemukan "
                            f"pada baris {line_number}."
                        )

                    capture_name = (row.get("source_capture") or "").strip()
                    if capture_name.startswith(CAPTURE_PREFIX):
                        seen_public_capture = True
                        public_windows += 1
                        window_start_ns = _parse_utc_window(
                            row.get("window_start_utc") or "", line_number
                        )
                        subtype = _attack_subtype(window_start_ns, capture_start_ns)
                        if subtype is not None:
                            row["label"] = "-1"
                            labeled_windows += 1
                            subtype_counts[subtype] += 1
                    writer.writerow(row)

        if not seen_public_capture:
            raise ValueError(
                f"Tidak ada capture dengan awalan yang diharapkan: {CAPTURE_PREFIX}"
            )
        if not labeled_windows or any(count == 0 for count in subtype_counts.values()):
            raise ValueError(
                "Rentang serangan publik tidak cocok dengan window input. "
                "Periksa sumber capture dan waktu mulai sebelum menggunakan label."
            )
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return {
        "public_windows": public_windows,
        "labeled_attack_windows": labeled_windows,
        "unlabeled_windows": public_windows - labeled_windows,
        "subtype_counts": subtype_counts,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Tandai hanya window serangan publik yang memiliki rentang waktu "
            "terdokumentasi; window lainnya tetap tanpa label."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "processed" / "training_features_week2_unlabeled.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "training_features_public_attack_labeled.csv",
    )
    parser.add_argument(
        "--first-capture",
        type=Path,
        default=project_root / "data" / "raw" / "SAT-03-11-2018_0",
        help="Capture pertama untuk menghitung offset waktu jadwal serangan.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Izinkan mengganti file output yang sudah ada.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = label_feature_csv(
            args.input,
            args.output,
            args.first_capture,
            args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Window publik diperiksa: {result['public_windows']}")
    print(f"Window serangan diberi label: {result['labeled_attack_windows']}")
    print(f"Window publik tetap belum diketahui: {result['unlabeled_windows']}")
    for subtype, count in result["subtype_counts"].items():
        print(f"{subtype}: {count}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
