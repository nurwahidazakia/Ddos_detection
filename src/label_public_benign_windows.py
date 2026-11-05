"""Label public feature windows as benign only from verified time intervals."""

from __future__ import annotations

import argparse
import csv
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


CAPTURE_PREFIX = "SAT-03-11-2018_"
REQUIRED_FEATURE_COLUMNS = {"source_capture", "window_start_utc", "label"}
REQUIRED_INTERVAL_COLUMNS = {
    "source_capture",
    "start_utc",
    "end_utc",
    "evidence_reference",
    "review_status",
}
WINDOW_NS = 1_000_000_000


@dataclass(frozen=True)
class BenignInterval:
    start_ns: int
    end_ns: int
    evidence_reference: str


def _parse_utc(value: str, context: str) -> int:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Timestamp tidak valid pada {context}: {value}") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError(f"Timestamp harus menyertakan zona waktu UTC pada {context}.")
    utc_timestamp = timestamp.astimezone(timezone.utc)
    return int(utc_timestamp.timestamp()) * WINDOW_NS + utc_timestamp.microsecond * 1_000


def _load_benign_intervals(path: Path) -> dict[str, list[BenignInterval]]:
    intervals: dict[str, list[BenignInterval]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        missing_columns = REQUIRED_INTERVAL_COLUMNS - set(fieldnames)
        if missing_columns:
            raise ValueError(
                "Kolom manifest interval tidak lengkap: "
                + ", ".join(sorted(missing_columns))
            )

        for line_number, row in enumerate(reader, start=2):
            context = f"manifest baris {line_number}"
            capture_name = (row.get("source_capture") or "").strip()
            evidence_reference = (row.get("evidence_reference") or "").strip()
            review_status = (row.get("review_status") or "").strip().casefold()
            if not capture_name.startswith(CAPTURE_PREFIX):
                raise ValueError(
                    f"{context}: source_capture bukan capture publik yang didukung."
                )
            if not evidence_reference:
                raise ValueError(f"{context}: evidence_reference wajib diisi.")
            if review_status != "verified":
                raise ValueError(
                    f"{context}: review_status harus 'verified' sebelum memberi label."
                )

            start_ns = _parse_utc(row.get("start_utc") or "", context)
            end_ns = _parse_utc(row.get("end_utc") or "", context)
            if start_ns >= end_ns:
                raise ValueError(f"{context}: end_utc harus setelah start_utc.")
            intervals.setdefault(capture_name, []).append(
                BenignInterval(start_ns, end_ns, evidence_reference)
            )

    for capture_name, capture_intervals in intervals.items():
        capture_intervals.sort(key=lambda interval: interval.start_ns)
        for previous, current in zip(capture_intervals, capture_intervals[1:]):
            if current.start_ns < previous.end_ns:
                raise ValueError(
                    f"Interval benign tumpang tindih untuk {capture_name}."
                )
    if not intervals:
        raise ValueError("Manifest tidak berisi interval benign terverifikasi.")
    return intervals


def label_benign_windows(
    input_path: Path,
    intervals_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"CSV fitur tidak ditemukan: {input_path}")
    if not intervals_path.is_file():
        raise FileNotFoundError(f"Manifest interval tidak ditemukan: {intervals_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("File output harus berbeda dari file input.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
        )

    intervals = _load_benign_intervals(intervals_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    benign_labeled = 0
    public_windows = 0
    preserved_attack = 0

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames or []
            missing_columns = REQUIRED_FEATURE_COLUMNS - set(fieldnames)
            if missing_columns:
                raise ValueError(
                    "Kolom CSV fitur tidak lengkap: "
                    + ", ".join(sorted(missing_columns))
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
                    capture_name = (row.get("source_capture") or "").strip()
                    if not capture_name.startswith(CAPTURE_PREFIX):
                        writer.writerow(row)
                        continue

                    public_windows += 1
                    label = (row.get("label") or "").strip()
                    if label not in {"", "-1"}:
                        raise ValueError(
                            f"Label tidak didukung pada baris {line_number}: {label}"
                        )
                    matching_intervals = intervals.get(capture_name, [])
                    window_start_ns = _parse_utc(
                        row.get("window_start_utc") or "",
                        f"CSV fitur baris {line_number}",
                    )
                    window_end_ns = window_start_ns + WINDOW_NS
                    is_verified_benign = any(
                        window_start_ns >= interval.start_ns
                        and window_end_ns <= interval.end_ns
                        for interval in matching_intervals
                    )
                    if label == "-1":
                        if is_verified_benign:
                            raise ValueError(
                                f"Konflik label attack/benign pada baris {line_number}."
                            )
                        preserved_attack += 1
                        writer.writerow(row)
                        continue
                    if is_verified_benign:
                        row["label"] = "1"
                        benign_labeled += 1
                    writer.writerow(row)

        if not public_windows:
            raise ValueError(
                f"Tidak ada capture publik dengan awalan yang diharapkan: "
                f"{CAPTURE_PREFIX}"
            )
        if not benign_labeled:
            raise ValueError(
                "Tidak ada window publik yang sepenuhnya tercakup interval benign "
                "terverifikasi. Periksa nama capture dan rentang UTC."
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
        "benign_labeled_windows": benign_labeled,
        "preserved_attack_windows": preserved_attack,
        "remaining_unlabeled_windows": public_windows
        - benign_labeled
        - preserved_attack,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Labeli window benign hanya jika tercakup penuh oleh interval "
            "ground truth publik yang berstatus verified."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "training_features_public_attack_labeled.csv",
    )
    parser.add_argument(
        "--intervals",
        type=Path,
        required=True,
        help="CSV interval benign terverifikasi; gunakan template manifest publik.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "training_features_public_labeled.csv",
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
        result = label_benign_windows(
            args.input,
            args.intervals,
            args.output,
            args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Window publik diperiksa: {result['public_windows']}")
    print(f"Window benign berlabel: {result['benign_labeled_windows']}")
    print(f"Label attack yang dipertahankan: {result['preserved_attack_windows']}")
    print(f"Window publik tetap belum diketahui: {result['remaining_unlabeled_windows']}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
