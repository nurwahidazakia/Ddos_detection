"""Profile delimited text files incrementally without exposing row contents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import sqlite3
import sys
import tempfile
import warnings
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .pcap_profiling import PCAPNG_MAGIC, is_pcap_signature, profile_pcap


LOGGER = logging.getLogger("data_profiling")
DELIMITER_CANDIDATES = ",;\t|"
SAMPLE_BYTES = 64 * 1024


def detect_encoding(sample: bytes, requested_encoding: str = "auto") -> str:
    """Choose an explicit supported encoding, preferring BOMs and strict UTF-8."""
    if requested_encoding != "auto":
        sample.decode(requested_encoding)
        return requested_encoding

    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith(b"\xff\xfe"):
        return "utf-16"
    if sample.startswith(b"\xfe\xff"):
        return "utf-16"

    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        try:
            sample.decode("cp1252")
        except UnicodeDecodeError as error:
            raise UnicodeError(
                "Encoding sampel bukan UTF-8 atau Windows-1252; tentukan --encoding secara eksplisit."
            ) from error
        return "cp1252"


def inspect_file_format(path: Path) -> tuple[str, str | None]:
    """Identify common capture signatures or determine whether a file is text."""
    with path.open("rb") as file_handle:
        sample = file_handle.read(SAMPLE_BYTES)

    if not sample:
        return "empty", None

    if is_pcap_signature(sample[:4]):
        format_name = "pcapng" if sample[:4] == PCAPNG_MAGIC else "classic PCAP"
        return "packet_capture", format_name

    if b"\x00" in sample:
        return "binary", None

    try:
        sample.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "non_utf8_text_or_binary", None

    return "text_candidate", None


def _read_csv_options(
    path: Path,
    requested_encoding: str = "auto",
    requested_delimiter: str = "auto",
    requested_header: str = "auto",
) -> tuple[str, int | None, str, bool]:
    with path.open("rb") as file_handle:
        sample_bytes = file_handle.read(SAMPLE_BYTES)
    encoding = detect_encoding(sample_bytes, requested_encoding)
    sample = sample_bytes.decode(encoding)

    if requested_delimiter == "auto":
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=DELIMITER_CANDIDATES)
            delimiter = dialect.delimiter
        except csv.Error as error:
            delimiter = _infer_delimiter(sample)
            if delimiter is None:
                raise ValueError(
                    "Delimiter tidak dapat dikenali dari sampel. "
                    "Tentukan --delimiter secara eksplisit."
                ) from error
    else:
        delimiter = "\t" if requested_delimiter == "tab" else requested_delimiter

    if requested_header == "auto":
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False
        rows = [
            row
            for row in csv.reader(sample.splitlines(), delimiter=delimiter)
            if row and any(value.strip() for value in row)
        ]
        if len(rows) > 1 and rows[0]:
            first_row_numeric = all(_is_numeric(value) for value in rows[0])
            following_rows_numeric = all(
                _is_numeric(value)
                for row in rows[1:21]
                if len(row) == len(rows[0])
                for value in row
            )
            if first_row_numeric and following_rows_numeric:
                has_header = False
    else:
        has_header = requested_header == "present"

    return delimiter, 0 if has_header else None, encoding, requested_header == "auto"


def _infer_delimiter(sample: str) -> str | None:
    lines = sample.splitlines()
    best_candidate: tuple[float, int, str] | None = None
    for candidate in DELIMITER_CANDIDATES:
        try:
            rows = [
                row
                for row in csv.reader(lines, delimiter=candidate)
                if row and any(value.strip() for value in row)
            ]
        except csv.Error:
            continue
        if not rows:
            continue
        counts = Counter(len(row) for row in rows)
        modal_columns, modal_rows = counts.most_common(1)[0]
        if modal_columns < 2:
            continue
        score = (modal_rows / len(rows), modal_columns, candidate)
        if best_candidate is None or score > best_candidate:
            best_candidate = score
    return best_candidate[2] if best_candidate else None


def _is_numeric(value: str) -> bool:
    try:
        float(value.strip())
    except ValueError:
        return False
    return True


def _column_candidates(columns: list[str]) -> tuple[list[str], list[str]]:
    time_candidates = []
    protocol_candidates = []
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_")
        if (
            "timestamp" in normalized
            or "datetime" in normalized
            or "date_time" in normalized
            or normalized in {"time", "date"}
            or normalized.endswith("_time")
        ):
            time_candidates.append(column)
        if "protocol" in normalized or normalized in {"proto", "ip_proto"}:
            protocol_candidates.append(column)
    return time_candidates, protocol_candidates


def _label_candidates(columns: list[str]) -> list[str]:
    candidates = []
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_")
        if (
            normalized in {"label", "class", "target", "attack", "is_attack"}
            or normalized.endswith("_label")
            or normalized.endswith("_class")
        ):
            candidates.append(column)
    return candidates


def _row_digest(row: tuple[Any, ...]) -> bytes:
    normalized = []
    for value in row:
        if pd.isna(value):
            normalized.append(None)
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            if math.isinf(value):
                value = "Infinity" if value > 0 else "-Infinity"
            elif value.is_integer():
                value = int(value)
        normalized.append(value if isinstance(value, (bool, int, float)) else str(value))
    serialized = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).digest()


def profile_csv(
    path: Path,
    chunk_size: int,
    encoding: str = "auto",
    delimiter: str = "auto",
    header: str = "auto",
) -> dict[str, Any]:
    """Profile a delimited text file using bounded-memory Pandas chunks."""
    detected_delimiter, detected_header, detected_encoding, header_is_inferred = _read_csv_options(
        path, encoding, delimiter, header
    )
    read_options = {
        "sep": detected_delimiter,
        "header": detected_header,
        "encoding": detected_encoding,
        "encoding_errors": "strict",
        "index_col": False,
        "on_bad_lines": "error",
    }

    columns = [str(column) for column in pd.read_csv(path, nrows=0, **read_options).columns]
    time_candidates, protocol_candidates = _column_candidates(columns)
    label_candidates = _label_candidates(columns)
    statistics: dict[str, dict[str, Any]] = {
        column: {
            "observed_dtypes": set(),
            "null_count": 0,
            "infinity_count": 0,
            "numeric_min": None,
            "numeric_max": None,
        }
        for column in columns
    }
    row_count = 0
    duplicate_rows = 0
    time_ranges: dict[str, dict[str, Any]] = {
        column: {"min": None, "max": None, "parsed_values": 0} for column in time_candidates
    }
    protocol_counts: dict[str, Counter[str]] = {
        column: Counter() for column in protocol_candidates
    }
    label_counts: dict[str, Counter[str]] = {column: Counter() for column in label_candidates}

    with tempfile.TemporaryDirectory(prefix="ddos-profile-") as temporary_directory:
        database_path = Path(temporary_directory) / "row_hashes.sqlite3"
        with closing(sqlite3.connect(database_path)) as duplicate_database:
            duplicate_database.execute("PRAGMA journal_mode=OFF")
            duplicate_database.execute("PRAGMA synchronous=OFF")
            duplicate_database.execute(
                "CREATE TABLE row_hashes (digest BLOB PRIMARY KEY) WITHOUT ROWID"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error", pd.errors.ParserWarning)
                for chunk in pd.read_csv(path, chunksize=chunk_size, **read_options):
                    row_count += len(chunk)
                    for original_column in chunk.columns:
                        column = str(original_column)
                        values = chunk[original_column]
                        column_stats = statistics[column]
                        column_stats["observed_dtypes"].add(str(values.dtype))
                        column_stats["null_count"] += int(values.isna().sum())

                        numeric_values = pd.to_numeric(values, errors="coerce")
                        infinite_values = numeric_values.isin([float("inf"), float("-inf")])
                        column_stats["infinity_count"] += int(infinite_values.sum())
                        finite_values = numeric_values[numeric_values.notna() & ~infinite_values]
                        if not finite_values.empty:
                            chunk_min = finite_values.min().item()
                            chunk_max = finite_values.max().item()
                            current_min = column_stats["numeric_min"]
                            current_max = column_stats["numeric_max"]
                            column_stats["numeric_min"] = (
                                chunk_min if current_min is None else min(current_min, chunk_min)
                            )
                            column_stats["numeric_max"] = (
                                chunk_max if current_max is None else max(current_max, chunk_max)
                            )

                    for column in time_candidates:
                        values = chunk[column].dropna()
                        if values.empty:
                            continue
                        if "epoch" in column.casefold() and is_numeric_dtype(values.dtype):
                            timestamps = pd.to_datetime(values, unit="s", errors="coerce", utc=True)
                        else:
                            timestamps = pd.to_datetime(
                                values, errors="coerce", utc=True, format="mixed"
                            )
                        timestamps = timestamps.dropna()
                        if timestamps.empty:
                            continue
                        current = time_ranges[column]
                        chunk_min = timestamps.min().isoformat()
                        chunk_max = timestamps.max().isoformat()
                        current["min"] = (
                            chunk_min if current["min"] is None else min(current["min"], chunk_min)
                        )
                        current["max"] = (
                            chunk_max if current["max"] is None else max(current["max"], chunk_max)
                        )
                        current["parsed_values"] += int(len(timestamps))

                    for column in protocol_candidates:
                        values = chunk[column].dropna().astype(str).str.strip()
                        counts = {
                            str(value): int(count)
                            for value, count in values.value_counts().items()
                        }
                        protocol_counts[column].update(counts)
                    for column in label_candidates:
                        values = chunk[column].dropna().astype(str).str.strip()
                        counts = {
                            str(value): int(count)
                            for value, count in values.value_counts().items()
                        }
                        label_counts[column].update(counts)

                    row_hashes = (
                        _row_digest(row) for row in chunk.itertuples(index=False, name=None)
                    )
                    cursor = duplicate_database.executemany(
                        "INSERT OR IGNORE INTO row_hashes(digest) VALUES (?)",
                        ((digest,) for digest in row_hashes),
                    )
                    duplicate_rows += len(chunk) - cursor.rowcount
            duplicate_database.commit()

    column_reports = []
    for column, column_stats in statistics.items():
        missing = column_stats["null_count"]
        numeric_min = column_stats["numeric_min"]
        numeric_max = column_stats["numeric_max"]
        column_reports.append(
            {
                "name": column,
                "observed_dtypes": sorted(column_stats["observed_dtypes"]),
                "null_count": missing,
                "null_percentage": round((missing / row_count) * 100, 4) if row_count else 0.0,
                "infinity_count": column_stats["infinity_count"],
                "numeric_min": _json_number(numeric_min),
                "numeric_max": _json_number(numeric_max),
            }
        )

    return {
        "status": "profiled",
        "format": "delimited_text",
        "encoding": detected_encoding,
        "delimiter": "TAB" if detected_delimiter == "\t" else detected_delimiter,
        "header_present": detected_header == 0,
        "header_inferred": detected_header == 0 and header_is_inferred,
        "header_detection": "inferred" if header_is_inferred else "configured",
        "row_count": row_count,
        "column_count": len(columns),
        "exact_duplicate_rows": duplicate_rows,
        "columns": column_reports,
        "time_ranges": time_ranges,
        "protocol_distribution": _value_distribution(protocol_counts),
        "label_candidates": label_candidates,
        "label_distribution": _value_distribution(label_counts),
    }


def _value_distribution(counts_by_column: dict[str, Counter[str]]) -> dict[str, Any]:
    distribution = {}
    for column, counts in counts_by_column.items():
        most_common = counts.most_common(50)
        distribution[column] = {
            "unique_values": len(counts),
            "value_counts": dict(most_common),
            "other_count": sum(count for _, count in counts.most_common()[50:]),
        }
    return distribution


def _json_number(value: Any) -> int | float | None:
    if value is None:
        return None
    result = value.item() if hasattr(value, "item") else value
    if isinstance(result, float) and not math.isfinite(result):
        return None
    return result


def profile_file(
    path: Path,
    chunk_size: int,
    max_packets: int | None = None,
    encoding: str = "auto",
    delimiter: str = "auto",
    header: str = "auto",
    decode_pcap: bool = False,
) -> dict[str, Any]:
    """Inspect one file and profile it only when it is UTF-8 text."""
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "name": path.name,
        "size_bytes": path.stat().st_size,
    }

    try:
        kind, description = inspect_file_format(path)
        result["detected_format"] = description or kind

        if kind == "packet_capture":
            if description == "classic PCAP":
                result.update(
                    profile_pcap(
                        path,
                        max_packets=max_packets,
                        decode_packets=decode_pcap,
                    )
                )
                result["detected_format"] = description
                result["message"] = (
                    "Metadata paket dihitung secara streaming; payload paket tidak dimuat "
                    "ke memori dan alamat jaringan tidak disimpan."
                )
            else:
                result.update(
                    {
                        "status": "identified_non_csv",
                        "message": (
                            "PCAPNG teridentifikasi. Profiler saat ini hanya membaca classic PCAP."
                        ),
                    }
                )
        elif kind == "binary":
            result.update(
                {
                    "status": "identified_non_csv",
                    "message": "Byte NUL terdeteksi; file biner tidak diproses sebagai CSV.",
                }
            )
        elif kind == "non_utf8_text_or_binary":
            result.update(
                {
                    "status": "unsupported_encoding",
                    "message": "Sampel bukan UTF-8 yang valid; encoding perlu dikonfirmasi.",
                }
            )
        elif kind == "empty":
            result.update({"status": "empty_file", "message": "File kosong."})
        else:
            result.update(profile_csv(path, chunk_size, encoding, delimiter, header))
    except (
        OSError,
        UnicodeError,
        ValueError,
        pd.errors.ParserError,
        pd.errors.ParserWarning,
    ) as error:
        LOGGER.error("Gagal memprofilkan %s: %s", path, error)
        result.update({"status": "error", "message": str(error)})

    return result


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted((path for path in input_path.iterdir() if path.is_file()), key=lambda p: p.name.lower())
    raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")


def run(
    input_path: Path,
    report_dir: Path,
    chunk_size: int,
    max_packets: int | None = None,
    encoding: str = "auto",
    delimiter: str = "auto",
    header: str = "auto",
) -> tuple[Path, dict[str, Any]]:
    files = collect_input_files(input_path)
    if not files:
        raise ValueError(f"Tidak ada file langsung di direktori: {input_path}")

    profiles = [
        profile_file(path, chunk_size, max_packets, encoding, delimiter, header)
        for path in files
    ]
    successfully_profiled = {"profiled", "partial_profile"}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.resolve()),
        "csv_chunk_size_rows": chunk_size,
        "csv_encoding_override": encoding,
        "csv_delimiter_override": delimiter,
        "csv_header_override": header,
        "file_count": len(profiles),
        "summary": {
            "profiled": sum(item["status"] in successfully_profiled for item in profiles),
            "partial_profiles": sum(item["status"] == "partial_profile" for item in profiles),
            "identified_non_csv": sum(item["status"] == "identified_non_csv" for item in profiles),
            "other_statuses": {
                status: sum(item["status"] == status for item in profiles)
                for status in sorted(
                    {item["status"] for item in profiles}
                    - successfully_profiled
                    - {"identified_non_csv"}
                )
            },
        },
        "files": profiles,
        "privacy_note": "Nilai baris sampel tidak disimpan dalam laporan ini.",
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_name = f"profile_{input_path.stem or input_path.name}_{timestamp}.json"
    report_path = report_dir / report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profil CSV/teks secara bertahap; deteksi file biner tanpa mengubahnya."
    )
    parser.add_argument("input", type=Path, help="Path ke file atau direktori yang berisi file input.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Jumlah baris per chunk untuk CSV (default: 100000).",
    )
    parser.add_argument(
        "--max-packets-per-file",
        type=int,
        default=None,
        help="Batasi pemindaian PCAP per file; laporan akan ditandai parsial.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "reports",
        help="Direktori keluaran laporan JSON.",
    )
    parser.add_argument(
        "--encoding",
        choices=("auto", "utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"),
        default="auto",
        help="Encoding CSV; auto memeriksa BOM lalu UTF-8/Windows-1252.",
    )
    parser.add_argument(
        "--delimiter",
        choices=("auto", ",", ";", "tab", "|"),
        default="auto",
        help="Delimiter CSV atau auto untuk deteksi dari sampel.",
    )
    parser.add_argument(
        "--header",
        choices=("auto", "present", "absent"),
        default="auto",
        help="Status header; auto memakai dugaan awal yang perlu diverifikasi.",
    )
    return parser.parse_args(argv)


def display_sample(path: Path, encoding: str, delimiter: str, header: str) -> None:
    detected_delimiter, detected_header, detected_encoding, _ = _read_csv_options(
        path, encoding, delimiter, header
    )
    sample = pd.read_csv(
        path,
        sep=detected_delimiter,
        header=detected_header,
        encoding=detected_encoding,
        encoding_errors="strict",
        on_bad_lines="error",
        nrows=5,
    )
    print(f"\nSampel 5 baris pertama: {path.name}")
    print(sample.to_string(index=False, max_cols=30, max_colwidth=80))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    if args.chunk_size < 1:
        LOGGER.error("--chunk-size harus lebih besar dari nol.")
        return 2
    if args.max_packets_per_file is not None and args.max_packets_per_file < 1:
        LOGGER.error("--max-packets-per-file harus lebih besar dari nol.")
        return 2

    try:
        report_path, report = run(
            args.input,
            args.report_dir,
            args.chunk_size,
            args.max_packets_per_file,
            args.encoding,
            args.delimiter,
            args.header,
        )
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2

    print(f"Laporan: {report_path}")
    print(f"Ringkasan: {json.dumps(report['summary'], ensure_ascii=False)}")
    for file_path in collect_input_files(args.input):
        file_report = next(item for item in report["files"] if item["path"] == str(file_path.resolve()))
        if file_report["status"] == "profiled" and file_report.get("format") == "delimited_text":
            try:
                display_sample(file_path, args.encoding, args.delimiter, args.header)
            except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
                LOGGER.error("Gagal menampilkan sampel %s: %s", file_path, error)
                return 2
            break
    if report["summary"]["profiled"] == 0:
        if report["summary"]["identified_non_csv"] == report["file_count"]:
            LOGGER.warning(
                "Tidak ditemukan CSV; semua input berhasil diidentifikasi sebagai format non-CSV."
            )
            return 0
        LOGGER.error("Tidak ada file teks/CSV yang berhasil diprofilkan.")
        return 2
    if report["summary"]["other_statuses"]:
        LOGGER.error("Satu atau lebih file memerlukan pemeriksaan format/encoding.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
