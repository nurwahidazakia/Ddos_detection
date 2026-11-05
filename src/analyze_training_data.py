"""Summarize labeled window features and check whether grouped evaluation is viable."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FEATURE_COLUMNS = (
    "packets_total",
    "bytes_total",
    "ipv4_packets",
    "ipv6_packets",
    "tcp_packets",
    "udp_packets",
    "icmp_packets",
    "other_transport_packets",
    "tcp_syn",
    "tcp_ack",
    "tcp_rst",
    "tcp_fin",
    "unique_source_ips",
    "unique_destination_ips",
    "mean_packet_bytes",
    "max_packet_bytes",
)
LABEL_NAMES = {"1": "normal", "-1": "attack"}
REQUIRED_COLUMNS = {"source_capture", "window_start", "label", *FEATURE_COLUMNS}


def analyze_training_data(input_path: Path) -> dict[str, Any]:
    """Compute per-class feature summaries and capture-level evaluation checks."""
    if not input_path.is_file():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    values: dict[str, dict[str, list[float]]] = {
        label: {feature: [] for feature in FEATURE_COLUMNS} for label in LABEL_NAMES
    }
    capture_rows: dict[str, Counter[str]] = defaultdict(Counter)
    row_counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip()
            capture = (row.get("source_capture") or "").strip()
            if label not in LABEL_NAMES:
                raise ValueError(f"Label harus 1 atau -1 pada baris {line_number}.")
            if not capture:
                raise ValueError(f"source_capture kosong pada baris {line_number}.")

            parsed_features: dict[str, float] = {}
            for feature in FEATURE_COLUMNS:
                text_value = (row.get(feature) or "").strip()
                try:
                    value = float(text_value)
                except ValueError as error:
                    raise ValueError(
                        f"Nilai fitur {feature} tidak valid pada baris {line_number}."
                    ) from error
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Nilai fitur {feature} harus finite dan non-negatif "
                        f"pada baris {line_number}."
                    )
                parsed_features[feature] = value

            row_counts[label] += 1
            capture_rows[capture][label] += 1
            for feature, value in parsed_features.items():
                values[label][feature].append(value)

    if not sum(row_counts.values()):
        raise ValueError(f"Tidak ada baris fitur di dalam file: {input_path}")

    capture_counts = {
        label: sum(counts[label] > 0 for counts in capture_rows.values())
        for label in LABEL_NAMES
    }
    capture_label_confounded = (
        all(capture_counts[label] == 1 for label in LABEL_NAMES)
        and all(len(counts) == 1 for counts in capture_rows.values())
    )
    class_counts = {LABEL_NAMES[label]: row_counts[label] for label in LABEL_NAMES}
    warnings: list[str] = []
    if any(row_counts[label] == 0 for label in LABEL_NAMES):
        warnings.append("Kedua kelas (normal dan attack) harus tersedia untuk evaluasi supervised.")
    if any(capture_counts[label] < 2 for label in LABEL_NAMES):
        warnings.append(
            "Minimal dua source_capture independen per kelas diperlukan agar evaluasi "
            "train/test berbasis capture dapat dilakukan."
        )
    if capture_label_confounded:
        warnings.append(
            "Setiap source_capture hanya memuat satu label; pola kelas dapat tercampur "
            "dengan identitas capture."
        )

    nonzero_counts = [count for count in row_counts.values() if count]
    imbalance_ratio = (
        max(nonzero_counts) / min(nonzero_counts) if len(nonzero_counts) > 1 else None
    )
    feature_summary = {
        LABEL_NAMES[label]: {
            feature: {
                "count": len(feature_values),
                "min": min(feature_values) if feature_values else None,
                "median": statistics.median(feature_values) if feature_values else None,
                "mean": statistics.fmean(feature_values) if feature_values else None,
                "max": max(feature_values) if feature_values else None,
            }
            for feature, feature_values in label_values.items()
        }
        for label, label_values in values.items()
    }
    captures = {
        capture: {
            "rows": sum(counts.values()),
            "labels": {
                LABEL_NAMES[label]: counts[label]
                for label in LABEL_NAMES
                if counts[label]
            },
        }
        for capture, counts in sorted(capture_rows.items())
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": input_path.name,
        "summary": {
            "window_count": sum(row_counts.values()),
            "capture_count": len(capture_rows),
            "class_counts": class_counts,
            "class_imbalance_ratio": imbalance_ratio,
            "independent_captures_per_class": {
                LABEL_NAMES[label]: capture_counts[label] for label in LABEL_NAMES
            },
            "capture_label_confounded": capture_label_confounded,
            "grouped_evaluation_ready": (
                all(row_counts[label] > 0 for label in LABEL_NAMES)
                and all(capture_counts[label] >= 2 for label in LABEL_NAMES)
            ),
        },
        "captures": captures,
        "feature_summary_by_class": feature_summary,
        "warnings": warnings,
    }


def run(input_path: Path, output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    """Analyze a feature CSV and persist the JSON report atomically."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
        )

    report = analyze_training_data(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(report, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buat ringkasan EDA dan cek kesiapan evaluasi dataset fitur berlabel."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/training_features_labeled.csv"),
        help="CSV fitur berlabel yang dianalisis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/training_features_eda.json"),
        help="Path laporan JSON.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Izinkan mengganti laporan yang sudah ada."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run(args.input, args.output, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 2

    summary = report["summary"]
    print(
        f"EDA selesai: {summary['window_count']} window, {summary['capture_count']} capture. "
        f"Normal={summary['class_counts']['normal']}, "
        f"attack={summary['class_counts']['attack']}. "
        f"Evaluasi grouped siap: {summary['grouped_evaluation_ready']}. "
        f"Laporan: {args.output}"
    )
    for warning in report["warnings"]:
        print(f"Peringatan: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
