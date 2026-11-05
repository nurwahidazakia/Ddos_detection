"""Create a descriptive, label-free baseline report from window features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prepare_training_data import FEATURE_COLUMNS


REQUIRED_COLUMNS = {"source_capture", "window_start_utc", "label", *FEATURE_COLUMNS}
PERCENTILES = (50, 95, 99)


def _capture_family(capture: str) -> str:
    if capture.startswith("DATA"):
        return "DATA"
    if capture.startswith("SAT-"):
        return "SAT"
    return "other"


def _percentile(sorted_values: list[float], percentile: int) -> float:
    position = (len(sorted_values) - 1) * percentile / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1 - fraction) + sorted_values[upper_index] * fraction


def _summarize(values: list[float]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "zero_count": sum(value == 0 for value in ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def analyze_unlabeled_features(input_path: Path) -> dict[str, Any]:
    """Summarize unlabelled windows without assigning classes or alert thresholds."""
    if not input_path.is_file():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    values_by_group: dict[str, dict[str, list[float]]] = {}
    capture_stats: dict[str, dict[str, Any]] = {}
    seen_windows: set[tuple[str, str]] = set()
    window_count = 0
    blank_label_count = 0
    protocol_partition_checks = 0
    invalid_rows: list[str] = []
    first_window: str | None = None
    last_window: str | None = None

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            capture = (row.get("source_capture") or "").strip()
            timestamp = (row.get("window_start_utc") or "").strip()
            if not capture:
                raise ValueError(f"source_capture kosong pada baris {line_number}.")
            if not timestamp:
                raise ValueError(f"window_start_utc kosong pada baris {line_number}.")
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(
                    f"window_start_utc tidak valid pada baris {line_number}."
                ) from error
            if parsed_timestamp.tzinfo is None:
                raise ValueError(
                    f"window_start_utc harus mencantumkan zona waktu pada baris {line_number}."
                )
            timestamp = parsed_timestamp.astimezone(timezone.utc).isoformat()

            label = (row.get("label") or "").strip()
            if label:
                raise ValueError(
                    "Analisis ini hanya untuk fitur tanpa label; "
                    f"label ditemukan pada baris {line_number}."
                )
            blank_label_count += 1

            window_key = (capture, timestamp)
            if window_key in seen_windows:
                raise ValueError(
                    f"Window capture/timestamp duplikat pada baris {line_number}: "
                    f"{capture}, {timestamp}"
                )
            seen_windows.add(window_key)

            parsed_features: dict[str, float] = {}
            for feature in FEATURE_COLUMNS:
                try:
                    value = float(row.get(feature) or "")
                except ValueError as error:
                    raise ValueError(
                        f"Fitur {feature} tidak valid pada baris {line_number}."
                    ) from error
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Fitur {feature} harus finite dan non-negatif "
                        f"pada baris {line_number}."
                    )
                parsed_features[feature] = value

            packet_count = parsed_features["packets_total"]
            protocol_total = sum(
                parsed_features[feature]
                for feature in (
                    "tcp_packets",
                    "udp_packets",
                    "icmp_packets",
                    "other_transport_packets",
                )
            )
            if packet_count <= 0 or packet_count != protocol_total:
                invalid_rows.append(f"{line_number}:packet_protocol_total")
            if parsed_features["ipv4_packets"] + parsed_features["ipv6_packets"] > packet_count:
                invalid_rows.append(f"{line_number}:ip_version_total")
            if any(
                parsed_features[flag] > parsed_features["tcp_packets"]
                for flag in ("tcp_syn", "tcp_ack", "tcp_rst", "tcp_fin")
            ):
                invalid_rows.append(f"{line_number}:tcp_flag_total")
            if any(
                parsed_features[address_count] > packet_count
                for address_count in ("unique_source_ips", "unique_destination_ips")
            ):
                invalid_rows.append(f"{line_number}:unique_ip_total")
            expected_mean = parsed_features["bytes_total"] / packet_count if packet_count else 0
            if not math.isclose(
                parsed_features["mean_packet_bytes"],
                expected_mean,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                invalid_rows.append(f"{line_number}:mean_packet_bytes")

            family = _capture_family(capture)
            if family not in values_by_group:
                values_by_group[family] = {feature: [] for feature in FEATURE_COLUMNS}
            for feature, value in parsed_features.items():
                values_by_group[family][feature].append(value)

            stats = capture_stats.setdefault(
                capture,
                {
                    "family": family,
                    "windows": 0,
                    "packets_total": 0,
                    "bytes_total": 0,
                    "first_window_utc": timestamp,
                    "last_window_utc": timestamp,
                },
            )
            stats["windows"] += 1
            stats["packets_total"] += int(packet_count)
            stats["bytes_total"] += int(parsed_features["bytes_total"])
            stats["first_window_utc"] = min(stats["first_window_utc"], timestamp)
            stats["last_window_utc"] = max(stats["last_window_utc"], timestamp)
            first_window = timestamp if first_window is None else min(first_window, timestamp)
            last_window = timestamp if last_window is None else max(last_window, timestamp)
            protocol_partition_checks += 1
            window_count += 1

    if not window_count:
        raise ValueError(f"Tidak ada baris fitur di dalam file: {input_path}")
    if invalid_rows:
        raise ValueError(
            f"Invariant fitur gagal pada {len(invalid_rows)} baris; "
            "contoh: " + ", ".join(invalid_rows[:5])
        )

    overall_values = {
        feature: [
            value
            for group_values in values_by_group.values()
            for value in group_values[feature]
        ]
        for feature in FEATURE_COLUMNS
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": input_path.name,
        "method": {
            "purpose": "Descriptive baseline only; no labels, model predictions, or alert thresholds.",
            "percentiles": "Linear interpolation at p50, p95, and p99.",
            "capture_families": "Filename prefixes DATA and SAT- are descriptive grouping only.",
        },
        "summary": {
            "window_count": window_count,
            "capture_count": len(capture_stats),
            "unlabeled_rows": blank_label_count,
            "first_window_utc": first_window,
            "last_window_utc": last_window,
            "protocol_partition_checks": protocol_partition_checks,
            "quality_invariant_failures": 0,
        },
        "feature_summary": {
            feature: _summarize(feature_values)
            for feature, feature_values in overall_values.items()
        },
        "feature_summary_by_capture_family": {
            family: {
                "window_count": len(group_values[FEATURE_COLUMNS[0]]),
                "capture_count": sum(
                    stats["family"] == family for stats in capture_stats.values()
                ),
                "features": {
                    feature: _summarize(feature_values)
                    for feature, feature_values in group_values.items()
                },
            }
            for family, group_values in sorted(values_by_group.items())
        },
        "captures": dict(sorted(capture_stats.items())),
        "limitations": [
            "Without ground-truth labels this report does not evaluate detection performance.",
            "Percentiles are descriptive, not validated anomaly thresholds.",
            "Capture-family differences may reflect source or time domain shift, not attack behavior.",
        ],
    }


def run(input_path: Path, output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    """Analyze an unlabeled feature CSV and write the JSON report atomically."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
        )

    report = analyze_unlabeled_features(input_path)
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
        description="Buat baseline deskriptif tanpa label; tidak melatih atau mengevaluasi model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/training_features_week2_unlabeled.csv"),
        help="CSV fitur window tanpa label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/week3_unlabeled_baseline.json"),
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
    except (FileNotFoundError, FileExistsError, ValueError, OSError, csv.Error) as error:
        print(f"Error: {error}")
        return 2

    summary = report["summary"]
    print(
        f"Baseline deskriptif selesai: {summary['window_count']} window, "
        f"{summary['capture_count']} capture, "
        f"{summary['unlabeled_rows']} label kosong. "
        f"Tidak ada metrik deteksi. Laporan: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
