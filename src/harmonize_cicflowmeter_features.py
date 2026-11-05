"""Align CICFlowMeter feature names without assigning unverified labels."""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


NORMAL_ALIASES = {
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Fwd Packets Length Total",
    "Total Length of Bwd Packet": "Bwd Packets Length Total",
    "Average Packet Size": "Avg Packet Size",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init Fwd Win Bytes",
    "Bwd Init Win Bytes": "Init Bwd Win Bytes",
    "Fwd Act Data Pkts": "Fwd Act Data Packets",
}
IDENTIFIER_COLUMNS = {
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Timestamp",
    "Unnamed: 0",
}
LABEL_COLUMNS = {"Label", "Class"}
UNVERIFIED_FLAG_COLUMNS = {"CWR Flag Count", "CWE Flag Count"}
NORMAL_LABEL_PLACEHOLDER = "NeedManualLabel"
NORMAL_OUTPUT_NAME = "normal_features_unlabeled.csv"
ATTACK_OUTPUT_NAME = "attack_features_labeled.csv"


def _read_header(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV input tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            columns = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV kosong: {path}") from error
    if not columns or any(not column for column in columns):
        raise ValueError(f"Header CSV kosong atau tidak valid: {path}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"Nama kolom duplikat: {path}")
    return columns


def _feature_sources(
    columns: list[str], aliases: dict[str, str], path: Path
) -> dict[str, str]:
    sources: dict[str, str] = {}
    for column in columns:
        if column in IDENTIFIER_COLUMNS or column in LABEL_COLUMNS:
            continue
        if column in UNVERIFIED_FLAG_COLUMNS:
            continue
        feature = aliases.get(column, column)
        if feature in sources:
            raise ValueError(
                f"Alias fitur bertabrakan di {path.name}: "
                f"{sources[feature]} dan {column}."
            )
        sources[feature] = column
    if not sources:
        raise ValueError(f"Tidak ada kolom fitur yang dapat digunakan: {path}")
    return sources


def _convert_csv(
    input_path: Path,
    temporary_path: Path,
    feature_columns: tuple[str, ...],
    feature_sources: dict[str, str],
    *,
    is_normal: bool,
) -> dict[str, Any]:
    row_count = 0
    labels: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    negative_values: Counter[str] = Counter()
    missing_values: Counter[str] = Counter()
    nonfinite_values: Counter[str] = Counter()
    output_columns = ["source_file", *feature_columns, "Label", "Class"]

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Header CSV tidak ditemukan: {input_path}")
        missing_features = set(feature_sources.values()) - set(reader.fieldnames)
        if missing_features:
            raise ValueError(
                f"Kolom fitur hilang di {input_path.name}: "
                + ", ".join(sorted(missing_features))
            )
        required_labels = {"Label"} if is_normal else {"Label", "Class"}
        missing_labels = required_labels - set(reader.fieldnames)
        if missing_labels:
            raise ValueError(
                f"Kolom label hilang di {input_path.name}: "
                + ", ".join(sorted(missing_labels))
            )

        with temporary_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=output_columns)
            writer.writeheader()
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"Jumlah kolom tidak sesuai di {input_path.name}:{line_number}."
                    )

                label = row["Label"].strip()
                if not label:
                    raise ValueError(
                        f"Label kosong di {input_path.name}:{line_number}."
                    )
                if is_normal and label != NORMAL_LABEL_PLACEHOLDER:
                    raise ValueError(
                        f"Label normal harus tetap {NORMAL_LABEL_PLACEHOLDER!r}; "
                        f"ditemukan {label!r} di {input_path.name}:{line_number}."
                    )
                class_name = "" if is_normal else row["Class"].strip()
                if not is_normal and class_name != "Attack":
                    raise ValueError(
                        f"Dataset attack-only harus memiliki Class='Attack'; "
                        f"ditemukan {class_name!r} di {input_path.name}:{line_number}."
                    )

                output_row: dict[str, str] = {
                    "source_file": input_path.name,
                    "Label": label,
                    "Class": class_name,
                }
                for feature in feature_columns:
                    value = row[feature_sources[feature]].strip()
                    if not value:
                        missing_values[feature] += 1
                        output_row[feature] = value
                        continue
                    try:
                        numeric_value = float(value)
                    except ValueError as error:
                        raise ValueError(
                            f"Nilai fitur bukan angka di "
                            f"{input_path.name}:{line_number}, {feature}={value!r}."
                        ) from error
                    if not math.isfinite(numeric_value):
                        nonfinite_values[feature] += 1
                    elif numeric_value < 0:
                        negative_values[feature] += 1
                    output_row[feature] = value
                writer.writerow(output_row)
                labels[label] += 1
                if class_name:
                    classes[class_name] += 1
                row_count += 1

    if row_count == 0:
        raise ValueError(f"Tidak ada baris data di {input_path}.")
    return {
        "rows": row_count,
        "labels": dict(sorted(labels.items())),
        "classes": dict(sorted(classes.items())),
        "negative_values": dict(sorted(negative_values.items())),
        "missing_values": dict(sorted(missing_values.items())),
        "nonfinite_values": dict(sorted(nonfinite_values.items())),
    }


def harmonize_cicflowmeter_features(
    normal_csv: Path,
    attack_csv: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Write schema-aligned files while keeping unverified normal labels pending."""
    normal_columns = _read_header(normal_csv)
    attack_columns = _read_header(attack_csv)
    normal_sources = _feature_sources(normal_columns, NORMAL_ALIASES, normal_csv)
    attack_sources = _feature_sources(attack_columns, {}, attack_csv)

    normal_features = set(normal_sources)
    attack_features = set(attack_sources)
    if normal_features != attack_features:
        normal_only = sorted(normal_features - attack_features)
        attack_only = sorted(attack_features - normal_features)
        raise ValueError(
            "Skema fitur tidak sepadan setelah alias; "
            f"hanya normal={normal_only}, hanya attack={attack_only}."
        )
    feature_columns = tuple(
        feature
        for feature in attack_columns
        if feature in attack_sources
    )
    if not feature_columns:
        raise ValueError("Tidak ada fitur bersama setelah harmonisasi.")

    output_directory.mkdir(parents=True, exist_ok=True)
    normal_output = output_directory / NORMAL_OUTPUT_NAME
    attack_output = output_directory / ATTACK_OUTPUT_NAME
    if normal_output.exists() or attack_output.exists():
        raise FileExistsError(
            f"Output sudah ada di {output_directory}; pilih direktori output baru."
        )

    temporary_paths: list[Path] = []
    try:
        for _ in range(2):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=output_directory,
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_paths.append(Path(temporary.name))

        normal_stats = _convert_csv(
            normal_csv,
            temporary_paths[0],
            feature_columns,
            normal_sources,
            is_normal=True,
        )
        attack_stats = _convert_csv(
            attack_csv,
            temporary_paths[1],
            feature_columns,
            attack_sources,
            is_normal=False,
        )
        temporary_paths[0].replace(normal_output)
        temporary_paths[1].replace(attack_output)
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()

    return {
        "feature_count": len(feature_columns),
        "features": list(feature_columns),
        "excluded_unverified_flag_columns": sorted(
            (set(normal_columns) | set(attack_columns)) & UNVERIFIED_FLAG_COLUMNS
        ),
        "normal": {
            **normal_stats,
            "output": str(normal_output),
            "class_pending": True,
        },
        "attack": {**attack_stats, "output": str(attack_output)},
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Selaraskan nama fitur CICFlowMeter tanpa mengisi label normal "
            "yang belum tercatat pada manifest."
        )
    )
    parser.add_argument("--normal-csv", type=Path, required=True)
    parser.add_argument("--attack-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cicflowmeter_harmonized"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = harmonize_cicflowmeter_features(
            args.normal_csv, args.attack_csv, args.output_dir
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError, csv.Error) as error:
        print(f"Error: {error}")
        return 2
    print(
        f"Harmonisasi selesai: {result['feature_count']} fitur bersama. "
        f"Normal={result['normal']['rows']} baris (Class pending); "
        f"attack={result['attack']['rows']} baris. "
        f"Kolom flag yang dikecualikan: "
        f"{result['excluded_unverified_flag_columns']}. "
        f"Output: {result['normal']['output']}, {result['attack']['output']}"
    )
    quality_issues = (
        "negative_values",
        "missing_values",
        "nonfinite_values",
    )
    if any(
        result[dataset][issue]
        for dataset in ("normal", "attack")
        for issue in quality_issues
    ):
        print(
            "Peringatan: nilai negatif/kosong/non-finite dipertahankan "
            "untuk ditinjau; "
            f"normal={{'negative': {result['normal']['negative_values']}, "
            f"'missing': {result['normal']['missing_values']}, "
            f"'nonfinite': {result['normal']['nonfinite_values']}}}, "
            f"attack={{'negative': {result['attack']['negative_values']}, "
            f"'missing': {result['attack']['missing_values']}, "
            f"'nonfinite': {result['attack']['nonfinite_values']}}}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
