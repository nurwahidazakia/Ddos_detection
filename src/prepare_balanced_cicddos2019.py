"""Prepare a deterministic, balanced CIC-DDoS2019 development dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .prepare_csv_train_test import _load_label_mapping


DEFAULT_INPUT = Path("data/raw/cicddos2019_dataset.csv")
DEFAULT_OUTPUT = Path(
    "data/processed/cicflowmeter_harmonized/cicddos2019_balanced_training.csv"
)
DEFAULT_REPORT = Path("data/reports/cicddos2019_balanced_training.json")
EXCLUDED_COLUMNS = {
    "CWE Flag Count",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Seg Size Min",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
}
EXCLUDED_REASONS = {
    "CWE Flag Count": "flag name/semantics have not been reconciled with extractor",
    "Fwd Header Length": "negative int32-like overflow values in source",
    "Bwd Header Length": "negative int32-like overflow values in source",
    "Fwd Seg Size Min": "negative int32-like overflow values in source",
    "Init Fwd Win Bytes": "-1 sentinel is common in source",
    "Init Bwd Win Bytes": "-1 sentinel is common in source",
}
INDEX_COLUMN_CANDIDATES = {"", "Unnamed: 0"}
RANDOM_STATE = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV sumber tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            columns = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV kosong: {path}") from error
    if len(set(columns)) != len(columns):
        raise ValueError(f"Header CSV memiliki kolom duplikat: {path}")
    return columns


def _source_layout(path: Path) -> tuple[list[str], str, tuple[str, ...]]:
    columns = _read_header(path)
    index_columns = [column for column in columns if column in INDEX_COLUMN_CANDIDATES]
    if len(index_columns) != 1:
        raise ValueError(
            "CSV sumber harus memiliki tepat satu kolom indeks kosong/Unnamed: 0."
        )
    index_column = index_columns[0]
    required = {"Label", "Class"}
    missing = required - set(columns)
    if missing:
        raise ValueError(
            "Kolom label sumber tidak lengkap: " + ", ".join(sorted(missing))
        )
    feature_columns = tuple(
        column
        for column in columns
        if column not in required
        and column not in INDEX_COLUMN_CANDIDATES
        and column not in EXCLUDED_COLUMNS
    )
    if not feature_columns:
        raise ValueError("Tidak ada fitur numerik untuk dataset training.")
    if any(not column for column in feature_columns):
        raise ValueError("Nama fitur kosong tidak dapat digunakan.")
    return columns, index_column, feature_columns


def _validate_row(
    row: dict[str, str],
    *,
    line_number: int,
    feature_columns: tuple[str, ...],
    label_mapping: dict[str, int],
) -> tuple[str, str, dict[str, float]]:
    if None in row or any(value is None for value in row.values()):
        raise ValueError(f"Jumlah kolom tidak sesuai pada baris {line_number}.")
    raw_label = row["Label"].strip()
    label = raw_label.upper()
    if label not in label_mapping:
        raise ValueError(
            f"Label tidak ada di config/label_mapping.json pada baris "
            f"{line_number}: {raw_label!r}."
        )
    class_name = row["Class"].strip().casefold()
    if class_name not in {"attack", "benign"}:
        raise ValueError(
            f"Class harus Attack atau Benign pada baris {line_number}: {class_name!r}."
        )
    expected_class = "attack" if label_mapping[label] == -1 else "benign"
    if class_name != expected_class:
        raise ValueError(
            f"Class dan Label bertentangan pada baris {line_number}: "
            f"{class_name}/{raw_label}."
        )
    values: dict[str, float] = {}
    for column in feature_columns:
        raw_value = row[column].strip()
        if not raw_value:
            raise ValueError(
                f"Fitur kosong pada baris {line_number}: {column}."
            )
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Fitur bukan angka pada baris {line_number}: "
                f"{column}={raw_value!r}."
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                f"Fitur non-finite pada baris {line_number}: "
                f"{column}={raw_value!r}."
            )
        if value < 0:
            raise ValueError(
                f"Fitur negatif yang tidak ditinjau pada baris {line_number}: "
                f"{column}={raw_value!r}."
            )
        values[column] = value
    return class_name, raw_label, values


def _scan_source(
    path: Path,
    index_column: str,
    feature_columns: tuple[str, ...],
    label_mapping: dict[str, int],
) -> tuple[Counter[str], Counter[str], Counter[tuple[str, str]], int]:
    class_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    index_values: set[str] = set()
    duplicate_index_rows = 0
    excluded_negative_counts: Counter[tuple[str, str]] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for line_number, row in enumerate(reader, start=2):
            class_name, raw_label, _ = _validate_row(
                row,
                line_number=line_number,
                feature_columns=feature_columns,
                label_mapping=label_mapping,
            )
            source_index = row[index_column].strip()
            if source_index in index_values:
                duplicate_index_rows += 1
            else:
                index_values.add(source_index)
            class_counts[class_name] += 1
            subtype_counts[f"{class_name}:{raw_label}"] += 1
            for column in EXCLUDED_COLUMNS.intersection(row):
                raw_value = row[column].strip()
                if not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if math.isfinite(value) and value < 0:
                    excluded_negative_counts[(class_name, column)] += 1
    if not class_counts["benign"] or not class_counts["attack"]:
        raise ValueError(
            "CSV sumber harus memuat setidaknya satu baris BENIGN dan Attack."
        )
    return (
        class_counts,
        subtype_counts,
        excluded_negative_counts,
        duplicate_index_rows,
    )


def _allocate_attack_quotas(
    subtype_counts: Counter[str], target_count: int
) -> dict[str, int]:
    attack_counts = {
        key.removeprefix("attack:"): count
        for key, count in subtype_counts.items()
        if key.startswith("attack:")
    }
    total = sum(attack_counts.values())
    if target_count > total:
        raise ValueError(
            f"Jumlah attack {total} kurang dari target seimbang {target_count}."
        )
    exact = {
        label: count * target_count / total
        for label, count in attack_counts.items()
    }
    quotas = {label: math.floor(count) for label, count in exact.items()}
    remaining = target_count - sum(quotas.values())
    order = sorted(
        exact,
        key=lambda label: (-(exact[label] - quotas[label]), label),
    )
    for label in order[:remaining]:
        quotas[label] += 1
    if any(quotas[label] == 0 for label in attack_counts):
        raise ValueError(
            "Target seimbang terlalu kecil untuk mempertahankan semua subtype attack."
        )
    return quotas


def _sample_attack_indices(
    path: Path,
    *,
    quotas: dict[str, int],
    feature_columns: tuple[str, ...],
    label_mapping: dict[str, int],
    seed: int,
) -> set[str]:
    reservoirs: dict[str, list[str]] = {
        label: [] for label in quotas
    }
    seen: Counter[str] = Counter()
    randomizers = {
        label: random.Random(
            seed
            ^ int.from_bytes(
                hashlib.sha256(label.encode("utf-8")).digest()[:8], "big"
            )
        )
        for label in quotas
    }
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for line_number, row in enumerate(reader, start=2):
            class_name, raw_label, _ = _validate_row(
                row,
                line_number=line_number,
                feature_columns=feature_columns,
                label_mapping=label_mapping,
            )
            if class_name != "attack":
                continue
            reservoir = reservoirs[raw_label]
            quota = quotas[raw_label]
            seen[raw_label] += 1
            if len(reservoir) < quota:
                reservoir.append(str(line_number))
            else:
                replacement = randomizers[raw_label].randrange(seen[raw_label])
                if replacement < quota:
                    reservoir[replacement] = str(line_number)
    if any(len(reservoirs[label]) != quota for label, quota in quotas.items()):
        raise ValueError("Sampling attack tidak mencapai kuota subtype yang ditentukan.")
    return {index for reservoir in reservoirs.values() for index in reservoir}


def _write_balanced_csv(
    path: Path,
    temporary_path: Path,
    *,
    feature_columns: tuple[str, ...],
    label_mapping: dict[str, int],
    selected_attack_indices: set[str],
    expected_class_count: int,
) -> tuple[int, Counter[str], Counter[str]]:
    class_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    written_attack_indices: set[str] = set()
    output_columns = ["Unnamed: 0", *feature_columns, "Label"]
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        with temporary_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=output_columns)
            writer.writeheader()
            for line_number, row in enumerate(reader, start=2):
                class_name, raw_label, _ = _validate_row(
                    row,
                    line_number=line_number,
                    feature_columns=feature_columns,
                    label_mapping=label_mapping,
                )
                source_index = str(line_number)
                if class_name == "attack" and source_index not in selected_attack_indices:
                    continue
                if class_name == "attack":
                    if source_index in written_attack_indices:
                        raise ValueError(
                            f"Indeks attack terpilih muncul ganda: {source_index}."
                        )
                    written_attack_indices.add(source_index)
                writer.writerow(
                    {
                        "Unnamed: 0": source_index,
                        **{column: row[column].strip() for column in feature_columns},
                        "Label": raw_label,
                    }
                )
                class_counts[class_name] += 1
                subtype_counts[f"{class_name}:{raw_label}"] += 1
    if (
        class_counts["benign"] != expected_class_count
        or class_counts["attack"] != expected_class_count
    ):
        raise ValueError(
            f"Dataset hasil tidak seimbang: {dict(class_counts)}; "
            f"target tiap kelas {expected_class_count}."
        )
    if written_attack_indices != selected_attack_indices:
        raise ValueError("Sebagian indeks attack terpilih tidak ditulis ke output.")
    return sum(class_counts.values()), class_counts, subtype_counts


def prepare_balanced_dataset(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    if output_path.resolve() == report_path.resolve():
        raise ValueError("Path dataset dan laporan harus berbeda.")
    if output_path.exists() or report_path.exists():
        raise FileExistsError(
            f"Dataset atau laporan sudah ada: {output_path}, {report_path}."
        )

    _, index_column, feature_columns = _source_layout(input_path)
    label_mapping = _load_label_mapping()
    (
        class_counts,
        source_subtype_counts,
        excluded_negative_counts,
        duplicate_index_rows,
    ) = _scan_source(input_path, index_column, feature_columns, label_mapping)
    target_count = min(class_counts.values())
    quotas = _allocate_attack_quotas(source_subtype_counts, target_count)
    selected_attack_indices = _sample_attack_indices(
        input_path,
        quotas=quotas,
        feature_columns=feature_columns,
        label_mapping=label_mapping,
        seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_temporary: Path | None = None
    report_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            dataset_temporary = Path(temporary.name)
        row_count, written_class_counts, written_subtype_counts = _write_balanced_csv(
            input_path,
            dataset_temporary,
            feature_columns=feature_columns,
            label_mapping=label_mapping,
            selected_attack_indices=selected_attack_indices,
            expected_class_count=target_count,
        )
        report: dict[str, Any] = {
            "status": "provisional_balanced_development_dataset_not_independent",
            "input_file": str(input_path),
            "input_sha256": _sha256_file(input_path),
            "input_index_column": index_column or "(empty header)",
            "input_duplicate_index_rows": duplicate_index_rows,
            "output_index_semantics": "1-based physical CSV line number; original index is not unique",
            "output_file": str(output_path),
            "output_sha256": _sha256_file(dataset_temporary),
            "random_state": seed,
            "sampling": "all benign rows plus stratified attack-family reservoir sample without replacement",
            "source_rows_by_class": dict(sorted(class_counts.items())),
            "output_rows": row_count,
            "output_rows_by_class": dict(sorted(written_class_counts.items())),
            "source_rows_by_subtype": dict(sorted(source_subtype_counts.items())),
            "output_rows_by_subtype": dict(sorted(written_subtype_counts.items())),
            "feature_count": len(feature_columns),
            "features": list(feature_columns),
            "excluded_columns_and_reasons": EXCLUDED_REASONS,
            "excluded_negative_value_counts": {
                f"{class_name}:{column}": count
                for (class_name, column), count in sorted(
                    excluded_negative_counts.items()
                )
            },
            "dropped_rows": 0,
            "duplicated_rows_added": 0,
            "label_source_status": "pending_verification",
            "limitations": [
                "The source file identity and provenance are not independently verified.",
                "A balanced sample changes class prevalence and is for development only.",
                "The source has no capture/day IDs; random rows are not an independent test.",
                "Excluded anomalous features require source-level review before any later inclusion.",
                "No production accuracy or operational readiness is established by preparing this file.",
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=report_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            report_temporary = Path(temporary.name)
            temporary.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        dataset_temporary.replace(output_path)
        try:
            report_temporary.replace(report_path)
        except OSError:
            output_path.unlink(missing_ok=True)
            raise
    finally:
        if dataset_temporary is not None and dataset_temporary.exists():
            dataset_temporary.unlink()
        if report_temporary is not None and report_temporary.exists():
            report_temporary.unlink()
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Buat CSV CIC-DDoS2019 seimbang dengan sampling tanpa duplikasi; "
            "hasil tetap provisional sampai provenance dan test independen tersedia."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = prepare_balanced_dataset(
            args.input,
            args.output,
            args.report,
            seed=args.seed,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError, csv.Error) as error:
        print(f"Error: {error}")
        return 2
    print(
        f"Dataset seimbang dibuat: {report['output_rows']} baris, "
        f"{report['feature_count']} fitur, "
        f"{report['output_rows_by_class']}; "
        f"status={report['status']}. "
        f"File: {report['output_file']}; laporan: {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
