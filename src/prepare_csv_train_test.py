"""Clean labeled flow CSVs and create per-schema train/test datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


LABELS = (1, -1)
INVALID_LABELS = {"", "LABEL", "NAN", "NONE", "NULL", "UNKNOWN", "N/A"}
IDENTIFIER_COLUMNS = {
    "flow_id",
    "src_ip",
    "source_ip",
    "dst_ip",
    "destination_ip",
    "timestamp",
    "unnamed_0",
    "index",
}
CHUNK_SIZE = 100_000
LABEL_MAPPING_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "label_mapping.json"
)


@dataclass
class CsvSource:
    path: Path
    columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    class_counts: Counter[int] = field(default_factory=Counter)
    raw_rows: int = 0
    dropped_rows: int = 0
    source_labels: set[str] = field(default_factory=set)


def _load_label_mapping(path: Path = LABEL_MAPPING_PATH) -> dict[str, int]:
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Gagal membaca pemetaan label: {path}") from error

    mapping: dict[str, int] = {}
    for group, labels in payload.items():
        if group.startswith("_"):
            continue
        if group != "normal" and not group.startswith("attack_"):
            raise ValueError(f"Grup label tidak dikenal di {path}: {group}")
        if not isinstance(labels, dict):
            raise ValueError(f"Grup label harus berupa objek: {group}")

        expected_value = 0 if group == "normal" else 1
        model_label = 1 if expected_value == 0 else -1
        for raw_label, value in labels.items():
            if raw_label.startswith("_"):
                continue
            if isinstance(value, bool) or value != expected_value:
                raise ValueError(
                    f"Nilai label {raw_label!r} tidak sesuai grup {group}."
                )
            normalized = str(raw_label).strip().upper()
            if not normalized:
                raise ValueError(f"Nama label kosong di grup {group}.")
            if normalized in mapping and mapping[normalized] != model_label:
                raise ValueError(f"Label memiliki pemetaan yang bertentangan: {raw_label}")
            mapping[normalized] = model_label

    if mapping.get("BENIGN") != 1 or not any(
        label == -1 for label in mapping.values()
    ):
        raise ValueError(
            f"Pemetaan harus memuat BENIGN=normal dan minimal satu label attack: {path}"
        )
    return mapping


def _normalize_column(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return normalized.casefold()


def _read_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, skipinitialspace=True)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV kosong: {path}") from error
    columns = tuple(_normalize_column(column) for column in header)
    if len(set(columns)) != len(columns):
        raise ValueError(f"Nama kolom duplikat setelah normalisasi: {path}")
    if "label" not in columns:
        raise ValueError(f"Kolom Label tidak ditemukan: {path}")
    feature_columns = [
        column for column in columns
        if column != "label" and column not in IDENTIFIER_COLUMNS
    ]
    if not feature_columns:
        raise ValueError(f"Tidak ada kolom fitur numerik untuk diproses: {path}")
    return columns


def _clean_chunks(
    path: Path,
    columns: tuple[str, ...],
    feature_columns: tuple[str, ...],
    label_mapping: dict[str, int],
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[tuple[pd.DataFrame, int, set[str]]]:
    for raw_chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        skipinitialspace=True,
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        raw_chunk.columns = list(columns)
        raw_count = len(raw_chunk)
        label_text = raw_chunk["label"].astype("string").str.strip()
        normalized_labels = label_text.str.upper()
        present_labels = set(normalized_labels.unique()) - INVALID_LABELS
        unknown_labels = sorted(present_labels - label_mapping.keys())
        if unknown_labels:
            raise ValueError(
                f"{path}: label tidak dipetakan untuk klasifikasi DDoS: "
                + ", ".join(unknown_labels)
            )
        valid_label = normalized_labels.isin(label_mapping)
        numeric_features = raw_chunk.loc[:, feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        valid_features = numeric_features.notna().all(axis=1)
        valid_rows = valid_label & valid_features
        source_labels = set(label_text.loc[valid_rows].unique().tolist())
        cleaned = numeric_features.loc[valid_rows].copy()
        cleaned["label"] = (
            normalized_labels.loc[valid_rows]
            .map(label_mapping)
            .to_numpy(dtype=np.int8)
        )
        yield cleaned, raw_count - len(cleaned), source_labels


def _scan_source(path: Path, label_mapping: dict[str, int]) -> CsvSource:
    columns = _read_header(path)
    feature_columns = tuple(
        column
        for column in columns
        if column != "label" and column not in IDENTIFIER_COLUMNS
    )
    source = CsvSource(path, columns, feature_columns)
    for cleaned, dropped, source_labels in _clean_chunks(
        path, columns, feature_columns, label_mapping
    ):
        source.raw_rows += len(cleaned) + dropped
        source.dropped_rows += dropped
        labels, counts = np.unique(
            cleaned["label"].to_numpy(dtype=np.int8), return_counts=True
        )
        for label, count in zip(labels, counts):
            source.class_counts[int(label)] += int(count)
        source.source_labels.update(source_labels)
    if not sum(source.class_counts.values()):
        raise ValueError(f"Tidak ada baris berlabel valid setelah cleaning: {path}")
    return source


def _has_both_classes(counts: Counter[int]) -> bool:
    return all(counts[label] > 0 for label in LABELS)


def _group_split(
    sources: list[CsvSource],
    test_size: float,
    seed: int,
) -> tuple[set[Path], set[Path], dict[int, np.ndarray]]:
    if len(sources) == 1:
        source = sources[0]
        if not _has_both_classes(source.class_counts):
            raise ValueError(
                f"{source.path.name}: train/test memerlukan label normal dan attack."
            )
        rng = np.random.default_rng(seed)
        test_positions: dict[int, np.ndarray] = {}
        for label in LABELS:
            count = source.class_counts[label]
            if count < 2:
                raise ValueError(
                    f"{source.path.name}: minimal dua baris per kelas diperlukan "
                    "untuk split train/test."
                )
            test_count = min(max(1, math.ceil(count * test_size)), count - 1)
            test_positions[label] = np.sort(
                rng.choice(count, size=test_count, replace=False)
            )
        return set(), {source.path}, test_positions

    test_count = min(
        max(1, math.ceil(len(sources) * test_size)),
        len(sources) - 1,
    )
    rng = random.Random(seed)
    source_by_path = {source.path: source for source in sources}
    paths = sorted(source_by_path, key=str)
    for _ in range(1_000):
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        test_paths = set(shuffled[:test_count])
        train_paths = set(paths) - test_paths
        train_counts: Counter[int] = Counter()
        test_counts: Counter[int] = Counter()
        for path in train_paths:
            train_counts.update(source_by_path[path].class_counts)
        for path in test_paths:
            test_counts.update(source_by_path[path].class_counts)
        if _has_both_classes(train_counts) and _has_both_classes(test_counts):
            return train_paths, test_paths, {}
    raise ValueError(
        "Split berbasis file tidak dapat menempatkan kedua kelas di train dan "
        "test. Tambahkan file sumber berlabel yang mewakili kedua kelas."
    )


def _membership_mask(positions: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if not len(selected):
        return np.zeros(len(positions), dtype=bool)
    offsets = np.searchsorted(selected, positions)
    in_range = offsets < len(selected)
    matched = np.zeros(len(positions), dtype=bool)
    matched[in_range] = selected[offsets[in_range]] == positions[in_range]
    return matched


def _class_counts(
    sources: list[CsvSource],
    paths: set[Path],
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for source in sources:
        if source.path in paths:
            counts.update(source.class_counts)
    return counts


def _write_schema(
    sources: list[CsvSource],
    train_paths: set[Path],
    test_paths: set[Path],
    test_positions: dict[int, np.ndarray],
    output_directory: Path,
    schema_number: int,
    seed: int,
    label_mapping: dict[str, int],
    overwrite: bool,
) -> dict[str, Any]:
    feature_columns = sources[0].feature_columns
    train_counts = (
        Counter(
            {
                label: sources[0].class_counts[label]
                - len(test_positions[label])
                for label in LABELS
            }
        )
        if test_positions
        else _class_counts(sources, train_paths)
    )
    if not _has_both_classes(train_counts):
        raise ValueError("Train harus memiliki baris dari kelas normal dan attack.")
    balanced_count = min(train_counts.values())

    rng = np.random.default_rng(seed + schema_number)
    train_positions: dict[int, np.ndarray] = {}
    for label in LABELS:
        available_count = train_counts[label]
        if test_positions:
            available_positions = np.setdiff1d(
                np.arange(sources[0].class_counts[label], dtype=np.int64),
                test_positions[label],
                assume_unique=True,
            )
            train_positions[label] = np.sort(
                rng.choice(available_positions, size=balanced_count, replace=False)
            )
        else:
            train_positions[label] = np.sort(
                rng.choice(available_count, size=balanced_count, replace=False)
            )

    train_output = output_directory / f"schema_{schema_number:02d}_train.csv"
    test_output = output_directory / f"schema_{schema_number:02d}_test.csv"
    if not overwrite and (train_output.exists() or test_output.exists()):
        raise FileExistsError(
            f"Output schema {schema_number:02d} sudah ada; gunakan --overwrite."
        )
    train_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    train_temporary: Path | None = None
    test_temporary: Path | None = None
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
        train_temporary, test_temporary = temporary_paths

        seen_train: Counter[int] = Counter()
        seen_test: Counter[int] = Counter()
        train_written = False
        test_written = False
        columns = (*feature_columns, "label")
        for source in sources:
            if source.path not in train_paths and source.path not in test_paths:
                continue
            for cleaned, _, _ in _clean_chunks(
                source.path,
                source.columns,
                source.feature_columns,
                label_mapping,
            ):
                row_count = len(cleaned)
                labels = cleaned["label"].to_numpy(dtype=np.int8)
                train_mask = np.zeros(row_count, dtype=bool)
                test_mask = np.zeros(row_count, dtype=bool)
                is_grouped_test = source.path in test_paths and not test_positions
                is_grouped_train = source.path in train_paths
                for label in LABELS:
                    row_indexes = np.flatnonzero(labels == label)
                    if not len(row_indexes):
                        continue
                    if is_grouped_test:
                        test_mask[row_indexes] = True
                        seen_test[label] += len(row_indexes)
                    elif is_grouped_train:
                        positions = seen_train[label] + np.arange(
                            len(row_indexes), dtype=np.int64
                        )
                        train_mask[row_indexes] = _membership_mask(
                            positions, train_positions[label]
                        )
                        seen_train[label] += len(row_indexes)
                    else:
                        positions = seen_test[label] + np.arange(
                            len(row_indexes), dtype=np.int64
                        )
                        test_mask[row_indexes] = _membership_mask(
                            positions, test_positions[label]
                        )
                        train_positions_for_file = train_positions[label]
                        train_mask[row_indexes] = (
                            ~test_mask[row_indexes]
                            & _membership_mask(positions, train_positions_for_file)
                        )
                        seen_test[label] += len(row_indexes)

                if train_mask.any():
                    train_rows = cleaned.loc[train_mask, list(columns)].copy()
                    train_rows.insert(0, "source_file", source.path.name)
                    train_rows.to_csv(
                        train_temporary,
                        mode="a",
                        index=False,
                        header=not train_written,
                    )
                    train_written = True
                if test_mask.any():
                    test_rows = cleaned.loc[test_mask, list(columns)].copy()
                    test_rows.insert(0, "source_file", source.path.name)
                    test_rows.to_csv(
                        test_temporary,
                        mode="a",
                        index=False,
                        header=not test_written,
                    )
                    test_written = True

        if not train_written or not test_written:
            raise ValueError("Split menghasilkan train atau test kosong.")
        if not overwrite and (train_output.exists() or test_output.exists()):
            raise FileExistsError(
                f"Output schema {schema_number:02d} sudah ada; gunakan --overwrite."
            )
        train_temporary.replace(train_output)
        test_temporary.replace(test_output)
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()

    test_counts = (
        Counter(
            {
                label: len(test_positions[label])
                for label in LABELS
            }
        )
        if test_positions
        else _class_counts(sources, test_paths)
    )
    return {
        "schema": schema_number,
        "files": [source.path.name for source in sources],
        "feature_count": len(feature_columns),
        "features": list(feature_columns),
        "split_strategy": (
            "stratified_rows_single_source"
            if test_positions
            else "source_file_group"
        ),
        "input_rows": sum(source.raw_rows for source in sources),
        "dropped_rows": sum(source.dropped_rows for source in sources),
        "clean_rows": sum(sum(source.class_counts.values()) for source in sources),
        "train_rows": balanced_count * len(LABELS),
        "test_rows": sum(test_counts.values()),
        "train_class_counts": {"normal": balanced_count, "attack": balanced_count},
        "test_class_counts": {
            "normal": test_counts[1],
            "attack": test_counts[-1],
        },
        "dropped_per_file": {
            source.path.name: source.dropped_rows for source in sources
        },
        "source_labels": sorted(
            {label for source in sources for label in source.source_labels}
        ),
        "train_output": str(train_output),
        "test_output": str(test_output),
    }


def prepare_csv_train_test(
    input_directory: Path,
    output_directory: Path,
    test_size: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create cleaned, per-schema CSV splits from labeled source CSVs."""
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Direktori input tidak ditemukan: {input_directory}")
    if not 0 < test_size < 1:
        raise ValueError("test_size harus lebih besar dari 0 dan lebih kecil dari 1.")

    input_paths = sorted(
        (path for path in input_directory.glob("*.csv") if path.is_file()),
        key=lambda path: path.name.casefold(),
    )
    if not input_paths:
        raise ValueError(f"Tidak ada file CSV di dalam: {input_directory}")

    label_mapping = _load_label_mapping()
    groups: dict[tuple[str, ...], list[CsvSource]] = {}
    for path in input_paths:
        source = _scan_source(path, label_mapping)
        groups.setdefault(source.columns, []).append(source)

    for schema_number in range(1, len(groups) + 1):
        train_output = output_directory / f"schema_{schema_number:02d}_train.csv"
        test_output = output_directory / f"schema_{schema_number:02d}_test.csv"
        if not overwrite and (train_output.exists() or test_output.exists()):
            raise FileExistsError(
                f"Output schema {schema_number:02d} sudah ada; gunakan --overwrite."
            )

    schemas: list[dict[str, Any]] = []
    for schema_number, sources in enumerate(groups.values(), start=1):
        train_paths, test_paths, test_positions = _group_split(
            sources, test_size, seed + schema_number
        )
        result = _write_schema(
            sources,
            train_paths,
            test_paths,
            test_positions,
            output_directory,
            schema_number,
            seed,
            label_mapping,
            overwrite,
        )
        schemas.append(result)
    return {"schema_count": len(schemas), "schemas": schemas}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bersihkan CSV berlabel, pisahkan per schema, buat train/test, "
            "dan seimbangkan kelas training dengan undersampling."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Direktori CSV input berlabel (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/csv_train_test"),
        help="Direktori output (default: data/processed/csv_train_test).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi test (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed split dan sampling (default: 42).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Izinkan mengganti berkas output yang sudah ada.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = prepare_csv_train_test(
            args.input_dir,
            args.output_dir,
            args.test_size,
            args.seed,
            args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError, csv.Error) as error:
        print(f"Error: {error}")
        return 2
    for schema in result["schemas"]:
        print(
            f"Schema {schema['schema']:02d}: sumber={schema['files']}; "
            f"training={schema['train_class_counts']} ({schema['train_rows']} baris), "
            f"test={schema['test_class_counts']} ({schema['test_rows']} baris); "
            f"baris dibuang={schema['dropped_rows']}; "
            f"split={schema['split_strategy']}; "
            f"output={schema['train_output']}, {schema['test_output']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
