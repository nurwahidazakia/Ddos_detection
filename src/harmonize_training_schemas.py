"""Combine schema-specific flow datasets using their shared semantic features."""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALIASES = {
    "dst_port": "destination_port",
    "tot_fwd_pkts": "total_fwd_packets",
    "tot_bwd_pkts": "total_backward_packets",
    "totlen_fwd_pkts": "total_length_of_fwd_packets",
    "totlen_bwd_pkts": "total_length_of_bwd_packets",
    "fwd_pkt_len_max": "fwd_packet_length_max",
    "fwd_pkt_len_min": "fwd_packet_length_min",
    "fwd_pkt_len_mean": "fwd_packet_length_mean",
    "fwd_pkt_len_std": "fwd_packet_length_std",
    "bwd_pkt_len_max": "bwd_packet_length_max",
    "bwd_pkt_len_min": "bwd_packet_length_min",
    "bwd_pkt_len_mean": "bwd_packet_length_mean",
    "bwd_pkt_len_std": "bwd_packet_length_std",
    "flow_byts_s": "flow_bytes_s",
    "fwd_iat_tot": "fwd_iat_total",
    "bwd_iat_tot": "bwd_iat_total",
    "fwd_header_len": "fwd_header_length",
    "bwd_header_len": "bwd_header_length",
    "fwd_pkts_s": "fwd_packets_s",
    "bwd_pkts_s": "bwd_packets_s",
    "pkt_len_min": "min_packet_length",
    "pkt_len_max": "max_packet_length",
    "pkt_len_mean": "packet_length_mean",
    "pkt_len_std": "packet_length_std",
    "pkt_len_var": "packet_length_variance",
    "fin_flag_cnt": "fin_flag_count",
    "syn_flag_cnt": "syn_flag_count",
    "rst_flag_cnt": "rst_flag_count",
    "psh_flag_cnt": "psh_flag_count",
    "ack_flag_cnt": "ack_flag_count",
    "urg_flag_cnt": "urg_flag_count",
    "ece_flag_cnt": "ece_flag_count",
    "pkt_size_avg": "average_packet_size",
    "fwd_seg_size_avg": "avg_fwd_segment_size",
    "bwd_seg_size_avg": "avg_bwd_segment_size",
    "fwd_byts_b_avg": "fwd_avg_bytes_bulk",
    "fwd_pkts_b_avg": "fwd_avg_packets_bulk",
    "fwd_blk_rate_avg": "fwd_avg_bulk_rate",
    "bwd_byts_b_avg": "bwd_avg_bytes_bulk",
    "bwd_pkts_b_avg": "bwd_avg_packets_bulk",
    "bwd_blk_rate_avg": "bwd_avg_bulk_rate",
    "subflow_fwd_pkts": "subflow_fwd_packets",
    "subflow_fwd_byts": "subflow_fwd_bytes",
    "subflow_bwd_pkts": "subflow_bwd_packets",
    "subflow_bwd_byts": "subflow_bwd_bytes",
    "init_fwd_win_byts": "init_win_bytes_forward",
    "init_bwd_win_byts": "init_win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
}

SCHEMA_FILES = {
    1: ("schema_01_train.csv", "schema_01_test.csv"),
    2: ("schema_02_train.csv", "schema_02_test.csv"),
    3: ("schema_03_train.csv", "schema_03_test.csv"),
}
SCHEMA_SPLIT_FILES = {
    "train": {
        1: ("schema_01_train.csv",),
        2: ("schema_02_train.csv",),
        3: (),
    },
    "test": {
        1: ("schema_01_test.csv",),
        2: (),
        3: ("schema_03_test.csv",),
    },
}
OUTPUT_COLUMNS_PREFIX = ("source_file",)
OUTPUT_LABEL_COLUMN = "label"
CHUNK_SIZE = 50_000


def _canonicalize_feature(name: str) -> str:
    return ALIASES.get(name, name)


def _feature_columns(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            columns = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV kosong: {path}") from error
    if "source_file" not in columns or "label" not in columns:
        raise ValueError(f"Kolom source_file dan label wajib ada: {path}")
    feature_columns = tuple(
        column for column in columns if column not in {"source_file", "label"}
    )
    canonical = tuple(_canonicalize_feature(column) for column in feature_columns)
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"Alias fitur bertabrakan di {path.name}.")
    return feature_columns


def _load_schema_features(input_directory: Path) -> tuple[str, ...]:
    schema_features: dict[int, tuple[str, ...]] = {}
    for schema, filenames in SCHEMA_FILES.items():
        present_paths = [input_directory / name for name in filenames]
        for path in present_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Input dataset tidak ditemukan: {path}")
        file_features = [_feature_columns(path) for path in present_paths]
        canonical_files = [
            tuple(_canonicalize_feature(name) for name in features)
            for features in file_features
        ]
        if any(features != canonical_files[0] for features in canonical_files[1:]):
            raise ValueError(
                f"Fitur train/test Schema {schema:02d} tidak sama; "
                "periksa input sebelum harmonisasi."
            )
        schema_features[schema] = canonical_files[0]

    common = set(schema_features[1])
    for features in schema_features.values():
        common.intersection_update(features)
    ordered_common = tuple(name for name in schema_features[1] if name in common)
    if not ordered_common:
        raise ValueError("Tidak ada fitur bersama antar schema.")
    return ordered_common


def _write_combined_split(
    input_directory: Path,
    output_path: Path,
    split: str,
    features: tuple[str, ...],
    overwrite: bool,
) -> tuple[int, dict[int, int], set[str]]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    row_count = 0
    label_counts = {1: 0, -1: 0}
    source_files: set[str] = set()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        first_chunk = True
        for filenames in SCHEMA_SPLIT_FILES[split].values():
            for filename in filenames:
                path = input_directory / filename
                original_features = _feature_columns(path)
                keep_features = [
                    name for name in original_features
                    if _canonicalize_feature(name) in features
                ]
                rename_map = {
                    original: _canonicalize_feature(original)
                    for original in keep_features
                }

                for chunk_number, chunk in enumerate(
                    pd.read_csv(
                        path,
                        encoding="utf-8-sig",
                        usecols=["source_file", "label", *keep_features],
                        chunksize=CHUNK_SIZE,
                    ),
                    start=1,
                ):
                    labels = pd.to_numeric(chunk["label"], errors="coerce")
                    if labels.isna().any() or not labels.isin({1, -1}).all():
                        raise ValueError(
                            f"Label tidak valid di {path.name}, chunk {chunk_number}."
                        )
                    source_column = chunk["source_file"].astype("string").str.strip()
                    if source_column.isna().any() or source_column.eq("").any():
                        raise ValueError(
                            f"source_file kosong di {path.name}, chunk {chunk_number}."
                        )

                    values = chunk.loc[:, keep_features].apply(
                        pd.to_numeric, errors="coerce"
                    )
                    numeric_values = values.to_numpy(dtype=np.float64)
                    if not np.isfinite(numeric_values).all():
                        raise ValueError(
                            f"Fitur nonnumerik atau non-finite di {path.name}, "
                            f"chunk {chunk_number}."
                        )
                    values = values.rename(columns=rename_map).loc[:, features]
                    output_chunk = pd.concat(
                        [
                            source_column.rename("source_file"),
                            values,
                            labels.astype("int8").rename("label"),
                        ],
                        axis=1,
                    )
                    output_chunk.to_csv(
                        temporary_path,
                        mode="a",
                        index=False,
                        header=first_chunk,
                    )
                    first_chunk = False
                    row_count += len(output_chunk)
                    unique_labels, counts = np.unique(
                        labels.to_numpy(dtype=np.int8), return_counts=True
                    )
                    for label, count in zip(unique_labels, counts):
                        label_counts[int(label)] += int(count)
                    source_files.update(source_column.unique().tolist())

        if first_chunk:
            raise ValueError(f"Tidak ada baris untuk output split {split}.")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return row_count, label_counts, source_files


def harmonize_training_schemas(
    input_directory: Path,
    output_directory: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create combined, semantically aligned train/test CSVs."""
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Direktori input tidak ditemukan: {input_directory}")

    features = _load_schema_features(input_directory)
    train_output = output_directory / "harmonized_train.csv"
    test_output = output_directory / "harmonized_test.csv"
    if not overwrite and (train_output.exists() or test_output.exists()):
        raise FileExistsError(
            f"Output harmonisasi sudah ada di {output_directory}; gunakan --overwrite."
        )

    train_rows, train_counts, train_sources = _write_combined_split(
        input_directory, train_output, "train", features, overwrite
    )
    test_rows, test_counts, test_sources = _write_combined_split(
        input_directory, test_output, "test", features, overwrite
    )
    overlap = train_sources & test_sources
    if overlap:
        raise ValueError(
            "Kebocoran source_file antar train/test: " + ", ".join(sorted(overlap))
        )
    if train_counts[1] != train_counts[-1]:
        raise ValueError(
            "Data training gabungan tidak seimbang setelah harmonisasi: "
            f"normal={train_counts[1]}, attack={train_counts[-1]}."
        )
    return {
        "feature_count": len(features),
        "features": list(features),
        "train_rows": train_rows,
        "train_class_counts": {
            "normal": train_counts[1],
            "attack": train_counts[-1],
        },
        "test_rows": test_rows,
        "test_class_counts": {
            "normal": test_counts[1],
            "attack": test_counts[-1],
        },
        "train_source_files": sorted(train_sources),
        "test_source_files": sorted(test_sources),
        "train_output": str(train_output),
        "test_output": str(test_output),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Selaraskan fitur bersama antar schema dan gabungkan dataset train/test "
            "tanpa menempatkan source_file yang sama di kedua split."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/csv_train_test"),
        help="Folder CSV per-schema yang sudah dibuat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/harmonized"),
        help="Folder output train/test gabungan.",
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
        result = harmonize_training_schemas(
            args.input_dir, args.output_dir, args.overwrite
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError, csv.Error) as error:
        print(f"Error: {error}")
        return 2
    print(
        f"Harmonisasi selesai: {result['feature_count']} fitur bersama. "
        f"Train={result['train_rows']} baris "
        f"{result['train_class_counts']}; "
        f"test={result['test_rows']} baris {result['test_class_counts']}. "
        f"Output: {result['train_output']}, {result['test_output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
