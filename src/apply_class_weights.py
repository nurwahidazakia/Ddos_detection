"""Add inverse-frequency sample weights without changing training labels or rows."""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = {"1", "-1"}
WEIGHT_COLUMN = "sample_weight"


def apply_class_weights(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a copy of a two-class CSV with equal total sample weight per class."""
    if not input_path.is_file():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("File input dan output harus berbeda.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    counts: Counter[str] = Counter()
    weights: dict[str, float] = {}
    row_count = 0
    fieldnames: list[str] | None = None
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or "label" not in reader.fieldnames:
                raise ValueError("CSV input harus memiliki kolom 'label'.")
            if WEIGHT_COLUMN in reader.fieldnames:
                raise ValueError(
                    f"CSV input sudah memiliki kolom '{WEIGHT_COLUMN}'; "
                    "gunakan dataset tanpa bobot sebagai input."
                )

            fieldnames = list(reader.fieldnames)
            for line_number, row in enumerate(reader, start=2):
                label = (row.get("label") or "").strip()
                if label not in LABELS:
                    raise ValueError(f"Label harus 1 atau -1 pada baris {line_number}.")
                counts[label] += 1
            total_rows = sum(counts.values())
            if not total_rows:
                raise ValueError(f"Tidak ada baris data di dalam file: {input_path}")
            if any(counts[label] == 0 for label in LABELS):
                raise ValueError("Data harus memiliki baris dari kedua kelas: 1 dan -1.")

            weights = {
                label: total_rows / (len(LABELS) * counts[label])
                for label in LABELS
            }

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=output_path.parent,
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                writer = csv.DictWriter(
                    temporary_file,
                    fieldnames=[*fieldnames, WEIGHT_COLUMN],
                    extrasaction="raise",
                )
                writer.writeheader()
                with input_path.open("r", encoding="utf-8-sig", newline="") as source:
                    second_reader = csv.DictReader(source)
                    for line_number, row in enumerate(second_reader, start=2):
                        label = (row.get("label") or "").strip()
                        if label not in LABELS:
                            raise ValueError(
                                f"Label harus 1 atau -1 pada baris {line_number}."
                            )
                        row["label"] = label
                        writer.writerow(
                            {**row, WEIGHT_COLUMN: format(weights[label], ".12g")}
                        )
                        row_count += 1

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    class_weight_totals = {
        label: counts[label] * weights[label] for label in sorted(LABELS)
    }
    return {
        "rows": row_count,
        "class_counts": dict(counts),
        "class_sample_weights": weights,
        "class_weight_totals": class_weight_totals,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tambahkan bobot sample berimbang tanpa mengubah label atau jumlah "
            "baris dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/training_dataset.csv"),
        help="CSV training tanpa sample_weight.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/training_dataset_weighted.csv"),
        help="CSV output dengan kolom sample_weight.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Izinkan mengganti file output."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = apply_class_weights(args.input, args.output, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, OSError, csv.Error) as error:
        print(f"Error: {error}")
        return 2

    print(
        f"Bobot kelas ditambahkan: {result['rows']} baris tidak berubah. "
        f"Jumlah label={result['class_counts']}; "
        f"total bobot per kelas={result['class_weight_totals']}. "
        f"Output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
