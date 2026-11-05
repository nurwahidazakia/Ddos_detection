"""Combine public labeled features and selected, ground-truth-labeled BPS PCAPs."""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
from pathlib import Path
from typing import Any

from .prepare_training_data import FEATURE_COLUMNS, _load_labels, prepare_training_data


OUTPUT_COLUMNS = (
    "source_capture",
    "source_dataset",
    "source_sha256",
    "window_start",
    *FEATURE_COLUMNS,
    "label",
)
PUBLIC_COLUMNS = {
    "source_capture",
    "source_sha256",
    "window_start",
    "label",
    *FEATURE_COLUMNS,
}


def prepare_training_dataset(
    public_features_path: Path,
    bps_captures_path: Path,
    bps_labels_path: Path,
    output_path: Path,
    window_seconds: int = 1,
    overwrite: bool = False,
    public_labels_path: Path | None = None,
) -> dict[str, Any]:
    """Create a unified feature CSV without including unlabeled BPS captures."""
    if public_labels_path is None:
        raise ValueError("Manifest ground truth publik terverifikasi wajib disediakan.")
    for path in (
        public_features_path,
        bps_captures_path,
        bps_labels_path,
        public_labels_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Input tidak ditemukan: {path}")
    if (
        not public_features_path.is_file()
        or not bps_labels_path.is_file()
        or not public_labels_path.is_file()
    ):
        raise ValueError("Input fitur dan manifest label harus berupa file.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti.")

    bps_labels = _load_labels(bps_labels_path, expected_source_dataset="bps")
    if not bps_labels:
        raise ValueError("Manifest label BPS tidak memuat capture.")
    invalid_labels = sorted(
        f"{name}={label.label}"
        for name, label in bps_labels.items()
        if label.label not in {"1", "-1"}
    )
    if invalid_labels:
        raise ValueError(
            "Label BPS harus konsisten: 1=normal dan -1=attack. Nilai tidak valid: "
            + ", ".join(invalid_labels)
        )
    public_labels = _load_labels(public_labels_path, expected_source_dataset="public")
    if not public_labels:
        raise ValueError("Manifest ground truth publik tidak memuat capture.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path: Path | None = None
    temporary_path: Path | None = None
    public_windows = 0
    public_captures: set[str] = set()
    bps_result: dict[str, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            suffix=".bps.tmp",
            delete=False,
        ) as intermediate_file:
            intermediate_path = Path(intermediate_file.name)
        intermediate_path.unlink()
        bps_result = prepare_training_data(
            bps_captures_path,
            intermediate_path,
            bps_labels_path,
            window_seconds,
            include_captures=set(bps_labels),
            expected_source_dataset="bps",
        )

        with public_features_path.open("r", encoding="utf-8-sig", newline="") as public_file:
            public_reader = csv.DictReader(public_file)
            missing_columns = sorted(PUBLIC_COLUMNS - set(public_reader.fieldnames or ()))
            if missing_columns:
                raise ValueError(
                    "Kolom fitur publik tidak ditemukan: " + ", ".join(missing_columns)
                )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=output_path.parent,
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                fieldnames: list[str] = list(OUTPUT_COLUMNS)
                writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
                writer.writeheader()
                for line_number, row in enumerate(public_reader, start=2):
                    label = (row["label"] or "").strip()
                    if label not in {"1", "-1"}:
                        raise ValueError(
                            f"Label fitur publik harus 1 atau -1 pada baris {line_number}."
                        )
                    source_capture = (row["source_capture"] or "").strip()
                    if not source_capture:
                        raise ValueError(
                            f"source_capture kosong pada baris {line_number} fitur publik."
                        )
                    if source_capture not in public_labels:
                        raise ValueError(
                            f"Capture publik tidak ditemukan pada manifest terverifikasi: "
                            f"{source_capture} (baris {line_number})."
                        )
                    if public_labels[source_capture].label != label:
                        raise ValueError(
                            f"Label fitur publik tidak cocok dengan manifest untuk "
                            f"{source_capture} pada baris {line_number}."
                        )
                    if (
                        (row["source_sha256"] or "").strip().casefold()
                        != public_labels[source_capture].sha256
                    ):
                        raise ValueError(
                            f"SHA-256 fitur publik tidak cocok dengan manifest untuk "
                            f"{source_capture} pada baris {line_number}."
                        )
                    public_captures.add(source_capture)
                    window_start = (row["window_start"] or "").strip()
                    if not window_start:
                        raise ValueError(
                            f"window_start kosong pada baris {line_number} fitur publik."
                        )
                    for feature in FEATURE_COLUMNS:
                        try:
                            value = float(row[feature])
                        except (TypeError, ValueError) as error:
                            raise ValueError(
                                f"Fitur {feature} tidak valid pada baris {line_number}."
                            ) from error
                        if not math.isfinite(value) or value < 0:
                            raise ValueError(
                                f"Fitur {feature} harus finite dan non-negatif "
                                f"pada baris {line_number}."
                            )
                    writer.writerow(
                        {
                            **row,
                            "source_capture": f"public::{source_capture}",
                            "source_dataset": "public",
                            "source_sha256": public_labels[source_capture].sha256,
                            "window_start": window_start,
                            "label": label,
                        }
                    )
                    public_windows += 1
                if public_windows == 0:
                    raise ValueError("CSV fitur publik tidak memiliki baris data.")
                missing_public_captures = sorted(set(public_labels) - public_captures)
                if missing_public_captures:
                    raise ValueError(
                        "Capture manifest publik tidak ditemukan pada fitur: "
                        + ", ".join(missing_public_captures)
                    )

                with intermediate_path.open(
                    "r", encoding="utf-8-sig", newline=""
                ) as bps_file:
                    bps_reader = csv.DictReader(bps_file)
                    for row in bps_reader:
                        writer.writerow(
                            {
                                "source_capture": f"bps::{row['source_capture']}",
                                "source_dataset": "bps",
                                "source_sha256": bps_labels[row["source_capture"]].sha256,
                                "window_start": row["window_start_utc"],
                                **{feature: row[feature] for feature in FEATURE_COLUMNS},
                                "label": row["label"],
                            }
                        )

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
            )
        temporary_path.replace(output_path)
    finally:
        for temporary in (intermediate_path, temporary_path):
            if temporary is not None and temporary.exists():
                temporary.unlink()

    assert bps_result is not None
    return {
        "public_windows": public_windows,
        "bps_windows": bps_result["windows"],
        "total_windows": public_windows + bps_result["windows"],
        "bps_captures": bps_result["capture_count"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Satukan fitur publik berlabel dan capture BPS yang label ground-truth-nya "
            "sudah tercatat di manifest."
        )
    )
    parser.add_argument(
        "--public-features",
        type=Path,
        default=Path("data/processed/training_features_labeled.csv"),
        help="CSV fitur publik berlabel.",
    )
    parser.add_argument(
        "--bps-captures",
        type=Path,
        default=Path("data/raw"),
        help="File atau direktori capture BPS.",
    )
    parser.add_argument(
        "--bps-labels",
        type=Path,
        default=Path("data/labels/bps_ground_truth_manifest.csv"),
        help="Manifest ground truth BPS dengan bukti, verifier, dan cakupan waktu UTC.",
    )
    parser.add_argument(
        "--public-labels",
        type=Path,
        default=Path("data/labels/public_ground_truth_manifest.csv"),
        help="Manifest ground truth publik terverifikasi yang terpisah dari BPS.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/training_dataset.csv"),
        help="CSV fitur gabungan untuk training.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=1,
        help="Durasi window PCAP BPS dalam detik (default: 1).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Izinkan mengganti output yang sudah ada."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = prepare_training_dataset(
            args.public_features,
            args.bps_captures,
            args.bps_labels,
            args.output,
            args.window_seconds,
            args.overwrite,
            args.public_labels,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 2
    print(
        f"Dataset siap: {result['total_windows']} window "
        f"({result['public_windows']} publik, {result['bps_windows']} BPS), "
        f"{result['bps_captures']} capture BPS. Output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
