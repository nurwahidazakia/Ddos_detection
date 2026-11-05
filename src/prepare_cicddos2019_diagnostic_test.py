"""Prepare a clearly marked cross-source diagnostic CSV, not a final test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib

from .harmonize_cicflowmeter_features import NORMAL_ALIASES
from .prepare_csv_train_test import _load_label_mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMAL = Path(
    "data/processed/cicflowmeter_harmonized/normal_features_unlabeled.csv"
)
DEFAULT_ATTACK = Path("data/raw/attack/cicddos2019_attack.csv")
DEFAULT_TRAINING = Path(
    "data/processed/cicflowmeter_harmonized/cicddos2019_balanced_training.csv"
)
DEFAULT_MODEL = Path("models/cicddos2019_development_candidate.joblib")
DEFAULT_OUTPUT = Path(
    "data/processed/cicflowmeter_harmonized/"
    "cicddos2019_cross_source_diagnostic_test.csv"
)
DEFAULT_REPORT = Path(
    "data/reports/cicddos2019_cross_source_diagnostic_test.json"
)
DEFAULT_ROWS_PER_CLASS = 20_000
DEFAULT_RANDOM_STATE = 20261105
METADATA_COLUMNS = (
    "Label",
    "CaptureID",
    "SourceDataset",
    "ExtractorVersion",
    "LabelEvidence",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_columns(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        try:
            header = next(csv.reader(source))
        except StopIteration as error:
            raise ValueError(f"CSV kosong: {path}") from error
    columns = [column.strip().replace("\ufeff", "") for column in header]
    normalized = [column.casefold() for column in columns]
    if not columns or any(not column for column in columns):
        raise ValueError(f"Header CSV kosong atau tidak valid: {path}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Nama kolom duplikat: {path}")
    return columns, dict(zip(normalized, columns, strict=True))


def _feature_sources(path: Path, features: tuple[str, ...]) -> dict[str, str]:
    _columns, normalized = _normalized_columns(path)
    aliases = {
        source.casefold(): target
        for source, target in NORMAL_ALIASES.items()
    }
    requested = {feature.casefold(): feature for feature in features}
    sources: dict[str, str] = {}
    for column in normalized.values():
        canonical = aliases.get(column.casefold(), column)
        feature = requested.get(canonical.casefold())
        if feature is None:
            continue
        if feature in sources:
            raise ValueError(
                f"Alias fitur bertabrakan di {path.name}: {feature}."
            )
        sources[feature] = column
    missing = [feature for feature in features if feature not in sources]
    if missing:
        raise ValueError(
            f"Fitur model tidak ditemukan di {path.name}: " + ", ".join(missing)
        )
    return sources


def _row_fingerprint(
    row: dict[str, str], features: tuple[str, ...], label: str | None = None
) -> str:
    digest = hashlib.sha256()
    for feature in features:
        raw_value = row.get(feature, "").strip()
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Fitur training tidak numerik: {feature}={raw_value!r}."
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"Fitur training non-finite: {feature}.")
        digest.update(value.hex().encode("ascii"))
        digest.update(b"\0")
    fingerprint_label = row["Label"] if label is None else label
    digest.update(fingerprint_label.strip().upper().encode("utf-8"))
    return digest.hexdigest()


def _training_attack_fingerprints(
    path: Path,
    features: tuple[str, ...],
    label_mapping: dict[str, int],
) -> set[str]:
    columns, normalized = _normalized_columns(path)
    label_column = normalized.get("label")
    if label_column is None or any(feature not in columns for feature in features):
        raise ValueError(f"Skema training tidak sesuai model: {path}")

    fingerprints: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            label = row[label_column].strip().upper()
            if label_mapping.get(label) == -1:
                fingerprints.add(_row_fingerprint(row, features))
    if not fingerprints:
        raise ValueError("Tidak ditemukan baris attack di dataset training.")
    return fingerprints


def _reservoir_sample(
    path: Path,
    feature_sources: dict[str, str],
    features: tuple[str, ...],
    rows_per_class: int,
    random_state: int,
    *,
    label_mapping: dict[str, int],
    excluded_attack_fingerprints: set[str] | None = None,
    class_column: str | None = None,
    class_value: str | None = None,
    is_normal: bool,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    columns, normalized = _normalized_columns(path)
    label_column = normalized.get("label")
    if label_column is None:
        raise ValueError(f"Kolom Label tidak ditemukan: {path}")
    if class_column is not None and class_column.casefold() not in normalized:
        raise ValueError(f"Kolom {class_column!r} tidak ditemukan: {path}")
    resolved_class_column = (
        normalized[class_column.casefold()] if class_column is not None else None
    )
    if resolved_class_column is not None and class_value is None:
        raise ValueError("Nilai kelas sumber harus ditentukan.")
    if any(column not in columns for column in feature_sources.values()):
        raise ValueError(f"Kolom fitur tidak cocok: {path}")

    rng = random.Random(random_state)
    selected: list[dict[str, str]] = []
    counts = {
        "source_rows": 0,
        "eligible_rows": 0,
        "duplicate_training_rows_excluded": 0,
        "duplicate_source_rows_excluded": 0,
        "rows_with_invalid_features_excluded": 0,
    }
    source_fingerprints: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            counts["source_rows"] += 1
            if resolved_class_column is not None:
                if row[resolved_class_column].strip().casefold() != str(
                    class_value
                ).casefold():
                    continue

            source_label = row[label_column].strip()
            normalized_label = source_label.upper()
            if is_normal:
                output_label = "BENIGN"
            else:
                if label_mapping.get(normalized_label) != -1:
                    continue
                output_label = source_label

            raw_features = {
                feature: row[source]
                for feature, source in feature_sources.items()
            }
            valid_features = 0
            for value in raw_features.values():
                try:
                    if math.isfinite(float(value)):
                        valid_features += 1
                except ValueError:
                    continue
            if valid_features != len(raw_features):
                counts["rows_with_invalid_features_excluded"] += 1
                continue

            fingerprint = _row_fingerprint(
                raw_features, features, label=normalized_label
            )
            if (
                not is_normal
                and excluded_attack_fingerprints is not None
                and fingerprint in excluded_attack_fingerprints
            ):
                counts["duplicate_training_rows_excluded"] += 1
                continue
            if not is_normal and fingerprint in source_fingerprints:
                counts["duplicate_source_rows_excluded"] += 1
                continue

            counts["eligible_rows"] += 1
            if not is_normal:
                source_fingerprints.add(fingerprint)
            record = {
                feature: raw_features[feature].strip()
                for feature in features
            }
            record.update(
                {
                    "Label": output_label,
                    "CaptureID": (
                        "CIC-IDS2017-Monday-WorkingHours"
                        if is_normal
                        else "unknown-no-capture-day-id"
                    ),
                    "SourceDataset": (
                        "CIC-IDS2017"
                        if is_normal
                        else "CIC-DDoS2019-identified-source-CSV"
                    ),
                    "ExtractorVersion": (
                        "CICFlowMeter 4.0"
                        if is_normal
                        else "CICFlowMeter-V3 (source CSV documentation)"
                    ),
                    "LabelEvidence": (
                        "User-confirmed normal; official CIC-IDS2017 documentation "
                        "states Monday is benign-only; exact local capture identity "
                        "checksum remains pending"
                        if is_normal
                        else "Inherited source Label and Class=Attack; user-confirmed; "
                        "source capture/day ID is absent"
                    ),
                }
            )

            if len(selected) < rows_per_class:
                selected.append(record)
            else:
                replacement = rng.randrange(counts["eligible_rows"])
                if replacement < rows_per_class:
                    selected[replacement] = record

    return selected, counts


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        suffix=".tmp",
        delete=False,
    ) as temporary:
        return Path(temporary.name)


def prepare_diagnostic_test_set(
    normal_path: Path = DEFAULT_NORMAL,
    attack_path: Path = DEFAULT_ATTACK,
    training_path: Path = DEFAULT_TRAINING,
    model_path: Path = DEFAULT_MODEL,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    *,
    rows_per_class: int = DEFAULT_ROWS_PER_CLASS,
    random_state: int = DEFAULT_RANDOM_STATE,
    overwrite: bool = False,
) -> dict[str, Any]:
    if rows_per_class < 1:
        raise ValueError("Jumlah baris per kelas harus lebih besar dari nol.")
    for path, description in (
        (normal_path, "CSV normal"),
        (attack_path, "CSV attack"),
        (training_path, "CSV training"),
        (model_path, "model kandidat"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} tidak ditemukan: {path}")

    protected_paths = {
        normal_path.resolve(),
        attack_path.resolve(),
        training_path.resolve(),
        model_path.resolve(),
    }
    if output_path.resolve() in protected_paths or report_path.resolve() in protected_paths:
        raise ValueError("Output tidak boleh menimpa input, training, atau model.")
    if output_path.resolve() == report_path.resolve():
        raise ValueError("Path dataset test dan laporan harus berbeda.")
    if not overwrite and (output_path.exists() or report_path.exists()):
        raise FileExistsError(
            "Dataset test atau laporan sudah ada; gunakan --overwrite untuk mengganti."
        )

    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or not payload.get("features"):
        raise ValueError("Payload model tidak memiliki daftar fitur yang valid.")
    features = tuple(payload["features"])
    label_mapping = _load_label_mapping()
    training_fingerprints = _training_attack_fingerprints(
        training_path, features, label_mapping
    )

    normal_features = _feature_sources(normal_path, features)
    attack_features = _feature_sources(attack_path, features)
    normal_rows, normal_counts = _reservoir_sample(
        normal_path,
        normal_features,
        features,
        rows_per_class,
        random_state,
        label_mapping=label_mapping,
        class_column="Class" if "class" in _normalized_columns(normal_path)[1] else None,
        class_value="",
        is_normal=True,
    )
    attack_rows, attack_counts = _reservoir_sample(
        attack_path,
        attack_features,
        features,
        rows_per_class,
        random_state + 1,
        label_mapping=label_mapping,
        excluded_attack_fingerprints=training_fingerprints,
        class_column="Class" if "class" in _normalized_columns(attack_path)[1] else None,
        class_value="Attack",
        is_normal=False,
    )
    if len(normal_rows) < rows_per_class:
        raise ValueError(
            f"Baris normal eligible hanya {len(normal_rows)}; "
            f"dibutuhkan {rows_per_class}."
        )
    if len(attack_rows) < rows_per_class:
        raise ValueError(
            f"Baris attack unik di luar training hanya {len(attack_rows)}; "
            f"dibutuhkan {rows_per_class}."
        )

    records = normal_rows + attack_rows
    random.Random(random_state + 2).shuffle(records)
    output_temporary = _temporary_path(output_path)
    report_temporary: Path | None = None
    try:
        output_columns = [*features, *METADATA_COLUMNS]
        with output_temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=output_columns,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)

        class_counts: dict[str, int] = {}
        for record in records:
            label = record["Label"]
            normalized = label.upper()
            class_name = "normal" if label_mapping.get(normalized) == 1 else "attack"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        report: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "diagnostic_cross_source_not_independent_capture_test",
            "purpose": "Exercise the offline prediction runner and expose cross-source/extractor problems; not production-readiness evidence.",
            "output_file": str(output_path),
            "output_sha256": _sha256_file(output_temporary),
            "rows": len(records),
            "rows_per_class_requested": rows_per_class,
            "class_counts": class_counts,
            "feature_count": len(features),
            "features": list(features),
            "random_state": random_state,
            "sampling": "Deterministic reservoir sampling without replacement; equal diagnostic class counts.",
            "sources": {
                "normal": {
                    "file": str(normal_path),
                    "sha256": _sha256_file(normal_path),
                    "rows_seen": normal_counts["source_rows"],
                    "eligible_rows": normal_counts["eligible_rows"],
                    "sampled_rows": len(normal_rows),
                    "capture_id": "CIC-IDS2017-Monday-WorkingHours",
                    "label": "BENIGN",
                    "label_evidence": "User-confirmed and supported by official CIC-IDS2017 Monday description; exact local capture file identity/checksum remains pending.",
                    "extractor": "CICFlowMeter 4.0",
                },
                "attack": {
                    "file": str(attack_path),
                    "sha256": _sha256_file(attack_path),
                    "rows_seen": attack_counts["source_rows"],
                    "eligible_rows_after_training_exclusion": attack_counts["eligible_rows"],
                    "sampled_rows": len(attack_rows),
                    "duplicate_feature_rows_matching_training_excluded": attack_counts[
                        "duplicate_training_rows_excluded"
                    ],
                    "duplicate_feature_rows_within_source_excluded": attack_counts[
                        "duplicate_source_rows_excluded"
                    ],
                    "capture_id": "unknown-no-capture-day-id",
                    "label": "source attack subtype labels",
                    "label_evidence": "Source Label and Class=Attack; user-confirmed; exact source release and capture/day ID are not recorded.",
                    "extractor": "CICFlowMeter-V3-derived source CSV, as documented for CIC-DDoS2019.",
                },
            },
            "training_exclusion": {
                "file": str(training_path),
                "sha256": _sha256_file(training_path),
                "attack_feature_label_fingerprints": len(training_fingerprints),
                "exact_attack_feature_and_label_matches_excluded": attack_counts[
                    "duplicate_training_rows_excluded"
                ],
                "note": "Feature-identical attack rows with the same label were excluded. This does not prove capture/day independence; train CSV lacks capture IDs and timestamps.",
            },
            "limitations": [
                "This is not an independent capture/day test. Attack rows come from the same source CSV family used for training, and capture/day IDs are absent.",
                "Normal features were extracted with CICFlowMeter 4.0 while attack source features are CICFlowMeter-V3-derived.",
                "Class counts were deliberately balanced for a diagnostic and do not represent operational prevalence.",
                "The normal source capture identity is not independently checksum-verified.",
                "Do not tune the model or threshold on this set and do not claim production accuracy from it.",
            ],
            "model_file": str(model_path),
            "model_sha256": _sha256_file(model_path),
        }

        report_temporary = _temporary_path(report_path)
        report_temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_temporary.replace(output_path)
        report_temporary.replace(report_path)
        return report
    finally:
        if output_temporary.exists():
            output_temporary.unlink()
        if report_temporary is not None and report_temporary.exists():
            report_temporary.unlink()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a labeled cross-source diagnostic test CSV, not a final independent test."
    )
    parser.add_argument("--normal", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--attack", type=Path, default=DEFAULT_ATTACK)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--rows-per-class", type=int, default=DEFAULT_ROWS_PER_CLASS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = prepare_diagnostic_test_set(
            args.normal,
            args.attack,
            args.training,
            args.model,
            args.output,
            args.report,
            rows_per_class=args.rows_per_class,
            random_state=args.random_state,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 2
    print(
        f"Status={report['status']}; rows={report['rows']}; "
        f"normal={report['class_counts'].get('normal', 0)}; "
        f"attack={report['class_counts'].get('attack', 0)}; "
        f"output={args.output}; laporan={args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
