"""Run offline predictions with the provisional CIC-DDoS2019 candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
)

from .harmonize_cicflowmeter_features import NORMAL_ALIASES
from .prepare_csv_train_test import _load_label_mapping


DEFAULT_MODEL = Path("models/cicddos2019_development_candidate.joblib")
OUTPUT_COLUMNS = {
    "predictedlabel",
    "attackscore",
    "threshold",
    "predictionstatus",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path) -> list[str]:
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
    return columns


def _load_model(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")

    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("Format model tidak didukung; diperlukan payload kandidat.")

    required_keys = {"model", "imputer", "features", "threshold", "status"}
    missing_keys = required_keys - payload.keys()
    if missing_keys:
        raise ValueError(
            "Payload model tidak lengkap: " + ", ".join(sorted(missing_keys))
        )

    features = payload["features"]
    if not isinstance(features, (tuple, list)) or not features:
        raise ValueError("Payload model tidak memiliki daftar fitur yang valid.")
    if len(set(features)) != len(features):
        raise ValueError("Daftar fitur model mengandung nama duplikat.")

    threshold = float(payload["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold model harus berada dalam rentang 0 sampai 1.")

    classes = set(np.asarray(payload["model"].classes_).tolist())
    if not {-1, 1}.issubset(classes):
        raise ValueError("Model harus memiliki kelas -1 (attack) dan 1 (normal).")
    if not hasattr(payload["model"], "predict_proba"):
        raise ValueError("Model tidak mendukung predict_proba.")
    return payload


def _resolve_features(
    columns: list[str], model_features: tuple[str, ...] | list[str]
) -> tuple[list[str], dict[str, str]]:
    aliases = {
        source.casefold(): target
        for source, target in NORMAL_ALIASES.items()
    }
    requested = {feature.casefold(): feature for feature in model_features}
    sources: dict[str, str] = {}

    for column in columns:
        canonical = aliases.get(column.casefold(), column)
        feature = requested.get(canonical.casefold())
        if feature is None:
            continue
        if feature in sources:
            raise ValueError(
                f"Kolom input memiliki lebih dari satu alias untuk fitur {feature!r}."
            )
        sources[feature] = column

    missing = [feature for feature in model_features if feature not in sources]
    if missing:
        raise ValueError(
            "Fitur model tidak ditemukan pada CSV: " + ", ".join(missing)
        )
    return [sources[feature] for feature in model_features], sources


def _resolve_label_column(columns: list[str], requested: str | None) -> str | None:
    if requested is None:
        return None
    matches = [column for column in columns if column.casefold() == requested.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Kolom label {requested!r} tidak ditemukan secara unik.")
    return matches[0]


def _map_labels(values: pd.Series, column: str) -> np.ndarray:
    mapping = _load_label_mapping()
    mapping.update({"NORMAL": 1, "ATTACK": -1})
    normalized = values.astype("string").str.strip().str.upper()
    invalid = normalized.isna() | normalized.eq("")
    unknown = sorted(set(normalized[~invalid].unique()) - mapping.keys())
    if invalid.any() or unknown:
        details = []
        if invalid.any():
            details.append(f"{int(invalid.sum())} label kosong")
        if unknown:
            details.append("label tidak dikenal: " + ", ".join(unknown))
        raise ValueError(
            f"Label pada kolom {column!r} tidak dapat dipetakan: "
            + "; ".join(details)
        )
    return normalized.map(mapping).to_numpy(dtype=np.int8)


def _classification_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    matrix = confusion_matrix(truth, predictions, labels=[-1, 1])
    true_attack, missed_attack = (int(value) for value in matrix[0])
    false_positives = int(matrix[1, 0])
    true_normal = int(matrix[1, 1])
    normal_support = false_positives + true_normal
    predicted_attack = true_attack + false_positives
    attack_support = true_attack + missed_attack
    attack_precision = (
        true_attack / predicted_attack if predicted_attack else 0.0
    )
    attack_recall = true_attack / attack_support if attack_support else 0.0
    attack_f1 = (
        2 * attack_precision * attack_recall / (attack_precision + attack_recall)
        if attack_precision + attack_recall
        else 0.0
    )
    normal_recall = true_normal / normal_support if normal_support else 0.0
    total = attack_support + normal_support
    return {
        "rows_evaluated": total,
        "accuracy": (true_attack + true_normal) / total if total else 0.0,
        "balanced_accuracy": (attack_recall + normal_recall) / 2,
        "precision_attack": attack_precision,
        "recall_attack": attack_recall,
        "f1_attack": attack_f1,
        "normal_false_positive_count": false_positives,
        "normal_sample_count": normal_support,
        "false_positive_rate_normal": (
            float(false_positives / normal_support) if normal_support else None
        ),
        "confusion_matrix_labels": ["attack", "normal"],
        "confusion_matrix": matrix.tolist(),
    }


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        suffix=".tmp",
        delete=False,
    ) as temporary:
        return Path(temporary.name)


def predict_csv(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    model_path: Path = DEFAULT_MODEL,
    *,
    label_column: str | None = None,
    capture_id: str | None = None,
    threshold: float | None = None,
    chunk_size: int = 50_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write per-flow predictions and an audit report for a CICFlowMeter CSV."""
    for path, description in (
        (input_path, "CSV input"),
        (model_path, "model"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} tidak ditemukan: {path}")
    if chunk_size < 1:
        raise ValueError("Ukuran chunk harus lebih besar dari nol.")

    resolved_paths = {
        input_path.resolve(),
        model_path.resolve(),
    }
    if output_path.resolve() in resolved_paths or report_path.resolve() in resolved_paths:
        raise ValueError("Path output tidak boleh menimpa input atau model.")
    if output_path.resolve() == report_path.resolve():
        raise ValueError("Path CSV output dan laporan harus berbeda.")
    if not overwrite and (output_path.exists() or report_path.exists()):
        raise FileExistsError(
            "Output atau laporan sudah ada; gunakan --overwrite untuk mengganti."
        )

    payload = _load_model(model_path)
    columns = _read_header(input_path)
    source_columns, feature_sources = _resolve_features(
        columns, payload["features"]
    )
    resolved_label = _resolve_label_column(columns, label_column)
    if OUTPUT_COLUMNS.intersection(column.casefold() for column in columns):
        raise ValueError("CSV input sudah memiliki kolom output prediksi.")

    selected_threshold = (
        float(payload["threshold"]) if threshold is None else float(threshold)
    )
    if not math.isfinite(selected_threshold) or not 0 <= selected_threshold <= 1:
        raise ValueError("Threshold harus berada dalam rentang 0 sampai 1.")

    output_temporary = _temporary_path(output_path)
    report_temporary: Path | None = None
    try:
        rows = 0
        scored_rows = 0
        insufficient_rows = 0
        attack_predictions = 0
        normal_predictions = 0
        partial_feature_rows = 0
        invalid_feature_values = 0
        truth_values: list[np.ndarray] = []
        prediction_values: list[np.ndarray] = []
        first_chunk = True
        attack_index = list(payload["model"].classes_).index(-1)

        with output_temporary.open("w", encoding="utf-8", newline="") as output:
            for frame in pd.read_csv(
                input_path,
                encoding="utf-8-sig",
                chunksize=chunk_size,
                low_memory=False,
            ):
                frame.columns = [
                    str(column).strip().replace("\ufeff", "")
                    for column in frame.columns
                ]
                raw_features = frame.loc[:, source_columns].apply(
                    pd.to_numeric, errors="coerce"
                )
                raw_features.columns = list(payload["features"])
                numeric = raw_features.to_numpy(dtype=np.float64)
                valid = np.isfinite(numeric)
                valid_counts = valid.sum(axis=1)
                invalid_feature_values += int((~valid).sum())
                partial_feature_rows += int(
                    ((valid_counts > 0) & (valid_counts < len(payload["features"]))).sum()
                )
                usable = valid_counts > 0

                scores = np.full(len(frame), np.nan, dtype=np.float64)
                if usable.any():
                    model_input = raw_features.loc[usable].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    transformed = payload["imputer"].transform(model_input)
                    scores[usable] = payload["model"].predict_proba(transformed)[
                        :, attack_index
                    ]

                predictions = np.zeros(len(frame), dtype=np.int8)
                predictions[usable] = np.where(
                    scores[usable] >= selected_threshold, -1, 1
                )
                frame["PredictedLabel"] = np.where(
                    usable,
                    np.where(predictions == -1, "ATTACK", "NORMAL"),
                    "NOT_SCORED",
                )
                frame["AttackScore"] = scores
                frame["Threshold"] = selected_threshold
                frame["PredictionStatus"] = np.where(
                    usable, "scored", "no_valid_features"
                )

                if resolved_label is not None:
                    truth = _map_labels(frame[resolved_label], resolved_label)
                    truth_values.append(truth[usable])
                    prediction_values.append(predictions[usable])

                scored_rows += int(usable.sum())
                insufficient_rows += int((~usable).sum())
                attack_predictions += int((predictions[usable] == -1).sum())
                normal_predictions += int((predictions[usable] == 1).sum())
                rows += len(frame)
                frame.to_csv(
                    output,
                    index=False,
                    header=first_chunk,
                    lineterminator="\n",
                )
                first_chunk = False

        if rows == 0:
            raise ValueError(f"Tidak ada baris data di CSV: {input_path}")

        metrics: dict[str, Any] | None = None
        if resolved_label is not None and scored_rows:
            metrics = _classification_metrics(
                np.concatenate(truth_values),
                np.concatenate(prediction_values),
            )

        status = (
            "provisional_test_metrics_not_production_validation"
            if resolved_label is not None
            else "offline_inference_only_not_production_validation"
        )
        report: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "model_status": payload["status"],
            "model_file": str(model_path),
            "model_sha256": _sha256_file(model_path),
            "model_type": type(payload["model"]).__name__,
            "model_source_sha256": payload.get("source_sha256"),
            "input_file": str(input_path),
            "input_sha256": _sha256_file(input_path),
            "capture_id": capture_id,
            "rows": rows,
            "scored_rows": scored_rows,
            "rows_without_valid_features": insufficient_rows,
            "partial_feature_rows_imputed": partial_feature_rows,
            "invalid_feature_values_imputed": invalid_feature_values,
            "feature_count": len(payload["features"]),
            "feature_columns": list(payload["features"]),
            "feature_aliases_used": {
                feature: source
                for feature, source in feature_sources.items()
                if feature != source
            },
            "threshold": selected_threshold,
            "threshold_source": (
                "command_line"
                if threshold is not None
                else "model_internal_validation"
            ),
            "prediction_counts": {
                "attack": attack_predictions,
                "normal": normal_predictions,
                "not_scored": insufficient_rows,
            },
            "label_column": resolved_label,
            "metrics": metrics,
            "warnings": [
                "This candidate is provisional and is not approved for operational detection or mitigation.",
                "Predictions are not validation evidence unless the input labels and capture independence are verified.",
                "The model was trained using CICFlowMeter-V3-derived features; check extractor version and settings before interpreting scores.",
            ],
            "output_file": str(output_path),
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
        description=(
            "Jalankan prediksi offline kandidat DDoS pada CSV flow CICFlowMeter; "
            "bukan integrasi produksi."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--label-column")
    parser.add_argument("--capture-id")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_path = args.report or args.output.with_suffix(".report.json")
    try:
        report = predict_csv(
            args.input,
            args.output,
            report_path,
            args.model,
            label_column=args.label_column,
            capture_id=args.capture_id,
            threshold=args.threshold,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 2

    print(
        f"Status={report['status']}; rows={report['rows']}; "
        f"attack={report['prediction_counts']['attack']}; "
        f"normal={report['prediction_counts']['normal']}; "
        f"tidak_dinilai={report['prediction_counts']['not_scored']}; "
        f"threshold={report['threshold']:.6f}; "
        f"output={args.output}; laporan={report_path}"
    )
    if report["metrics"] is None:
        print("Metrik evaluasi tidak dihitung karena label test tidak diberikan.")
    else:
        print(
            f"Akurasi={report['metrics']['accuracy']:.4f}; "
            f"recall_attack={report['metrics']['recall_attack']:.4f}; "
            f"FPR_normal={report['metrics']['false_positive_rate_normal']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
