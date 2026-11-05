"""Train and evaluate a streaming binary traffic classifier."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


LABELS = (-1, 1)
LABEL_NAMES = {-1: "attack", 1: "normal"}
METADATA_COLUMNS = {"source_file", "label"}
WINDOW_SENTINELS = {"init_win_bytes_forward", "init_win_bytes_backward"}
CHUNK_SIZE = 50_000
PREPROCESSING_VERSION = "signed_log1p_nonnegative_v1"


def _read_columns(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    columns = tuple(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    if not METADATA_COLUMNS.issubset(columns):
        raise ValueError(f"{path}: wajib memiliki kolom source_file dan label.")
    features = tuple(column for column in columns if column not in METADATA_COLUMNS)
    if not features:
        raise ValueError(f"{path}: tidak memiliki fitur input.")
    return columns


def _validate_pair(
    train_path: Path,
    test_path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train_columns = _read_columns(train_path)
    test_columns = _read_columns(test_path)
    if train_columns != test_columns:
        raise ValueError("Kolom training dan test harus sama nama dan urutannya.")
    features = tuple(
        column for column in train_columns if column not in METADATA_COLUMNS
    )
    return train_columns, features


def _load_clean_chunks(
    path: Path,
    features: tuple[str, ...],
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    use_columns = ["source_file", "label", *features]
    for chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=use_columns,
        chunksize=chunk_size,
        low_memory=False,
    ):
        labels = pd.to_numeric(chunk["label"], errors="coerce")
        numeric = chunk.loc[:, features].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        finite_rows = np.isfinite(values).all(axis=1) & labels.isin(LABELS).to_numpy()
        source_values = chunk["source_file"].astype("string").str.strip()
        source_valid = source_values.notna().to_numpy() & source_values.ne("").to_numpy()
        valid = finite_rows & source_valid

        invalid_count = int((~valid).sum())
        values = values[valid]
        valid_labels = labels.to_numpy(dtype=np.float64)[valid].astype(np.int8)
        valid_sources = source_values.to_numpy()[valid].astype(str)

        negative_rows = np.zeros(len(values), dtype=bool)
        for index, feature in enumerate(features):
            if feature in WINDOW_SENTINELS:
                negative_rows |= values[:, index] < -1
            else:
                negative_rows |= values[:, index] < 0
        negative_count = int(negative_rows.sum())
        keep = ~negative_rows
        values = values[keep]
        valid_labels = valid_labels[keep]
        valid_sources = valid_sources[keep]

        transformed = np.sign(values) * np.log1p(np.abs(values))
        if not np.isfinite(transformed).all():
            raise ValueError(f"Transform fitur menghasilkan nilai tidak finite: {path}")
        yield transformed, valid_labels, valid_sources, invalid_count, negative_count


def _scan_dataset(
    path: Path,
    features: tuple[str, ...],
) -> dict[str, Any]:
    class_counts = {label: 0 for label in LABELS}
    total_rows = 0
    invalid_rows = 0
    negative_rows = 0
    source_files: set[str] = set()
    for values, labels, sources, invalid, negative in _load_clean_chunks(path, features):
        total_rows += len(labels)
        invalid_rows += invalid
        negative_rows += negative
        source_files.update(sources.tolist())
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            class_counts[int(label)] += int(count)
        if values.shape[1] != len(features):
            raise ValueError(f"Jumlah fitur berubah ketika membaca {path}.")
    if total_rows == 0:
        raise ValueError(f"Tidak ada baris valid setelah cleaning: {path}")
    if any(class_counts[label] == 0 for label in LABELS):
        raise ValueError(f"Kedua kelas harus ada di {path} setelah cleaning.")
    return {
        "rows": total_rows,
        "class_counts": class_counts,
        "invalid_rows_dropped": invalid_rows,
        "negative_feature_rows_dropped": negative_rows,
        "source_files": source_files,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_joblib(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        joblib.dump(payload, temporary_path, compress=3)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def train_and_evaluate(
    train_path: Path,
    test_path: Path,
    model_path: Path,
    report_path: Path,
    *,
    epochs: int = 3,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit a scaled log-loss SGD model and evaluate it on a separate test CSV."""
    if epochs < 1:
        raise ValueError("epochs harus minimal 1.")
    if train_path.resolve() == test_path.resolve():
        raise ValueError("File train dan test harus berbeda.")
    if not overwrite and (model_path.exists() or report_path.exists()):
        raise FileExistsError(
            "Model atau laporan sudah ada; gunakan --overwrite untuk mengganti."
        )

    _, features = _validate_pair(train_path, test_path)
    train_data = _scan_dataset(train_path, features)
    test_data = _scan_dataset(test_path, features)
    source_overlap = train_data["source_files"] & test_data["source_files"]
    if source_overlap:
        raise ValueError(
            "source_file muncul pada train dan test: "
            + ", ".join(sorted(source_overlap))
        )

    train_count = train_data["rows"]
    class_weights = {
        label: train_count / (len(LABELS) * train_data["class_counts"][label])
        for label in LABELS
    }
    scaler = StandardScaler()
    for values, _, _, _, _ in _load_clean_chunks(train_path, features):
        scaler.partial_fit(values)

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        learning_rate="optimal",
        average=True,
        random_state=seed,
    )
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for values, labels, _, _, _ in _load_clean_chunks(train_path, features):
            order = rng.permutation(len(labels))
            values = scaler.transform(values[order])
            labels = labels[order]
            sample_weights = np.fromiter(
                (class_weights[int(label)] for label in labels),
                dtype=np.float64,
                count=len(labels),
            )
            classifier.partial_fit(
                values,
                labels,
                classes=np.asarray(LABELS, dtype=np.int8),
                sample_weight=sample_weights,
            )

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    attack_score_parts: list[np.ndarray] = []
    for values, labels, _, _, _ in _load_clean_chunks(test_path, features):
        scaled = scaler.transform(values)
        y_true_parts.append(labels)
        y_pred_parts.append(classifier.predict(scaled))
        decision = classifier.decision_function(scaled)
        attack_score_parts.append(-decision)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    attack_scores = np.concatenate(attack_score_parts)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[-1, 1],
        zero_division=0,
    )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "SGDClassifier(log_loss, L2, averaged)",
        "seed": seed,
        "epochs": epochs,
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "transform": "sign(x) * log1p(abs(x))",
            "invalid_or_nonfinite_rows_dropped": True,
            "negative_physical_features_dropped": True,
            "initial_window_minus_one_sentinel_preserved": True,
            "class_weighting": {
                LABEL_NAMES[label]: class_weights[label] for label in LABELS
            },
        },
        "features": list(features),
        "feature_count": len(features),
        "train": {
            "file": train_path.name,
            "rows_used": train_data["rows"],
            "class_counts": {
                LABEL_NAMES[label]: train_data["class_counts"][label]
                for label in LABELS
            },
            "invalid_rows_dropped": train_data["invalid_rows_dropped"],
            "negative_feature_rows_dropped": train_data[
                "negative_feature_rows_dropped"
            ],
            "source_files": sorted(train_data["source_files"]),
        },
        "test": {
            "file": test_path.name,
            "rows_used": test_data["rows"],
            "class_counts": {
                LABEL_NAMES[label]: test_data["class_counts"][label]
                for label in LABELS
            },
            "invalid_rows_dropped": test_data["invalid_rows_dropped"],
            "negative_feature_rows_dropped": test_data[
                "negative_feature_rows_dropped"
            ],
            "source_files": sorted(test_data["source_files"]),
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_attack": float(precision[0]),
            "recall_attack": float(recall[0]),
            "f1_attack": float(f1[0]),
            "precision_normal": float(precision[1]),
            "recall_normal": float(recall[1]),
            "f1_normal": float(f1[1]),
            "roc_auc_attack": float(roc_auc_score(y_true == -1, attack_scores)),
            "confusion_matrix_labels": ["attack", "normal"],
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=[-1, 1]
            ).tolist(),
            "test_rows_evaluated": len(y_true),
        },
        "model_file": str(model_path),
    }
    model_payload = {
        "model": classifier,
        "scaler": scaler,
        "features": features,
        "label_mapping": {"1": "normal", "-1": "attack"},
        "preprocessing": PREPROCESSING_VERSION,
    }

    if not overwrite and (model_path.exists() or report_path.exists()):
        raise FileExistsError(
            "Model atau laporan sudah ada; gunakan --overwrite untuk mengganti."
        )
    _atomic_joblib(model_path, model_payload)
    try:
        _atomic_json(report_path, report)
    except OSError:
        if model_path.exists():
            model_path.unlink()
        raise
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Latih baseline classifier secara streaming dan evaluasi pada "
            "dataset test berbasis file sumber terpisah."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/harmonized/harmonized_train.csv"),
        help="Dataset training harmonized.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/harmonized/harmonized_test.csv"),
        help="Dataset test harmonized.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/traffic_sgd.joblib"),
        help="Path model tersimpan.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/reports/traffic_model_evaluation.json"),
        help="Path laporan evaluasi JSON.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Izinkan mengganti model dan laporan yang sudah ada.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = train_and_evaluate(
            args.train,
            args.test,
            args.model,
            args.report,
            epochs=args.epochs,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 2
    metrics = report["metrics"]
    print(
        f"Model dilatih dengan {report['train']['rows_used']} baris training "
        f"dan dievaluasi pada {metrics['test_rows_evaluated']} baris test. "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"precision_attack={metrics['precision_attack']:.4f}, "
        f"recall_attack={metrics['recall_attack']:.4f}, "
        f"F1_attack={metrics['f1_attack']:.4f}, "
        f"ROC-AUC_attack={metrics['roc_auc_attack']:.4f}. "
        f"Model: {args.model}; laporan: {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
