"""Train a provisional DDoS classifier from the mixed public CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .prepare_csv_train_test import _load_label_mapping


DEFAULT_INPUT = Path("data/raw/Random_combine_final.csv")
DEFAULT_MODEL = Path("models/traffic_ddos_candidate.joblib")
DEFAULT_REPORT = Path("data/reports/traffic_ddos_candidate_evaluation.json")
EXCLUDED_COLUMNS = {
    "flow_id",
    "source_ip",
    "src_ip",
    "destination_ip",
    "dst_ip",
    "timestamp",
    "simillarhttp",
    "unnamed_0",
    "index",
}
RANDOM_STATE = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_column(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", column.strip().casefold()).strip("_")


def _load_dataset(
    path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [
        str(column).strip().replace("\ufeff", "") for column in frame.columns
    ]
    label_columns = [column for column in frame.columns if column.casefold() == "label"]
    if len(label_columns) != 1:
        raise ValueError(f"Dataset harus memiliki tepat satu kolom Label: {path}")
    label_column = label_columns[0]

    label_mapping = _load_label_mapping()
    label_text = frame[label_column].astype("string").str.strip().str.upper()
    unknown_labels = sorted(set(label_text.dropna().unique()) - label_mapping.keys())
    if unknown_labels:
        raise ValueError(
            "Label tidak terpetakan; training dibatalkan: "
            + ", ".join(unknown_labels)
        )
    if label_text.isna().any() or label_text.eq("").any():
        raise ValueError("Dataset berisi label kosong.")

    mapped = label_text.map(label_mapping)
    labels = mapped.to_numpy(dtype=np.int8)
    attack_families = label_text.to_numpy(dtype=str)
    class_counts = {
        "attack": int(np.count_nonzero(labels == -1)),
        "normal": int(np.count_nonzero(labels == 1)),
    }
    if min(class_counts.values()) < 50:
        raise ValueError(
            "Diperlukan minimal 50 baris untuk masing-masing kelas; "
            f"ditemukan {class_counts}."
        )

    duplicate_count = int(frame.duplicated().sum())
    if duplicate_count:
        keep = ~frame.duplicated()
        frame = frame.loc[keep].reset_index(drop=True)
        labels = labels[keep.to_numpy()]
        attack_families = attack_families[keep.to_numpy()]
        class_counts = {
            "attack": int(np.count_nonzero(labels == -1)),
            "normal": int(np.count_nonzero(labels == 1)),
        }

    feature_columns = [
        column
        for column in frame.columns
        if column != label_column
        and _normalize_column(column) not in EXCLUDED_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("Tidak ditemukan kolom fitur setelah metadata dihapus.")

    features = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")
    feature_columns = list(features.columns)
    if not feature_columns:
        raise ValueError("Semua fitur numerik kosong atau tidak valid.")
    valid_rows = features.notna().any(axis=1).to_numpy()
    dropped_rows = int((~valid_rows).sum())
    features = features.loc[valid_rows].reset_index(drop=True)
    labels = labels[valid_rows]
    attack_families = attack_families[valid_rows]
    if min(np.count_nonzero(labels == -1), np.count_nonzero(labels == 1)) < 50:
        raise ValueError("Terlalu sedikit sampel per kelas setelah pembersihan.")

    return features, labels, attack_families, {
        "rows": int(len(labels)),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "class_counts": {
            "attack": int(np.count_nonzero(labels == -1)),
            "normal": int(np.count_nonzero(labels == 1)),
        },
        "exact_duplicate_rows_dropped": duplicate_count,
        "rows_with_no_numeric_features_dropped": dropped_rows,
        "raw_label_counts": {
            str(label): int(count)
            for label, count in label_text.value_counts().items()
        },
    }


def _grouped_split(
    labels: np.ndarray,
    attack_families: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    ratios = {"train": 0.64, "validation": 0.16, "test": 0.20}
    attack_mask = labels == -1
    family_counts = pd.Series(attack_families[attack_mask]).value_counts()
    if len(family_counts) < 3:
        raise ValueError(
            "Minimal tiga keluarga label serangan diperlukan untuk split "
            "train/validation/test tanpa tumpang tindih."
        )

    targets = {
        split: int(ratio * int(attack_mask.sum()))
        for split, ratio in ratios.items()
    }
    assigned: dict[str, list[str]] = {split: [] for split in ratios}
    assigned_counts = {split: 0 for split in ratios}
    for family, count in sorted(
        family_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
    ):
        split = min(
            ratios,
            key=lambda name: assigned_counts[name] / max(targets[name], 1),
        )
        assigned[split].append(str(family))
        assigned_counts[split] += int(count)
    if not assigned["validation"] or not assigned["test"]:
        raise ValueError("Tidak dapat membagi keluarga label attack ke validation/test.")

    normal_indices = np.flatnonzero(labels == 1)
    normal_train_validation, normal_test = train_test_split(
        normal_indices,
        test_size=ratios["test"],
        random_state=RANDOM_STATE,
    )
    normal_train, normal_validation = train_test_split(
        normal_train_validation,
        test_size=ratios["validation"]
        / (ratios["train"] + ratios["validation"]),
        random_state=RANDOM_STATE,
    )
    attack_train = np.flatnonzero(
        attack_mask & np.isin(attack_families, assigned["train"])
    )
    attack_validation = np.flatnonzero(
        attack_mask & np.isin(attack_families, assigned["validation"])
    )
    attack_test = np.flatnonzero(
        attack_mask & np.isin(attack_families, assigned["test"])
    )
    train_indices = np.concatenate((normal_train, attack_train))
    validation_indices = np.concatenate((normal_validation, attack_validation))
    test_indices = np.concatenate((normal_test, attack_test))
    for indices in (train_indices, validation_indices, test_indices):
        np.random.default_rng(RANDOM_STATE).shuffle(indices)
    return train_indices, validation_indices, test_indices, assigned


def _best_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels == -1, scores)
    if not len(thresholds):
        return 0.5, 0.0
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    )
    best_index = int(np.argmax(f1))
    return float(thresholds[best_index]), float(f1[best_index])


def _evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = np.where(scores >= threshold, -1, 1).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[-1, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[-1, 1],
        zero_division=0,
    )
    normal_support = int(np.count_nonzero(labels == 1))
    false_positives = int(matrix[1, 0])
    false_positive_rate = (
        float(false_positives / normal_support) if normal_support else 0.0
    )
    if normal_support:
        z_score = 1.959963984540054
        denominator = 1 + z_score**2 / normal_support
        center = (
            false_positive_rate + z_score**2 / (2 * normal_support)
        ) / denominator
        margin = (
            z_score
            * (
                false_positive_rate * (1 - false_positive_rate) / normal_support
                + z_score**2 / (4 * normal_support**2)
            )
            ** 0.5
            / denominator
        )
        false_positive_ci = [
            max(0.0, center - margin),
            min(1.0, center + margin),
        ]
    else:
        false_positive_ci = None
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_attack": float(precision[0]),
        "recall_attack": float(recall[0]),
        "f1_attack": float(f1[0]),
        "precision_normal": float(precision[1]),
        "recall_normal": float(recall[1]),
        "f1_normal": float(f1[1]),
        "normal_false_positive_count": false_positives,
        "normal_sample_count": normal_support,
        "false_positive_rate_normal": false_positive_rate,
        "false_positive_rate_normal_wilson_95_ci": false_positive_ci,
        "roc_auc_attack": float(roc_auc_score(labels == -1, scores)),
        "pr_auc_attack": float(average_precision_score(labels == -1, scores)),
        "confusion_matrix_labels": ["attack", "normal"],
        "confusion_matrix": matrix.tolist(),
    }


def _atomic_save(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        writer(temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def train_candidate(
    input_path: Path = DEFAULT_INPUT,
    model_path: Path = DEFAULT_MODEL,
    report_path: Path = DEFAULT_REPORT,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if model_path.resolve() == report_path.resolve():
        raise ValueError("Path model dan laporan harus berbeda.")
    if not overwrite and (model_path.exists() or report_path.exists()):
        raise FileExistsError(
            "Kandidat atau laporan sudah ada; gunakan --overwrite untuk mengganti."
        )

    features, labels, attack_families, dataset_info = _load_dataset(input_path)
    (
        train_indices,
        validation_indices,
        test_indices,
        attack_families_by_split,
    ) = _grouped_split(labels, attack_families)

    imputer = SimpleImputer(strategy="median")
    train_features = imputer.fit_transform(features.iloc[train_indices])
    validation_features = imputer.transform(features.iloc[validation_indices])
    test_features = imputer.transform(features.iloc[test_indices])
    model_specs = {
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=20,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=30,
            class_weight="balanced",
            max_features="sqrt",
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }

    candidates: dict[str, dict[str, Any]] = {}
    for name, estimator in model_specs.items():
        estimator.fit(train_features, labels[train_indices])
        attack_index = int(np.flatnonzero(estimator.classes_ == -1)[0])
        validation_scores = estimator.predict_proba(validation_features)[:, attack_index]
        threshold, validation_f1 = _best_f1_threshold(
            labels[validation_indices], validation_scores
        )
        candidates[name] = {
            "estimator": estimator,
            "threshold": threshold,
            "validation_f1_attack": validation_f1,
            "validation_pr_auc_attack": float(
                average_precision_score(
                    labels[validation_indices] == -1, validation_scores
                )
            ),
        }

    selected_name = max(
        candidates,
        key=lambda name: candidates[name]["validation_pr_auc_attack"],
    )
    selected = candidates[selected_name]
    estimator = selected["estimator"]
    attack_index = int(np.flatnonzero(estimator.classes_ == -1)[0])
    test_scores = estimator.predict_proba(test_features)[:, attack_index]
    test_metrics = {
        "default_threshold_0_5": _evaluate(labels[test_indices], test_scores, 0.5),
        "validation_tuned_threshold": _evaluate(
            labels[test_indices], test_scores, selected["threshold"]
        ),
    }
    for metrics in test_metrics.values():
        threshold = metrics["threshold"]
        metrics["attack_family_detection"] = {
            family: {
                "rows": int(
                    np.count_nonzero(
                        (labels[test_indices] == -1)
                        & (attack_families[test_indices] == family)
                    )
                ),
                "recall": float(
                    np.mean(
                        test_scores[
                            (labels[test_indices] == -1)
                            & (attack_families[test_indices] == family)
                        ]
                        >= threshold
                    )
                ),
            }
            for family in attack_families_by_split["test"]
        }
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_experiment_not_production_ready",
        "source_file": str(input_path),
        "source_sha256": _sha256_file(input_path),
        "label_mapping": {
            "normal": 1,
            "mapped_ddos_attack": -1,
        },
        "dataset": dataset_info,
        "split": {
            "method": "attack_label_family_holdout_and_random_benign_split",
            "random_state": RANDOM_STATE,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "test_rows": len(test_indices),
            "train_class_counts": {
                "attack": int(np.count_nonzero(labels[train_indices] == -1)),
                "normal": int(np.count_nonzero(labels[train_indices] == 1)),
            },
            "validation_class_counts": {
                "attack": int(np.count_nonzero(labels[validation_indices] == -1)),
                "normal": int(np.count_nonzero(labels[validation_indices] == 1)),
            },
            "test_class_counts": {
                "attack": int(np.count_nonzero(labels[test_indices] == -1)),
                "normal": int(np.count_nonzero(labels[test_indices] == 1)),
            },
            "attack_families_by_split": attack_families_by_split,
            "attack_families_disjoint_across_splits": True,
            "independent_source_file_test": False,
        },
        "candidate_validation": {
            name: {
                "validation_f1_attack": candidate["validation_f1_attack"],
                "validation_pr_auc_attack": candidate[
                    "validation_pr_auc_attack"
                ],
                "threshold_selected_on_validation": candidate["threshold"],
            }
            for name, candidate in candidates.items()
        },
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "limitations": [
            "Source dataset provenance and construction are not independently verified.",
            "Attack families are held out by source label, but the holdout is not an independent capture/day.",
            "Normal rows are split randomly; no independent benign capture is represented.",
            "A balanced development sample does not represent operational class prevalence.",
            "Do not interpret this experiment as validated production DDoS detection.",
        ],
        "model_file": str(model_path),
    }
    model_payload = {
        "model": estimator,
        "imputer": imputer,
        "features": tuple(dataset_info["features"]),
        "threshold": selected["threshold"],
        "label_mapping": {"1": "normal", "-1": "attack"},
        "source_sha256": report["source_sha256"],
        "status": report["status"],
    }
    _atomic_save(model_path, lambda path: joblib.dump(model_payload, path, compress=3))

    def write_report(path: Path) -> None:
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    try:
        _atomic_save(report_path, write_report)
    except OSError:
        if model_path.exists():
            model_path.unlink()
        raise
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate model kandidat DDoS secara provisional."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = train_candidate(
            args.input,
            args.model,
            args.report,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 2
    print(
        f"Kandidat={report['selected_model']}; "
        f"test={report['split']['test_rows']} baris; "
        f"akurasi={report['test_metrics']['validation_tuned_threshold']['accuracy']:.4f}; "
        f"recall_attack={report['test_metrics']['validation_tuned_threshold']['recall_attack']:.4f}; "
        f"F1_attack={report['test_metrics']['validation_tuned_threshold']['f1_attack']:.4f}; "
        f"model={args.model}; laporan={args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
