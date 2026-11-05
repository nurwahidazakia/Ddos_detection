"""
Tahap 2 — Script Pemeriksaan Dataset Publik (Chunked, Hemat Memori)

Memeriksa seluruh file CSV di data/raw/ dan subfolder.
Menghasilkan laporan lengkap ke data/reports/.

Penggunaan:
    python -m src.data.profiling
    python -m src.data.profiling --dir data/raw/cicddos2019
    python -m src.data.profiling --file data/raw/Random_combine_final.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import hashlib
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
CHUNK_SIZE = 50_000  # Baris per chunk (hemat RAM ~7 GB)

# Label column name variants
LABEL_COL_CANDIDATES = ["Label", " Label", "label"]


def find_label_col(columns: list[str]) -> str | None:
    """Cari nama kolom label dari daftar kolom."""
    for candidate in LABEL_COL_CANDIDATES:
        if candidate in columns:
            return candidate
    # Fallback: strip whitespace
    for col in columns:
        if col.strip().lower() == "label":
            return col
    return None


def profile_single_csv(filepath: Path) -> dict[str, Any]:
    """
    Profiling satu file CSV secara chunked.

    Returns dict berisi:
      - file_name, file_path, file_size_mb
      - num_rows, num_columns
      - columns (list nama kolom)
      - dtypes (dict kolom -> tipe yang terdeteksi pandas)
      - label_column, label_distribution
      - missing_values (dict kolom -> jumlah NaN)
      - infinity_values (dict kolom -> jumlah Inf)
      - duplicate_rows
      - numeric_columns, non_numeric_columns
      - header_repeat_rows (baris yang berisi header terulang)
      - sample_values (5 baris pertama per kolom)
      - issues (list masalah yang ditemukan)
    """
    result: dict[str, Any] = {
        "file_name": filepath.name,
        "file_path": str(filepath),
        "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
        "num_rows": 0,
        "num_columns": 0,
        "columns": [],
        "dtypes": {},
        "label_column": None,
        "label_distribution": {},
        "missing_values": {},
        "infinity_values": {},
        "infinity_total": 0,
        "missing_total": 0,
        "duplicate_rows": 0,
        "numeric_columns": [],
        "non_numeric_columns": [],
        "header_repeat_rows": 0,
        "negative_value_columns": [],
        "constant_columns": [],
        "sample_values": {},
        "basic_stats": {},
        "issues": [],
    }

    print(f"\n{'='*70}")
    print(f"Profiling: {filepath.name} ({result['file_size_mb']:.1f} MB)")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Langkah 1: Baca header dan deteksi kolom
    # ------------------------------------------------------------------
    try:
        header_df = pd.read_csv(filepath, nrows=0, encoding="utf-8")
    except Exception as e:
        result["issues"].append(f"Gagal membaca header: {e}")
        print(f"  ERROR: {e}")
        return result

    columns = list(header_df.columns)
    result["columns"] = columns
    result["num_columns"] = len(columns)

    label_col = find_label_col(columns)
    result["label_column"] = label_col

    print(f"  Kolom: {len(columns)}")
    print(f"  Label column: {label_col}")

    # ------------------------------------------------------------------
    # Langkah 2: Chunked reading untuk statistik
    # ------------------------------------------------------------------
    total_rows = 0
    label_counter: Counter = Counter()
    missing_counter: dict[str, int] = {col: 0 for col in columns}
    inf_counter: dict[str, int] = {col: 0 for col in columns}
    header_repeats = 0
    sample_collected = False
    sample_values: dict[str, list] = {}
    dtype_detected: dict[str, str] = {}
    numeric_cols_set: set[str] = set()
    non_numeric_cols_set: set[str] = set()
    neg_cols_set: set[str] = set()
    constant_cols: dict[str, set] = {col: set() for col in columns}

    # Untuk deteksi duplikat: hash baris
    row_hashes: set[int] = set()
    duplicate_count = 0

    try:
        chunk_iter = pd.read_csv(
            filepath,
            chunksize=CHUNK_SIZE,
            encoding="utf-8",
            low_memory=False,
            on_bad_lines="warn",
        )
    except Exception as e:
        result["issues"].append(f"Gagal membuka file untuk chunked reading: {e}")
        print(f"  ERROR: {e}")
        return result

    chunk_num = 0
    print(f"  Membaca dengan chunk size {CHUNK_SIZE:,}...")

    for chunk in tqdm(chunk_iter, desc=f"  {filepath.name}", unit="chunk"):
        chunk_num += 1
        nrows = len(chunk)
        total_rows += nrows

        # Ambil sample dari chunk pertama
        if not sample_collected:
            for col in columns:
                if col in chunk.columns:
                    vals = chunk[col].head(5).tolist()
                    sample_values[col] = [
                        str(v) if not (isinstance(v, float) and math.isnan(v)) else "NaN"
                        for v in vals
                    ]
            sample_collected = True

        # Deteksi header terulang
        if label_col and label_col in chunk.columns:
            header_mask = chunk[label_col].astype(str).str.strip() == label_col.strip()
            n_header_repeats = header_mask.sum()
            header_repeats += n_header_repeats
            if n_header_repeats > 0:
                chunk = chunk[~header_mask]
                nrows = len(chunk)

        # Label distribution
        if label_col and label_col in chunk.columns:
            for lbl, cnt in chunk[label_col].value_counts(dropna=False).items():
                key = str(lbl).strip() if pd.notna(lbl) else "<NaN>"
                label_counter[key] += cnt

        # Missing values
        for col in columns:
            if col in chunk.columns:
                n_missing = chunk[col].isna().sum()
                missing_counter[col] += n_missing

        # Numeric check & infinity
        for col in columns:
            if col not in chunk.columns:
                continue
            try:
                numeric_series = pd.to_numeric(chunk[col], errors="coerce")
                coerced_nan = numeric_series.isna().sum() - chunk[col].isna().sum()
                if coerced_nan < len(chunk) * 0.5:  # Lebih dari 50% bisa numeric
                    numeric_cols_set.add(col)
                    # Infinity check
                    n_inf = np.isinf(numeric_series.dropna().values).sum()
                    inf_counter[col] += int(n_inf)
                    # Negative check
                    if (numeric_series.dropna() < 0).any():
                        neg_cols_set.add(col)
                else:
                    non_numeric_cols_set.add(col)
            except (TypeError, ValueError):
                non_numeric_cols_set.add(col)

        # Constant columns (track unique values, max 10)
        for col in columns:
            if col in chunk.columns and len(constant_cols[col]) <= 10:
                unique_vals = chunk[col].dropna().unique()
                for v in unique_vals[:10]:
                    constant_cols[col].add(str(v))

        # Duplikat (approximate via hashing)
        for _, row in chunk.iterrows():
            h = hash(tuple(row.values))
            if h in row_hashes:
                duplicate_count += 1
            else:
                row_hashes.add(h)
            # Batasi ukuran set untuk hemat memori
            if len(row_hashes) > 5_000_000:
                break

        # Safety: clear hash set jika terlalu besar
        if len(row_hashes) > 5_000_000:
            result["issues"].append(
                "Pengecekan duplikat dihentikan setelah 5 juta baris unik (hemat memori). "
                "Jumlah duplikat yang dilaporkan adalah batas bawah."
            )
            # Lanjutkan loop tapi skip duplikat check
            row_hashes.clear()

    # ------------------------------------------------------------------
    # Langkah 3: Compile hasil
    # ------------------------------------------------------------------
    result["num_rows"] = total_rows
    result["label_distribution"] = dict(label_counter.most_common())
    result["missing_values"] = {k: v for k, v in missing_counter.items() if v > 0}
    result["missing_total"] = sum(missing_counter.values())
    result["infinity_values"] = {k: v for k, v in inf_counter.items() if v > 0}
    result["infinity_total"] = sum(inf_counter.values())
    result["duplicate_rows"] = duplicate_count
    result["header_repeat_rows"] = header_repeats
    result["numeric_columns"] = sorted(numeric_cols_set)
    result["non_numeric_columns"] = sorted(non_numeric_cols_set)
    result["negative_value_columns"] = sorted(neg_cols_set)
    result["sample_values"] = sample_values

    # Constant columns
    constant_list = [col for col, vals in constant_cols.items() if len(vals) <= 1 and col in numeric_cols_set]
    result["constant_columns"] = constant_list

    # Detect dtype from sample
    try:
        sample_df = pd.read_csv(filepath, nrows=1000, encoding="utf-8", low_memory=False)
        dtype_detected = {col: str(sample_df[col].dtype) for col in sample_df.columns}
    except Exception:
        dtype_detected = {}
    result["dtypes"] = dtype_detected

    # ------------------------------------------------------------------
    # Langkah 4: Identifikasi masalah
    # ------------------------------------------------------------------
    if header_repeats > 0:
        result["issues"].append(
            f"Ditemukan {header_repeats} baris header terulang di tengah data. "
            "Baris ini harus dibuang saat preprocessing."
        )

    if result["missing_total"] > 0:
        pct = result["missing_total"] / (total_rows * len(columns)) * 100
        result["issues"].append(
            f"Missing values: {result['missing_total']:,} sel "
            f"({pct:.2f}% dari total sel). "
            f"Kolom terparah: {', '.join(list(result['missing_values'].keys())[:5])}"
        )

    if result["infinity_total"] > 0:
        result["issues"].append(
            f"Infinity values: {result['infinity_total']:,} sel. "
            f"Kolom: {', '.join(list(result['infinity_values'].keys())[:5])}"
        )

    if duplicate_count > 0:
        result["issues"].append(
            f"Ditemukan {duplicate_count:,} baris duplikat (approximate)."
        )

    if constant_list:
        result["issues"].append(
            f"Kolom konstan (hanya 1 nilai unik): {', '.join(constant_list)}. "
            "Kolom ini tidak berguna untuk model."
        )

    if not label_col:
        result["issues"].append("Kolom label tidak ditemukan!")

    # Print summary
    print(f"\n  Hasil profiling {filepath.name}:")
    print(f"    Baris         : {total_rows:,}")
    print(f"    Kolom         : {len(columns)}")
    print(f"    Label column  : {label_col}")
    print(f"    Label unik    : {len(label_counter)}")
    print(f"    Missing total : {result['missing_total']:,}")
    print(f"    Infinity total: {result['infinity_total']:,}")
    print(f"    Header repeats: {header_repeats}")
    print(f"    Duplikat      : {duplicate_count:,}")
    print(f"    Fitur numerik : {len(numeric_cols_set)}")
    print(f"    Fitur non-num : {len(non_numeric_cols_set)}")
    if result["issues"]:
        print(f"    Masalah ({len(result['issues'])}):")
        for issue in result["issues"]:
            print(f"      - {issue}")
    print(f"    Label distribution:")
    for lbl, cnt in label_counter.most_common(15):
        pct = cnt / total_rows * 100
        print(f"      {lbl}: {cnt:,} ({pct:.2f}%)")

    return result


def identify_dataset_source(profile: dict) -> dict[str, Any]:
    """
    Identifikasi sumber dataset berdasarkan kolom dan label.

    Returns dict dengan:
      - dataset_source: nama dataset yang teridentifikasi
      - confidence: tingkat keyakinan
      - ddos_relevant: apakah file relevan untuk deteksi DDoS
      - relevant_labels: label yang relevan sebagai DDoS
      - irrelevant_labels: label yang BUKAN DDoS
    """
    columns = profile.get("columns", [])
    labels = profile.get("label_distribution", {})
    col_set = {c.strip() for c in columns}

    # CIC-DDoS2019 markers
    cicddos2019_markers = {"SimillarHTTP", "Inbound"}
    cicddos2019_labels = {
        "DrDoS_DNS", "DrDoS_LDAP", "DrDoS_MSSQL", "DrDoS_NetBIOS",
        "DrDoS_NTP", "DrDoS_SNMP", "DrDoS_SSDP", "DrDoS_UDP",
        "Syn", "TFTP", "UDP-lag", "UDPLag", "WebDDoS",
        "LDAP", "MSSQL", "NetBIOS", "Portmap", "UDP",
        "BENIGN",
    }

    # CIC-IDS2017/2018 markers (no SimillarHTTP/Inbound, 80 cols)
    cicids_non_ddos_labels = {
        "FTP-BruteForce", "SSH-Bruteforce", "Brute Force",
        "Infilteration", "Infiltration", "Bot", "Web Attack",
        "Heartbleed", "PortScan", "DoS Hulk", "DoS GoldenEye",
        "DoS slowloris", "DoS Slowhttptest",
    }
    cicids_ddos_label = {"DDoS"}

    has_ddos2019_markers = cicddos2019_markers.issubset(col_set)
    label_keys = set(labels.keys())

    result = {
        "dataset_source": "unknown",
        "confidence": "low",
        "ddos_relevant": False,
        "relevant_labels": [],
        "irrelevant_labels": [],
        "recommendation": "",
    }

    if has_ddos2019_markers or label_keys.intersection(cicddos2019_labels - {"BENIGN"}):
        result["dataset_source"] = "CIC-DDoS2019"
        result["confidence"] = "high" if has_ddos2019_markers else "medium"
        for lbl in label_keys:
            if lbl in cicddos2019_labels and lbl != "BENIGN":
                result["relevant_labels"].append(lbl)
            elif lbl == "BENIGN":
                pass  # Normal traffic
            else:
                result["irrelevant_labels"].append(lbl)
        result["ddos_relevant"] = True
        result["recommendation"] = "GUNAKAN — Dataset utama untuk training DDoS detection."

    elif label_keys.intersection(cicids_non_ddos_labels):
        result["dataset_source"] = "CIC-IDS2017/2018 (CSE-CIC-IDS2018)"
        result["confidence"] = "high"
        for lbl in label_keys:
            if lbl in cicids_ddos_label:
                result["relevant_labels"].append(lbl)
            elif lbl in {"Benign", "BENIGN"}:
                pass
            else:
                result["irrelevant_labels"].append(lbl)
        if result["relevant_labels"]:
            result["ddos_relevant"] = True
            result["recommendation"] = (
                "OPSIONAL — Hanya gunakan baris berlabel 'DDoS' dan 'Benign'. "
                "Label lain (Brute Force, Infiltration, Bot, dll.) BUKAN DDoS."
            )
        else:
            result["ddos_relevant"] = False
            result["recommendation"] = (
                "TIDAK RELEVAN — File ini tidak mengandung label DDoS. "
                "Hanya berisi label non-DDoS. Bisa digunakan sebagai sumber "
                "data BENIGN tambahan jika diperlukan, dengan catatan jelas."
            )

    elif label_keys.intersection(cicids_ddos_label):
        result["dataset_source"] = "CIC-IDS2017 (subset DDoS)"
        result["confidence"] = "medium"
        result["relevant_labels"] = ["DDoS"]
        result["ddos_relevant"] = True
        result["recommendation"] = (
            "OPSIONAL — Berisi label DDoS. Hanya gunakan baris 'DDoS' dan 'Benign'."
        )
    else:
        result["dataset_source"] = "Unknown"
        result["recommendation"] = (
            "PERIKSA MANUAL — Sumber dataset tidak teridentifikasi. "
            "Jangan gunakan tanpa verifikasi provenance."
        )

    return result


def generate_report(
    profiles: list[dict],
    identifications: list[dict],
    output_dir: Path,
) -> Path:
    """Generate laporan pemeriksaan dataset ke JSON dan TXT."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json_path = output_dir / f"dataset_profiling_{timestamp}.json"
    report_txt_path = output_dir / f"dataset_profiling_{timestamp}.txt"

    # JSON report
    report_data = {
        "report_timestamp": timestamp,
        "tool": "src.data.profiling",
        "total_files_scanned": len(profiles),
        "files": [],
    }

    for prof, ident in zip(profiles, identifications):
        entry = {**prof, "identification": ident}
        # Remove sample_values from JSON to keep it small
        entry.pop("sample_values", None)
        report_data["files"].append(entry)

    # Summary
    total_rows = sum(p["num_rows"] for p in profiles)
    ddos_relevant = [
        p["file_name"] for p, i in zip(profiles, identifications) if i["ddos_relevant"]
    ]
    not_relevant = [
        p["file_name"] for p, i in zip(profiles, identifications) if not i["ddos_relevant"]
    ]
    all_issues = []
    for p in profiles:
        for iss in p["issues"]:
            all_issues.append(f"[{p['file_name']}] {iss}")

    # All unique labels across files
    all_labels: Counter = Counter()
    for p in profiles:
        for lbl, cnt in p["label_distribution"].items():
            all_labels[lbl] += cnt

    report_data["summary"] = {
        "total_rows_all_files": total_rows,
        "files_ddos_relevant": ddos_relevant,
        "files_not_relevant": not_relevant,
        "all_labels": dict(all_labels.most_common()),
        "total_issues": len(all_issues),
        "issues": all_issues,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

    # TXT report (human-readable)
    lines = []
    lines.append("=" * 70)
    lines.append("LAPORAN PEMERIKSAAN DATASET PUBLIK")
    lines.append(f"Tanggal: {timestamp}")
    lines.append(f"Total file diperiksa: {len(profiles)}")
    lines.append(f"Total baris semua file: {total_rows:,}")
    lines.append("=" * 70)

    for prof, ident in zip(profiles, identifications):
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"FILE: {prof['file_name']}")
        lines.append(f"Path: {prof['file_path']}")
        lines.append(f"Ukuran: {prof['file_size_mb']:.1f} MB")
        lines.append(f"Baris: {prof['num_rows']:,}")
        lines.append(f"Kolom: {prof['num_columns']}")
        lines.append(f"Sumber teridentifikasi: {ident['dataset_source']} "
                      f"(confidence: {ident['confidence']})")
        lines.append(f"Relevan untuk DDoS: {'YA' if ident['ddos_relevant'] else 'TIDAK'}")
        lines.append(f"Rekomendasi: {ident['recommendation']}")
        lines.append("")
        lines.append("  Label distribution:")
        for lbl, cnt in sorted(prof["label_distribution"].items(), key=lambda x: -x[1]):
            pct = cnt / prof["num_rows"] * 100 if prof["num_rows"] > 0 else 0
            lines.append(f"    {lbl}: {cnt:,} ({pct:.2f}%)")

        if ident["relevant_labels"]:
            lines.append(f"  Label DDoS relevan: {', '.join(ident['relevant_labels'])}")
        if ident["irrelevant_labels"]:
            lines.append(f"  Label BUKAN DDoS: {', '.join(ident['irrelevant_labels'])}")

        lines.append(f"  Missing values: {prof['missing_total']:,}")
        if prof["missing_values"]:
            for col, cnt in list(prof["missing_values"].items())[:10]:
                lines.append(f"    {col}: {cnt:,}")

        lines.append(f"  Infinity values: {prof['infinity_total']:,}")
        if prof["infinity_values"]:
            for col, cnt in list(prof["infinity_values"].items())[:10]:
                lines.append(f"    {col}: {cnt:,}")

        lines.append(f"  Duplikat: {prof['duplicate_rows']:,}")
        lines.append(f"  Header terulang: {prof['header_repeat_rows']}")
        lines.append(f"  Fitur numerik: {len(prof['numeric_columns'])}")
        lines.append(f"  Fitur non-numerik: {len(prof['non_numeric_columns'])}")

        if prof["constant_columns"]:
            lines.append(f"  Kolom konstan: {', '.join(prof['constant_columns'])}")
        if prof["negative_value_columns"]:
            lines.append(f"  Kolom dengan nilai negatif: {', '.join(prof['negative_value_columns'])}")

        if prof["issues"]:
            lines.append(f"  Masalah ({len(prof['issues'])}):")
            for iss in prof["issues"]:
                lines.append(f"    - {iss}")

    # Global summary
    lines.append("")
    lines.append("=" * 70)
    lines.append("RINGKASAN GLOBAL")
    lines.append("=" * 70)
    lines.append(f"File relevan untuk DDoS: {', '.join(ddos_relevant) if ddos_relevant else 'TIDAK ADA'}")
    lines.append(f"File TIDAK relevan: {', '.join(not_relevant) if not_relevant else '-'}")
    lines.append("")
    lines.append("Distribusi label gabungan (semua file):")
    for lbl, cnt in all_labels.most_common():
        lines.append(f"  {lbl}: {cnt:,}")

    lines.append("")
    lines.append("DAFTAR FITUR (dari file pertama yang relevan):")
    for prof, ident in zip(profiles, identifications):
        if ident["ddos_relevant"]:
            for i, col in enumerate(prof["columns"]):
                cat = "numeric" if col in prof["numeric_columns"] else "non-numeric"
                lines.append(f"  {i+1:3d}. {col.strip():40s} [{cat}]")
            break

    if all_issues:
        lines.append("")
        lines.append(f"SEMUA MASALAH ({len(all_issues)}):")
        for iss in all_issues:
            lines.append(f"  - {iss}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("REKOMENDASI DATASET YANG LAYAK DIGUNAKAN:")
    lines.append("=" * 70)
    for prof, ident in zip(profiles, identifications):
        status = "✓ GUNAKAN" if ident["ddos_relevant"] else "✗ LEWATI"
        lines.append(f"  [{status}] {prof['file_name']} — {ident['recommendation']}")

    lines.append("")
    lines.append("--- Akhir laporan ---")

    txt_content = "\n".join(lines)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    print(f"\n{'='*70}")
    print("LAPORAN DISIMPAN:")
    print(f"  JSON: {report_json_path}")
    print(f"  TXT:  {report_txt_path}")
    print(f"{'='*70}")

    return report_json_path


def main():
    parser = argparse.ArgumentParser(
        description="Profiling dataset CSV untuk proyek DDoS Monitoring BPS"
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Direktori berisi file CSV (default: data/raw/)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Satu file CSV spesifik untuk diperiksa"
    )
    parser.add_argument(
        "--skip-duplicates", action="store_true",
        help="Lewati pengecekan duplikat (hemat memori)"
    )
    args = parser.parse_args()

    # Tentukan file yang akan diproses
    csv_files: list[Path] = []

    if args.file:
        fp = Path(args.file)
        if not fp.is_absolute():
            fp = PROJECT_ROOT / fp
        if fp.is_file():
            csv_files.append(fp)
        else:
            print(f"ERROR: File tidak ditemukan: {fp}")
            sys.exit(1)
    else:
        scan_dir = Path(args.dir) if args.dir else DEFAULT_RAW_DIR
        if not scan_dir.is_absolute():
            scan_dir = PROJECT_ROOT / scan_dir
        if not scan_dir.is_dir():
            print(f"ERROR: Direktori tidak ditemukan: {scan_dir}")
            sys.exit(1)

        # Cari CSV di direktori dan 1 level subfolder
        for item in sorted(scan_dir.iterdir()):
            if item.is_file() and item.suffix.lower() == ".csv":
                csv_files.append(item)
            elif item.is_dir():
                for subitem in sorted(item.iterdir()):
                    if subitem.is_file() and subitem.suffix.lower() == ".csv":
                        csv_files.append(subitem)

    if not csv_files:
        print("Tidak ada file CSV yang ditemukan!")
        sys.exit(1)

    print(f"\nDitemukan {len(csv_files)} file CSV:")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    # Proses setiap file
    profiles: list[dict] = []
    identifications: list[dict] = []

    for csv_file in csv_files:
        profile = profile_single_csv(csv_file)
        identification = identify_dataset_source(profile)
        profiles.append(profile)
        identifications.append(identification)

        print(f"\n  >> Identifikasi: {identification['dataset_source']} "
              f"(confidence: {identification['confidence']})")
        print(f"  >> Relevan DDoS: {'YA' if identification['ddos_relevant'] else 'TIDAK'}")
        print(f"  >> {identification['recommendation']}")

    # Generate laporan
    report_path = generate_report(profiles, identifications, REPORTS_DIR)

    print(f"\nProfiling selesai. {len(profiles)} file diperiksa.")
    return report_path


if __name__ == "__main__":
    main()

