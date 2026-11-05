"""Inspect local dataset files and save privacy-conscious CSV/TXT reports."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_profiling import profile_file
from .pcap_profiling import PCAPNG_MAGIC, PCAP_MAGICS, profile_pcap_stream


LOGGER = logging.getLogger("inspect_data")
REPORT_FIELDS: tuple[str, ...] = (
    "source_dataset",
    "source_path",
    "member_name",
    "size_bytes",
    "detected_format",
    "status",
    "label_status",
    "row_count",
    "column_count",
    "columns",
    "exact_duplicate_rows",
    "label_candidates",
    "label_distribution",
    "protocol_distribution",
    "protocol_decode_status",
    "data_granularity",
    "packet_length_bytes",
    "packets_scanned",
    "captured_packet_bytes_scanned",
    "original_packet_bytes",
    "truncated_packets",
    "invalid_timestamp_count",
    "capture_complete",
    "pcap_version",
    "snaplen_bytes",
    "linktype",
    "timestamp_precision",
    "timestamp_min_utc",
    "timestamp_max_utc",
    "message",
)


def _profile_source(
    source_name: str,
    path: Path,
    chunk_size: int,
    max_packets: int | None,
) -> list[dict[str, Any]]:
    files = sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: str(candidate).casefold(),
    ) if path.is_dir() else [path]
    rows = []
    for file_path in files:
        if file_path.name == ".gitkeep":
            continue
        result = profile_file(
            file_path,
            chunk_size,
            max_packets=max_packets,
            decode_pcap=True,
        )
        if result.get("label_candidates"):
            label_status = "candidate_column_found_not_verified"
        else:
            label_status = "unlabeled"
        if result.get("format") == "classic_pcap":
            result["message"] = (
                "Statistik ukuran dan protokol dihitung dari paket yang dipindai; flow tidak "
                "dibentuk dan label tidak diinferensikan."
            )
        rows.append(
            {
                "source_dataset": source_name,
                "source_path": str(file_path.resolve()),
                "member_name": "",
                "label_status": label_status,
                **result,
            }
        )
    return rows


def _profile_zip(
    path: Path,
    source_name: str,
    max_packets: int | None,
) -> list[dict[str, Any]]:
    rows = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            with archive.open(member) as member_file:
                header = member_file.read(24)
            signature = header[:4]
            if signature == PCAPNG_MAGIC:
                detected_format = "PCAPNG"
                rows.append(
                    {
                        "source_dataset": source_name,
                        "source_path": str(path.resolve()),
                        "member_name": member.filename,
                        "size_bytes": member.file_size,
                        "detected_format": detected_format,
                        "status": "identified_non_csv",
                        "label_status": "unlabeled",
                        "message": "PCAPNG teridentifikasi di arsip; format ini belum didukung.",
                    }
                )
                continue
            if signature not in PCAP_MAGICS or len(header) != 24:
                kind, description = inspect_file_format_from_signature(signature)
                rows.append(
                    {
                        "source_dataset": source_name,
                        "source_path": str(path.resolve()),
                        "member_name": member.filename,
                        "size_bytes": member.file_size,
                        "detected_format": description or kind,
                        "status": "archive_member_not_profiled",
                        "label_status": "unlabeled",
                        "message": "Anggota arsip bukan classic PCAP; belum diprofilkan.",
                    }
                )
                continue

            try:
                with archive.open(member) as member_file:
                    member_file.seek(0)
                    profile = profile_pcap_stream(
                        member_file,
                        member.file_size,
                        max_packets=max_packets,
                        decode_packets=True,
                    )
            except (OSError, ValueError) as error:
                profile = {
                    "status": "error",
                    "message": str(error),
                }
            rows.append(
                {
                    "source_dataset": source_name,
                    "source_path": str(path.resolve()),
                    "member_name": member.filename,
                    "size_bytes": member.file_size,
                    "detected_format": "classic PCAP",
                    "label_status": "unlabeled",
                    "message": (
                        "PCAP dalam ZIP diprofilkan secara streaming; tidak diekstrak ke disk."
                    ),
                    **profile,
                }
            )
    return rows


def inspect_file_format_from_signature(signature: bytes) -> tuple[str, str | None]:
    from .pcap_profiling import is_pcap_signature

    if is_pcap_signature(signature):
        return "packet_capture", "classic PCAP"
    return "archive_member", None


def _report_row(row: dict[str, Any]) -> dict[str, Any]:
    report_row = {}
    for field in REPORT_FIELDS:
        value = row.get(field, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        report_row[field] = value
    return report_row


def run(
    inputs: dict[str, Path],
    report_dir: Path,
    chunk_size: int = 100_000,
    max_packets: int | None = 1_000,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    if chunk_size < 1:
        raise ValueError("--chunk-size harus lebih besar dari nol.")
    if max_packets is not None and max_packets < 1:
        raise ValueError("--max-packets-per-file harus lebih besar dari nol.")
    if not inputs:
        raise ValueError("Sediakan setidaknya satu input --bps atau --public.")

    rows = []
    for source_name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Input tidak ditemukan: {path}")
        if path.is_file() and zipfile.is_zipfile(path):
            rows.extend(_profile_zip(path, source_name, max_packets))
        else:
            rows.extend(_profile_source(source_name, path, chunk_size, max_packets))
    if not rows:
        raise ValueError("Tidak ada file yang dapat diperiksa dari input.")

    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = report_dir / f"laporan_pemeriksaan_{timestamp}.csv"
    text_path = report_dir / f"laporan_pemeriksaan_{timestamp}.txt"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(_report_row(row) for row in rows)

    scan_description = (
        "Semua paket classic PCAP dipindai; anggota PCAP dalam ZIP diprofilkan langsung tanpa ekstraksi."
        if max_packets is None
        else (
            f"Profil PCAP dibatasi maksimal {max_packets} paket per file; "
            "anggota PCAP dalam ZIP diprofilkan langsung tanpa ekstraksi."
        )
    )
    lines = [
        "Laporan pemeriksaan dataset",
        f"Dibuat (UTC): {timestamp}",
        f"Jumlah file/anggota arsip: {len(rows)}",
        scan_description,
        "Anggota arsip non-classic-PCAP hanya didaftar, tidak diprofilkan.",
        "Distribusi protokol PCAP berasal dari paket yang dipindai dan didekode; window/flow belum dibentuk.",
        "Label tidak diinferensikan. Capture tanpa kolom/manifest label berstatus unlabeled.",
        "",
        "source | file/anggota | format | status | ukuran (byte) | baris/paket | kolom",
    ]
    for row in rows:
        source = str(row.get("source_dataset", ""))
        path = str(row.get("member_name") or row.get("name") or row.get("source_path", ""))
        count = row.get("row_count", row.get("packets_scanned", ""))
        columns = row.get("column_count", "")
        lines.append(
            f"{source} | {path} | {row.get('detected_format', '')} | "
            f"{row.get('status', '')} | {row.get('size_bytes', '')} | {count} | {columns}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, text_path, rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periksa dataset BPS/publik tanpa mengubah sumber atau menebak label."
    )
    parser.add_argument("--bps", type=Path, help="File atau direktori dataset BPS.")
    parser.add_argument("--public", type=Path, help="File, direktori, atau ZIP dataset publik.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "reports",
        help="Folder laporan CSV/TXT.",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--max-packets-per-file",
        type=int,
        default=1_000,
        help="Batas paket PCAP per file; gunakan --full-pcap-scan untuk pemindaian penuh.",
    )
    parser.add_argument(
        "--full-pcap-scan",
        action="store_true",
        help="Pindai seluruh paket di setiap PCAP, bukan hanya sampel awal.",
    )
    args = parser.parse_args(argv)
    if not args.bps and not args.public:
        parser.error("Sediakan minimal satu dari --bps atau --public.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = {}
    if args.bps:
        inputs["BPS"] = args.bps
    if args.public:
        inputs["PUBLIC"] = args.public
    max_packets = None if args.full_pcap_scan else args.max_packets_per_file
    try:
        csv_path, text_path, rows = run(
            inputs, args.report_dir, args.chunk_size, max_packets
        )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        LOGGER.error("%s", error)
        return 2
    print(f"File diperiksa: {len(rows)}")
    print(f"Laporan CSV: {csv_path}")
    print(f"Laporan TXT: {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
