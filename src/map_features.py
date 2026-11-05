"""Create a cautious feature-mapping CSV from an inspection report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FEATURES = (
    ("packets_total", "Jumlah record paket dalam satu window 1 detik.", "paket/window"),
    ("bytes_total", "Jumlah byte frame yang tertangkap dalam satu window.", "byte/window"),
    ("ipv4_packets", "Jumlah paket IPv4 dalam satu window.", "paket/window"),
    ("ipv6_packets", "Jumlah paket IPv6 dalam satu window.", "paket/window"),
    ("tcp_packets", "Jumlah paket TCP dalam satu window.", "paket/window"),
    ("udp_packets", "Jumlah paket UDP dalam satu window.", "paket/window"),
    ("icmp_packets", "Jumlah paket ICMP/ICMPv6 dalam satu window.", "paket/window"),
    (
        "other_transport_packets",
        "Jumlah paket non-TCP/UDP/ICMP atau yang tidak terdekode.",
        "paket/window",
    ),
    ("tcp_syn", "Jumlah paket TCP dengan bit SYN aktif dalam satu window.", "paket/window"),
    ("tcp_ack", "Jumlah paket TCP dengan bit ACK aktif dalam satu window.", "paket/window"),
    ("tcp_rst", "Jumlah paket TCP dengan bit RST aktif dalam satu window.", "paket/window"),
    ("tcp_fin", "Jumlah paket TCP dengan bit FIN aktif dalam satu window.", "paket/window"),
    (
        "unique_source_ips",
        "Jumlah alamat sumber unik yang dihitung dalam window; alamat tidak disimpan.",
        "alamat/window",
    ),
    (
        "unique_destination_ips",
        "Jumlah alamat tujuan unik yang dihitung dalam window; alamat tidak disimpan.",
        "alamat/window",
    ),
    ("mean_packet_bytes", "Rata-rata panjang frame tertangkap per paket dalam window.", "byte/paket"),
    ("max_packet_bytes", "Panjang frame tertangkap maksimum dalam window.", "byte"),
)

FLOW_CANDIDATES = (
    ("Flow Duration", "Durasi flow berdasarkan definisi dan unit dataset sumber.", "per flow"),
    ("Total Fwd Packets", "Jumlah paket arah forward menurut definisi endpoint flow.", "paket/flow"),
    ("Total Bwd Packets", "Jumlah paket arah backward menurut definisi endpoint flow.", "paket/flow"),
    (
        "Total Length of Fwd Packets",
        "Jumlah panjang paket forward; lapisan panjang harus dikonfirmasi.",
        "byte/flow",
    ),
    (
        "Total Length of Bwd Packets",
        "Jumlah panjang paket backward; lapisan panjang harus dikonfirmasi.",
        "byte/flow",
    ),
    ("Flow Bytes/s", "Laju byte flow; perlu definisi durasi dan panjang yang sama.", "byte/detik"),
    ("Flow Packets/s", "Laju paket flow; perlu definisi durasi flow yang sama.", "paket/detik"),
    ("Packet Length Mean", "Rata-rata panjang paket pada populasi dan lapisan yang sama.", "byte/paket"),
    ("Packet Length Max", "Panjang paket maksimum pada populasi dan lapisan yang sama.", "byte"),
    ("Packet Length Min", "Panjang paket minimum pada populasi dan lapisan yang sama.", "byte"),
    ("Packet Length Std", "Simpangan baku panjang paket dengan konvensi yang sama.", "byte"),
    ("Source/Destination Port", "Port transport; tidak dihitung oleh ekstraktor window saat ini.", "port"),
    ("Label", "Ground truth yang diverifikasi dari sumber atau catatan insiden.", "kategori"),
)

FIELDS: tuple[str, ...] = (
    "Nama fitur pada dataset BPS",
    "Nama fitur pada dataset publik",
    "Definisi fitur",
    "Satuan",
    "Tipe data",
    "Metode ekstraksi",
    "Status kesesuaian",
    "Keputusan penggunaan",
)


def _shared_packet_capture_support(report_path: Path) -> tuple[bool, dict[str, int]]:
    counts = {"BPS": 0, "PUBLIC": 0}
    supported = {"BPS": True, "PUBLIC": True}
    with report_path.open("r", encoding="utf-8-sig", newline="") as report_file:
        for row in csv.DictReader(report_file):
            source = row.get("source_dataset", "")
            if source not in counts:
                continue
            counts[source] += 1
            if row.get("detected_format") != "classic PCAP" or row.get("linktype") != "1":
                supported[source] = False
    both_present = all(counts[source] > 0 for source in counts)
    return both_present and all(supported.values()), counts


def build_mapping(report_path: Path) -> list[dict[str, str]]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Laporan inspeksi tidak ditemukan: {report_path}")
    packet_compatible, source_counts = _shared_packet_capture_support(report_path)
    mapping: list[dict[str, str]] = []

    for feature_name, definition, unit in FEATURES:
        if packet_compatible:
            status = "kompatibel bersyarat: perlu ekstraksi bersama dan validasi window"
            public_name = feature_name
            decision = "kandidat; gunakan hanya setelah diekstrak dengan implementasi dan parameter sama"
        else:
            status = "belum terverifikasi: kedua sumber belum dikonfirmasi classic PCAP Ethernet"
            public_name = ""
            decision = "tunda; jangan gabungkan sampai format dan definisi fitur diverifikasi"
        mapping.append(
            {
                FIELDS[0]: feature_name,
                FIELDS[1]: public_name,
                FIELDS[2]: definition,
                FIELDS[3]: unit,
                FIELDS[4]: "integer" if feature_name not in {"mean_packet_bytes"} else "float",
                FIELDS[5]: "Agregasi paket ke window 1 detik memakai src.prepare_training_data",
                FIELDS[6]: status,
                FIELDS[7]: decision,
            }
        )

    for feature_name, definition, unit in FLOW_CANDIDATES:
        if feature_name == "Label":
            status = "ground truth tidak dapat ditentukan dari fitur PCAP"
            decision = "jangan beri label otomatis; pertahankan Unlabeled sampai terverifikasi"
        else:
            status = "tidak tersedia pada ekstraktor BPS window saat ini"
            decision = "jangan sintesis; perlu ekstraksi flow-level dan definisi seragam"
        mapping.append(
            {
                FIELDS[0]: "",
                FIELDS[1]: feature_name,
                FIELDS[2]: definition,
                FIELDS[3]: unit,
                FIELDS[4]: "belum diketahui",
                FIELDS[5]: "tidak tersedia; perlu profil/ekstraksi sumber dan validasi definisi",
                FIELDS[6]: status,
                FIELDS[7]: decision,
            }
        )

    if not packet_compatible:
        source_summary: Any = f"Jumlah profil sumber: BPS={source_counts['BPS']}, PUBLIC={source_counts['PUBLIC']}"
        mapping.append(
            {
                FIELDS[0]: "",
                FIELDS[1]: "",
                FIELDS[2]: source_summary,
                FIELDS[3]: "",
                FIELDS[4]: "",
                FIELDS[5]: "",
                FIELDS[6]: "peringatan",
                FIELDS[7]: "periksa laporan inspeksi; fitur PCAP tidak otomatis membuktikan kesetaraan dataset",
            }
        )
    return mapping


def run(report_path: Path, output_dir: Path) -> Path:
    mapping = build_mapping(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"pemetaan_fitur_{timestamp}.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(mapping)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buat pemetaan kandidat fitur tanpa memaksa skema atau label."
    )
    parser.add_argument(
        "--inspection-report",
        type=Path,
        required=True,
        help="CSV yang dihasilkan src.inspect_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "reports",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_path = run(args.inspection_report, args.output_dir)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Pemetaan fitur: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
