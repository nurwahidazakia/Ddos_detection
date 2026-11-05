"""Aggregate packet-level labeled CSV files into per-second training features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .prepare_training_data import _load_labels

FEATURE_COLUMNS = (
    "packets_total",
    "bytes_total",
    "ipv4_packets",
    "ipv6_packets",
    "tcp_packets",
    "udp_packets",
    "icmp_packets",
    "other_transport_packets",
    "tcp_syn",
    "tcp_ack",
    "tcp_rst",
    "tcp_fin",
    "unique_source_ips",
    "unique_destination_ips",
    "mean_packet_bytes",
    "max_packet_bytes",
)
OUTPUT_COLUMNS = (
    "source_capture",
    "source_sha256",
    "window_start",
    *FEATURE_COLUMNS,
    "label",
)
INPUT_COLUMNS = {"timestamp", "src_ip", "dst_ip", "protocol", "packet_length", "tcp_flags", "label"}


def _parse_timestamp(value: str, path: Path, line_number: int) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{path}: timestamp tidak valid pada baris {line_number}: {value!r}") from error
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.replace(microsecond=0)


def _parse_ip(value: str, path: Path, line_number: int) -> tuple[int | None, str | None]:
    candidate = value.strip()
    if not candidate:
        return None, None
    try:
        parsed = ipaddress.ip_address(candidate)
    except ValueError as error:
        raise ValueError(f"{path}: alamat IP tidak valid pada baris {line_number}: {candidate!r}") from error
    return parsed.version, candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class WindowStats:
    packets_total: int = 0
    bytes_total: int = 0
    ipv4_packets: int = 0
    ipv6_packets: int = 0
    tcp_packets: int = 0
    udp_packets: int = 0
    icmp_packets: int = 0
    other_transport_packets: int = 0
    tcp_syn: int = 0
    tcp_ack: int = 0
    tcp_rst: int = 0
    tcp_fin: int = 0
    max_packet_bytes: int = 0
    source_ips: set[str] = field(default_factory=set)
    destination_ips: set[str] = field(default_factory=set)

    def add(
        self,
        protocol_value: str,
        packet_length: int,
        tcp_flags: int | None,
        source_ip: str | None,
        destination_ip: str | None,
        ip_versions: set[int],
    ) -> None:
        self.packets_total += 1
        self.bytes_total += packet_length
        self.max_packet_bytes = max(self.max_packet_bytes, packet_length)
        self.ipv4_packets += 4 in ip_versions
        self.ipv6_packets += 6 in ip_versions
        if source_ip is not None:
            self.source_ips.add(source_ip)
        if destination_ip is not None:
            self.destination_ips.add(destination_ip)

        protocol = protocol_value.strip().upper()
        if protocol == "TCP":
            self.tcp_packets += 1
            if tcp_flags is not None:
                self.tcp_fin += bool(tcp_flags & 0x01)
                self.tcp_syn += bool(tcp_flags & 0x02)
                self.tcp_rst += bool(tcp_flags & 0x04)
                self.tcp_ack += bool(tcp_flags & 0x10)
        elif protocol == "UDP":
            self.udp_packets += 1
        elif protocol in {"ICMP", "ICMPV6", "ICMP6"}:
            self.icmp_packets += 1
        else:
            self.other_transport_packets += 1

    def to_row(self) -> dict[str, int | float]:
        return {
            "packets_total": self.packets_total,
            "bytes_total": self.bytes_total,
            "ipv4_packets": self.ipv4_packets,
            "ipv6_packets": self.ipv6_packets,
            "tcp_packets": self.tcp_packets,
            "udp_packets": self.udp_packets,
            "icmp_packets": self.icmp_packets,
            "other_transport_packets": self.other_transport_packets,
            "tcp_syn": self.tcp_syn,
            "tcp_ack": self.tcp_ack,
            "tcp_rst": self.tcp_rst,
            "tcp_fin": self.tcp_fin,
            "unique_source_ips": len(self.source_ips),
            "unique_destination_ips": len(self.destination_ips),
            "mean_packet_bytes": self.bytes_total / self.packets_total,
            "max_packet_bytes": self.max_packet_bytes,
        }


def _load_capture(path: Path, expected_label: int) -> dict[datetime, WindowStats]:
    windows: dict[datetime, WindowStats] = defaultdict(WindowStats)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not INPUT_COLUMNS.issubset(reader.fieldnames):
            missing_columns = sorted(INPUT_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"{path}: kolom wajib tidak ditemukan: {', '.join(missing_columns)}")
        for line_number, row in enumerate(reader, start=2):
            timestamp = _parse_timestamp(row["timestamp"], path, line_number)
            try:
                label = int(row["label"])
                packet_length = int(row["packet_length"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: label/panjang paket tidak valid pada baris {line_number}.") from error
            if label != expected_label:
                raise ValueError(
                    f"{path}: label harus {expected_label} untuk semua paket; "
                    f"ditemukan {label} di baris {line_number}."
                )
            if packet_length < 0:
                raise ValueError(f"{path}: panjang paket negatif pada baris {line_number}.")

            source_version, source_ip = _parse_ip(row["src_ip"] or "", path, line_number)
            destination_version, destination_ip = _parse_ip(row["dst_ip"] or "", path, line_number)
            ip_versions = {
                version for version in (source_version, destination_version) if version is not None
            }
            flag_text = (row["tcp_flags"] or "").strip()
            try:
                tcp_flags = (
                    int(flag_text, 0)
                    if flag_text and (row["protocol"] or "").strip().upper() == "TCP"
                    else None
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}: tcp_flags tidak valid pada baris {line_number}: {flag_text!r}"
                ) from error
            windows[timestamp].add(
                row["protocol"] or "",
                packet_length,
                tcp_flags,
                source_ip,
                destination_ip,
                ip_versions,
            )
    if not windows:
        raise ValueError(f"{path}: tidak ada baris paket untuk diproses.")
    return windows


def prepare_labeled_csv(
    baseline_path: Path,
    attack_path: Path,
    output_path: Path,
    overwrite: bool = False,
    labels_path: Path | None = None,
) -> dict[str, int]:
    for path in (baseline_path, attack_path):
        if not path.is_file():
            raise FileNotFoundError(f"File input tidak ditemukan: {path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti.")
    if labels_path is None:
        raise ValueError("Manifest ground truth publik terverifikasi wajib disediakan.")
    if baseline_path.name == attack_path.name:
        raise ValueError("Nama file input harus unik agar dapat dipetakan ke manifest.")

    labels = _load_labels(labels_path, expected_source_dataset="public")
    input_names = {baseline_path.name, attack_path.name}
    missing_labels = sorted(input_names - set(labels))
    if missing_labels:
        raise ValueError(
            "Manifest publik belum memuat label terverifikasi untuk: "
            + ", ".join(missing_labels)
        )

    datasets = [(path, int(labels[path.name].label)) for path in (baseline_path, attack_path)]
    for path in (baseline_path, attack_path):
        if _sha256_file(path) != labels[path.name].sha256:
            raise ValueError(
                f"{path.name}: SHA-256 tidak cocok dengan manifest ground truth publik."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    total_packets = 0
    total_windows = 0
    packet_counts = {1: 0, -1: 0}
    try:
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
            for path, label in datasets:
                windows = _load_capture(path, label)
                if _sha256_file(path) != labels[path.name].sha256:
                    raise ValueError(
                        f"{path.name}: file berubah saat diproses; SHA-256 tidak lagi cocok "
                        "dengan manifest ground truth publik."
                    )
                for window_start in sorted(windows):
                    stats = windows[window_start]
                    writer.writerow(
                        {
                            "source_capture": path.name,
                            "source_sha256": labels[path.name].sha256,
                            "window_start": window_start.isoformat(),
                            **stats.to_row(),
                            "label": label,
                        }
                    )
                    total_packets += stats.packets_total
                    total_windows += 1
                    packet_counts[label] += stats.packets_total
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return {
        "capture_count": len(datasets),
        "packets": total_packets,
        "windows": total_windows,
        "normal_packets": packet_counts[1],
        "attack_packets": packet_counts[-1],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agregasikan dua CSV paket; label kelas ditentukan hanya oleh manifest "
            "ground truth publik terverifikasi."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="CSV sumber pertama; kelas tidak ditentukan dari nama file.",
    )
    parser.add_argument(
        "--attack",
        required=True,
        type=Path,
        help="CSV sumber kedua; kelas tidak ditentukan dari nama file.",
    )
    parser.add_argument(
        "--labels-csv",
        required=True,
        type=Path,
        help="Manifest publik terverifikasi; label dan bukti sumber diambil dari manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/training_features_labeled.csv"),
        help="Path output (default: data/processed/training_features_labeled.csv).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Izinkan mengganti file output yang sudah ada."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = prepare_labeled_csv(
            args.baseline,
            args.attack,
            args.output,
            args.overwrite,
            args.labels_csv,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 2
    print(
        f"Selesai: {result['capture_count']} sumber, {result['packets']} paket, "
        f"{result['windows']} window. Normal={result['normal_packets']}, "
        f"attack={result['attack_packets']}. Output: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
