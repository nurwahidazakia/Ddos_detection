"""Create time-windowed training features from PCAP and PCAPNG captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import struct
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO
from zipfile import BadZipFile, ZipFile, ZipInfo

from .pcap_profiling import PCAPNG_MAGIC, PCAP_MAGICS


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
CSV_COLUMNS = ("source_capture", "window_start_utc", *FEATURE_COLUMNS, "label")
LABEL_MANIFEST_COLUMNS = {
    "file",
    "source_dataset",
    "sha256",
    "scope",
    "label",
    "subtype",
    "start_utc",
    "end_utc",
    "timezone",
    "evidence_type",
    "evidence_reference",
    "review_status",
    "verified_by",
    "verified_at_utc",
    "notes",
}
LABEL_MANIFEST_SOURCES = {"bps", "public"}
VLAN_TYPES = {0x8100, 0x88A8, 0x9100}
IPV6_EXTENSION_HEADERS = {0, 43, 44, 51, 60, 135, 139, 140}


@dataclass(frozen=True)
class CaptureLabel:
    label: str
    source_dataset: str
    sha256: str
    start_utc: datetime | None
    end_utc: datetime | None


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
    source_ips: set[bytes] = field(default_factory=set)
    destination_ips: set[bytes] = field(default_factory=set)

    def add(self, packet: bytes) -> None:
        self.packets_total += 1
        self.bytes_total += len(packet)
        self.max_packet_bytes = max(self.max_packet_bytes, len(packet))

        network_type = _ethernet_payload_offset(packet)
        if network_type is None:
            self.other_transport_packets += 1
            return

        offset, ethertype = network_type
        if ethertype == 0x0800:
            parsed = _parse_ipv4(packet, offset)
            if parsed is None:
                self.other_transport_packets += 1
                return
            protocol, source_ip, destination_ip, transport_offset = parsed
            self.ipv4_packets += 1
        elif ethertype == 0x86DD:
            parsed = _parse_ipv6(packet, offset)
            if parsed is None:
                self.other_transport_packets += 1
                return
            protocol, source_ip, destination_ip, transport_offset = parsed
            self.ipv6_packets += 1
        else:
            self.other_transport_packets += 1
            return

        self.source_ips.add(source_ip)
        self.destination_ips.add(destination_ip)
        if protocol == 6:
            self.tcp_packets += 1
            if len(packet) >= transport_offset + 14:
                flags = packet[transport_offset + 13]
                self.tcp_fin += bool(flags & 0x01)
                self.tcp_syn += bool(flags & 0x02)
                self.tcp_rst += bool(flags & 0x04)
                self.tcp_ack += bool(flags & 0x10)
        elif protocol == 17:
            self.udp_packets += 1
        elif protocol in (1, 58):
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


def _ethernet_payload_offset(packet: bytes) -> tuple[int, int] | None:
    if len(packet) < 14:
        return None
    ethertype = int.from_bytes(packet[12:14], "big")
    offset = 14
    while ethertype in VLAN_TYPES:
        if len(packet) < offset + 4:
            return None
        ethertype = int.from_bytes(packet[offset + 2 : offset + 4], "big")
        offset += 4
    return offset, ethertype


def _parse_ipv4(packet: bytes, offset: int) -> tuple[int, bytes, bytes, int] | None:
    if len(packet) < offset + 20 or packet[offset] >> 4 != 4:
        return None
    header_length = (packet[offset] & 0x0F) * 4
    if header_length < 20 or len(packet) < offset + header_length:
        return None
    fragment = int.from_bytes(packet[offset + 6 : offset + 8], "big")
    protocol = packet[offset + 9]
    if fragment & 0x1FFF:
        protocol = 255
    return (
        protocol,
        packet[offset + 12 : offset + 16],
        packet[offset + 16 : offset + 20],
        offset + header_length,
    )


def _parse_ipv6(packet: bytes, offset: int) -> tuple[int, bytes, bytes, int] | None:
    if len(packet) < offset + 40 or packet[offset] >> 4 != 6:
        return None
    protocol = packet[offset + 6]
    transport_offset = offset + 40
    noninitial_fragment = False
    while protocol in IPV6_EXTENSION_HEADERS:
        if len(packet) < transport_offset + 2:
            return None
        next_protocol = packet[transport_offset]
        extension_length = (
            (packet[transport_offset + 1] + 2) * 4
            if protocol == 51
            else (packet[transport_offset + 1] + 1) * 8
        )
        if len(packet) < transport_offset + extension_length:
            return None
        if protocol == 44:
            fragment = int.from_bytes(packet[transport_offset + 2 : transport_offset + 4], "big")
            noninitial_fragment = bool((fragment >> 3) & 0x1FFF)
        transport_offset += extension_length
        protocol = next_protocol
    if noninitial_fragment:
        protocol = 255
    return (
        protocol,
        packet[offset + 8 : offset + 24],
        packet[offset + 24 : offset + 40],
        transport_offset,
    )


def _parse_utc_timestamp(value: str, path: Path, line_number: int, field_name: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"{path}: {field_name} tidak valid pada baris {line_number}."
        ) from error
    if timestamp.tzinfo is None:
        raise ValueError(
            f"{path}: {field_name} harus memiliki zona waktu pada baris {line_number}."
        )
    return timestamp.astimezone(timezone.utc)


def _datetime_to_nanoseconds(timestamp: datetime) -> int:
    delta = timestamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


def _load_labels(
    path: Path | None,
    expected_source_dataset: str | None = None,
) -> dict[str, CaptureLabel]:
    if path is None:
        return {}
    if expected_source_dataset is not None:
        expected_source_dataset = expected_source_dataset.strip().casefold()
        if expected_source_dataset not in LABEL_MANIFEST_SOURCES:
            raise ValueError(f"Sumber manifest tidak didukung: {expected_source_dataset}")
    labels: dict[str, CaptureLabel] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing_columns = LABEL_MANIFEST_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "Manifest ground truth harus memiliki kolom: "
                + ", ".join(sorted(LABEL_MANIFEST_COLUMNS))
                + ". Kolom yang belum ada: "
                + ", ".join(sorted(missing_columns))
            )
        for line_number, row in enumerate(reader, start=2):
            filename = (row.get("file") or "").strip()
            source_dataset = (row.get("source_dataset") or "").strip().casefold()
            review_status = (row.get("review_status") or "").strip().casefold()
            label = (row.get("label") or "").strip()
            sha256 = (row.get("sha256") or "").strip().casefold()
            scope = (row.get("scope") or "").strip().casefold()
            timezone_name = (row.get("timezone") or "").strip().upper()
            evidence_type = (row.get("evidence_type") or "").strip()
            evidence_reference = (row.get("evidence_reference") or "").strip()
            verified_by = (row.get("verified_by") or "").strip()
            if not filename:
                raise ValueError(f"Nama file kosong pada baris {line_number} manifest.")
            if source_dataset not in LABEL_MANIFEST_SOURCES:
                raise ValueError(
                    f"source_dataset harus bps atau public pada baris {line_number}."
                )
            if (
                expected_source_dataset is not None
                and source_dataset != expected_source_dataset
            ):
                raise ValueError(
                    f"Manifest {path} memuat source_dataset={source_dataset} pada baris "
                    f"{line_number}; diharapkan {expected_source_dataset}."
                )
            if review_status != "verified":
                raise ValueError(
                    f"Baris {line_number} berstatus {review_status or 'unknown'}; "
                    "hanya baris verified yang dapat digunakan untuk training."
                )
            if label not in {"1", "-1"}:
                raise ValueError(
                    f"Label harus 1 (normal) atau -1 (attack) pada baris {line_number}."
                )
            if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
                raise ValueError(
                    f"SHA-256 harus berisi 64 digit heksadesimal pada baris {line_number}."
                )
            if scope != "capture":
                raise ValueError(
                    f"scope harus capture; label parsial belum didukung "
                    f"(baris {line_number})."
                )
            if not evidence_type or not evidence_reference or not verified_by:
                raise ValueError(
                    f"evidence_type, evidence_reference, dan verified_by wajib diisi "
                    f"pada baris {line_number}."
                )
            start_value = (row.get("start_utc") or "").strip()
            end_value = (row.get("end_utc") or "").strip()
            if source_dataset == "bps":
                if timezone_name != "UTC":
                    raise ValueError(
                        f"timezone harus UTC pada baris {line_number}; "
                        "jangan mengasumsikan zona waktu."
                    )
                coverage_start = _parse_utc_timestamp(
                    start_value, path, line_number, "start_utc"
                )
                coverage_end = _parse_utc_timestamp(
                    end_value, path, line_number, "end_utc"
                )
                if coverage_end < coverage_start:
                    raise ValueError(
                        f"end_utc lebih awal dari start_utc "
                        f"pada baris {line_number}."
                    )
            elif (
                start_value.casefold() in {"unknown", "not_provided"}
                and end_value.casefold() in {"unknown", "not_provided"}
                and timezone_name.casefold() in {"unknown", "not_provided"}
            ):
                coverage_start = None
                coverage_end = None
            elif timezone_name == "UTC":
                coverage_start = _parse_utc_timestamp(
                    start_value, path, line_number, "start_utc"
                )
                coverage_end = _parse_utc_timestamp(
                    end_value, path, line_number, "end_utc"
                )
                if coverage_end < coverage_start:
                    raise ValueError(
                        f"end_utc lebih awal dari start_utc "
                        f"pada baris {line_number}."
                    )
            else:
                raise ValueError(
                    f"Timestamp publik harus UTC atau ditandai unknown pada "
                    f"baris {line_number}."
                )
            _parse_utc_timestamp(
                row.get("verified_at_utc") or "",
                path,
                line_number,
                "verified_at_utc",
            )
            if filename in labels:
                raise ValueError(f"Nama file duplikat pada baris {line_number}: {filename}")
            labels[filename] = CaptureLabel(
                label=label,
                source_dataset=source_dataset,
                sha256=sha256,
                start_utc=coverage_start,
                end_utc=coverage_end,
            )
    return labels


def _write_capture(
    source: IO[bytes],
    capture_name: str,
    size_bytes: int,
    writer: csv.DictWriter,
    window_seconds: int,
    capture_label: CaptureLabel | None,
) -> tuple[int, int]:
    signature = source.read(4)
    source.seek(0)
    if signature == PCAPNG_MAGIC:
        return _write_pcapng_capture(
            source,
            capture_name,
            size_bytes,
            writer,
            window_seconds,
            capture_label,
        )

    global_header = source.read(24)
    if len(global_header) != 24:
        raise ValueError(f"{capture_name}: header PCAP global tidak lengkap.")
    digest = hashlib.sha256()
    if capture_label is not None:
        digest.update(global_header)
    metadata = PCAP_MAGICS.get(global_header[:4])
    if metadata is None:
        raise ValueError(f"{capture_name}: hanya classic PCAP yang didukung.")
    byte_order, ticks_per_second, _precision = metadata
    major, minor, _offset, _accuracy, snaplen, linktype = struct.unpack(
        f"{byte_order}HHiIII", global_header[4:]
    )
    if (major, minor) != (2, 4):
        raise ValueError(f"{capture_name}: versi PCAP tidak didukung: {major}.{minor}.")
    if linktype != 1:
        raise ValueError(
            f"{capture_name}: linktype {linktype} tidak didukung; diperlukan Ethernet (1)."
        )

    windows: dict[int, WindowStats] = {}
    packet_count = 0
    capture_start_ns: int | None = None
    capture_end_ns: int | None = None
    while True:
        record_offset = source.tell()
        record_header = source.read(16)
        if not record_header:
            break
        if len(record_header) != 16:
            raise ValueError(f"{capture_name}: header paket terpotong pada offset {record_offset}.")
        if capture_label is not None:
            digest.update(record_header)
        seconds, fraction, captured_length, original_length = struct.unpack(
            f"{byte_order}IIII", record_header
        )
        if fraction >= ticks_per_second:
            raise ValueError(f"{capture_name}: timestamp tidak valid pada offset {record_offset}.")
        if captured_length > original_length or (snaplen and captured_length > snaplen):
            raise ValueError(f"{capture_name}: panjang paket tidak valid pada offset {record_offset}.")
        if captured_length > size_bytes - source.tell():
            raise ValueError(f"{capture_name}: payload paket terpotong pada offset {record_offset}.")
        packet = source.read(captured_length)
        if len(packet) != captured_length:
            raise ValueError(f"{capture_name}: payload paket tidak lengkap pada offset {record_offset}.")
        if capture_label is not None:
            digest.update(packet)

        timestamp_ns = seconds * 1_000_000_000 + fraction * (
            1_000_000_000 // ticks_per_second
        )
        capture_start_ns = (
            timestamp_ns if capture_start_ns is None else min(capture_start_ns, timestamp_ns)
        )
        capture_end_ns = (
            timestamp_ns if capture_end_ns is None else max(capture_end_ns, timestamp_ns)
        )
        window = seconds // window_seconds
        windows.setdefault(window, WindowStats()).add(packet)
        packet_count += 1

    if capture_label is not None:
        if capture_start_ns is None or capture_end_ns is None:
            raise ValueError(f"{capture_name}: capture berlabel tidak memiliki paket.")
        if capture_label.start_utc is None or capture_label.end_utc is None:
            raise ValueError(
                f"{capture_name}: cakupan waktu UTC wajib untuk memberi label PCAP."
            )
        if (
            _datetime_to_nanoseconds(capture_label.start_utc) > capture_start_ns
            or _datetime_to_nanoseconds(capture_label.end_utc) < capture_end_ns
        ):
            raise ValueError(
                f"{capture_name}: cakupan waktu manifest tidak mencakup seluruh timestamp "
                "capture; label interval belum didukung."
            )
        if digest.hexdigest() != capture_label.sha256:
            raise ValueError(
                f"{capture_name}: SHA-256 tidak cocok dengan manifest; "
                "capture mungkin berubah atau salah dipasangkan."
            )

    for window in sorted(windows):
        _write_window(
            writer,
            capture_name,
            window,
            windows[window],
            capture_label.label if capture_label is not None else "",
            window_seconds,
        )
    return packet_count, len(windows)


def _pcapng_options(
    payload: bytes,
    byte_order: str,
    capture_name: str,
) -> dict[int, list[bytes]]:
    options: dict[int, list[bytes]] = {}
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < 4:
            raise ValueError(f"{capture_name}: opsi PCAPNG terpotong.")
        code, length = struct.unpack(
            f"{byte_order}HH", payload[offset : offset + 4]
        )
        offset += 4
        if code == 0:
            if length != 0:
                raise ValueError(f"{capture_name}: opsi akhir PCAPNG tidak valid.")
            if any(payload[offset:]):
                raise ValueError(f"{capture_name}: data setelah opsi akhir tidak kosong.")
            break
        padded_length = (length + 3) & ~3
        if padded_length > len(payload) - offset:
            raise ValueError(f"{capture_name}: nilai opsi PCAPNG terpotong.")
        options.setdefault(code, []).append(payload[offset : offset + length])
        offset += padded_length
    return options


def _write_pcapng_capture(
    source: IO[bytes],
    capture_name: str,
    size_bytes: int,
    writer: csv.DictWriter,
    window_seconds: int,
    capture_label: CaptureLabel | None,
) -> tuple[int, int]:
    interfaces: list[tuple[int, int, int, int]] = []
    byte_order: str | None = None
    digest = hashlib.sha256()
    windows: dict[int, WindowStats] = {}
    packet_count = 0
    capture_start_ns: int | None = None
    capture_end_ns: int | None = None
    maximum_block_size = 64 * 1024 * 1024

    while True:
        block_offset = source.tell()
        header = source.read(8)
        if not header:
            break
        if len(header) != 8:
            raise ValueError(
                f"{capture_name}: header blok PCAPNG terpotong pada offset {block_offset}."
            )

        is_section_header = header[:4] == PCAPNG_MAGIC
        if is_section_header:
            byte_order_magic = source.read(4)
            byte_orders = {
                b"\x4d\x3c\x2b\x1a": "<",
                b"\x1a\x2b\x3c\x4d": ">",
            }
            byte_order = byte_orders.get(byte_order_magic)
            if byte_order is None:
                raise ValueError(
                    f"{capture_name}: byte-order magic PCAPNG tidak valid "
                    f"pada offset {block_offset}."
                )
            block_length = struct.unpack(f"{byte_order}I", header[4:8])[0]
            prefix = header + byte_order_magic
            minimum_length = 28
        else:
            if byte_order is None:
                raise ValueError(
                    f"{capture_name}: section header PCAPNG tidak ditemukan."
                )
            block_length = struct.unpack(f"{byte_order}I", header[4:8])[0]
            prefix = header
            minimum_length = 12

        if (
            block_length < minimum_length
            or block_length % 4
            or block_length > maximum_block_size
            or block_length > size_bytes - block_offset
        ):
            raise ValueError(
                f"{capture_name}: panjang blok PCAPNG tidak valid "
                f"pada offset {block_offset}."
            )
        remainder = source.read(block_length - len(prefix))
        if len(remainder) != block_length - len(prefix):
            raise ValueError(
                f"{capture_name}: blok PCAPNG terpotong pada offset {block_offset}."
            )
        block = prefix + remainder
        if struct.unpack(f"{byte_order}I", block[-4:])[0] != block_length:
            raise ValueError(
                f"{capture_name}: panjang blok PCAPNG tidak cocok "
                f"pada offset {block_offset}."
            )
        digest.update(block)
        block_type = struct.unpack(f"{byte_order}I", block[:4])[0]

        if is_section_header:
            major, _minor = struct.unpack(f"{byte_order}HH", block[12:16])
            if major != 1:
                raise ValueError(
                    f"{capture_name}: versi PCAPNG tidak didukung pada offset "
                    f"{block_offset}."
                )
            interfaces = []
            continue

        body = block[8:-4]
        if block_type == 1:
            if len(body) < 8:
                raise ValueError(
                    f"{capture_name}: interface description PCAPNG tidak lengkap."
                )
            linktype, _reserved, snaplen = struct.unpack(
                f"{byte_order}HHI", body[:8]
            )
            if linktype != 1:
                raise ValueError(
                    f"{capture_name}: linktype {linktype} tidak didukung; "
                    "diperlukan Ethernet (1)."
                )
            timestamp_denominator = 1_000_000
            timestamp_offset_seconds = 0
            options = _pcapng_options(body[8:], byte_order, capture_name)
            if 9 in options:
                if len(options[9]) != 1 or len(options[9][0]) != 1:
                    raise ValueError(
                        f"{capture_name}: if_tsresol PCAPNG tidak valid."
                    )
                resolution = options[9][0][0]
                exponent = resolution & 0x7F
                timestamp_denominator = (
                    1 << exponent if resolution & 0x80 else 10**exponent
                )
            if 14 in options:
                if len(options[14]) != 1 or len(options[14][0]) != 8:
                    raise ValueError(
                        f"{capture_name}: if_tsoffset PCAPNG tidak valid."
                    )
                timestamp_offset_seconds = struct.unpack(
                    f"{byte_order}q", options[14][0]
                )[0]
            interfaces.append(
                (
                    linktype,
                    snaplen,
                    timestamp_denominator,
                    timestamp_offset_seconds,
                )
            )
            continue

        if block_type not in {2, 6}:
            continue
        if len(body) < 20:
            raise ValueError(
                f"{capture_name}: packet block PCAPNG tidak lengkap "
                f"pada offset {block_offset}."
            )
        if block_type == 6:
            interface_id, timestamp_high, timestamp_low, captured_length, original_length = (
                struct.unpack(f"{byte_order}IIIII", body[:20])
            )
        else:
            interface_id, _drops, timestamp_high, timestamp_low, captured_length, original_length = (
                struct.unpack(f"{byte_order}HHIIII", body[:20])
            )
        if interface_id >= len(interfaces):
            raise ValueError(
                f"{capture_name}: interface ID PCAPNG tidak valid "
                f"pada offset {block_offset}."
            )
        linktype, snaplen, timestamp_denominator, timestamp_offset_seconds = (
            interfaces[interface_id]
        )
        if captured_length > original_length or (
            snaplen and captured_length > snaplen
        ):
            raise ValueError(
                f"{capture_name}: panjang paket PCAPNG tidak valid "
                f"pada offset {block_offset}."
            )
        padded_length = (captured_length + 3) & ~3
        if 20 + padded_length > len(body):
            raise ValueError(
                f"{capture_name}: payload paket PCAPNG terpotong "
                f"pada offset {block_offset}."
            )
        packet = body[20 : 20 + captured_length]
        timestamp_ticks = (timestamp_high << 32) | timestamp_low
        timestamp_ns = (
            timestamp_ticks * 1_000_000_000 // timestamp_denominator
            + timestamp_offset_seconds * 1_000_000_000
        )
        capture_start_ns = (
            timestamp_ns if capture_start_ns is None else min(capture_start_ns, timestamp_ns)
        )
        capture_end_ns = (
            timestamp_ns if capture_end_ns is None else max(capture_end_ns, timestamp_ns)
        )
        window = timestamp_ns // (window_seconds * 1_000_000_000)
        windows.setdefault(window, WindowStats()).add(packet)
        packet_count += 1

    if packet_count == 0:
        raise ValueError(f"{capture_name}: tidak ada packet block bertimestamp di PCAPNG.")
    if capture_label is not None:
        if capture_label.start_utc is None or capture_label.end_utc is None:
            raise ValueError(
                f"{capture_name}: cakupan waktu UTC wajib untuk memberi label PCAP."
            )
        if (
            _datetime_to_nanoseconds(capture_label.start_utc) > capture_start_ns
            or _datetime_to_nanoseconds(capture_label.end_utc) < capture_end_ns
        ):
            raise ValueError(
                f"{capture_name}: cakupan waktu manifest tidak mencakup seluruh timestamp "
                "capture; label interval belum didukung."
            )
        if digest.hexdigest() != capture_label.sha256:
            raise ValueError(
                f"{capture_name}: SHA-256 tidak cocok dengan manifest; "
                "capture mungkin berubah atau salah dipasangkan."
            )

    for window in sorted(windows):
        _write_window(
            writer,
            capture_name,
            window,
            windows[window],
            capture_label.label if capture_label is not None else "",
            window_seconds,
        )
    return packet_count, len(windows)


def _write_window(
    writer: csv.DictWriter,
    capture_name: str,
    window_index: int,
    stats: WindowStats,
    label: str,
    window_seconds: int,
) -> None:
    window_start = datetime.fromtimestamp(
        window_index * window_seconds, timezone.utc
    ).isoformat()
    writer.writerow(
        {
            "source_capture": capture_name,
            "window_start_utc": window_start,
            **stats.to_row(),
            "label": label,
        }
    )


def prepare_training_data(
    input_path: Path,
    output_path: Path,
    labels_path: Path | None = None,
    window_seconds: int = 1,
    overwrite: bool = False,
    include_captures: set[str] | None = None,
    expected_source_dataset: str | None = None,
) -> dict[str, int]:
    if window_seconds < 1:
        raise ValueError("Durasi window harus minimal 1 detik.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    is_zip_input = input_path.is_file() and input_path.suffix.lower() == ".zip"
    captures: list[Path] = []
    archive_entries: list[ZipInfo] = []
    if is_zip_input:
        try:
            with ZipFile(input_path) as archive:
                archive_entries = sorted(
                    (entry for entry in archive.infolist() if not entry.is_dir()),
                    key=lambda entry: entry.filename,
                )
        except BadZipFile as error:
            raise ValueError(f"File ZIP tidak valid: {input_path}") from error
        capture_names = [Path(entry.filename).name for entry in archive_entries]
        if len(capture_names) != len(set(capture_names)):
            raise ValueError("Nama file di dalam ZIP harus unik agar manifest label tidak ambigu.")
    else:
        captures = sorted(
            path
            for path in (input_path.iterdir() if input_path.is_dir() else [input_path])
            if path.is_file()
        )
        capture_names = [path.name for path in captures]
    if not capture_names:
        raise ValueError(f"Tidak ada file input di {input_path}.")
    if include_captures is not None:
        unknown_captures = sorted(include_captures - set(capture_names))
        if unknown_captures:
            raise ValueError(
                "Capture yang diminta tidak ditemukan: " + ", ".join(unknown_captures)
            )
        if is_zip_input:
            archive_entries = [
                entry for entry in archive_entries
                if Path(entry.filename).name in include_captures
            ]
        else:
            captures = [capture for capture in captures if capture.name in include_captures]
        capture_names = sorted(include_captures)
    if not capture_names:
        raise ValueError(f"Tidak ada capture terpilih di {input_path}.")
    labels = _load_labels(labels_path, expected_source_dataset)
    input_names = set(capture_names)
    missing_labels = sorted(input_names - labels.keys()) if labels_path else []
    unknown_labels = sorted(labels.keys() - input_names)
    if unknown_labels:
        raise ValueError(f"CSV label memuat file yang tidak ditemukan: {', '.join(unknown_labels)}")
    if missing_labels:
        raise ValueError(
            "Label belum tersedia untuk: "
            + ", ".join(missing_labels)
            + "; lengkapi CSV label agar seluruh capture berlabel."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    total_packets = 0
    total_windows = 0
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
            writer = csv.DictWriter(temporary_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            if is_zip_input:
                try:
                    with ZipFile(input_path) as archive:
                        for entry in archive_entries:
                            capture_name = Path(entry.filename).name
                            with archive.open(entry) as source:
                                packets, windows = _write_capture(
                                    source,
                                    capture_name,
                                    entry.file_size,
                                    writer,
                                    window_seconds,
                                    labels.get(capture_name),
                                )
                            total_packets += packets
                            total_windows += windows
                except BadZipFile as error:
                    raise ValueError(f"File ZIP tidak valid atau rusak: {input_path}") from error
            else:
                for capture in captures:
                    with capture.open("rb") as source:
                        packets, windows = _write_capture(
                            source,
                            capture.name,
                            capture.stat().st_size,
                            writer,
                            window_seconds,
                            labels.get(capture.name),
                        )
                    total_packets += packets
                    total_windows += windows
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output sudah ada: {output_path}; gunakan --overwrite untuk mengganti."
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return {"capture_count": len(capture_names), "packets": total_packets, "windows": total_windows}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ekstrak fitur per-window dari PCAP/PCAPNG atau ZIP berisi capture "
            "untuk menyiapkan dataset training."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="File PCAP/PCAPNG, direktori berisi capture, atau ZIP berisi capture.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/training_features.csv"),
        help="Path CSV output (default: data/processed/training_features.csv).",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help=(
            "Manifest ground truth terverifikasi dengan kolom file,label, "
            "cakupan waktu UTC, referensi bukti, dan verifier."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=1,
        help="Durasi bin waktu dalam detik (default: 1).",
    )
    parser.add_argument(
        "--include-captures",
        nargs="+",
        metavar="NAME",
        help="Proses hanya nama capture yang disebutkan (pilihan; exact filename).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Izinkan mengganti file output yang sudah ada."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = prepare_training_data(
            args.input,
            args.output,
            args.labels_csv,
            args.window_seconds,
            args.overwrite,
            set(args.include_captures) if args.include_captures else None,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError, BadZipFile) as error:
        print(f"Error: {error}")
        return 2
    print(
        f"Selesai: {result['capture_count']} capture, {result['packets']} paket, "
        f"{result['windows']} window. Output: {args.output}"
    )
    if args.labels_csv is None:
        print("Label belum diisi; gunakan --labels-csv sebelum melatih model supervised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
