"""Bounded-memory profiling for classic PCAP capture files."""

from __future__ import annotations

import math
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, IO


PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000_000, "microsecond"),
    b"\xa1\xb2\xc3\xd4": (">", 1_000_000, "microsecond"),
    b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000, "nanosecond"),
    b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000, "nanosecond"),
}
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"


def is_pcap_signature(signature: bytes) -> bool:
    return signature in PCAP_MAGICS or signature == PCAPNG_MAGIC


def _timestamp_string(seconds: int, fraction: int, precision: str) -> str:
    whole_seconds = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    digits = 6 if precision == "microsecond" else 9
    return f"{whole_seconds}.{fraction:0{digits}d}Z"


def profile_pcap(
    path: Path,
    max_packets: int | None = None,
    decode_packets: bool = False,
) -> dict[str, Any]:
    """Read classic PCAP packet headers sequentially; packet payloads are skipped."""
    file_size = path.stat().st_size
    with path.open("rb") as file_handle:
        return profile_pcap_stream(
            file_handle,
            file_size,
            max_packets=max_packets,
            decode_packets=decode_packets,
        )


def profile_pcap_stream(
    file_handle: IO[bytes],
    file_size: int,
    max_packets: int | None = None,
    decode_packets: bool = False,
) -> dict[str, Any]:
    """Profile a seekable binary stream containing classic PCAP data."""
    global_header = file_handle.read(24)
    if len(global_header) != 24:
        raise ValueError("Header global PCAP tidak lengkap (diperlukan 24 byte).")

    magic = global_header[:4]
    metadata = PCAP_MAGICS.get(magic)
    if metadata is None:
        if magic == PCAPNG_MAGIC:
            raise ValueError("PCAPNG terdeteksi, tetapi profiler saat ini hanya mendukung classic PCAP.")
        raise ValueError("Signature file tidak cocok dengan classic PCAP yang didukung.")

    byte_order, ticks_per_second, timestamp_precision = metadata
    major, minor, _, _, snaplen, linktype = struct.unpack(
        f"{byte_order}HHiIII", global_header[4:]
    )
    if major != 2:
        raise ValueError(f"Versi major PCAP tidak didukung: {major}.{minor}.")
    packet_parsers = None
    if decode_packets and linktype == 1:
        from .prepare_training_data import (
            _ethernet_payload_offset,
            _parse_ipv4,
            _parse_ipv6,
        )

        packet_parsers = (_ethernet_payload_offset, _parse_ipv4, _parse_ipv6)

    packet_count = 0
    captured_bytes = 0
    original_bytes = 0
    captured_length_squared_total = 0
    captured_length_min: int | None = None
    captured_length_max = 0
    truncated_packets = 0
    invalid_timestamps = 0
    protocol_counts: Counter[str] = Counter()
    first_timestamp: tuple[int, int] | None = None
    last_timestamp: tuple[int, int] | None = None

    while max_packets is None or packet_count < max_packets:
        record_offset = file_handle.tell()
        record_header = file_handle.read(16)
        if not record_header:
            break
        if len(record_header) != 16:
            raise ValueError(f"Header record terpotong pada offset byte {record_offset}.")

        timestamp_seconds, timestamp_fraction, captured_length, original_length = struct.unpack(
            f"{byte_order}IIII", record_header
        )
        payload_offset = file_handle.tell()
        remaining_bytes = file_size - payload_offset
        if captured_length > remaining_bytes:
            raise ValueError(
                f"Payload paket {packet_count + 1} melebihi sisa file pada offset byte {record_offset}."
            )
        if snaplen and captured_length > snaplen:
            raise ValueError(
                f"Panjang capture paket {packet_count + 1} melebihi snaplen pada offset byte {record_offset}."
            )
        if captured_length > original_length:
            raise ValueError(
                f"Panjang capture paket {packet_count + 1} melebihi panjang asli pada offset byte {record_offset}."
            )

        packet = file_handle.read(captured_length) if decode_packets and linktype == 1 else None
        if packet is None:
            file_handle.seek(captured_length, 1)
        packet_count += 1
        captured_bytes += captured_length
        captured_length_squared_total += captured_length * captured_length
        captured_length_min = (
            captured_length
            if captured_length_min is None
            else min(captured_length_min, captured_length)
        )
        captured_length_max = max(captured_length_max, captured_length)
        original_bytes += original_length
        truncated_packets += captured_length < original_length

        if packet is not None:
            assert packet_parsers is not None
            _count_packet_protocols(packet, protocol_counts, packet_parsers)

        if timestamp_fraction >= ticks_per_second:
            invalid_timestamps += 1
        else:
            timestamp = (timestamp_seconds, timestamp_fraction)
            if first_timestamp is None or timestamp < first_timestamp:
                first_timestamp = timestamp
            if last_timestamp is None or timestamp > last_timestamp:
                last_timestamp = timestamp

    complete = file_handle.tell() == file_size
    mean_packet_length = None
    packet_length_std = None
    if packet_count:
        mean_packet_length = captured_bytes / packet_count
        packet_length_variance = max(
            0.0,
            captured_length_squared_total / packet_count - mean_packet_length**2,
        )
        packet_length_std = math.sqrt(packet_length_variance)
    return {
        "status": "profiled" if complete else "partial_profile",
        "format": "classic_pcap",
        "capture_complete": complete,
        "packets_scanned": packet_count,
        "captured_packet_bytes_scanned": captured_bytes,
        "captured_payload_bytes_scanned": captured_bytes,
        "original_packet_bytes": original_bytes,
        "packet_length_bytes": {
            "min": captured_length_min,
            "max": captured_length_max if packet_count else None,
            "mean": mean_packet_length,
            "std": packet_length_std,
            "length_basis": "captured frame length",
        },
        "protocol_distribution": dict(protocol_counts),
        "protocol_decode_status": (
            "decoded" if decode_packets and linktype == 1 else "not_decoded"
        ),
        "data_granularity": "packet_level_capture",
        "truncated_packets": truncated_packets,
        "invalid_timestamp_count": invalid_timestamps,
        "pcap_version": f"{major}.{minor}",
        "snaplen_bytes": snaplen,
        "linktype": linktype,
        "timestamp_precision": timestamp_precision,
        "timestamp_min_utc": (
            _timestamp_string(*first_timestamp, timestamp_precision) if first_timestamp else None
        ),
        "timestamp_max_utc": (
            _timestamp_string(*last_timestamp, timestamp_precision) if last_timestamp else None
        ),
    }


def _count_packet_protocols(
    packet: bytes,
    counts: Counter[str],
    parsers: tuple[
        Callable[[bytes], tuple[int, int] | None],
        Callable[[bytes, int], tuple[int, bytes, bytes, int] | None],
        Callable[[bytes, int], tuple[int, bytes, bytes, int] | None],
    ],
) -> None:
    ethernet_payload_offset, parse_ipv4, parse_ipv6 = parsers
    network_type = ethernet_payload_offset(packet)
    if network_type is None:
        counts["other_or_truncated"] += 1
        return
    offset, ethertype = network_type
    if ethertype == 0x0800:
        parsed = parse_ipv4(packet, offset)
        if parsed is None:
            counts["other_or_truncated"] += 1
            return
        protocol = parsed[0]
        counts["IPv4"] += 1
    elif ethertype == 0x86DD:
        parsed = parse_ipv6(packet, offset)
        if parsed is None:
            counts["other_or_truncated"] += 1
            return
        protocol = parsed[0]
        counts["IPv6"] += 1
    else:
        counts["other_or_truncated"] += 1
        return

    if protocol == 6:
        counts["TCP"] += 1
    elif protocol == 17:
        counts["UDP"] += 1
    elif protocol in (1, 58):
        counts["ICMP/ICMPv6"] += 1
    else:
        counts["other_ip_protocol"] += 1
