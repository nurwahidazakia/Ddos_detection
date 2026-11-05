"""Run a locally configured CICFlowMeter command on an uploaded capture."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .pcap_profiling import PCAPNG_MAGIC, is_pcap_signature


COMMAND_ENV = "CICFLOWMETER_COMMAND"
HOME_ENV = "CICFLOWMETER_HOME"
VERSION_ENV = "CICFLOWMETER_VERSION"
COMMAND_PLACEHOLDERS = {"{input}", "{output_dir}"}
EXTRACTION_TIMEOUT_SECONDS = 30 * 60
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _capture_format(path: Path) -> str:
    with path.open("rb") as capture:
        signature = capture.read(4)
    if not is_pcap_signature(signature):
        raise ValueError("File bukan PCAP/PCAPNG yang dikenali.")
    if signature == PCAPNG_MAGIC:
        raise ValueError(
            "CICFlowMeter memerlukan file PCAP klasik. PCAPNG belum didukung "
            "oleh konverter ini; simpan ulang sebagai PCAP klasik terlebih dahulu."
        )
    return "pcap"


def _custom_command() -> list[str] | None:
    raw_command = os.environ.get(COMMAND_ENV, "").strip()
    if not raw_command:
        return None
    try:
        command = json.loads(raw_command)
    except json.JSONDecodeError as error:
        raise ValueError(f"{COMMAND_ENV} harus berisi JSON array argumen.") from error
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) or not argument for argument in command)
    ):
        raise ValueError(f"{COMMAND_ENV} harus berupa JSON array string yang tidak kosong.")
    placeholders = {
        placeholder
        for argument in command
        for placeholder in COMMAND_PLACEHOLDERS
        if placeholder in argument
    }
    if placeholders != COMMAND_PLACEHOLDERS:
        raise ValueError(
            f"{COMMAND_ENV} harus menyertakan placeholder {{input}} dan {{output_dir}}."
        )

    executable = command[0]
    if Path(executable).is_absolute() or any(separator in executable for separator in ("/", "\\")):
        if not Path(executable).is_file():
            raise FileNotFoundError(f"Program CICFlowMeter tidak ditemukan: {executable}")
    elif shutil.which(executable) is None:
        raise FileNotFoundError(
            f"Perintah {executable!r} tidak ditemukan di PATH; periksa {COMMAND_ENV}."
        )
    return command


def _java_executable() -> str:
    executable_name = "java.exe" if os.name == "nt" else "java"
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        executable = Path(java_home) / "bin" / executable_name
        if executable.is_file():
            return str(executable)
        raise FileNotFoundError(f"Java tidak ditemukan di JAVA_HOME: {executable}")
    executable = shutil.which("java")
    if executable is None:
        raise FileNotFoundError("Java tidak ditemukan. Pasang Java dan atur PATH atau JAVA_HOME.")
    return executable


def _distribution_command() -> list[str]:
    configured_home = os.environ.get(HOME_ENV, "").strip()
    home = Path(configured_home).expanduser() if configured_home else (
        PROJECT_ROOT / "tools" / "CICFlowMeter"
    )
    if not home.is_dir():
        raise RuntimeError(
            f"CICFlowMeter belum ditemukan. Atur {HOME_ENV} ke folder instalasi "
            "CICFlowMeter, atau letakkan distribusinya di tools\\CICFlowMeter."
        )

    library_directory = home / "lib"
    native_directory = library_directory / "native"
    if not library_directory.is_dir():
        raise FileNotFoundError(
            f"Folder lib CICFlowMeter tidak ditemukan: {library_directory}"
        )
    native_pattern = "*.dll" if os.name == "nt" else "*.dylib" if platform.system() == "Darwin" else "*.so"
    if not any(native_directory.glob(native_pattern)):
        raise FileNotFoundError(
            f"Library native jNetPcap tidak ditemukan di {native_directory}."
        )
    if not any(library_directory.glob("*.jar")):
        raise FileNotFoundError(
            f"File JAR CICFlowMeter tidak ditemukan di {library_directory}."
        )

    return [
        _java_executable(),
        f"-Djava.library.path={native_directory.resolve()}",
        "-cp",
        str((library_directory / "*").resolve()),
        "cic.cs.unb.ca.ifm.Cmd",
        "{input}",
        "{output_dir}",
    ]


def _configured_command() -> list[str]:
    custom = _custom_command()
    if custom is not None:
        return custom
    return _distribution_command()


def _extractor_version() -> str:
    configured_version = os.environ.get(VERSION_ENV, "").strip()
    if configured_version:
        return configured_version
    configured_home = os.environ.get(HOME_ENV, "").strip()
    if configured_home:
        return Path(configured_home).name
    return "unspecified"


def extractor_configuration_status() -> tuple[bool, str]:
    try:
        custom = _custom_command()
        if custom is None:
            _distribution_command()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        return False, str(error)
    return (
        True,
        "Distribusi CICFlowMeter lengkap ditemukan; unggah PCAP untuk mulai mengekstrak.",
    )


def extract_pcap_to_csv(
    capture_path: Path,
    output_directory: Path,
    *,
    timeout_seconds: int = EXTRACTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Extract one capture to exactly one CSV using a configured local command."""
    if not capture_path.is_file():
        raise FileNotFoundError(f"File PCAP tidak ditemukan: {capture_path}")
    capture_format = _capture_format(capture_path)
    if timeout_seconds < 1:
        raise ValueError("Batas waktu ekstraksi harus lebih besar dari nol.")

    configured = _configured_command()
    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        argument.replace("{input}", str(capture_path.resolve())).replace(
            "{output_dir}", str(output_directory.resolve())
        )
        for argument in configured
    ]

    log_path = output_directory / "cicflowmeter.log"
    try:
        with log_path.open("wb") as log_file:
            completed = subprocess.run(
                command,
                cwd=output_directory,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Ekstraksi PCAP melewati batas waktu {timeout_seconds} detik."
        ) from error
    except OSError as error:
        raise RuntimeError(f"Gagal menjalankan CICFlowMeter: {error}") from error

    if completed.returncode != 0:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - 8000))
            details = log_file.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"CICFlowMeter berhenti dengan kode {completed.returncode}."
            + (f" Pesan terakhir: {details}" if details else "")
        )

    csv_files = sorted(
        path for path in output_directory.rglob("*.csv") if path.is_file()
    )
    if len(csv_files) != 1:
        raise RuntimeError(
            "Ekstraksi harus menghasilkan tepat satu CSV, "
            f"tetapi ditemukan {len(csv_files)}."
        )
    if csv_files[0].stat().st_size == 0:
        raise RuntimeError("CICFlowMeter menghasilkan CSV kosong.")

    return {
        "csv_path": csv_files[0],
        "capture_format": capture_format,
        "extractor_version": _extractor_version(),
        "extractor_executable": (
            "CICFlowMeter Cmd"
            if Path(configured[0]).name.casefold().startswith("java")
            else Path(configured[0]).name
        ),
    }
