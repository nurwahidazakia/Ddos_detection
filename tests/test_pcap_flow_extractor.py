import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from src.pcap_flow_extractor import extract_pcap_to_csv


class PcapFlowExtractorTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.capture_path = self.root / "capture.pcap"
        self.capture_path.write_bytes(b"\xd4\xc3\xb2\xa1" + b"\x00" * 20)
        self.output_directory = self.root / "output"
        self.script = (
            "import pathlib,sys; "
            "capture=pathlib.Path(sys.argv[1]); "
            "out=pathlib.Path(sys.argv[2]); "
            "out.mkdir(parents=True,exist_ok=True); "
            "(out/'flows.csv').write_text('Protocol\\n6\\n',encoding='utf-8')"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _set_command(self):
        command = [
            sys.executable,
            "-c",
            self.script,
            "{input}",
            "{output_dir}",
        ]
        return patch.dict(
            os.environ,
            {
                "CICFLOWMETER_COMMAND": json.dumps(command),
                "CICFLOWMETER_VERSION": "test-version",
            },
        )

    def test_runs_configured_extractor_and_returns_single_csv(self):
        with self._set_command():
            result = extract_pcap_to_csv(self.capture_path, self.output_directory)

        self.assertEqual(result["capture_format"], "pcap")
        self.assertEqual(result["extractor_version"], "test-version")
        self.assertEqual(result["csv_path"].name, "flows.csv")
        self.assertEqual(result["csv_path"].read_text(encoding="utf-8"), "Protocol\n6\n")

    def test_requires_extractor_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CICFLOWMETER_HOME"):
                extract_pcap_to_csv(self.capture_path, self.output_directory)

    def test_rejects_non_pcap_input_before_running_extractor(self):
        invalid_path = self.root / "capture.pcap"
        invalid_path.write_bytes(b"not a capture")

        with self._set_command():
            with self.assertRaisesRegex(ValueError, "bukan PCAP/PCAPNG"):
                extract_pcap_to_csv(invalid_path, self.output_directory)

    def test_rejects_pcapng_with_supported_conversion_guidance(self):
        pcapng_path = self.root / "capture.pcapng"
        pcapng_path.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 20)

        with self.assertRaisesRegex(ValueError, "PCAP klasik"):
            extract_pcap_to_csv(pcapng_path, self.output_directory)

    def test_requires_exactly_one_nonempty_csv(self):
        script = (
            "import pathlib,sys; out=pathlib.Path(sys.argv[2]); "
            "out.mkdir(parents=True,exist_ok=True); "
            "(out/'one.csv').write_text('a\\n1\\n'); "
            "(out/'two.csv').write_text('b\\n2\\n')"
        )
        command = [sys.executable, "-c", script, "{input}", "{output_dir}"]
        with patch.dict(
            os.environ,
            {"CICFLOWMETER_COMMAND": json.dumps(command)},
        ):
            with self.assertRaisesRegex(RuntimeError, "tepat satu CSV"):
                extract_pcap_to_csv(self.capture_path, self.output_directory)

    def test_uses_installed_distribution_without_custom_command_json(self):
        home = self.root / "CICFlowMeter-4.0"
        library_directory = home / "lib"
        native_directory = library_directory / "native"
        java_directory = self.root / "java" / "bin"
        native_directory.mkdir(parents=True)
        java_directory.mkdir(parents=True)
        (library_directory / "CICFlowMeter.jar").write_bytes(b"jar")
        (native_directory / "jnetpcap.dll").write_bytes(b"dll")
        (java_directory / "java.exe").write_bytes(b"java")

        def fake_run(command, **_kwargs):
            output_directory = Path(command[-1])
            (output_directory / "flows.csv").write_text(
                "Protocol\n6\n", encoding="utf-8"
            )
            return CompletedProcess(command, 0)

        with patch.dict(
            os.environ,
            {
                "CICFLOWMETER_HOME": str(home),
                "JAVA_HOME": str(java_directory.parent),
                "CICFLOWMETER_COMMAND": "",
                "CICFLOWMETER_VERSION": "",
            },
            clear=True,
        ), patch("src.pcap_flow_extractor.subprocess.run", side_effect=fake_run) as run:
            result = extract_pcap_to_csv(self.capture_path, self.output_directory)

        command = run.call_args.args[0]
        self.assertEqual(command[0], str(java_directory / "java.exe"))
        self.assertIn("cic.cs.unb.ca.ifm.Cmd", command)
        self.assertEqual(command[-2], str(self.capture_path.resolve()))
        self.assertEqual(command[-1], str(self.output_directory.resolve()))
        self.assertEqual(result["extractor_executable"], "CICFlowMeter Cmd")
        self.assertEqual(result["extractor_version"], "CICFlowMeter-4.0")
        self.assertTrue(result["csv_path"].is_file())


if __name__ == "__main__":
    unittest.main()
