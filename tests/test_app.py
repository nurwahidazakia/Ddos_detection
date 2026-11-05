import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as prediction_app


class PredictionAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_directory = Path(self.temporary_directory.name)
        self.model_path = self.output_directory / "model.joblib"
        self.model_path.write_bytes(b"test-model")
        prediction_app.app.config.update(TESTING=True)
        self.output_directory_original = prediction_app.PREDICTION_OUTPUT_DIR
        self.model_path_original = prediction_app.MODEL_PATH
        prediction_app.PREDICTION_OUTPUT_DIR = self.output_directory
        prediction_app.MODEL_PATH = self.model_path
        self.client = prediction_app.app.test_client()

    def tearDown(self):
        prediction_app.PREDICTION_OUTPUT_DIR = self.output_directory_original
        prediction_app.MODEL_PATH = self.model_path_original
        self.temporary_directory.cleanup()

    def _fake_prediction(self, input_path, output_path, report_path, _model_path, **kwargs):
        output_path.write_text("PredictedLabel\nNORMAL\n", encoding="utf-8")
        report_path.write_text('{"warnings": []}', encoding="utf-8")
        return {
            "rows": 1,
            "prediction_counts": {"attack": 0, "normal": 1, "not_scored": 0},
            "threshold": 0.3849,
            "metrics": None,
            "model_type": "TestModel",
            "model_status": "provisional",
            "input_sha256": "test-hash",
            "received_label_column": kwargs.get("label_column"),
        }

    def test_home_page_lists_the_local_test_datasets(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Flow normal Monday", response.data)
        self.assertIn(b"Dataset diagnostik lintas-sumber", response.data)
        self.assertIn(b"Jalankan prediksi", response.data)

    def test_upload_runs_prediction_and_removes_temporary_input(self):
        captured_input = []

        def fake_prediction(input_path, output_path, report_path, model_path, **kwargs):
            captured_input.append(Path(input_path))
            self.assertTrue(Path(input_path).is_file())
            self.assertEqual(kwargs["label_column"], "Label")
            return self._fake_prediction(
                input_path, output_path, report_path, model_path, **kwargs
            )

        with patch.object(prediction_app, "predict_csv", side_effect=fake_prediction):
            response = self.client.post(
                "/",
                data={
                    "source": "normal",
                    "label_column": "Label",
                    "capture_id": "pilot-1",
                    "csv_file": (io.BytesIO(b"Protocol\n6\n"), "flows.csv"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Ringkasan hasil", response.data)
        self.assertIn(b"Unduh CSV prediksi", response.data)
        self.assertEqual(len(captured_input), 1)
        self.assertFalse(captured_input[0].exists())
        self.assertEqual(
            len(list(self.output_directory.glob("app_predictions_*_report.json"))), 1
        )

    def test_download_rejects_unrecognized_filename(self):
        response = self.client.get("/download/../model.joblib")

        self.assertEqual(response.status_code, 404)

    def test_pcap_upload_reports_missing_extractor_configuration(self):
        with patch.dict(os.environ, {"CICFLOWMETER_COMMAND": ""}):
            response = self.client.post(
                "/",
                data={
                    "source": "normal",
                    "label_column": "",
                    "csv_file": (
                        io.BytesIO(b"\xd4\xc3\xb2\xa1" + b"\x00" * 20),
                        "today.pcap",
                    ),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Konversi PCAP belum siap", response.data)
        self.assertIn(b"CICFLOWMETER_HOME", response.data)

    def test_pcap_upload_is_extracted_then_prediction_report_tracks_capture(self):
        uploaded_capture_paths = []

        def fake_extractor(capture_path, output_directory):
            uploaded_capture_paths.append(Path(capture_path))
            extracted = Path(output_directory) / "flows.csv"
            extracted.parent.mkdir(parents=True, exist_ok=True)
            extracted.write_text("Protocol\n6\n", encoding="utf-8")
            return {
                "csv_path": extracted,
                "capture_format": "pcap",
                "extractor_version": "test-version",
                "extractor_executable": "CICFlowMeter",
            }

        with (
            patch.object(prediction_app, "extract_pcap_to_csv", side_effect=fake_extractor),
            patch.object(prediction_app, "predict_csv", side_effect=self._fake_prediction),
        ):
            response = self.client.post(
                "/",
                data={
                    "source": "normal",
                    "label_column": "",
                    "capture_id": "today-pcap",
                    "csv_file": (
                        io.BytesIO(b"\xd4\xc3\xb2\xa1" + b"\x00" * 20),
                        "today.pcap",
                    ),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PCAP dikonversi otomatis", response.data)
        self.assertEqual(len(uploaded_capture_paths), 1)
        self.assertFalse(uploaded_capture_paths[0].exists())
        report_path = next(self.output_directory.glob("app_predictions_*_report.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            report["pcap_extraction"]["original_capture_filename"], "today.pcap"
        )
        self.assertEqual(
            report["pcap_extraction"]["extractor_version"], "test-version"
        )
        self.assertTrue(report["pcap_extraction"]["capture_sha256"])


if __name__ == "__main__":
    unittest.main()
