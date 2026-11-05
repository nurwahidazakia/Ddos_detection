import csv
import tempfile
import unittest
from pathlib import Path

from src.harmonize_training_schemas import (
    _canonicalize_feature,
    harmonize_training_schemas,
)


class HarmonizeTrainingSchemasTests(unittest.TestCase):
    def _write_csv(
        self,
        path: Path,
        source_file: str,
        *,
        extra_columns: tuple[str, ...] = (),
        labels: tuple[int, ...] = (1, -1),
    ) -> None:
        columns = ["source_file", "flow_duration", "protocol", *extra_columns, "label"]
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for index, label in enumerate(labels):
                writer.writerow(
                    {
                        "source_file": source_file,
                        "flow_duration": index + 10,
                        "protocol": 6,
                        **{column: 99 for column in extra_columns},
                        "label": label,
                    }
                )

    def test_feature_aliases_map_to_common_semantic_names(self):
        self.assertEqual(_canonicalize_feature("dst_port"), "destination_port")
        self.assertEqual(_canonicalize_feature("tot_fwd_pkts"), "total_fwd_packets")
        self.assertEqual(_canonicalize_feature("flow_duration"), "flow_duration")

    def test_combines_common_features_without_source_file_overlap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            self._write_csv(input_dir / "schema_01_train.csv", "day-train.csv")
            self._write_csv(
                input_dir / "schema_01_test.csv", "day-test.csv", labels=(1, -1, 1)
            )
            self._write_csv(
                input_dir / "schema_02_train.csv",
                "single-train.csv",
                extra_columns=("source_port",),
            )
            self._write_csv(
                input_dir / "schema_02_test.csv",
                "single-train.csv",
                extra_columns=("source_port",),
                labels=(1, -1, 1),
            )
            self._write_csv(
                input_dir / "schema_03_train.csv",
                "random-single.csv",
                extra_columns=("simillarhttp", "inbound"),
            )
            self._write_csv(
                input_dir / "schema_03_test.csv",
                "random-single.csv",
                extra_columns=("simillarhttp", "inbound"),
                labels=(1, -1, 1),
            )

            result = harmonize_training_schemas(input_dir, output_dir)

            self.assertEqual(result["features"], ["flow_duration", "protocol"])
            self.assertEqual(result["train_class_counts"], {"normal": 2, "attack": 2})
            self.assertEqual(result["test_class_counts"], {"normal": 4, "attack": 2})
            self.assertEqual(
                set(result["train_source_files"]) & set(result["test_source_files"]),
                set(),
            )
            with (output_dir / "harmonized_train.csv").open(
                "r", encoding="utf-8", newline=""
            ) as source:
                train_reader = csv.DictReader(source)
                train_rows = list(train_reader)
            with (output_dir / "harmonized_test.csv").open(
                "r", encoding="utf-8", newline=""
            ) as source:
                test_reader = csv.DictReader(source)
                test_rows = list(test_reader)
            self.assertEqual(
                train_reader.fieldnames,
                ["source_file", "flow_duration", "protocol", "label"],
            )
            self.assertEqual(train_reader.fieldnames, test_reader.fieldnames)
            self.assertEqual(len(train_rows), 4)
            self.assertEqual(len(test_rows), 6)
            self.assertNotIn("source_port", train_reader.fieldnames)
            self.assertNotIn("inbound", test_reader.fieldnames)


if __name__ == "__main__":
    unittest.main()
