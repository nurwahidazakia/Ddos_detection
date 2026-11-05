import csv
import tempfile
import unittest
from pathlib import Path

from src.train_ddos_candidate import _grouped_split, _load_dataset


class TrainDdosCandidateTests(unittest.TestCase):
    def _write_dataset(self, path: Path, attack_labels: tuple[str, ...]) -> None:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                ("Flow ID", " Source IP", " Timestamp", "Unnamed: 0", "Feature", "Label")
            )
            for index in range(60):
                writer.writerow(
                    (f"flow-{index}", "192.0.2.1", "2026-01-01", index, index, "BENIGN")
                )
            for label_index, label in enumerate(attack_labels):
                for row_index in range(60):
                    index = label_index * 60 + row_index
                    writer.writerow(
                        (
                            f"flow-{index}",
                            "192.0.2.2",
                            "2026-01-02",
                            index,
                            index + 100,
                            label,
                        )
                    )

    def test_load_dataset_drops_identifiers_and_maps_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dataset.csv"
            self._write_dataset(path, ("DDoS",))

            features, labels, families, info = _load_dataset(path)

            self.assertEqual(len(features), 120)
            self.assertEqual(info["exact_duplicate_rows_dropped"], 0)
            self.assertEqual(features.columns.tolist(), ["Feature"])
            self.assertEqual(int((labels == 1).sum()), 60)
            self.assertEqual(int((labels == -1).sum()), 60)
            self.assertEqual(set(families), {"BENIGN", "DDOS"})

    def test_grouped_split_keeps_attack_families_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dataset.csv"
            self._write_dataset(path, ("DDoS", "Syn", "NetBIOS"))
            _features, labels, families, _info = _load_dataset(path)

            train, validation, test, groups = _grouped_split(labels, families)

            self.assertFalse(set(groups["train"]) & set(groups["validation"]))
            self.assertFalse(set(groups["train"]) & set(groups["test"]))
            self.assertFalse(set(groups["validation"]) & set(groups["test"]))
            self.assertIn(1, labels[train])
            self.assertIn(-1, labels[train])
            self.assertIn(1, labels[validation])
            self.assertIn(-1, labels[validation])
            self.assertIn(1, labels[test])
            self.assertIn(-1, labels[test])
            self.assertEqual(set(families[test][labels[test] == -1]), set(groups["test"]))


if __name__ == "__main__":
    unittest.main()
