import csv
import tempfile
import unittest
from pathlib import Path

from wildwatch.inference import (
    Detection,
    validate_confidence,
    write_detections_csv,
    xyxy_from_values,
)


class InferenceUtilitiesTests(unittest.TestCase):
    def test_confidence_validation(self):
        self.assertEqual(validate_confidence(0.35), 0.35)
        with self.assertRaises(ValueError):
            validate_confidence(1.2)

    def test_xyxy_validation(self):
        self.assertEqual(xyxy_from_values([1, 2, 8, 9]), (1.0, 2.0, 8.0, 9.0))
        with self.assertRaises(ValueError):
            xyxy_from_values([8, 9, 1, 2])

    def test_csv_export(self):
        row = Detection("frame.png", None, 0, 2, "fox", 0.91, 1, 2, 30, 40)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detections.csv"
            count = write_detections_csv([row], path)
            self.assertEqual(count, 1)
            with path.open(encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(records[0]["class_name"], "fox")
            self.assertEqual(records[0]["confidence"], "0.91")
            self.assertEqual(records[0]["frame_index"], "")


if __name__ == "__main__":
    unittest.main()
