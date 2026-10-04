import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ocr_core import lines_to_text, load_image, normalized, write_result


class CoreContractTests(unittest.TestCase):
    def test_challenge_normalization_only(self):
        self.assertEqual(normalized(" ab-12.c_d · 3 "), "AB12CD3")
        self.assertEqual(normalized("粤Z·90123"), "粤Z90123")
        self.assertNotEqual(normalized("ABC/123"), "ABC123")

    def test_plate_candidate_filters_banner(self):
        observations = [
            {"text": "CALIFORNIA", "x": 0.3, "y": 0.05, "width": 0.4, "height": 0.08},
            {"text": "8XYZ456", "x": 0.2, "y": 0.3, "width": 0.6, "height": 0.25},
            {"text": "dmv.ca.gov", "x": 0.35, "y": 0.7, "width": 0.3, "height": 0.05},
        ]
        self.assertEqual(lines_to_text(observations, aspect=3.0), "8XYZ456")

    def test_sign_reading_order(self):
        observations = [
            {"text": "42", "x": 0.4, "y": 0.6, "width": 0.2, "height": 0.2},
            {"text": "POINT", "x": 0.55, "y": 0.2, "width": 0.3, "height": 0.1},
            {"text": "CHECK", "x": 0.1, "y": 0.2, "width": 0.35, "height": 0.1},
        ]
        self.assertEqual(lines_to_text(observations, aspect=1.0), "CHECK POINT 42")

    def test_supported_formats_and_resize(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for suffix, fmt in ((".png", "PNG"), (".jpg", "JPEG"), (".tiff", "TIFF")):
                path = root / ("sample" + suffix)
                Image.new("RGBA" if fmt != "JPEG" else "RGB", (3000, 1000), (255, 0, 0, 128) if fmt != "JPEG" else "red").save(path, format=fmt)
                loaded = load_image(path)
                self.assertEqual(loaded.mode, "RGB")
                self.assertLessEqual(max(loaded.size), 2048)

    def test_atomic_json_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_result(Path(temp), "image_01", "粤Z90123")
            self.assertEqual(path.name, "image_01_output.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"text": "粤Z90123"})


if __name__ == "__main__":
    unittest.main()
