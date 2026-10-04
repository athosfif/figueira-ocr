import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app


class FakeVision:
    name = "test-double"

    def read(self, image):
        return "ABC1D23", {"fixture": True}


class CliContractTests(unittest.TestCase):
    def test_cli_writes_expected_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "image_01.png"
            output = root / "output"
            report = root / "report.json"
            Image.new("RGB", (200, 100), "white").save(source)
            with patch("backends.VisionOCR", FakeVision):
                code = app.main([
                    "--input-image", str(source), "--output-dir", str(output),
                    "--backend", "vision", "--report", str(report),
                ])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((output / "image_01_output.json").read_text()), {"text": "ABC1D23"})
            diagnostics = json.loads(report.read_text())
            self.assertEqual(diagnostics["backend"], "test-double")
            self.assertFalse(diagnostics["competition_gpu_validated"])


if __name__ == "__main__":
    unittest.main()
