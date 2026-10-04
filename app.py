#!/usr/bin/env python3
"""CLI contract: app.py --input-image path, writes <stem>_output.json."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ocr_core import load_image, write_result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/app/output"))
    parser.add_argument("--backend", choices=("vision", "rocm"), default="rocm")
    parser.add_argument("--report", type=Path, help="Optional diagnostics; not the graded JSON.")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        image = load_image(args.input_image)
        from backends import VisionOCR, ROCmWorkerClient
        model_started = time.perf_counter()
        backend = (VisionOCR() if args.backend == "vision" else
                   ROCmWorkerClient())
        ready = time.perf_counter()
        text, details = backend.read(image)
        inferred = time.perf_counter()
        output = write_result(args.output_dir, args.input_image.stem, text)
        report = {
            "backend": backend.name, "output": str(output), "text": text,
            "input_pixels": list(image.size),
            "load_seconds": ready - model_started,
            "inference_seconds": inferred - ready,
            "total_seconds": time.perf_counter() - started,
            "competition_gpu_validated": False,
            **details,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "observations"},
                         ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"OCR error: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
