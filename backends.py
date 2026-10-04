"""Native development baseline and an AMD-only OCR backend with a persistent worker."""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ocr_core import PROMPT, lines_to_text

ROOT = Path(__file__).resolve().parent
SOCKET_PATH = Path(os.environ.get("OCR_WORKER_SOCKET", "/tmp/figueira-ocr.sock"))
WORKER_LOG = Path(os.environ.get("OCR_WORKER_LOG", "/tmp/figueira-ocr-worker.log"))

class VisionOCR:
    name = "apple-vision-local-baseline"

    def __init__(self):
        if platform.system() != "Darwin":
            raise RuntimeError("Vision is a macOS development baseline only.")
        self.binary = ROOT / ".build" / "vision-ocr"
        if not self.binary.is_file():
            raise RuntimeError("Build the local baseline with: sh scripts/build_vision.sh")

    def read(self, image):
        with tempfile.TemporaryDirectory(prefix="figueira-ocr-") as temp:
            path = Path(temp) / "input.png"
            image.save(path)
            result = subprocess.run([str(self.binary), str(path)], capture_output=True,
                                    text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Native OCR failed.")
        lines = json.loads(result.stdout)
        return lines_to_text(lines, image.width / image.height), {"observations": lines}

class QwenROCmOCR:
    name = "qwen2.5-vl-7b-rocm-candidate"

    def __init__(self, model_dir):
        if platform.system() != "Linux":
            raise RuntimeError("Challenge backend requires Linux and an AMD ROCm GPU; no CPU fallback.")
        import torch
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("A working AMD ROCm GPU is required; no CPU fallback.")
        if not model_dir.is_dir():
            raise RuntimeError("OCR_MODEL_DIR must point to prepared local model weights.")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
        )
        # Keep torch from the event image. Never install a replacement CUDA wheel.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), torch_dtype=torch.float16,
            local_files_only=True, trust_remote_code=False, attn_implementation="sdpa",
        ).to("cuda").eval()

    def read(self, image):
        torch = self.torch
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}
        ]}]
        template = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[template], images=[image],
                                padding=True, return_tensors="pt").to("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        torch.cuda.synchronize()
        text = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return " ".join(text.split()), {
            "gpu": torch.cuda.get_device_name(0),
            "hip": torch.version.hip,
            "torch": torch.__version__,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "generated_tokens": generated.shape[1] - inputs.input_ids.shape[1],
            "token_limit_reached": generated.shape[1] - inputs.input_ids.shape[1] >= 128,
        }


def _request_worker(payload: dict, timeout: float) -> dict:
    """Exchange one newline-delimited JSON message with the local worker."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(SOCKET_PATH))
        stream = client.makefile("rwb")
        stream.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        stream.flush()
        raw = stream.readline()
    if not raw:
        raise RuntimeError("OCR worker closed the connection without a response.")
    response = json.loads(raw)
    if not response.get("ok"):
        raise RuntimeError(response.get("error") or "OCR worker failed.")
    return response


class ROCmWorkerClient:
    """Wait for the foreground worker; CLI calls never launch another model."""

    name = "qwen2.5-vl-7b-rocm-worker-candidate"

    def __init__(self):
        if platform.system() != "Linux":
            raise RuntimeError("Challenge backend requires Linux and an AMD ROCm GPU; no CPU fallback.")
        started = time.perf_counter()
        deadline = started + float(os.environ.get("OCR_STARTUP_TIMEOUT", "570"))
        last_error = "worker not ready"
        while time.perf_counter() < deadline:
            try:
                _request_worker({"op": "ping"}, timeout=min(2, deadline - time.perf_counter()))
                self.startup_seconds = time.perf_counter() - started
                self.reused_worker = True
                return
            except (OSError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                time.sleep(min(0.1, max(0, deadline - time.perf_counter())))
        raise RuntimeError(f"Foreground OCR worker not ready: {last_error}. Start worker.py before CLI calls.")

    def read(self, image):
        with tempfile.TemporaryDirectory(prefix="figueira-rocm-ocr-") as temp:
            path = Path(temp) / "input.png"
            image.save(path)
            response = _request_worker(
                {"op": "ocr", "path": str(path)},
                timeout=float(os.environ.get("OCR_INFERENCE_TIMEOUT", "30")),
            )
        details = response.get("details") or {}
        details.update({
            "worker_reused": self.reused_worker,
            "worker_startup_seconds": self.startup_seconds,
        })
        return response["text"], details
