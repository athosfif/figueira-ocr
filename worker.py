#!/usr/bin/env python3
"""Foreground worker: preload once, signal readiness, serialize model inference."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import traceback
from pathlib import Path
from threading import Lock, Thread

from backends import QwenROCmOCR, SOCKET_PATH
from ocr_core import load_image


def respond(stream, payload):
    try:
        stream.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        stream.flush()
        return True
    except (OSError, TimeoutError):
        return False


def model_files_ready(directory):
    if not (directory / "config.json").is_file():
        return False
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            return bool(shards) and all((directory / name).is_file() for name in shards)
        except (ValueError, KeyError, OSError):
            return False
    return (directory / "model.safetensors").is_file()


def prepare_model(directory):
    if model_files_ready(directory):
        return
    if os.environ.get("OCR_ALLOW_MODEL_DOWNLOAD", "0") != "1":
        raise RuntimeError("Prepared model weights required; runtime downloads are disabled.")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=os.environ.get("OCR_MODEL_REPO", "Qwen/Qwen2.5-VL-7B-Instruct"),
        revision=os.environ.get("OCR_MODEL_REVISION", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
        local_dir=directory,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "LICENSE*", "NOTICE*", "README.md"],
    )
    if not model_files_ready(directory):
        raise RuntimeError("Model download finished without all required weight files.")


def main(model_factory=QwenROCmOCR):
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = SOCKET_PATH.with_suffix(".lock")
    with lock_path.open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An OCR worker already owns this socket; refusing a second model.") from exc
        server = None
        bound = False
        previous_term = signal.getsignal(signal.SIGTERM)

        def stop(signum, frame):
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, stop)
        try:
            directory = Path(os.environ.get("OCR_MODEL_DIR", "/models/qwen"))
            if model_factory is QwenROCmOCR:
                prepare_model(directory)
            model = model_factory(directory)
            inference_lock = Lock()
            SOCKET_PATH.unlink(missing_ok=True)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(SOCKET_PATH))
            bound = True
            os.chmod(SOCKET_PATH, 0o600)
            server.listen(8)
            print("OCR worker ready", flush=True)

            def handle(connection):
                with connection:
                    connection.settimeout(30)
                    with connection.makefile("rwb") as stream:
                        try:
                            raw = stream.readline(4097)
                            if len(raw) > 4096 or not raw.endswith(b"\n"):
                                raise ValueError("Invalid or oversized worker request.")
                            request = json.loads(raw)
                            if request.get("op") == "ping":
                                respond(stream, {"ok": True})
                            elif request.get("op") == "ocr":
                                image = load_image(Path(request["path"]))
                                with inference_lock:
                                    text, details = model.read(image)
                                respond(stream, {"ok": True, "text": text, "details": details})
                            else:
                                respond(stream, {"ok": False, "error": "Unsupported worker operation."})
                        except Exception as exc:
                            traceback.print_exc()
                            respond(stream, {"ok": False, "error": str(exc)})

            while True:
                connection, _ = server.accept()
                Thread(target=handle, args=(connection,), daemon=True).start()
        finally:
            if server is not None:
                server.close()
            if bound:
                SOCKET_PATH.unlink(missing_ok=True)
            signal.signal(signal.SIGTERM, previous_term)
            fcntl.flock(owner, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
