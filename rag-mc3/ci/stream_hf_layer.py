#!/usr/bin/env python3
"""CI-only draft: replay an audited AMD layer using pinned, hashed HF streams.

No model cache or TAR file is created. This is a different execution environment,
not a way around the AMD notebook's egress policy. Run only after CI publication
and its Docker Hub device login have been authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import platform
import stat
import sys
import tarfile
import urllib.parse
import urllib.request

REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
RUNTIME = {"app.py", "worker.py", "rag_core.py", "parsers.py", "model_backend.py", "healthcheck.py", "requirements.txt"}
MAX_UNCOMPRESSED = 60 * 1024**3
BASE_UNCOMPRESSED = 31_190_284_288


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan(stage):
    if stage.is_symlink() or any(parent.is_symlink() for parent in stage.parents):
        raise ValueError("CI stage and its ancestors must not be symlinks")
    stage = stage.resolve()
    receipt = json.loads((stage / "APP-LAYER-RECEIPT.json").read_text())
    files = json.loads((stage / "APP-LAYER-FILES.json").read_text())
    quality = json.loads((stage / "GPU-QUALITY-GATE.json").read_text())
    if (receipt.get("status") != "stream_plan_prepared" or not receipt.get("weights_included")
            or receipt.get("model_revision") != REVISION or not receipt.get("dependency_closure_verified")):
        raise ValueError("A complete pinned AMD prepare receipt is required")
    smoke = receipt.get("native_vendor_import_smoke", {})
    if smoke.get("model_loaded") is not False or smoke.get("gpu_inference") is not False:
        raise ValueError("Preserved native AMD import smoke is required; it is not a final-image test")
    total_queries = quality.get("queries_total", 0)
    if (quality.get("quality_verified") is not True or not isinstance(total_queries, int)
            or total_queries < 14 or quality.get("queries_passed") != total_queries):
        raise ValueError("The completed official and independent GPU quality gate is required")
    if set(quality.get("runtime_sha256", {})) != RUNTIME:
        raise ValueError("GPU quality gate must bind all seven exact runtime files")
    seen = set()
    models = []
    payload_bytes = 0
    for row in files:
        path = pathlib.PurePosixPath(row["path"])
        if path.is_absolute() or ".." in path.parts or path.as_posix() != row["path"] or row["path"] in seen:
            raise ValueError("Unsafe or duplicate layer manifest path")
        seen.add(row["path"])
        if row["kind"] not in {"file", "directory"}:
            raise ValueError("Only regular files and directories may enter the layer")
        is_model = path.parts[:2] == ("models", "qwen")
        is_vendor = path.parts[:2] == ("app", "vendor")
        is_runtime = len(path.parts) == 2 and path.parts[0] == "app" and path.name in RUNTIME
        if row["kind"] == "file" and not (is_model or is_vendor or is_runtime):
            raise ValueError("Only exact runtime, audited vendor and pinned model files may enter CI")
        if row["kind"] == "directory" and not (is_vendor or row['path'] in {'app', 'app/output', 'models', 'models/qwen'}):
            raise ValueError("Unexpected layer directory outside the audited allowlist")
        if is_model and row["kind"] == "file":
            if (len(path.parts) != 3 or row["bytes"] <= 0 or len(row.get("sha256", "")) != 64
                    or set(row['sha256']) - set('0123456789abcdef')):
                raise ValueError("Pinned flat model file, size and SHA256 required")
            models.append(row)
        elif row["kind"] == "file":
            local = stage / "layer" / path
            if (local.is_symlink() or any(p.is_symlink() for p in local.parents)
                    or not local.is_file() or local.stat().st_size != row["bytes"] or sha_file(local) != row["sha256"]):
                raise ValueError(f"Audited local layer member changed: {path}")
        payload_bytes += row.get("bytes", 0) if row["kind"] == "file" else 0
    names = {pathlib.PurePosixPath(row["path"]).name for row in models}
    if (sum(row["bytes"] for row in models) != receipt.get("weights_bytes")
            or len([n for n in names if n.endswith(".safetensors")]) != 5
            or not {"config.json", "tokenizer_config.json", "model.safetensors.index.json"} <= names):
        raise ValueError("Complete five-shard pinned model and configs must match AMD receipt")
    # TAR headers/PAX/end padding add bytes; the Go assembler measures the actual
    # full stream again before remote.Write. This bound is only an early gate.
    if BASE_UNCOMPRESSED + payload_bytes >= MAX_UNCOMPRESSED:
        raise ValueError("Payload already exceeds the uncompressed image limit")
    for name, expected in quality["runtime_sha256"].items():
        actual = sha_file(stage / "layer" / "app" / name)
        if actual != expected:
            raise ValueError(f"Runtime changed after successful GPU pilot: {name}")
    return files


class CheckedStream:
    def __init__(self, source, row):
        self.source = source
        self.row = row
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        # tarfile normally requests 16 KiB; cap arbitrary callers to bounded RAM.
        if size < 0 or size > 1024 * 1024:
            size = 1024 * 1024
        chunk = self.source.read(size)
        self.count += len(chunk)
        self.digest.update(chunk)
        if self.count > self.row["bytes"]:
            raise ValueError("Source exceeded the pinned file size")
        return chunk

    def finish(self):
        if self.source.read(1):
            raise ValueError("Source contains extra bytes after the pinned file")
        if self.count != self.row["bytes"] or self.digest.hexdigest() != self.row["sha256"]:
            raise ValueError(f"Pinned source size/hash mismatch: {self.row['path']}")


def emit(stage, files):
    if platform.system() != "Linux":
        raise ValueError("Model streaming is allowed only on the remote Linux CI runner")
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        for row in files:
            path = pathlib.PurePosixPath(row["path"])
            header = tarfile.TarInfo(path.as_posix())
            header.uid = header.gid = 0
            header.uname = header.gname = "root"
            header.mtime = 1791158400
            header.mode = row.get("mode", 0o644)
            header.pax_headers = {}
            if row["kind"] == "directory":
                header.type = tarfile.DIRTYPE
                header.mode = 0o777 if path.as_posix() == "app/output" else 0o755
                archive.addfile(header)
                continue
            header.size = row["bytes"]
            if path.parts[:2] == ("models", "qwen"):
                filename = urllib.parse.quote(path.name, safe="")
                url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}"
                request = urllib.request.Request(url, headers={"User-Agent": "Figueira-RAG-audited-CI/1", "Accept-Encoding": "identity"})
                source = urllib.request.urlopen(request, timeout=120)
                # Model origin is fixed above. Redirects are normal HF public
                # object delivery; TLS verification remains enabled.
                length = source.headers.get("Content-Length")
                if length is not None and int(length) != row["bytes"]:
                    source.close()
                    raise ValueError("HF pinned file Content-Length differs from AMD manifest")
            else:
                local = stage / "layer" / path
                source = local.open("rb")
            with source:
                checked = CheckedStream(source, row)
                archive.addfile(header, checked)
                checked.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=pathlib.Path, required=True)
    parser.add_argument("--stream-tar", action="store_true")
    args = parser.parse_args()
    files = plan(args.stage)
    if args.stream_tar:
        emit(args.stage.resolve(), files)
    else:
        print(json.dumps({"status": "verified_plan", "files": len(files), "weights_downloaded": False,
                          "tar_written": False, "registry_contacted": False,
                          "native_amd_proof_reused": True, "final_container_gpu_execution_verified": False}))


if __name__ == "__main__":
    main()
