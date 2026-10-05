#!/usr/bin/env python3
"""Prepare a small RAG OCI layer; default dry run never contacts a registry.

Use the pip --report JSON produced by the remote, ROCm-preserving bootstrap.
The base is never pulled or extracted by this program. Offline evaluation
requires weights in the layer: they are streamed from the existing remote
directory, never copied into staging. A new empty staging directory is required.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import pathlib
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.parse
import zipfile
from email.parser import BytesParser

RUNTIME = ("app.py", "worker.py", "rag_core.py", "parsers.py", "model_backend.py", "healthcheck.py", "requirements.txt")
DENY = {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm"}
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_items(report):
    items = json.loads(report.read_text())["install"]
    selected = []
    for item in items:
        name = item["metadata"]["name"].lower().replace("_", "-")
        if name in DENY:
            raise ValueError(f"Refusing a replacement of a base ROCm package: {name}")
        download = item["download_info"]
        parsed = urllib.parse.urlparse(download["url"])
        # Existing AMD pip may use the public Tsinghua mirror. It supplies only
        # the filename/hash evidence here; downloads below always use official
        # PyPI metadata and files.pythonhosted.org, never the report URL.
        if parsed.scheme != "https" or parsed.hostname not in {"files.pythonhosted.org", "pypi.tuna.tsinghua.edu.cn"} or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError(f"Recognized public wheel provenance required for {name}")
        filename = pathlib.PurePosixPath(parsed.path).name
        if not filename.endswith(".whl"):
            raise ValueError(f"Binary wheel required for {name}")
        expected = download["archive_info"]["hashes"]["sha256"]
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid wheel SHA256 for {name}")
        selected.append({"name": name, "version": item["metadata"]["version"], "filename": filename,
                         "sha256": expected, "report_source_host": parsed.hostname})
    if len({item["filename"] for item in selected}) != len(selected):
        raise ValueError("Duplicate wheel filename")
    return selected


def fetch(item, wheel_dir):
    meta = wheel_dir / (item["name"] + "-pypi.json")
    subprocess.run(["curl", "-fsSL", "--max-time", "60", "--max-filesize", "10000000",
                    f"https://pypi.org/pypi/{item['name']}/{item['version']}/json", "-o", str(meta)], check=True)
    public = json.loads(meta.read_text())
    artifact = next(row for row in public["urls"] if row["filename"] == item["filename"])
    if artifact["digests"]["sha256"] != item["sha256"] or urllib.parse.urlparse(artifact["url"]).hostname != "files.pythonhosted.org":
        raise ValueError(f"PyPI metadata differs from the successful remote install: {item['name']}")
    if artifact["size"] > 200 * 1024**2:
        raise ValueError(f"Wheel exceeds this lightweight layer budget: {item['name']}")
    path = wheel_dir / item["filename"]
    subprocess.run(["curl", "-fsSL", "--max-time", "180", "--max-filesize", str(200 * 1024**2),
                    artifact["url"], "-o", str(path)], check=True)
    if sha256(path) != item["sha256"]:
        raise ValueError(f"Downloaded wheel hash mismatch: {item['name']}")
    with zipfile.ZipFile(path) as archive:
        metadata = BytesParser().parsebytes(archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))))
        wheel = BytesParser().parsebytes(archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/WHEEL"))))
        tags = wheel.get_all("Tag", [])
        compatible = []
        for tag in tags:
            python_tag, abi, target = tag.split('-', 2)
            linux = target == 'linux_x86_64' or target.startswith('manylinux') and target.endswith('_x86_64')
            stable = python_tag.startswith('cp') and python_tag[2:].isdigit() and int(python_tag[2:]) <= 314 and abi == 'abi3'
            if tag == 'py3-none-any' or linux and (python_tag == 'cp314' and abi in {'cp314', 'none'} or stable):
                compatible.append(tag)
        if not compatible or metadata["Version"] != item["version"]:
            raise ValueError(f"Unsupported Python 3.14/Linux x86_64 wheel: {item['name']}: {tags}")
    return dict(item, source_url=artifact["url"], bytes=path.stat().st_size, tags=tags, requires_python=metadata.get("Requires-Python"))


def extract_wheel(path, vendor):
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            rel = pathlib.PurePosixPath(member.filename)
            if rel.is_absolute() or ".." in rel.parts or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unsafe wheel member")
            if member.is_dir():
                continue
            if rel.parts[0].endswith(".data"):
                if len(rel.parts) > 2 and rel.parts[1] in {"purelib", "platlib"}:
                    rel = pathlib.PurePosixPath(*rel.parts[2:])
                else:
                    continue
            dest = vendor.joinpath(*rel.parts)
            if dest.exists():
                raise ValueError(f"Vendored package overlap: {rel}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(member))
            dest.chmod((member.external_attr >> 16) & 0o777 or 0o644)


def image_config():
    return {"Cmd": ["python3", "/app/worker.py"], "WorkingDir": "/app",
            "Env": ["PYTHONPATH=/app/vendor", "PYTHONUNBUFFERED=1", "PIP_NO_CACHE_DIR=1",
                    "RAG_MODEL_DIR=/models/qwen", "RAG_MODEL_REPO=Qwen/Qwen2.5-VL-7B-Instruct",
                    f"RAG_MODEL_REVISION={REVISION}", "RAG_ALLOW_MODEL_DOWNLOAD=0",
                    "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                    "RAG_WORKER_SOCKET=/tmp/figueira-rag.sock", "RAG_STARTUP_TIMEOUT=570",
                    "RAG_INDEX_TIMEOUT=570", "RAG_QUERY_TIMEOUT=28"],
            "Healthcheck": {"Test": ["CMD", "python3", "/app/healthcheck.py"],
                            "Interval": 2000000000, "Timeout": 3000000000, "StartPeriod": 600000000000, "Retries": 3}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--pip-report", type=pathlib.Path)
    parser.add_argument("--closure", type=pathlib.Path, help="Full dependency graph evidence/wheel-closure.json, required for assembly")
    parser.add_argument("--stage", type=pathlib.Path, required=True)
    parser.add_argument("--models-dir", type=pathlib.Path, help="Prepared, pinned model directory on the remote Linux host")
    parser.add_argument("--assemble", action="store_true", help="Prepare audited app/wheels and a model stream manifest on remote Linux; no TAR or registry upload")
    parser.add_argument("--stream-tar", action="store_true", help="Emit the prepared layer on stdout, only inside remote Linux; no TAR is saved")
    args = parser.parse_args()
    if args.stream_tar:
        stream_tar(args.stage)
        return
    if not args.pip_report:
        parser.error("--pip-report is required for preparation/dry run")
    root = args.root.resolve()
    items = selected_items(args.pip_report)
    closure = None
    if args.closure:
        closure = json.loads(args.closure.read_text())
        actual = {r['name']: r['version'] for r in items}
        planned = {r['name']: r['version'] for r in closure['wheels']}
        if actual != planned:
            raise ValueError('Wheel report must contain the complete frozen dependency graph, not an install delta')
        if sha256(root / 'requirements.txt') != closure['requirements_sha256']:
            raise ValueError('Requirements changed after closure capture')
    for filename in RUNTIME:
        path = root / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular runtime source: {filename}")
        if filename.endswith(".py"):
            ast.parse(path.read_text(), filename=filename)
    plan = {"mode": "assemble" if args.assemble else "dry_run", "root": str(root), "stage": str(args.stage),
            "runtime_files": list(RUNTIME), "wheels": items, "config": image_config(),
            "base_downloaded": False, "model_downloaded": False, "registry_contacted": False,
            "model_weights_required_in_image": True, "model_dir": str(args.models_dir) if args.models_dir else None}
    if not args.assemble:
        print(json.dumps(plan, indent=2))
        return
    stage = args.stage.resolve()
    if platform.system() != "Linux":
        raise ValueError("Model packaging runs on remote Linux only, never on the Mac")
    if not args.models_dir:
        parser.error("--models-dir is required; MC3 evaluation cannot download weights")
    if not closure:
        parser.error("--closure is required; incremental pip install reports do not prove image dependencies")
    model_entries = validate_model(args.models_dir.resolve())
    if stage.exists() and any(stage.iterdir()):
        raise ValueError("A new empty staging directory is required; existing files are preserved")
    if shutil.disk_usage(stage.parent).free < 1024**3:
        raise ValueError("At least 1 GiB remote staging space required; weights will not be copied")
    wheel_dir, tree = stage / "wheels", stage / "layer"
    vendor = tree / "app/vendor"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    vendor.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda item: fetch(item, wheel_dir), items))
    for item in receipts:
        extract_wheel(wheel_dir / item["filename"], vendor)
    for denied in DENY:
        if (vendor / denied.replace("-", "_")).exists():
            raise ValueError(f"Refusing shadow of a base ROCm package: {denied}")
    for filename in RUNTIME:
        shutil.copyfile(root / filename, tree / "app" / filename)
    smoke = verify_native_vendor_imports(tree / 'app', closure)
    (tree / "app/output").mkdir()
    (tree / "app/output").chmod(0o777)
    (tree / "models/qwen").mkdir(parents=True)
    files = [describe(path, path.relative_to(tree).as_posix()) for path in sorted(tree.rglob("*"))]
    if any(row.get("kind") == "symlink" for row in files):
        raise ValueError("Symlinks cannot be packaged")
    files += model_entries
    receipt = {"status": "stream_plan_prepared", "method": "audited runtime app and public PyPI wheels plus existing pinned weights streamed from remote Linux",
               "wheels": receipts, "base_torch_untouched": True, "weights_included": True,
               "weights_bytes": sum(row.get("bytes", 0) for row in model_entries),
               "runtime_imports_verified_in_final_image": False, "tar_saved_to_disk": False,
               "model_weights_copied": False, "model_revision": REVISION}
    receipt.update(dependency_closure_verified=True, dependency_closure=closure)
    receipt['native_vendor_import_smoke'] = smoke
    for name, value in [("APP-LAYER-RECEIPT.json", receipt), ("APP-LAYER-FILES.json", files), ("IMAGE-CONFIG.json", image_config())]:
        (stage / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def verify_native_vendor_imports(app, closure):
    """Fresh base-Python imports using exactly the app/vendor layer, no model load."""
    if pathlib.Path(sys.prefix).resolve() == pathlib.Path(closure['runtime_venv']).resolve():
        raise ValueError('Use the base Python, not the pilot venv, to audit vendored imports')
    code = r'''
import importlib, importlib.metadata, json, pathlib, sys
app = pathlib.Path(sys.argv[1]).resolve()
sys.path[:0] = [str(app / 'vendor'), str(app)]
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
for name in ['accelerate', 'safetensors', 'huggingface_hub', 'tokenizers', 'pypdf', 'pymupdf', 'openpyxl', 'PIL', 'app', 'worker', 'rag_core', 'parsers', 'model_backend', 'healthcheck']:
    importlib.import_module(name)
print(json.dumps({'torch_file': str(pathlib.Path(torch.__file__).resolve()), 'torch_version': importlib.metadata.version('torch'), 'torch_hip': torch.version.hip,
 'source': 'fresh base Python with vendored layer modules', 'model_loaded': False, 'gpu_inference': False, 'final_container_imports_verified': False}))
'''
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', RAG_ALLOW_MODEL_DOWNLOAD='0')
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(app)], capture_output=True, text=True, timeout=120, env=environment)
    if result.returncode:
        raise ValueError('Native vendor import smoke failed: ' + result.stderr[-3000:])
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    baseline = closure['base_torch']
    if proof['torch_file'] != baseline['file'] or proof['torch_version'] != baseline['version'] or not proof['torch_hip']:
        raise ValueError('Vendored imports changed the original ROCm Torch')
    return proof


def describe(path, archive_path):
    metadata = path.lstat()
    kind = "symlink" if path.is_symlink() else "directory" if path.is_dir() else "file"
    result = {"path": archive_path, "source": str(path), "kind": kind,
              "mode": stat.S_IMODE(metadata.st_mode), "mtime_ns": metadata.st_mtime_ns}
    if kind == "file":
        result.update(bytes=metadata.st_size, sha256=sha256(path))
    return result


def validate_model(directory):
    if directory.is_symlink() or not (directory / "config.json").is_file():
        raise ValueError("Regular prepared model directory/config required")
    metadata = directory / ".cache/huggingface/download"
    revisions = set()
    if metadata.is_dir():
        for file in metadata.rglob("*.metadata"):
            lines = file.read_text().splitlines()
            if lines:
                revisions.add(lines[0].strip())
    if revisions != {REVISION} and directory.name != REVISION:
        raise ValueError("Prepared model metadata does not prove the pinned revision")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        required = set(json.loads(index.read_text())["weight_map"].values())
    else:
        required = {"model.safetensors"}
    if not required or any(not (directory / name).is_file() for name in required):
        raise ValueError("Incomplete model shards")
    allow_suffix = {".json", ".safetensors", ".model", ".txt", ".jinja", ".md"}
    files = [p for p in sorted(directory.iterdir()) if p.is_file() and (p.suffix in allow_suffix or p.name.startswith(("LICENSE", "NOTICE")))]
    if any(p.is_symlink() for p in files):
        raise ValueError("Resolve model snapshot symlinks into a prepared local directory before packaging")
    names = {p.name for p in files}
    if not required <= names or "tokenizer_config.json" not in names:
        raise ValueError("Model shard/tokenizer configuration is incomplete")
    return [describe(p, "models/qwen/" + p.name) for p in files]


def stream_tar(stage):
    if platform.system() != "Linux":
        raise ValueError("Weight stream emitter runs only on remote Linux")
    manifest = json.loads((stage / "APP-LAYER-FILES.json").read_text())
    # Deterministic repeated opens: source, size and mtime must remain frozen.
    # SHA256 values were measured during preparation; the OCI SDK independently
    # hashes the full uncompressed and compressed stream without buffering it.
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        for row in manifest:
            path = pathlib.Path(row["source"])
            rel = pathlib.PurePosixPath(row["path"])
            if rel.is_absolute() or ".." in rel.parts or path.is_symlink():
                raise ValueError("Unsafe stream manifest member")
            metadata = path.stat()
            if metadata.st_mtime_ns != row["mtime_ns"] or (row["kind"] == "file" and metadata.st_size != row["bytes"]):
                raise ValueError(f"Prepared source changed: {rel}")
            info = archive.gettarinfo(str(path), arcname=rel.as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 1791158400
            info.pax_headers = {}
            if row["kind"] == "file":
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            elif row["kind"] == "directory":
                archive.addfile(info)
            else:
                raise ValueError("Unsupported stream member type")


if __name__ == "__main__":
    main()
