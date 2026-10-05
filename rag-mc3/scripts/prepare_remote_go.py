#!/usr/bin/env python3
"""Pinned official Go SDK installer for remote /tmp; default is read-only plan."""
import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
from pathlib import Path

URL = "https://go.dev/dl/go1.27.1.linux-amd64.tar.gz"
HASH = "63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445"
BYTES = 70553950


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("/tmp/figueira-rag-go-1.27.1"))
    p.add_argument("--install", action="store_true")
    args = p.parse_args()
    plan = {"version": "go1.27.1", "url": URL, "sha256": HASH, "archive_bytes": BYTES,
            "destination": str(args.directory), "system_install": False,
            "mode": "install" if args.install else "dry_run", "registry_contacted": False, "gpu_started": False}
    if not args.install:
        print(json.dumps(plan, indent=2))
        return
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        p.error("Install this SDK only on the remote Linux x86_64 host")
    root = args.directory.resolve()
    if not str(root).startswith("/tmp/"):
        p.error("Temporary /tmp destination required")
    if root.exists() and any(root.iterdir()):
        p.error("A new empty directory is required; existing files are preserved")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "sdk.tar.gz"
    subprocess.run(["curl", "-fsSL", "--max-time", "180", "--max-filesize", str(BYTES), URL, "-o", str(archive)], check=True)
    h = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != HASH or archive.stat().st_size != BYTES:
        raise RuntimeError("Official Go archive digest/size mismatch")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root, filter="data")
    plan["installed_go"] = str(root / "go/bin/go")
    (root / "GO-INSTALL-RECEIPT.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
