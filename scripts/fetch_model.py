#!/usr/bin/env python3
"""Pin and materialize public model weights in the challenge image."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=args.destination,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "LICENSE*", "NOTICE*", "README.md"],
    )


if __name__ == "__main__":
    main()
