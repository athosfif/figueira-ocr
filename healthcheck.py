#!/usr/bin/env python3
"""Ready only when the preloaded foreground OCR worker answers ping."""
import json
import os
import socket

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.settimeout(2)
    connection.connect(os.environ.get("OCR_WORKER_SOCKET", "/tmp/figueira-ocr.sock"))
    connection.sendall(b'{"op":"ping"}\n')
    with connection.makefile("rb") as stream:
        if not json.loads(stream.readline()).get("ok"):
            raise SystemExit(1)
