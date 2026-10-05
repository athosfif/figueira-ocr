#!/usr/bin/env python3
"""Load one model and serve serialized index/query operations over a Unix socket."""
from __future__ import annotations
import fcntl
import json
import os
import signal
import socket
import time
from pathlib import Path
from threading import Lock, Thread
from model_backend import QwenBackend
from rag_core import RagEngine


def main(model_factory=QwenBackend, stop_event=None):
    path = Path(os.environ.get('RAG_WORKER_SOCKET', '/tmp/figueira-rag.sock'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        model = model_factory(Path(os.environ.get('RAG_MODEL_DIR', '/models/qwen')))
        loaded_seconds = time.monotonic() - started
        engine = RagEngine(model)
        lock = Lock()
        path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        os.chmod(path, 0o600)
        server.listen(8)
        server.settimeout(.25)
        previous = None
        if stop_event is None:
            previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))

        def handle(connection):
            with connection:
                connection.settimeout(600)
                with connection.makefile('rwb') as stream:
                    try:
                        raw = stream.readline(65537)
                        if len(raw) > 65536 or not raw.endswith(b'\n'):
                            raise ValueError('Invalid or oversized request.')
                        request = json.loads(raw)
                        op = request.get('op')
                        if op == 'ping':
                            payload = {'ready': True, 'model_load_seconds': loaded_seconds}
                        else:
                            with lock:
                                if op == 'index':
                                    budget = max(1, float(os.environ.get('RAG_INDEX_TIMEOUT', '570')) - loaded_seconds)
                                    payload = engine.build_index(request['corpus'], timeout=budget)
                                elif op == 'query':
                                    payload = engine.answer(request['corpus'], request['query'],
                                        timeout=float(os.environ.get('RAG_QUERY_TIMEOUT', '28')) - 1)
                                else:
                                    raise ValueError('Unsupported operation.')
                        response = {'ok': True, 'result': payload}
                    except Exception as exc:
                        response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
                    try:
                        stream.write(json.dumps(response, ensure_ascii=False, allow_nan=False).encode() + b'\n')
                        stream.flush()
                    except (OSError, TimeoutError):
                        pass

        print('RAG worker ready', flush=True)
        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                Thread(target=handle, args=(connection,), daemon=True).start()
        finally:
            server.close()
            path.unlink(missing_ok=True)
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
