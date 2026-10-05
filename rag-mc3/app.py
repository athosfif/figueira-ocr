#!/usr/bin/env python3
"""Official two-phase CLI; all expensive work belongs to the resident worker."""
from __future__ import annotations
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from rag_core import write_result, validate_query_id, refusal


def request_worker(payload, timeout):
    path = os.environ.get('RAG_WORKER_SOCKET', '/tmp/figueira-rag.sock')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(path)
        with client.makefile('rwb') as stream:
            stream.write(json.dumps(payload, ensure_ascii=False).encode() + b'\n')
            stream.flush()
            raw = stream.readline(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024:
        raise RuntimeError('Invalid worker response.')
    response = json.loads(raw)
    if not response.get('ok'):
        raise RuntimeError(response.get('error', 'Worker failed.'))
    return response['result']


def wait_ready(timeout):
    deadline = time.monotonic() + timeout
    last = 'not ready'
    while time.monotonic() < deadline:
        try:
            return request_worker({'op': 'ping'}, min(2, max(.01, deadline - time.monotonic())))
        except (OSError, ValueError, RuntimeError) as exc:
            last = str(exc)
            time.sleep(.1)
    raise RuntimeError(f'Resident RAG worker not ready: {last}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--query-id')
    parser.add_argument('--query')
    parser.add_argument('--output-dir', type=Path, default=Path('/app/output'))
    args = parser.parse_args(argv)
    if args.index is not None:
        if args.corpus is not None or args.query is not None or args.query_id is not None:
            parser.error('--index cannot be combined with query arguments.')
        try:
            started = time.monotonic()
            budget = float(os.environ.get('RAG_STARTUP_TIMEOUT', '570'))
            wait_ready(budget)
            report = request_worker({'op': 'index', 'corpus': str(args.index.absolute())},
                                    max(1, budget - (time.monotonic() - started)))
            print(json.dumps(report, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(f'RAG index error: {exc}', file=sys.stderr)
            return 1
    if args.corpus is None or args.query_id is None or args.query is None:
        parser.error('Query mode requires --corpus, --query-id and --query.')
    try:
        validate_query_id(args.query_id)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        result = request_worker({'op': 'query', 'corpus': str(args.corpus.absolute()), 'query': args.query},
                                float(os.environ.get('RAG_QUERY_TIMEOUT', '28')))
        output = write_result(args.output_dir, args.query_id, result)
        print(json.dumps({'output': str(output), 'result': result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        write_result(args.output_dir, args.query_id, refusal())
        print(f'RAG query error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
