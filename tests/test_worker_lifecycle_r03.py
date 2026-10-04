"""R03 lifecycle checks with local subprocesses and an injected, GPU-free model.

Run from src with: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
    -s tests -p test_worker_lifecycle_r03.py -v
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import backends


ROOT = Path(__file__).resolve().parents[1]

FAKE_WORKER = r'''
import os
import time
from pathlib import Path
import worker

class FakeModel:
    def __init__(self, model_dir):
        with open(os.environ["R03_LOAD_MARKER"], "a", encoding="utf-8") as out:
            out.write(str(os.getpid()) + "\n")
        time.sleep(float(os.environ.get("R03_MODEL_LOAD_DELAY", "0")))

    def read(self, image):
        Path(os.environ["R03_READ_STARTED"]).touch()
        time.sleep(float(os.environ.get("R03_READ_DELAY", "0")))
        Path(os.environ["R03_READ_FINISHED"]).touch()
        return "FAKE OCR", {"fake_model": True, "worker_pid": os.getpid()}

raise SystemExit(worker.main(model_factory=FakeModel))
'''

FAKE_CLI = r'''
import sys
from unittest.mock import patch
import app
import backends

with patch("backends.platform.system", return_value="Linux"), \
     patch("subprocess.Popen", side_effect=AssertionError("CLI must never spawn a worker")):
    raise SystemExit(app.main(sys.argv[1:]))
'''


class WorkerLifecycleR03Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="figueira-ocr-r03-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.socket_path = self.directory / "worker.sock"
        self.load_marker = self.directory / "model-loads.txt"
        self.read_started = self.directory / "read-started"
        self.read_finished = self.directory / "read-finished"
        self.image_path = self.directory / "input.png"
        Image.new("RGB", (32, 16), "white").save(self.image_path)
        self.processes = []
        self.addCleanup(self._stop_processes)

    def _environment(self, *, load_delay=0, read_delay=0):
        return dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(ROOT),
            OCR_WORKER_SOCKET=str(self.socket_path),
            OCR_WORKER_LOG=str(self.directory / "worker.log"),
            OCR_MODEL_DIR=str(self.directory / "unused-fake-model"),
            OCR_STARTUP_TIMEOUT="5",
            OCR_INFERENCE_TIMEOUT="2",
            R03_LOAD_MARKER=str(self.load_marker),
            R03_READ_STARTED=str(self.read_started),
            R03_READ_FINISHED=str(self.read_finished),
            R03_MODEL_LOAD_DELAY=str(load_delay),
            R03_READ_DELAY=str(read_delay),
        )

    def _process(self, code, *, env, arguments=()):
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", code, *arguments],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.processes.append(process)
        return process

    def _worker(self, *, load_delay=0, read_delay=0):
        return self._process(
            FAKE_WORKER,
            env=self._environment(load_delay=load_delay, read_delay=read_delay),
        )

    def _stop_processes(self):
        for process in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            process.communicate(timeout=3)

    def _wait_for(self, predicate, *, description, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail(f"Timed out waiting for {description}")

    def _load_count(self):
        if not self.load_marker.exists():
            return 0
        return len(self.load_marker.read_text(encoding="utf-8").splitlines())

    def _ready(self):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(0.1)
                connection.connect(str(self.socket_path))
                connection.sendall(b'{"op":"ping"}\n')
                with connection.makefile("rb") as stream:
                    return json.loads(stream.readline()).get("ok") is True
        except (OSError, ValueError):
            return False

    def _wait_ready(self, process):
        self._wait_for(self._ready, description="the fake worker readiness")
        self.assertIsNone(process.poll(), "The foreground worker must remain alive")

    @contextmanager
    def _client_context(self, *, startup_timeout=3, inference_timeout=2):
        with patch.object(backends, "SOCKET_PATH", self.socket_path), \
             patch("backends.platform.system", return_value="Linux"), \
             patch.dict(os.environ, {
                 "OCR_STARTUP_TIMEOUT": str(startup_timeout),
                 "OCR_INFERENCE_TIMEOUT": str(inference_timeout),
             }), \
             patch("subprocess.Popen", side_effect=AssertionError(
                 "Client must never spawn a worker"
             )) as spawn:
            yield spawn

    def test_two_workers_reject_second_before_model_load(self):
        first = self._worker(load_delay=0.7)
        self._wait_for(lambda: self._load_count() == 1, description="the first model load")
        second = self._worker()
        second.communicate(timeout=3)
        self.assertNotEqual(second.returncode, 0, "A concurrent worker must be rejected")
        self.assertEqual(self._load_count(), 1, "The lock must precede model construction")
        self._wait_ready(first)
        with self._client_context() as spawn:
            client = backends.ROCmWorkerClient()
            with Image.open(self.image_path) as image:
                text, details = client.read(image)
            self.assertEqual(text, "FAKE OCR")
            self.assertEqual(details["worker_pid"], first.pid)
            spawn.assert_not_called()

    def test_two_cli_processes_wait_during_load_and_reuse_one_worker(self):
        worker = self._worker(load_delay=0.8)
        self._wait_for(lambda: self._load_count() == 1, description="the delayed model load")
        clients = []
        for index in range(2):
            run = self.directory / f"cli-{index}"
            arguments = (
                "--backend", "rocm", "--input-image", str(self.image_path),
                "--output-dir", str(run), "--report", str(run / "report.json"),
            )
            clients.append((self._process(FAKE_CLI, env=self._environment(),
                                          arguments=arguments), run))
        for client, run in clients:
            _, stderr = client.communicate(timeout=7)
            self.assertEqual(client.returncode, 0, f"GPU-free CLI failed: {stderr}")
            report = json.loads((run / "report.json").read_text(encoding="utf-8"))
            output = json.loads((run / "input_output.json").read_text(encoding="utf-8"))
            self.assertEqual(output, {"text": "FAKE OCR"})
            self.assertTrue(report["worker_reused"])
            self.assertEqual(report["worker_pid"], worker.pid)
        self.assertEqual(self._load_count(), 1)
        self.assertIsNone(worker.poll())

    def test_startup_timeout_never_spawns_a_worker(self):
        with self._client_context(startup_timeout=0.15) as spawn:
            started = time.monotonic()
            with self.assertRaises((RuntimeError, TimeoutError, OSError)):
                backends.ROCmWorkerClient()
            self.assertLess(time.monotonic() - started, 1.5)
            spawn.assert_not_called()
        self.assertEqual(self._load_count(), 0)

    def test_inference_timeout_does_not_spawn_and_worker_recovers(self):
        worker = self._worker(read_delay=0.35)
        self._wait_ready(worker)
        with self._client_context(inference_timeout=0.08) as spawn:
            client = backends.ROCmWorkerClient()
            with Image.open(self.image_path) as image:
                with self.assertRaises((RuntimeError, TimeoutError, OSError)):
                    client.read(image)
            spawn.assert_not_called()
        self._wait_for(self.read_finished.exists, description="the timed-out fake inference")
        self._wait_ready(worker)
        with self._client_context(inference_timeout=1) as spawn:
            client = backends.ROCmWorkerClient()
            with Image.open(self.image_path) as image:
                text, details = client.read(image)
            self.assertEqual(text, "FAKE OCR")
            self.assertEqual(details["worker_pid"], worker.pid)
            spawn.assert_not_called()
        self.assertEqual(self._load_count(), 1)

    def test_disconnected_client_does_not_kill_worker(self):
        worker = self._worker(read_delay=0.15)
        self._wait_ready(worker)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self.socket_path))
            request = {"op": "ocr", "path": str(self.image_path)}
            connection.sendall(json.dumps(request).encode("utf-8") + b"\n")
            self._wait_for(self.read_started.exists, description="the fake inference start")
        self._wait_for(self.read_finished.exists, description="the disconnected fake inference")
        self._wait_ready(worker)
        self.assertEqual(self._load_count(), 1)

    def test_sigterm_cleans_socket_and_releases_lock(self):
        first = self._worker()
        self._wait_ready(first)
        first.terminate()
        first.communicate(timeout=3)
        self.assertFalse(self.socket_path.exists(), "SIGTERM must remove the owned socket")
        second = self._worker()
        self._wait_ready(second)
        self.assertEqual(self._load_count(), 2, "A clean shutdown must release the lifetime lock")

    def test_docker_cmd_keeps_foreground_worker_alive(self):
        commands = [line.strip()[4:].strip() for line in
                    (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
                    if line.strip().startswith("CMD ")]
        self.assertEqual(len(commands), 1)
        self.assertEqual(json.loads(commands[0]), ["python3", "/app/worker.py"])
        process = self._worker()
        self._wait_ready(process)
        with self._client_context() as spawn:
            for _ in range(2):
                client = backends.ROCmWorkerClient()
                self.assertTrue(client.reused_worker)
            spawn.assert_not_called()
        self.assertIsNone(process.poll(), "Worker must keep serving after CLI clients exit")
        self.assertEqual(self._load_count(), 1)


if __name__ == "__main__":
    unittest.main()
