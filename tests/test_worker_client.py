import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import backends


class WorkerClientTests(unittest.TestCase):
    def test_rocm_client_refuses_non_linux_hosts(self):
        with patch("backends.platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                backends.ROCmWorkerClient()

    def test_worker_protocol_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            socket_path = Path(temp) / "worker.sock"
            ready = threading.Event()

            def serve_once():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                    server.bind(str(socket_path))
                    server.listen(1)
                    ready.set()
                    connection, _ = server.accept()
                    with connection:
                        stream = connection.makefile("rwb")
                        request = json.loads(stream.readline())
                        response = {"ok": True, "echo": request}
                        stream.write(json.dumps(response).encode("utf-8") + b"\n")
                        stream.flush()

            thread = threading.Thread(target=serve_once, daemon=True)
            thread.start()
            self.assertTrue(ready.wait(timeout=2))
            with patch("backends.SOCKET_PATH", socket_path):
                response = backends._request_worker({"op": "ping"}, timeout=2)
            thread.join(timeout=2)
            self.assertEqual(response["echo"], {"op": "ping"})


if __name__ == "__main__":
    unittest.main()
