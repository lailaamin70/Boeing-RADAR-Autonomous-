"""Serves the app over real HTTP on a free port, backed by the synthetic dataset, for browser tests."""
import logging
import threading
from pathlib import Path
from unittest import mock

from werkzeug.serving import make_server

from tests import fake_dataset

# the stub detector: one lidar detection 30 m dead ahead, nothing from radar
DETECTION = {"center": [30.0, 0.0, 0.9], "size": [2.0, 4.5, 1.8], "yaw": 0.0, "label": "car", "score": 0.9}


class LiveServer:
    def __init__(self, root):
        from app import create_app, detections_loader

        self.dataset, self._stack = fake_dataset.install(Path(root))
        self._stack.enter_context(mock.patch.object(
            detections_loader, "get_predictions",
            side_effect=lambda modality, token, threshold=None: [dict(DETECTION)] if modality == "lidar" else [],
        ))
        logging.getLogger("werkzeug").setLevel(logging.ERROR)  # no per-request noise in test output
        self._server = make_server("127.0.0.1", 0, create_app(), threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server.server_close()
        self._stack.close()
