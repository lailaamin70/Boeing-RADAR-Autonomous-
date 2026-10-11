import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tests.live_server import LiveServer


class LiveServerTest(unittest.TestCase):
    """The browser tests stand on this helper, so it is tested without needing a browser."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.server = LiveServer(Path(tmp.name) / "data").start()
        self.stopped = False
        self.addCleanup(lambda: None if self.stopped else self.server.stop())

    def fetch(self, path):
        with urllib.request.urlopen(self.server.url + path, timeout=10) as r:
            return r.status, r.headers.get_content_type(), r.read()

    def test_serves_pages_images_and_json_over_http(self):
        status, kind, body = self.fetch("/")
        self.assertEqual((status, kind), (200, "text/html"))
        self.assertIn(b"scene-0001", body)
        status, kind, body = self.fetch("/sample/A/topdown.png?sensor=lidar&ego=1")
        self.assertEqual((status, kind, body[:4]), (200, "image/png", b"\x89PNG"))
        self.assertEqual(json.loads(self.fetch("/api/overview")[2])["counts"]["annotations"], 4)

    def test_stub_detector_is_wired_in(self):
        points = json.loads(self.fetch("/sample/A/points.json?sensor=lidar&sensor=radar&detections=1")[2])
        self.assertEqual([d["modality"] for d in points["detections"]], ["lidar"])

    def test_handles_concurrent_requests(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex:
            statuses = list(ex.map(lambda p: self.fetch(p)[0], ["/", "/overview", "/detectability", "/analysis", "/scene/S", "/api/overview"] * 2))
        self.assertEqual(set(statuses), {200})

    def test_stop_releases_the_port_and_undoes_the_patches(self):
        from app import truckscenes_loader as tsl
        self.server.stop()
        self.stopped = True
        with self.assertRaises(urllib.error.URLError):
            self.fetch("/")
        self.assertNotEqual(getattr(tsl.get_trucksc, "__name__", ""), "<lambda>")  # real function restored


if __name__ == "__main__":
    unittest.main()
