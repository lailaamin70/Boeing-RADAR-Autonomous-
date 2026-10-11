"""
Runs the page scripts (overview, detectability, ego motion, analysis, scene
viewer) in Node against REAL API output from the Flask app, with a fake DOM,
Chart.js and Plotly.
"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import create_app, detections_loader
from tests.base import DatasetTestCase

HARNESS = Path(__file__).parent / "js" / "harness.js"
DETECTION = {"center": [30.0, 0.0, 0.9], "size": [2.0, 4.5, 1.8], "yaw": 0.0, "label": "car", "score": 0.9}


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class PageScriptsTest(DatasetTestCase):
    def test_page_scripts_work_against_real_api_output(self):
        app = create_app()
        app.testing = True
        client = app.test_client()

        def get(url):
            r = client.get(url)
            self.addCleanup(r.close)
            self.assertEqual(r.status_code, 200, url)
            return r

        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out, True)

        def save(name, url):
            (out / f"{name}.json").write_text(json.dumps(get(url).get_json()))

        save("overview", "/api/overview")
        save("volume", "/api/data_volume")
        save("detectability", "/api/detectability")
        save("detectability_camera", "/api/detectability?camera=1&min_visibility=2")
        save("ego", "/api/scene/S/ego_motion")
        save("analysis", "/api/analysis?metric=num_samples&metric=visibility&group_by=tag")
        (out / "ego_empty.json").write_text(json.dumps({"t": [], "note": "No ego-motion rows fall inside this scene's time window."}))
        (out / "viewer.html").write_text(get("/scene/S/viewer").get_data(as_text=True))

        with mock.patch.object(detections_loader, "get_predictions", return_value=[DETECTION]):
            save("points", "/sample/A/points.json?sensor=radar&sensor=lidar&annotations=1&detections=1&ego=1")
            save("overlay", "/sample/A/camera/CAMERA_FRONT/overlay.json?annotations=1&detections=1&sensor=lidar")

        result = subprocess.run(["node", str(HARNESS), str(out)], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, f"\n{result.stdout}\n{result.stderr}")
        print("\n" + result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
