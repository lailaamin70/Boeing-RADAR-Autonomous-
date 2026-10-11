"""
Browser end-to-end tests: needs selsnium, chrome, and internet connection
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    HAVE_SELENIUM = True
except ImportError:  # pragma: no cover
    HAVE_SELENIUM = False

from tests.live_server import LiveServer

# JavaScript run inside the page
CHART_JS = """
const canvas = document.getElementById(arguments[0]);
const chart = canvas && window.Chart ? Chart.getChart(canvas) : null;
if (!chart) return null;
return {type: chart.config.type, labels: chart.data.labels,
        datasets: chart.data.datasets.map(d => ({label: d.label, data: d.data})),
        yTitle: (chart.options.scales.y.title || {}).text || null};
"""
LIB_JS = "return typeof window[arguments[0]] !== 'undefined';"
WEBGL_JS = "const c = document.createElement('canvas'); return !!(c.getContext('webgl') || c.getContext('experimental-webgl'));"
IMG_LOADED_JS = "const i = document.getElementById(arguments[0]); return !!(i && i.complete && i.naturalWidth > 0);"
SAMPLE_IMGS_JS = "const i = [...document.querySelectorAll('.channel-card img')]; return i.length === 3 && i.every(x => x.complete && x.naturalWidth > 0);"
SET_SLIDER_JS = "const s = document.getElementById('frame-slider'); s.value = arguments[0]; s.dispatchEvent(new Event('input')); return s.value;"
CLICK_PLAY_JS = "document.getElementById('play-btn').click(); return document.getElementById('frame-label').textContent;"
FRAME_B_SELECTED_JS = "return document.getElementById('frame-label').textContent === '2 / 2' && document.getElementById('topdown-img').getAttribute('src').includes('/sample/B/');"
PLAYBACK_DONE_JS = "return document.getElementById('frame-label').textContent === '2 / 2' && document.getElementById('play-btn').textContent === 'Play';"
TOPDOWN_SRC_JS = "return document.getElementById('topdown-img').getAttribute('src');"
OVERLAY_LINES_JS = "return document.querySelectorAll('#camera-overlay line').length;"
DASHED_LINES_JS = "return document.querySelectorAll('#camera-overlay line[stroke-dasharray]').length;"
PLOT_NAMES_JS = "const gd = document.getElementById('plot3d-wrap'); return gd.data ? gd.data.map(t => t.name || t.type) : null;"
NO_EGO_TRUCK_JS = "const gd = document.getElementById('plot3d-wrap'); return !!gd.data && gd.data.length > 0 && !gd.data.some(t => t.name === 'ego truck');"
UIREVISION_JS = "const gd = document.getElementById('plot3d-wrap'); return gd.layout ? gd.layout.uirevision : null;"
EYE_JS = "const gd = document.getElementById('plot3d-wrap'); const c = gd.layout && gd.layout.scene && gd.layout.scene.camera; return c && c.eye ? [c.eye.x, c.eye.y, c.eye.z] : null;"
LIDAR_B_POINTS_JS = "const gd = document.getElementById('plot3d-wrap'); const t = gd.data && gd.data.find(t => t.name === 'lidar'); return !!t && t.x.length === 300;"
TABLE_HAS_EXPENSIVE_JS = "const t = document.getElementById('table-wrap').innerText; return t.includes('Points per sweep') && t.includes('200');"
ANALYSIS_CHARTS_JS = "return document.querySelectorAll('#charts canvas').length;"
SUMMARY_TOTAL_JS = "const e = document.querySelector('#summary .h4'); return e ? e.textContent.trim() : null;"

# a JS predicate per page that is true once the page has finished loading its data
READY = {
    "/": "return !!document.querySelector('.scene-card');",
    "/overview": "return !!(window.Chart && Chart.getChart(document.getElementById('c-volume')));",
    "/detectability": "return !!(window.Chart && Chart.getChart(document.getElementById('c-combos')));",
    "/analysis": "return document.querySelectorAll('#charts canvas').length > 0;",
    "/scene/S": "return !!(window.Chart && Chart.getChart(document.getElementById('c-path')));",
    "/scene/S/viewer": "return document.getElementById('frame-label').textContent === '1 / 2';",
    "/sample/A": SAMPLE_IMGS_JS,
}
LIBS = {"/overview": ["Chart"], "/detectability": ["Chart"], "/analysis": ["Chart"], "/scene/S": ["Chart"], "/scene/S/viewer": ["Plotly"]}

_server = None
_driver = None
_tmp = None
_cdn_ok = False


def _make_driver():
    opts = webdriver.ChromeOptions()
    if os.environ.get("E2E_HEADED") != "1":
        opts.add_argument("--headless=new")
    if os.environ.get("CHROME_BIN"):
        opts.binary_location = os.environ["CHROME_BIN"]
    for arg in ("--window-size=1500,1100", "--no-sandbox", "--disable-dev-shm-usage",
                # software WebGL so the 3D view works on machines without a GPU
                "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"):
        opts.add_argument(arg)
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    return webdriver.Chrome(options=opts)


def setUpModule():
    global _server, _driver, _tmp, _cdn_ok
    if not HAVE_SELENIUM:
        raise unittest.SkipTest("selenium is not installed (pip install selenium)")
    _tmp = tempfile.mkdtemp()
    _server = LiveServer(Path(_tmp) / "data").start()
    try:
        _driver = _make_driver()
    except WebDriverException as err:
        tearDownModule()
        raise unittest.SkipTest(f"could not start Chrome via Selenium: {(err.msg or str(err)).strip()[:200]}")
    # the pages pull Chart.js / Plotly from a CDN; probe once so an offline machine skips instead of failing everything
    _driver.get(_server.url + "/scene/S/viewer")
    plotly = _driver.execute_script(LIB_JS, "Plotly")
    _driver.get(_server.url + "/overview")
    _cdn_ok = bool(plotly and _driver.execute_script(LIB_JS, "Chart"))


def tearDownModule():
    global _server, _driver, _tmp
    if _driver is not None:
        _driver.quit()
        _driver = None
    if _server is not None:
        _server.stop()
        _server = None
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
        _tmp = None


class BrowserTest(unittest.TestCase):
    check_console = True

    @property
    def driver(self):
        return _driver

    def setUp(self):
        self.check_console = False  # nothing to check if we skip
        if not _cdn_ok:
            self.skipTest("Chart.js / Plotly did not load from the CDN - are you offline?")
        self.check_console = True
        self.driver.get("about:blank")
        self.console_errors()  # drain anything left from the previous test

    def tearDown(self):
        if self.check_console:
            errors = self.console_errors()
            if errors:
                self.fail("browser console errors:\n  " + "\n  ".join(errors))

    # ---- helpers ----
    def console_errors(self):
        try:
            entries = self.driver.get_log("browser")
        except Exception:  # not every driver exposes the log
            return []
        return [e["message"] for e in entries if e.get("level") == "SEVERE" and "favicon" not in e["message"]]

    def open(self, path):
        self.driver.get(_server.url + path)

    def js(self, script, *args):
        return self.driver.execute_script(script, *args)

    def wait(self, condition, what, timeout=15):
        try:
            return WebDriverWait(self.driver, timeout).until(condition)
        except TimeoutException:
            self.fail(f"timed out after {timeout}s waiting for: {what}")

    def wait_js(self, script, what, *args, timeout=15):
        return self.wait(lambda d: d.execute_script(script, *args), what, timeout)

    def chart(self, canvas_id):
        return self.wait_js(CHART_JS, f"chart #{canvas_id} to render", canvas_id)

    def require_libs(self, *names):
        for name in names:
            if not self.js(LIB_JS, name):
                self.check_console = False  # the failed CDN request would also be logged as an error
                self.skipTest(f"{name} did not load - are the CDN assets reachable?")

    def require_webgl(self):
        if not self.js(WEBGL_JS):
            self.skipTest("this browser has no WebGL, which the 3D view needs")

    def click(self, css):
        self.driver.find_element(By.CSS_SELECTOR, css).click()

    def text(self, css):
        return self.driver.find_element(By.CSS_SELECTOR, css).text

    def displayed(self, element_id):
        return self.driver.find_element(By.ID, element_id).is_displayed()

    def click_link(self, text):
        self.driver.find_element(By.LINK_TEXT, text).click()

    def wait_url(self, fragment):
        self.wait(lambda d: fragment in d.current_url, f"URL containing {fragment!r}")

    def open_viewer(self):
        self.open("/scene/S/viewer")
        self.require_libs("Plotly")
        self.wait_js(IMG_LOADED_JS, "the first top-down image to load", "topdown-img")


@unittest.skipUnless(HAVE_SELENIUM, "selenium is not installed")
class NavigationTest(BrowserTest):
    def test_sidebar_reaches_every_page(self):
        self.open("/")
        self.assertEqual(self.text("h1"), "Scenes")
        self.assertIn("scene-0001", self.text("main"))
        for link, heading in (("Analysis", "Cross-scene analysis"), ("Overview", "Dataset overview"),
                              ("Detectability", "Detectability"), ("Scenes", "Scenes")):
            with self.subTest(link=link):
                self.click_link(link)
                self.wait(lambda d, h=heading: d.find_element(By.CSS_SELECTOR, "h1").text == h, f"heading {heading!r}")

    def test_scene_card_opens_the_scene_page_and_tags_filter_the_list(self):
        self.open("/")
        self.click("a.tag[href*='tag=clear']")
        self.wait_url("tag=clear")
        self.assertIn("scene-0001", self.text("main"))
        self.click(".scene-card")
        self.wait_url("/scene/S")
        self.assertEqual(self.text("h1"), "scene-0001")

    def test_unknown_scene_shows_the_friendly_not_found_page(self):
        self.check_console = False  # a 404 document may be logged by the browser
        self.open("/scene/NOPE")
        self.assertIn("Not found", self.text("main"))

    def test_no_console_errors_on_any_page(self):
        for path, ready in READY.items():
            with self.subTest(page=path):
                self.open(path)
                self.require_libs(*LIBS.get(path, []))
                self.wait_js(ready, f"{path} to finish loading")
                self.assertEqual(self.console_errors(), [], path)

    def test_sample_page_images_all_load(self):
        self.open("/sample/A")
        self.wait_js(SAMPLE_IMGS_JS, "camera image and both BEV plots to load")


@unittest.skipUnless(HAVE_SELENIUM, "selenium is not installed")
class ScenePageTest(BrowserTest):
    def test_ego_motion_section(self):
        self.open("/scene/S")
        self.require_libs("Chart")
        self.wait(lambda d: "Mean speed 36" in d.find_element(By.ID, "ego-status").text, "ego-motion summary")
        self.assertTrue(self.displayed("ego-charts"))
        self.assertEqual(set(self.chart("c-speed")["datasets"][0]["data"]), {36.0})
        self.assertEqual([d["label"] for d in self.chart("c-tilt")["datasets"]], ["pitch", "roll"])
        self.assertEqual(self.chart("c-path")["type"], "scatter")

    def test_sample_strip_and_viewer_button(self):
        self.open("/scene/S")
        self.require_libs("Chart")
        self.click(".sample-dot")
        self.wait_url("/sample/A")
        self.open("/scene/S")
        self.click("a.btn-accent")
        self.wait_url("/scene/S/viewer")


@unittest.skipUnless(HAVE_SELENIUM, "selenium is not installed")
class OverviewPageTest(BrowserTest):
    def test_counts_charts_tables_and_volume(self):
        self.open("/overview")
        self.require_libs("Chart")
        cats = self.chart("c-categories")
        self.assertEqual(cats["labels"], ["vehicle.car", "vehicle.truck", "human.pedestrian.adult"])
        self.assertEqual(self.chart("c-distance")["datasets"][0]["data"].count(1), 4)
        self.assertEqual(self.chart("c-volume")["labels"], ["CAMERA_FRONT", "LIDAR_TOP", "RADAR_FRONT"])
        self.wait(lambda d: "CAMERA_FRONT" in d.find_element(By.ID, "volume-table").text, "volume table")
        table = self.text("#scenes-table")
        self.assertIn("scene-0001", table)
        self.assertIn("36", table)  # mean speed, km/h

    def test_scene_link_in_the_table_opens_the_scene(self):
        self.open("/overview")
        self.require_libs("Chart")
        self.wait(lambda d: d.find_elements(By.CSS_SELECTOR, "#scenes-table a"), "scenes table")
        self.click("#scenes-table a")
        self.wait_url("/scene/S")


@unittest.skipUnless(HAVE_SELENIUM, "selenium is not installed")
class DetectabilityPageTest(BrowserTest):
    def open_page(self):
        self.open("/detectability")
        self.require_libs("Chart")
        self.chart("c-combos")

    def total(self, expected):
        self.wait(lambda d: self.js(SUMMARY_TOTAL_JS) == str(expected), f"object total to become {expected}")

    def apply(self):
        self.click("#apply-btn")

    def test_defaults(self):
        self.open_page()
        self.total(4)
        self.assertEqual([d["label"] for d in self.chart("c-mean")["datasets"]], ["lidar", "radar"])
        self.assertEqual(self.chart("c-combos")["yTitle"], "% of objects")
        self.assertIn("20-30", self.text("#detail-table"))

    def test_category_filter(self):
        self.open_page()
        self.total(4)
        self.click("input.cat-check[value='vehicle.car']")
        self.apply()
        self.total(2)
        self.click("input.cat-check[value='vehicle.car']")
        self.apply()
        self.total(4)

    def test_threshold_is_sent_and_shown(self):
        self.open_page()
        self.total(4)
        box = self.driver.find_element(By.ID, "min-lidar")
        box.clear()
        box.send_keys("100")
        self.apply()
        self.wait(lambda d: self.chart("c-seen")["datasets"][0]["label"] == "lidar (\u2265 100)", "seen chart to use the new threshold")
        seen = [v for v in self.chart("c-seen")["datasets"][0]["data"] if v is not None]
        self.assertTrue(seen and all(v == 0 for v in seen), "nothing has 100 lidar points")

    def test_percent_switch_changes_the_overlap_chart_units(self):
        self.open_page()
        self.assertEqual(self.chart("c-combos")["yTitle"], "% of objects")
        self.click("#combo-percent")
        self.wait(lambda d: self.chart("c-combos")["yTitle"] == "objects", "overlap chart to switch to counts")

    def test_including_the_camera_adds_camera_classes_and_the_dropped_note(self):
        self.open_page()
        self.assertFalse(self.driver.find_element(By.ID, "min-vis").is_enabled())
        self.click("#include-camera")
        self.wait(lambda d: self.driver.find_element(By.ID, "min-vis").is_enabled(), "visibility select to enable")
        self.wait(lambda d: "lidar + radar + camera" in [x["label"] for x in self.chart("c-combos")["datasets"]], "camera classes")
        self.wait(lambda d: "left out" in d.find_element(By.ID, "combo-note").text, "dropped-objects note")


@unittest.skipUnless(HAVE_SELENIUM, "selenium is not installed")
class AnalysisPageTest(BrowserTest):
    def test_expensive_metrics_appear_after_opting_in(self):
        self.open("/analysis")
        self.require_libs("Chart")
        self.wait(lambda d: self.js(ANALYSIS_CHARTS_JS) == 3, "one chart per default metric")
        self.click("#include-expensive")
        self.click("#apply-btn")
        self.wait_js(TABLE_HAS_EXPENSIVE_JS, "point-cloud metrics (200 lidar points per sweep) in the table")


@unittest.skipUnless(HAVE_SELENIUM, "selenium is not installed")
class SceneViewerTest(BrowserTest):
    def test_first_frame_loads_a_real_top_down_image(self):
        self.open_viewer()
        self.assertEqual(self.text("#frame-label"), "1 / 2")
        self.assertEqual(self.text("#main-view-title"), "Top-down")

    def test_slider_jumps_frames_and_play_restarts_from_the_first(self):
        self.open_viewer()
        self.js(SET_SLIDER_JS, 1)
        self.wait_js(FRAME_B_SELECTED_JS, "frame 2 to be selected")
        self.wait_js(IMG_LOADED_JS, "frame 2's top-down image", "topdown-img")
        self.assertEqual(self.js(CLICK_PLAY_JS), "1 / 2", "pressing Play at the last frame must start over")
        self.wait_js(PLAYBACK_DONE_JS, "playback to reach the end and stop")

    def test_overlay_toggles_change_the_top_down_request(self):
        self.open_viewer()
        src = self.js(TOPDOWN_SRC_JS)
        self.assertIn("ego=1", src)
        self.assertIn("annotations=1", src)
        self.assertNotIn("detections", src)
        self.click("#show-ego")
        self.assertNotIn("ego=1", self.js(TOPDOWN_SRC_JS))
        self.click("#show-detections")
        self.assertIn("detections=1", self.js(TOPDOWN_SRC_JS))
        self.assertTrue(self.displayed("score-threshold-wrap"))
        self.click("#sensor-radar")
        self.assertNotIn("radar", self.js(TOPDOWN_SRC_JS))

    def test_camera_overlay_follows_the_toggles(self):
        self.open_viewer()
        self.wait_js(IMG_LOADED_JS, "the camera image", "camera-img")
        self.wait(lambda d: self.js(OVERLAY_LINES_JS) == 36, "3 projected boxes x 12 edges")  # a4 and the ego trailer are behind the camera
        self.assertEqual(self.js(DASHED_LINES_JS), 0)
        self.click("#show-annotations")
        self.wait(lambda d: self.js(OVERLAY_LINES_JS) == 0, "overlay to clear")
        self.click("#show-detections")
        self.wait(lambda d: self.js(OVERLAY_LINES_JS) == 12, "the one detection's 12 edges")
        self.assertEqual(self.js(DASHED_LINES_JS), 12, "detections are drawn dashed")

    def test_3d_view_draws_points_truck_sensors_and_boxes(self):
        self.open_viewer()
        self.require_webgl()
        self.click("#view-3d")
        names = self.wait_js(PLOT_NAMES_JS, "the 3D traces")
        for expected in ("lidar", "radar", "ego truck", "lidar sensors", "radar sensors", "camera sensors", "annotations"):
            self.assertIn(expected, names)
        self.assertTrue(self.displayed("plot3d-wrap"))
        self.assertFalse(self.displayed("topdown-wrap"))
        self.assertEqual(self.js(UIREVISION_JS), "scene-viewer-3d")
        self.click("#show-ego")
        self.wait_js(NO_EGO_TRUCK_JS, "the truck to disappear from the plot")

    def test_3d_orbit_survives_a_frame_change(self):
        self.open_viewer()
        self.require_webgl()
        self.click("#view-3d")
        self.wait_js(PLOT_NAMES_JS, "the 3D traces")
        plot = self.driver.find_element(By.ID, "plot3d-wrap")
        before = self.js(EYE_JS)
        ActionChains(self.driver).move_to_element(plot).click_and_hold().move_by_offset(150, 80).release().perform()
        orbited = self.js(EYE_JS)
        if not orbited or orbited == before:
            self.skipTest("could not orbit the plot with simulated mouse input in this browser")
        self.js(SET_SLIDER_JS, 1)
        self.wait_js(LIDAR_B_POINTS_JS, "frame 2's point cloud (300 lidar points) to be drawn")
        for now, then in zip(self.js(EYE_JS), orbited):
            self.assertAlmostEqual(now, then, places=6, msg="the camera angle was reset by the frame change")


if __name__ == "__main__":
    unittest.main()
