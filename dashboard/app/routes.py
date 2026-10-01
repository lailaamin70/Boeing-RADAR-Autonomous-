import json
import os

from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file
from werkzeug.security import safe_join

from app import analytics, scene_viewer, truckscenes_loader as tsl

bp = Blueprint("dashboard", __name__)

VALID_SENSORS = ("radar", "lidar")

@bp.route("/")
def index():
    """Scene list, with optional tag filter and simple pagination."""
    tag_filter = request.args.get("tag", "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = current_app.config["SCENES_PER_PAGE"]

    scenes = tsl.list_scenes()

    # Collect every distinct tag across all scenes for the filter dropdown
    all_tags = sorted({tag for s in scenes for tag in s["tags"]})

    if tag_filter:
        scenes = [s for s in scenes if tag_filter in s["tags"]]

    total = len(scenes)
    start = (page - 1) * per_page
    page_scenes = scenes[start : start + per_page]
    total_pages = max((total + per_page - 1) // per_page, 1)

    return render_template(
        "index.html",
        scenes=page_scenes,
        all_tags=all_tags,
        active_tag=tag_filter,
        page=page,
        total_pages=total_pages,
        total=total,
        version=current_app.config["TRUCKSCENES_VERSION"],
    )


@bp.route("/scene/<scene_token>")
def scene_detail(scene_token):
    try:
        scene = tsl.get_scene(scene_token)
    except KeyError:
        abort(404)

    samples = tsl.list_samples_for_scene(scene_token)
    tags = tsl.parse_scene_tags(scene["description"])

    return render_template(
        "scene_detail.html",
        scene=scene,
        tags=tags,
        samples=samples,
    )


@bp.route("/sample/<sample_token>")
def sample_detail(sample_token):
    try:
        sample = tsl.get_sample(sample_token)
    except KeyError:
        abort(404)

    channels = tsl.get_sample_data_by_channel(sample_token)
    scene = tsl.get_scene(sample["scene_token"])

    return render_template(
        "sample_detail.html",
        sample=sample,
        scene=scene,
        channels=channels,
    )


@bp.route("/sample/<sample_token>/plot/<channel>")
def sensor_plot(sample_token, channel):
    """Server-rendered BEV scatter plot (PNG) for a radar or lidar channel."""
    channels = tsl.get_sample_data_by_channel(sample_token)
    entry = channels.get(channel)
    if entry is None:
        abort(404)

    modality = entry["modality"]
    if modality not in ("radar", "lidar"):
        abort(400, description="Plot endpoint only supports radar/lidar channels")

    sd_token = entry["sample_data"]["token"]
    buf = tsl.render_bev_png(sd_token, modality)
    return send_file(buf, mimetype="image/png")


@bp.route("/media/<path:filename>")
def media(filename):
    """
    Raw sensor files (camera jpgs, point cloud binaries).
    """
    dataroot = current_app.config["TRUCKSCENES_DATAROOT"]
    safe_path = safe_join(dataroot, filename)
    if safe_path is None or not os.path.isfile(safe_path):
        abort(404)
    return send_file(safe_path)


@bp.route("/analysis")
def analysis():
    """Cross-scene comparison page"""
    scenes = tsl.list_scenes()
    all_tags = sorted({tag for s in scenes for tag in s["tags"]})
    return render_template(
        "analysis.html",
        all_tags=all_tags,
        metrics=analytics.metric_catalog(),
    )


@bp.route("/api/analysis")
def api_analysis():
    """JSON data behind the analysis page charts"""
    tags = request.args.getlist("tag")
    metrics = request.args.getlist("metric") or ["sweep_point_count"]
    group_by = request.args.get("group_by", "tag")
    if group_by not in ("tag", "scene"):
        group_by = "tag"
    include_expensive = request.args.get("expensive") == "1"

    data = analytics.aggregate_by_condition(
        selected_tags=tags,
        selected_metrics=metrics,
        group_by=group_by,
        include_expensive=include_expensive,
    )
    return jsonify(data)


@bp.route("/scene/<scene_token>/viewer")
def scene_viewer_page(scene_token):
    """Scene player: top-down/3D view, sensor toggles, annotation overlay."""
    try:
        scene = tsl.get_scene(scene_token)
    except KeyError:
        abort(404)

    samples = tsl.list_samples_for_scene(scene_token)

    # Per sample metadata for playback
    samples_payload = []
    camera_channels = set()
    for s in samples:
        channels = tsl.get_sample_data_by_channel(s["token"])
        entry = {"token": s["token"], "timestamp": s["timestamp"], "channels": {}}
        for channel, info in channels.items():
            entry["channels"][channel] = {
                "modality": info["modality"],
                "filename": info["sample_data"]["filename"] if info["modality"] == "camera" else None,
            }
            if info["modality"] == "camera":
                camera_channels.add(channel)
        samples_payload.append(entry)

    return render_template(
        "scene_viewer.html",
        scene=scene,
        samples_json=json.dumps(samples_payload),
        camera_channels=sorted(camera_channels),
    )


def _parse_sensors(args):
    sensors = [s for s in args.getlist("sensor") if s in VALID_SENSORS]
    return sensors or list(VALID_SENSORS)


@bp.route("/sample/<sample_token>/topdown.png")
def sample_topdown(sample_token):
    """Radar/Lidar top-down PNG for one frame."""
    sensors = _parse_sensors(request.args)
    show_annotations = request.args.get("annotations") == "1"
    try:
        buf = scene_viewer.render_topdown_png(sample_token, sensors, show_annotations)
    except KeyError:
        abort(404)
    return send_file(buf, mimetype="image/png")


@bp.route("/sample/<sample_token>/points.json")
def sample_points(sample_token):
    """Radar/Lidar points and annotation boxes for the 3D view."""
    sensors = _parse_sensors(request.args)
    show_annotations = request.args.get("annotations") == "1"
    try:
        data = scene_viewer.get_points_payload(sample_token, sensors, show_annotations)
    except KeyError:
        abort(404)
    return jsonify(data)


@bp.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404
