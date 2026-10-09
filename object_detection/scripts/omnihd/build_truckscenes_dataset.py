#!/usr/bin/env python3
"""Build an OmniHD-format dataset with the existing TruckScenes adapter."""

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.object_detection.omnihd_adapter.build_dataset import (  # noqa: E402
    BuildConfig,
    build_dataset,
)
from src.object_detection.omnihd_adapter.lidar_adapter import (  # noqa: E402
    TRUCKSCENES_LIDAR_CHANNELS,
)
from src.object_detection.omnihd_adapter.radar_adapter import (  # noqa: E402
    TRUCKSCENES_RADAR_CHANNELS,
)


# These are loader compatibility slots, not claims that the TruckScenes and
# OmniHD sensors have physically equivalent positions.
RADAR_CHANNEL_ALIASES = {
    "RADAR_LEFT_FRONT": "radar_left_front",
    "RADAR_RIGHT_FRONT": "radar_right_front",
    "RADAR_LEFT_BACK": "radar_left_back",
    "RADAR_RIGHT_BACK": "radar_right_back",
    "RADAR_LEFT_SIDE": "radar_front",
    "RADAR_RIGHT_SIDE": "radar_back",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse dataset-build arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw TruckScenes data into the OmniHD/NewScenes format "
            "using the existing Boeing adapter."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Raw TruckScenes root containing samples/, sweeps/, and metadata.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory in which to write the converted dataset.",
    )
    parser.add_argument(
        "--metadata-version",
        default="v1.2-trainval",
        help="TruckScenes metadata directory name (default: v1.2-trainval).",
    )
    parser.add_argument(
        "--metadata-path-prefix",
        default=None,
        help=(
            "Optional portable path prefix stored inside the generated PKL, "
            "for example data/TruckScenes_OmniHD_trainval_135."
        ),
    )
    parser.add_argument(
        "--reference-channel",
        choices=TRUCKSCENES_LIDAR_CHANNELS,
        default="LIDAR_TOP_FRONT",
        help="LiDAR channel selecting the reference timestamp and ego pose.",
    )
    parser.add_argument(
        "--radar-power-mode",
        choices=("rcs_proxy", "constant"),
        default="rcs_proxy",
        help="RADAR power policy (default: rcs_proxy).",
    )
    parser.add_argument(
        "--radar-snr-fill-value",
        type=float,
        default=0.0,
        help="Constant SNR placeholder because TruckScenes has no compatible SNR.",
    )
    parser.add_argument(
        "--radar-power-fill-value",
        type=float,
        default=None,
        help="Required only when --radar-power-mode=constant.",
    )
    parser.add_argument(
        "--radar-sweeps-num",
        type=int,
        default=3,
        help="Maximum RADAR sweeps per channel (default: 3).",
    )
    parser.add_argument(
        "--ego-velocity-source",
        choices=("cabin", "chassis"),
        default="chassis",
        help="TruckScenes ego-motion source (default: chassis).",
    )
    parser.add_argument(
        "--info-filename",
        default="truckscenes_omnihd_trainval_135_infos.pkl",
        help="Output information-pickle filename.",
    )
    parser.add_argument(
        "--manifest-filename",
        default="build_manifest.json",
        help="Output build-manifest filename.",
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all-samples",
        action="store_true",
        help="Build every sample in the selected metadata version.",
    )
    selection.add_argument(
        "--sample-token",
        dest="sample_tokens",
        action="append",
        metavar="TOKEN",
        help="Build one sample token; repeat to select multiple samples.",
    )
    selection.add_argument(
        "--scene-token",
        dest="scene_tokens",
        action="append",
        metavar="TOKEN",
        help="Build one scene token; repeat to select multiple scenes.",
    )

    parser.add_argument(
        "--lidar-channel",
        dest="lidar_channels",
        action="append",
        choices=TRUCKSCENES_LIDAR_CHANNELS,
        metavar="CHANNEL",
        help="LiDAR channel to include; repeat as needed. Default: all channels.",
    )
    parser.add_argument(
        "--radar-channel",
        dest="radar_channels",
        action="append",
        choices=TRUCKSCENES_RADAR_CHANNELS,
        metavar="CHANNEL",
        help="RADAR channel to include; repeat as needed. Default: all channels.",
    )
    return parser.parse_args(argv)


def resolve_roots(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Resolve and validate source/output roots, creating the output if needed."""
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(
            "TruckScenes data root does not exist or is not a directory: {}".format(
                data_root
            )
        )
    metadata_root = data_root / args.metadata_version
    if not metadata_root.is_dir():
        raise FileNotFoundError(
            "TruckScenes metadata directory does not exist: {}".format(metadata_root)
        )
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(
            "Output root exists but is not a directory: {}".format(output_root)
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return data_root, output_root


def make_config(
    args: argparse.Namespace, data_root: Path, output_root: Path
) -> BuildConfig:
    """Translate CLI arguments directly into the existing adapter configuration."""
    lidar_channels = tuple(args.lidar_channels or TRUCKSCENES_LIDAR_CHANNELS)
    radar_channels = tuple(args.radar_channels or TRUCKSCENES_RADAR_CHANNELS)
    radar_aliases = {
        channel: RADAR_CHANNEL_ALIASES[channel] for channel in radar_channels
    }

    return BuildConfig(
        data_root=data_root,
        output_root=output_root,
        metadata_version=args.metadata_version,
        lidar_channels=lidar_channels,
        radar_channels=radar_channels,
        reference_channel=args.reference_channel,
        radar_channel_aliases=radar_aliases,
        radar_power_mode=args.radar_power_mode,
        radar_snr_fill_value=args.radar_snr_fill_value,
        ego_velocity_source=args.ego_velocity_source,
        radar_power_fill_value=args.radar_power_fill_value,
        radar_sweeps_num=args.radar_sweeps_num,
        sample_tokens=(
            tuple(args.sample_tokens) if args.sample_tokens is not None else None
        ),
        scene_tokens=(
            tuple(args.scene_tokens) if args.scene_tokens is not None else None
        ),
        build_all_samples=args.all_samples,
        filter_missing_sensor_files=True,
        metadata_path_prefix=args.metadata_path_prefix,
        info_filename=args.info_filename,
        manifest_filename=args.manifest_filename,
    )


def _selection_description(config: BuildConfig) -> str:
    if config.build_all_samples:
        return "all samples"
    if config.scene_tokens is not None:
        return "{} scene(s)".format(len(config.scene_tokens))
    return "{} sample(s)".format(len(config.sample_tokens or ()))


def print_build_summary(config: BuildConfig) -> None:
    """Print the effective experiment choices before conversion starts."""
    print("TruckScenes -> OmniHD dataset build")
    print("------------------------------------")
    print("Source: {}".format(Path(config.data_root)))
    print("Output: {}".format(Path(config.output_root)))
    print("Metadata version: {}".format(config.metadata_version))
    print("Selection: {}".format(_selection_description(config)))
    print("Require complete local LiDAR/RADAR inputs: yes")
    print("LiDAR channels: {}".format(", ".join(config.lidar_channels)))
    print("RADAR channels: {}".format(", ".join(config.radar_channels)))
    print("Reference channel: {}".format(config.reference_channel))
    print("RADAR sweeps: {}".format(config.radar_sweeps_num))
    print("RADAR power policy: {}".format(config.radar_power_mode))
    print("RADAR SNR fill: {}".format(config.radar_snr_fill_value))
    print("Ego velocity source: {}".format(config.ego_velocity_source))
    print(
        "Metadata path prefix: {}".format(
            config.metadata_path_prefix
            if config.metadata_path_prefix is not None
            else "None (resolved local paths)"
        )
    )
    print("RADAR aliases:")
    for channel in config.radar_channels:
        print("  {} -> {}".format(channel, config.radar_channel_aliases[channel]))
    print(flush=True)


def print_build_result(result: dict) -> None:
    """Print adapter output paths and conversion counts."""
    counts = result["counts"]
    availability = result["availability"]
    print("Build complete")
    print("--------------")
    print("Metadata scenes: {}".format(availability["metadata_scenes"]))
    print("Metadata samples: {}".format(availability["metadata_samples"]))
    print(
        "Samples skipped for missing required sensor files: {}".format(
            availability["skipped_missing_sensor_files"]
        )
    )
    print("Usable local samples selected: {}".format(availability["usable_samples"]))
    print("Samples successfully converted: {}".format(counts["samples"]))
    print("LiDAR files: {}".format(counts["lidar_files"]))
    print("LiDAR points: {}".format(counts["lidar_points"]))
    print("RADAR files: {}".format(counts["radar_files"]))
    print("GT boxes: {}".format(counts["gt_boxes"]))
    print("Input annotations: {}".format(counts["input_annotations"]))
    print("Included annotations: {}".format(counts["included_annotations"]))
    print("Excluded annotations: {}".format(counts["excluded_annotations"]))
    print()
    print("Output dataset: {}".format(result["info_path"].parent))
    print("Info PKL: {}".format(result["info_path"]))
    print("Manifest: {}".format(result["manifest_path"]))


def run(args: argparse.Namespace) -> dict:
    """Create one adapter config and invoke the existing dataset builder once."""
    data_root, output_root = resolve_roots(args)
    config = make_config(args, data_root, output_root)
    print_build_summary(config)
    result = build_dataset(config)
    print_build_result(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
