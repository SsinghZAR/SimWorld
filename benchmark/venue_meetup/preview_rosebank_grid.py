#!/usr/bin/env python3
"""Spawn and capture a scalable Rosebank-inspired mixed-use district."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from benchmark.venue_meetup.coarse_map import render_coarse_map
from benchmark.venue_meetup.preview_runtime import (backend_reachable,
                                                    capture_hidden_camera,
                                                    spawn_hidden_camera)
from benchmark.venue_meetup.rosebank_grid import SUPPORTED_GRID_SIZES
from benchmark.venue_meetup.rosebank_roads import plan_rosebank_road_actors
from benchmark.venue_meetup.scene_builder import SceneBuilder
from benchmark.venue_meetup.templates.rosebank_grid_playtest import (
    build_scaled_scenario, plan_playtest_grid)
from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV


def _yaw_toward(
    start: tuple[float, float, float],
    target: tuple[float, float],
) -> float:
    return math.degrees(
        math.atan2(target[1] - start[1], target[0] - start[0])
    )


def run_preview(args: argparse.Namespace) -> dict[str, object]:
    """Build the full grid and capture district-scale navigation evidence."""

    if not backend_reachable(args.ip, args.port):
        raise RuntimeError(f"UnrealCV backend is not reachable at {args.ip}:{args.port}")
    if args.output_dir is None:
        raise ValueError("output_dir must be resolved before running a preview")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario = build_scaled_scenario(grid_size=args.grid_size, seed=args.seed)
    plan = plan_playtest_grid(args.grid_size)
    artifact_prefix = f"rosebank_grid_{plan.grid_size}x{plan.grid_size}"
    coarse_map_path = render_coarse_map(
        scenario,
        args.output_dir / f"{artifact_prefix}_coarse_map.png",
        size=1_400,
    )

    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=args.resolution)
    communicator = Communicator(unrealcv)
    try:
        communicator.unrealcv.set_mode("sync", 0.05)
        communicator.clear_env(keep_roads=True)
        Humanoid._id_counter = 0
        Humanoid._camera_id_counter = 1
        builder = SceneBuilder(
            communicator,
            scenario,
            resolution=args.resolution,
            spawn_settle_sec=0.0,
        )
        builder.setup_lighting()
        builder.spawn_static_scene()

        overview_distance = max(14_000.0, plan.extent_cm * 0.8)
        initial_position = (
            plan.center[0],
            plan.street_y[0] - overview_distance,
            max(12_000.0, plan.extent_cm * 0.82),
        )
        initial_yaw = _yaw_toward(initial_position, plan.center)
        camera = spawn_hidden_camera(
            communicator,
            position=initial_position,
            direction=(
                math.cos(math.radians(initial_yaw)),
                math.sin(math.radians(initial_yaw)),
            ),
            resolution=args.resolution,
        )
        alley_block = next(
            (block for block in plan.blocks if block.alley_axes),
            plan.block_at(plan.grid_size // 2, plan.grid_size // 2),
        )
        oxford_x = plan.street_x[plan.primary_street_index]
        tyrwhitt_y = plan.street_y[plan.primary_street_index]
        topdown_height_cm = args.topdown_height_cm or (
            plan.extent_cm
            * 2.0
            * 1.35
            / (2.0 * math.tan(math.radians(args.topdown_fov_deg / 2.0)))
        )
        center_index = plan.grid_size // 2
        views = {
            "district_overview": {
                "position": initial_position,
                "target": plan.center,
                "pitch": args.overview_pitch_deg,
                "fov": args.overview_fov_deg,
            },
            "district_top_down": {
                "position": (*plan.center, topdown_height_cm),
                "target": (plan.center[0] + 1.0, plan.center[1]),
                "pitch": -89.0,
                "fov": args.topdown_fov_deg,
                "resolution": args.topdown_resolution,
                "direct_camera_pitch": -90.0,
            },
            "oxford_transit_spine": {
                "position": (oxford_x, plan.street_y[0] - 3_500.0, 650.0),
                "target": (oxford_x, plan.street_y[-1]),
                "pitch": args.street_pitch_deg,
                "fov": args.street_fov_deg,
            },
            "tyrwhitt_high_street": {
                "position": (plan.street_x[0] - 3_500.0, tyrwhitt_y, 650.0),
                "target": (plan.street_x[-1], tyrwhitt_y),
                "pitch": args.street_pitch_deg,
                "fov": args.street_fov_deg,
            },
            "cross_alley": {
                "position": (
                    alley_block.center[0] - 2_500.0,
                    alley_block.center[1],
                    420.0,
                ),
                "target": (
                    alley_block.center[0] + 3_000.0,
                    alley_block.center[1],
                ),
                "pitch": args.street_pitch_deg,
                "fov": args.alley_fov_deg,
            },
            "mixed_use_core": {
                "position": (
                    plan.street_x[max(0, plan.primary_street_index - 1)],
                    plan.street_y[max(0, plan.primary_street_index - 2)],
                    1_200.0,
                ),
                "target": plan.block_at(center_index, center_index).center,
                "pitch": args.core_pitch_deg,
                "fov": args.core_fov_deg,
            },
        }
        artifacts: dict[str, dict[str, str]] = {}
        for view_name, view in views.items():
            position = view["position"]
            target = view["target"]
            assert isinstance(position, tuple) and isinstance(target, tuple)
            view_resolution = view.get("resolution", args.resolution)
            assert isinstance(view_resolution, tuple)
            communicator.unrealcv.set_camera_resolution(
                camera.camera_id,
                view_resolution,
            )
            image_path = args.output_dir / f"{artifact_prefix}_{view_name}.png"
            mask_path = args.output_dir / f"{artifact_prefix}_{view_name}_mask.png"
            capture_hidden_camera(
                communicator,
                camera,
                position=position,
                yaw_deg=_yaw_toward(position, target),
                actor_pitch_deg=float(view["pitch"]),
                direct_camera_pitch_deg=(
                    float(view["direct_camera_pitch"])
                    if "direct_camera_pitch" in view
                    else None
                ),
                fov_deg=float(view["fov"]),
                frame_gamma=args.frame_gamma,
                output_path=image_path,
                mask_path=mask_path,
            )
            artifacts[view_name] = {
                "image": str(image_path),
                "mask": str(mask_path),
            }

        report: dict[str, object] = {
            "status": "ok",
            "scenario_id": scenario.scenario_id,
            "geometry": {
                "grid_size": [plan.grid_size, plan.grid_size],
                "district_extent_cm": plan.extent_cm * 2.0,
                "block_count": len(plan.blocks),
                "zones": dict(Counter(block.zone for block in plan.blocks)),
                "horizontal_alley_blocks": sum(
                    "horizontal" in block.alley_axes for block in plan.blocks
                ),
                "vertical_alley_blocks": sum(
                    "vertical" in block.alley_axes for block in plan.blocks
                ),
                "venue_count": len(scenario.venues),
                "landmark_count": len(scenario.landmarks),
                "massing_actor_count": len(scenario.buildings),
                "solid_massing_count": sum(
                    building.collision for building in scenario.buildings
                ),
                "road_actor_count": len(
                    plan_rosebank_road_actors(scenario.layout)
                ),
            },
            "artifacts": {
                **artifacts,
                "coarse_map": str(coarse_map_path),
            },
        }
        (args.output_dir / f"{artifact_prefix}_preview_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        unrealcv.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a scalable Rosebank-inspired mixed-use district",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--resolution", default="960x540")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=SUPPORTED_GRID_SIZES,
        default=9,
        help="Number of blocks on each side of the square district",
    )
    parser.add_argument("--frame-gamma", type=float, default=0.35)
    parser.add_argument("--overview-pitch-deg", type=float, default=4.0)
    parser.add_argument("--street-pitch-deg", type=float, default=18.0)
    parser.add_argument("--core-pitch-deg", type=float, default=12.0)
    parser.add_argument("--overview-fov-deg", type=float, default=78.0)
    parser.add_argument(
        "--topdown-height-cm",
        type=float,
        default=None,
        help="Optional override; by default height is fitted to the grid extent",
    )
    parser.add_argument("--topdown-fov-deg", type=float, default=52.0)
    parser.add_argument("--topdown-resolution", default="1200x1200")
    parser.add_argument("--street-fov-deg", type=float, default=74.0)
    parser.add_argument("--alley-fov-deg", type=float, default=78.0)
    parser.add_argument("--core-fov-deg", type=float, default=78.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.resolution = tuple(
        int(part) for part in args.resolution.lower().split("x")
    )
    args.topdown_resolution = tuple(
        int(part) for part in args.topdown_resolution.lower().split("x")
    )
    if args.output_dir is None:
        args.output_dir = Path(
            "runs/city_landmark_redesign/"
            f"rosebank_grid_{args.grid_size}x{args.grid_size}"
        )
    report = run_preview(args)
    print(json.dumps(report["geometry"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
