#!/usr/bin/env python3
"""Spawn the four-entry playtest block and capture validation views."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmark.venue_meetup.coarse_map import render_coarse_map
from benchmark.venue_meetup.preview_runtime import (
    backend_reachable,
    capture_hidden_camera,
    spawn_hidden_camera,
)
from benchmark.venue_meetup.scene_builder import SceneBuilder
from benchmark.venue_meetup.templates.busy_street_playtest import (
    build_fixed_scenario,
    plan_playtest_block,
)
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
    """Build the actual benchmark scene and capture its key spatial views."""

    if not backend_reachable(args.ip, args.port):
        raise RuntimeError(f"UnrealCV backend is not reachable at {args.ip}:{args.port}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario = build_fixed_scenario(seed=args.seed)
    plan = plan_playtest_block()
    coarse_map_path = render_coarse_map(
        scenario,
        args.output_dir / "city_block_coarse_map.png",
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

        initial_position = (-8_700.0, -8_700.0, 2_000.0)
        initial_yaw = _yaw_toward(initial_position, plan.center)
        direction = (
            math.cos(math.radians(initial_yaw)),
            math.sin(math.radians(initial_yaw)),
        )
        camera = spawn_hidden_camera(
            communicator,
            position=initial_position,
            direction=direction,
            resolution=args.resolution,
        )

        views = {
            "overview_southwest": {
                "position": initial_position,
                "target": plan.center,
                "pitch": args.overview_pitch_deg,
                "fov": args.overview_fov_deg,
            },
            "north_entry": {
                "position": (0.0, 6_500.0, 420.0),
                "target": plan.center,
                "pitch": args.street_pitch_deg,
                "fov": args.street_fov_deg,
            },
            "northeast_corner": {
                "position": (6_600.0, 6_600.0, 600.0),
                "target": plan.center,
                "pitch": args.street_pitch_deg,
                "fov": args.street_fov_deg,
            },
            "courtyard_cross_block": {
                "position": (0.0, 2_100.0, 420.0),
                "target": (0.0, -2_600.0),
                "pitch": args.street_pitch_deg,
                "fov": args.courtyard_fov_deg,
            },
        }
        artifacts: dict[str, dict[str, str]] = {}
        for view_name, view in views.items():
            position = view["position"]
            target = view["target"]
            assert isinstance(position, tuple) and isinstance(target, tuple)
            image_path = args.output_dir / f"city_block_{view_name}.png"
            mask_path = args.output_dir / f"city_block_{view_name}_mask.png"
            capture_hidden_camera(
                communicator,
                camera,
                position=position,
                yaw_deg=_yaw_toward(position, target),
                actor_pitch_deg=float(view["pitch"]),
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
                "side_length_cm": plan.side_length_cm,
                "building_count": len(plan.buildings),
                "venue_count": len(scenario.venues),
                "residence_count": len(scenario.buildings),
                "portals": [
                    {
                        "portal_id": portal.portal_id,
                        "side": portal.side,
                        "boundary_position": portal.boundary_position,
                        "planned_width_cm": portal.planned_width_cm,
                        "conservative_clear_width_cm": (
                            portal.conservative_clear_width_cm
                        ),
                    }
                    for portal in plan.portals
                ],
            },
            "artifacts": {
                **artifacts,
                "coarse_map": str(coarse_map_path),
            },
        }
        (args.output_dir / "city_block_preview_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        unrealcv.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the navigable four-entry Venue Meetup city block",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--resolution", default="960x540")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--frame-gamma", type=float, default=0.35)
    parser.add_argument("--overview-pitch-deg", type=float, default=15.0)
    parser.add_argument("--street-pitch-deg", type=float, default=20.0)
    parser.add_argument("--overview-fov-deg", type=float, default=75.0)
    parser.add_argument("--street-fov-deg", type=float, default=86.0)
    parser.add_argument("--courtyard-fov-deg", type=float, default=96.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/city_landmark_redesign/city_block_v0"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.resolution = tuple(
        int(part) for part in args.resolution.lower().split("x")
    )
    report = run_preview(args)
    print(json.dumps(report["geometry"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
