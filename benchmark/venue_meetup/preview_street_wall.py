#!/usr/bin/env python3
"""Spawn and capture the isolated continuous street-wall primitive."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmark.venue_meetup.building_catalog import asset_path
from benchmark.venue_meetup.busy_street import (
    BusyStreetBuilding,
    plan_busy_street,
    plan_busy_street_props,
)
from benchmark.venue_meetup.preview_runtime import (
    backend_reachable,
    capture_hidden_camera,
    set_preview_lighting,
    spawn_hidden_camera,
)
from benchmark.venue_meetup.street_wall import (
    DEFAULT_FACADE_ASSETS,
    DEFAULT_PREFERRED_SCALES,
    plan_street_wall,
    street_wall_metrics,
)
from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV


def run_preview(args: argparse.Namespace) -> dict[str, object]:
    if not backend_reachable(args.ip, args.port):
        raise RuntimeError(f"UnrealCV backend is not reachable at {args.ip}:{args.port}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    half_length = args.length_cm / 2.0
    start = (-half_length, args.wall_y_cm)
    end = (half_length, args.wall_y_cm)
    outward = (0.0, 1.0 if args.facing == "north" else -1.0)
    view_direction = (-outward[0], -outward[1])
    view_yaw = math.degrees(math.atan2(view_direction[1], view_direction[0]))
    busy_buildings: tuple[BusyStreetBuilding, ...] = ()
    props = ()
    if args.profile == "busy":
        busy_buildings = plan_busy_street(
            start,
            end,
            outward=outward,
            facade_fill_ratio=args.facade_fill_ratio,
            setback_cm=args.setback_cm,
        )
        placements = tuple(item.placement for item in busy_buildings)
        props = plan_busy_street_props(
            busy_buildings,
            start,
            end,
            outward=outward,
            setback_cm=args.setback_cm,
        )
        roles_by_index = {
            item.placement.index: item.use for item in busy_buildings
        }
        buildings_by_index = {
            item.placement.index: item for item in busy_buildings
        }
        effective_gap_cm = 0.0
    else:
        asset_keys = tuple(args.asset_key) if args.asset_key else DEFAULT_FACADE_ASSETS
        preferred_scales = {
            asset: DEFAULT_PREFERRED_SCALES[asset] for asset in asset_keys
        }
        placements = plan_street_wall(
            start,
            end,
            outward=outward,
            asset_keys=asset_keys,
            preferred_scales=preferred_scales,
            gap_cm=args.gap_cm,
            setback_cm=args.setback_cm,
            height_factor=args.height_factor,
            facade_fill_ratio=args.facade_fill_ratio,
        )
        roles_by_index = {}
        buildings_by_index = {}
        effective_gap_cm = args.gap_cm
    metrics = street_wall_metrics(placements, args.length_cm)

    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=args.resolution)
    communicator = Communicator(unrealcv)
    try:
        communicator.unrealcv.set_mode("sync", 0.05)
        communicator.clear_env(keep_roads=True)
        Humanoid._id_counter = 0
        Humanoid._camera_id_counter = 1
        set_preview_lighting(communicator)

        for placement in placements:
            actor_name = f"GEN_BP_{args.profile.upper()}_WALL_{placement.index:02d}"
            communicator.spawn_object(
                actor_name,
                asset_path(placement.asset_key),
                (placement.position[0], placement.position[1], 0.0),
                (0.0, placement.yaw_deg, 0.0),
                scale=placement.scale,
            )
            role = roles_by_index.get(placement.index)
            building = buildings_by_index.get(placement.index)
            mask_color = (
                building.mask_color_rgb
                if building is not None and building.mask_color_rgb is not None
                else (255, 0, 255)
                if role == "house"
                else (255, 255, 255)
            )
            communicator.unrealcv.set_color(actor_name, mask_color)
            communicator.unrealcv.set_collision(actor_name, True)
            communicator.unrealcv.set_movable(actor_name, False)

        for prop in props:
            actor_name = f"GEN_BP_BUSY_STREET_PROP_{prop.index:02d}"
            communicator.spawn_object(
                actor_name,
                asset_path(prop.asset_key),
                prop.position,
                (0.0, prop.yaw_deg, 0.0),
                scale=prop.scale,
            )
            communicator.unrealcv.set_color(
                actor_name,
                {
                    "restaurant_seating": (255, 230, 40),
                    "book_display": (50, 220, 255),
                    "bar_seating": (255, 80, 80),
                    "street_furniture": (80, 255, 120),
                }[prop.use],
            )
            communicator.unrealcv.set_collision(actor_name, False)
            communicator.unrealcv.set_movable(actor_name, False)

        overview_position = (
            0.0,
            args.wall_y_cm + outward[1] * args.overview_distance_cm,
            args.overview_height_cm,
        )
        camera = spawn_hidden_camera(
            communicator,
            position=overview_position,
            direction=view_direction,
            resolution=args.resolution,
        )

        artifact_prefix = "busy_street" if args.profile == "busy" else "street_wall"
        overview_path = args.output_dir / f"{artifact_prefix}_overview.png"
        close_path = args.output_dir / f"{artifact_prefix}_close.png"
        overview_mask_path = args.output_dir / f"{artifact_prefix}_overview_mask.png"
        close_mask_path = args.output_dir / f"{artifact_prefix}_close_mask.png"
        close_x = (
            min(
                (
                    item.placement.position[0]
                    for item in busy_buildings
                    if item.use == "restaurant"
                ),
                key=abs,
            )
            if busy_buildings
            else -3500.0
        )
        capture_hidden_camera(
            communicator,
            camera,
            position=overview_position,
            yaw_deg=view_yaw,
            actor_pitch_deg=args.camera_pitch_deg,
            fov_deg=args.overview_fov_deg,
            frame_gamma=args.frame_gamma,
            output_path=overview_path,
            mask_path=overview_mask_path,
        )
        capture_hidden_camera(
            communicator,
            camera,
            position=(
                close_x,
                args.wall_y_cm + outward[1] * args.close_distance_cm,
                args.close_height_cm,
            ),
            yaw_deg=view_yaw,
            actor_pitch_deg=args.camera_pitch_deg,
            fov_deg=args.close_fov_deg,
            frame_gamma=args.frame_gamma,
            output_path=close_path,
            mask_path=close_mask_path,
        )

        gallery_artifacts: dict[str, dict[str, str]] = {}
        if busy_buildings:
            for use in ("bookshop", "bar", "skyscraper_lobby"):
                candidates = [item for item in busy_buildings if item.use == use]
                # The sign-covered Lantern Books module has the clearer
                # street-level frontage; keep Red Page Books in the scenario
                # and mask tests, but feature the more legible variant here.
                target = candidates[-1] if use == "bookshop" else candidates[0]
                gallery_name = use.removesuffix("_lobby")
                gallery_path = args.output_dir / f"busy_street_{gallery_name}_close.png"
                gallery_mask_path = (
                    args.output_dir / f"busy_street_{gallery_name}_close_mask.png"
                )
                is_tower = use == "skyscraper_lobby"
                distance = max(args.close_distance_cm, 6000.0) if is_tower else args.close_distance_cm
                height = max(args.overview_height_cm, 1200.0) if is_tower else args.close_height_cm
                capture_hidden_camera(
                    communicator,
                    camera,
                    position=(
                        target.placement.position[0],
                        args.wall_y_cm + outward[1] * distance,
                        height,
                    ),
                    yaw_deg=view_yaw,
                    actor_pitch_deg=args.camera_pitch_deg,
                    fov_deg=90.0 if is_tower else args.close_fov_deg,
                    frame_gamma=args.frame_gamma,
                    output_path=gallery_path,
                    mask_path=gallery_mask_path,
                )
                gallery_artifacts[use] = {
                    "venue_id": target.venue_id or "",
                    "image": str(gallery_path),
                    "mask": str(gallery_mask_path),
                }

        report: dict[str, object] = {
            "status": "ok",
            "wall": {
                "profile": args.profile,
                "start": start,
                "end": end,
                "facing": args.facing,
                "gap_cm": effective_gap_cm,
                "setback_cm": args.setback_cm,
                "facade_fill_ratio": args.facade_fill_ratio,
                "metrics": {
                    "length_cm": metrics.length_cm,
                    "actor_count": metrics.actor_count,
                    "covered_cm": metrics.covered_cm,
                    "coverage": metrics.coverage,
                    "maximum_gap_cm": metrics.maximum_gap_cm,
                    "leading_gap_cm": metrics.leading_gap_cm,
                    "trailing_gap_cm": metrics.trailing_gap_cm,
                },
                "placements": [
                    {
                        "asset_key": item.asset_key,
                        "use": roles_by_index.get(item.index),
                        "venue_id": (
                            buildings_by_index[item.index].venue_id
                            if item.index in buildings_by_index
                            else None
                        ),
                        "display_name": (
                            buildings_by_index[item.index].display_name
                            if item.index in buildings_by_index
                            else None
                        ),
                        "position": item.position,
                        "yaw_deg": item.yaw_deg,
                        "scale": item.scale,
                        "tangent_span_cm": (
                            item.tangent_start_cm,
                            item.tangent_end_cm,
                        ),
                        "measured_tangent_width_cm": item.measured_tangent_width_cm,
                    }
                    for item in placements
                ],
                "props": [
                    {
                        "asset_key": item.asset_key,
                        "building_index": item.building_index,
                        "position": item.position,
                        "yaw_deg": item.yaw_deg,
                        "scale": item.scale,
                        "use": item.use,
                    }
                    for item in props
                ],
            },
            "artifacts": {
                "overview": str(overview_path),
                "close": str(close_path),
                "overview_mask": str(overview_mask_path),
                "close_mask": str(close_mask_path),
                "venue_gallery": gallery_artifacts,
            },
        }
        (args.output_dir / "street_wall_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        unrealcv.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spawn and capture a continuous wall or mixed-use busy street",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--resolution", default="640x360")
    parser.add_argument(
        "--profile",
        choices=("continuous", "busy"),
        default="continuous",
        help="Render the base primitive or the restaurant-and-house composition",
    )
    parser.add_argument("--length-cm", type=float, default=24000.0)
    parser.add_argument("--wall-y-cm", type=float, default=3000.0)
    parser.add_argument("--facing", choices=("north", "south"), default="north")
    parser.add_argument("--gap-cm", type=float, default=0.0)
    parser.add_argument("--setback-cm", type=float, default=200.0)
    parser.add_argument("--height-factor", type=float, default=1.5)
    parser.add_argument("--frame-gamma", type=float, default=0.4)
    parser.add_argument(
        "--facade-fill-ratio",
        type=float,
        default=0.85,
        help="Visible facade width divided by the stored asset bound",
    )
    parser.add_argument(
        "--asset-key",
        action="append",
        choices=DEFAULT_FACADE_ASSETS,
        help="Facade asset to repeat; may be supplied more than once",
    )
    parser.add_argument("--overview-distance-cm", type=float, default=10500.0)
    parser.add_argument("--close-distance-cm", type=float, default=4500.0)
    parser.add_argument("--overview-height-cm", type=float, default=600.0)
    parser.add_argument("--close-height-cm", type=float, default=350.0)
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=20.0,
        help="Actor pitch correction that levels the attached third-person camera",
    )
    parser.add_argument("--overview-fov-deg", type=float, default=95.0)
    parser.add_argument("--close-fov-deg", type=float, default=80.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/venue_meetup/street_wall_preview"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.resolution = tuple(int(part) for part in args.resolution.lower().split("x"))
    report = run_preview(args)
    print(json.dumps(report["wall"]["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
