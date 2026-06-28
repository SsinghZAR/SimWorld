#!/usr/bin/env python3
"""Live UE capability spike for Venue Meetup."""

from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - spike reports this explicitly.
    cv2 = None

from benchmark.venue_meetup.building_catalog import MASK_COLORS, asset_path
from benchmark.venue_meetup.templates.central_square import build_fixed_scenario
from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from simworld.config import Config
from simworld.utils.vector import Vector

AGENT_BLUEPRINT = "/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C"


def backend_reachable(ip: str, port: int, timeout_seconds: float = 2.0) -> bool:
    """Return whether UnrealCV appears reachable."""

    try:
        with socket.create_connection((ip, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def write_json(path: Path, payload: dict) -> None:
    """Write JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def image_nonblank(path: Path) -> bool:
    """Return whether an image contains non-zero pixels."""

    if cv2 is None:
        return False
    image = cv2.imread(str(path))
    return bool(image is not None and image.size and image.max() > 0)


def run_spike(args: argparse.Namespace) -> dict:
    """Run the live capability spike."""

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ip": args.ip,
        "port": args.port,
        "status": "not_started",
        "checks": {},
        "artifacts": {},
    }
    if not backend_reachable(args.ip, args.port, args.connect_timeout):
        report["status"] = "backend_unreachable"
        write_json(output_dir / "spike_report.json", report)
        return report
    if cv2 is None:
        report["status"] = "opencv_missing"
        report["checks"]["opencv_available"] = False
        write_json(output_dir / "spike_report.json", report)
        return report

    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=args.resolution)
    communicator = Communicator(unrealcv)
    try:
        scenario = build_fixed_scenario(args.seed)
        venue = scenario.venues[0]
        communicator.unrealcv.set_mode("sync", args.tick_interval)
        communicator.clear_env(keep_roads=True)
        Humanoid._id_counter = 0
        Humanoid._camera_id_counter = 1

        venue_actor = "SPIKE_VENUE"
        communicator.spawn_object(venue_actor, asset_path(venue.asset_key), venue.position, (0.0, venue.yaw_deg, 0.0))
        communicator.unrealcv.set_color(venue_actor, MASK_COLORS["venue_0"])

        agent = Humanoid(position=Vector(-450, -450), direction=Vector(1, 1).normalize(), communicator=communicator, config=Config())
        communicator.spawn_agent(agent, name=None, position=(-450.0, -450.0, 600.0), model_path=AGENT_BLUEPRINT, type="humanoid")
        actor_name = communicator.get_humanoid_name(agent.id)
        communicator.unrealcv.set_orientation((0.0, 45.0, 0.0), actor_name)
        communicator.unrealcv.set_camera_resolution(agent.camera_id, args.resolution)
        communicator.unrealcv.tick()

        lit = communicator.get_camera_observation(agent.camera_id, "lit", mode=args.camera_mode)
        mask = communicator.get_camera_observation(agent.camera_id, "object_mask", mode=args.camera_mode)
        lit_path = output_dir / "agent_lit.png"
        mask_path = output_dir / "agent_object_mask.png"
        cv2.imwrite(str(lit_path), lit)
        cv2.imwrite(str(mask_path), mask)

        orientation_before = communicator.unrealcv.get_orientation(actor_name).tolist()
        communicator.unrealcv.set_orientation((0.0, 135.0, 0.0), actor_name)
        communicator.unrealcv.tick()
        orientation_after = communicator.unrealcv.get_orientation(actor_name).tolist()

        report["status"] = "ok"
        report["checks"] = {
            "lit_nonblank": image_nonblank(lit_path),
            "object_mask_nonblank": image_nonblank(mask_path),
            "set_color_requested_rgb": MASK_COLORS["venue_0"],
            "camera_yaw_before": orientation_before,
            "camera_yaw_after": orientation_after,
            "spawn_z_agent": 600.0,
            "spawn_z_building": venue.position[2],
        }
        report["artifacts"] = {"lit": str(lit_path), "object_mask": str(mask_path)}
        write_json(output_dir / "spike_report.json", report)
        return report
    finally:
        unrealcv.disconnect()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--connect-timeout", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=lambda value: tuple(int(part) for part in value.lower().split("x")), default=(640, 360))
    parser.add_argument("--camera-mode", choices=["direct", "fast", "file"], default="direct")
    parser.add_argument("--tick-interval", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/venue_meetup/spike"))
    return parser


def main() -> int:
    """CLI entrypoint."""

    report = run_spike(build_parser().parse_args())
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] in {"ok", "backend_unreachable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
