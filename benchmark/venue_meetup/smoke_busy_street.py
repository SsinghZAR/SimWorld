#!/usr/bin/env python3
"""Run targeted live interaction checks against the busy-street template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.venue_meetup._core.action_space import VenueAction, VenueAgentTurn
from benchmark.venue_meetup.templates import TEMPLATE_BUILDERS
from benchmark.venue_meetup.venue_env import VenueMeetupEnv
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV

DEFAULT_VENUE_TYPES = (
    "restaurant",
    "bookshop",
    "bar",
    "skyscraper_lobby",
)
PLAYTEST_TEMPLATES = (
    "busy_street_playtest_v0",
    "connected_blocks_playtest_v0",
)


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    """Navigate one agent to one venue of each requested type and inspect it."""

    scenario = TEMPLATE_BUILDERS[args.template_id](args.seed)
    targets = list(scenario.venues) if args.all_venues else []
    if not args.all_venues and args.venue_id:
        for venue_id in args.venue_id:
            try:
                targets.append(scenario.venue_by_id(venue_id))
            except KeyError as exc:
                raise ValueError(f"Unknown playtest venue id: {venue_id}") from exc
    elif not args.all_venues:
        for venue_type in args.venue_type:
            target = next(
                (venue for venue in scenario.venues if venue.venue_type == venue_type),
                None,
            )
            if target is None:
                raise ValueError(
                    f"No {venue_type!r} venue exists in the playtest template"
                )
            targets.append(target)

    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=args.resolution)
    env = VenueMeetupEnv(
        Communicator(unrealcv),
        scenario,
        resolution=args.resolution,
        info_partition="none",
        navigate_mode="teleport",
        frame_gamma=args.frame_gamma,
    )
    results: list[dict[str, object]] = []
    try:
        env.reset()
        for target in targets:
            env.step(
                {
                    "agent_0": VenueAgentTurn(
                        choice=VenueAction.NAVIGATE.value,
                        target_venue_id=target.venue_id,
                    ),
                    "agent_1": VenueAgentTurn(),
                }
            )
            navigation = env.last_actions_internal["agent_0"]
            env.step(
                {
                    "agent_0": VenueAgentTurn(
                        choice=VenueAction.INSPECT.value,
                        target_venue_id=target.venue_id,
                    ),
                    "agent_1": VenueAgentTurn(),
                }
            )
            inspection = env.last_actions_internal["agent_0"]
            results.append(
                {
                    "venue_id": target.venue_id,
                    "venue_type": target.venue_type,
                    "navigate_result": navigation.get("result"),
                    "inspect_result": inspection.get("result"),
                    "mask_pixels": inspection.get("mask_pixels_internal"),
                    "facts": inspection.get("facts"),
                }
            )
    finally:
        env.disconnect()

    report: dict[str, object] = {
        "status": (
            "ok"
            if all(
                item["navigate_result"] == "NAVIGATE_OK"
                and item["inspect_result"] == "INSPECT_OK"
                for item in results
            )
            else "failed"
        ),
        "scenario_id": scenario.scenario_id,
        "map_template_id": scenario.map_template_id,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live NAVIGATE + INSPECT smoke for city-block playtest venues",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--template-id",
        choices=PLAYTEST_TEMPLATES,
        default="busy_street_playtest_v0",
    )
    parser.add_argument("--resolution", default="640x360")
    parser.add_argument("--frame-gamma", type=float, default=0.35)
    parser.add_argument(
        "--venue-type",
        action="append",
        choices=DEFAULT_VENUE_TYPES,
        help="Venue type to check; repeat for multiple types",
    )
    parser.add_argument(
        "--venue-id",
        action="append",
        help="Exact venue id to check; repeat for venues on different blocks",
    )
    parser.add_argument(
        "--all-venues",
        action="store_true",
        help="Inspect every named venue instead of one representative per type",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/venue_meetup/busy_street_live_smoke.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.resolution = tuple(int(part) for part in args.resolution.lower().split("x"))
    args.venue_type = tuple(
        args.venue_type or (() if args.venue_id else DEFAULT_VENUE_TYPES)
    )
    report = run_smoke(args)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
