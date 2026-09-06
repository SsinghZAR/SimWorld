"""Live all-source visibility preflight (diagnostic placements, not a social score)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from benchmark.venue_meetup._core.action_space import VenueAgentTurn
from benchmark.venue_meetup.generator import generate_scenario
from benchmark.venue_meetup.interactions import INTERACTION_KINDS
from benchmark.venue_meetup.targeted_env import TargetedVenueEnv
from benchmark.venue_meetup.varied_profiles import varied_profile
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/venue_meetup/targeted_preflight"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--venue-id", help="Limit the diagnostic sweep to one venue")
    parser.add_argument("--grid-size", type=int, choices=(3, 5, 7), default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scenario = varied_profile(generate_scenario(seed=args.seed,
                              template_id=f"rosebank_grid_{args.grid_size}x{args.grid_size}_v0", hidden_profile=True))
    env = TargetedVenueEnv(Communicator(UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 360))),
                           scenario, info_partition="spatial", navigate_mode="walk")
    records = []
    try:
        env.reset()
        for venue in scenario.venues:
            if args.venue_id and venue.venue_id != args.venue_id:
                continue
            agent = next(agent.agent_id for agent in scenario.agents if agent.zone_id == venue.zone_id)
            env._teleport_navigate(agent, venue)
            observation = env._build_observations()[agent]
            cv2.imwrite(str(args.output / f"{venue.venue_id}.png"), observation["ego_view"])
            for kind in INTERACTION_KINDS:
                result = env.interactions.inspect(agent, VenueAgentTurn(choice=3,
                                     target_interactable_id=f"{venue.venue_id}__{kind.key}"))
                records.append({"agent": agent, **result})
                print(f"{venue.venue_id} {kind.key}: {result['result']} pixels={result.get('mask_pixels_internal', 0)}", flush=True)
        report = {"diagnostic_only": True, "placement": "teleport preflight", "checks": records,
                  "passed": all(record["result"] == "INSPECT_OK" for record in records)}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0 if report["passed"] else 1
    finally:
        env.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
