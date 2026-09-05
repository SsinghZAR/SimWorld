"""Offline tests for post-episode movement minimaps."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageChops

from benchmark.venue_meetup.coarse_map import with_rendered_coarse_map
from benchmark.venue_meetup.generator import generate_scenario
from benchmark.venue_meetup.trajectory_minimap import (
    movement_history, render_trajectory_minimap)


def _scenario_payload(tmp_path: Path) -> tuple[dict, Path]:
    scenario = generate_scenario(
        seed=7,
        template_id="rosebank_grid_3x3_v0",
        num_agents=2,
        randomize=False,
        hidden_profile=True,
    )
    scenario = with_rendered_coarse_map(scenario, tmp_path)
    payload = scenario.compact(include_hidden=True)
    (tmp_path / "scenario_hidden.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload, Path(scenario.coarse_map_path)


def _sample_trajectory(scenario: dict) -> list[dict]:
    starts = {
        agent["agent_id"]: agent["position"][:2]
        for agent in scenario["agents"]
    }
    venue_points = [venue["region"]["center"] for venue in scenario["venues"][:2]]
    midpoint = [
        0.5 * (starts["agent_0"][0] + venue_points[0][0]),
        0.5 * (starts["agent_0"][1] + venue_points[0][1]),
    ]
    return [
        {
            "step": 0,
            "turns": {
                "agent_0": {"choice": 5},
                "agent_1": {"choice": 0},
            },
            "info": {
                "movement_paths_internal": {
                    "agent_0": [starts["agent_0"], midpoint, venue_points[0]],
                    "agent_1": [starts["agent_1"]],
                },
                "positions_internal": {
                    "agent_0": venue_points[0],
                    "agent_1": starts["agent_1"],
                },
                "actions": {
                    "agent_0": {"result": "NAVIGATE_OK", "mode": "walk"},
                    "agent_1": {"result": "WAIT"},
                },
            },
        },
        {
            "step": 1,
            "turns": {
                "agent_0": {"choice": 0},
                "agent_1": {"choice": 5},
            },
            "info": {
                "movement_paths_internal": {
                    "agent_0": [venue_points[0]],
                    "agent_1": [starts["agent_1"], venue_points[1]],
                },
                "positions_internal": {
                    "agent_0": venue_points[0],
                    "agent_1": venue_points[1],
                },
                "actions": {
                    "agent_0": {"result": "WAIT"},
                    "agent_1": {"result": "NAVIGATE_OK", "mode": "teleport"},
                },
            },
        },
    ]


def test_movement_history_preserves_intra_turn_points_and_modes(tmp_path: Path) -> None:
    scenario, _ = _scenario_payload(tmp_path)
    trajectory = _sample_trajectory(scenario)

    histories = movement_history(scenario, trajectory)

    assert sorted(histories) == ["agent_0", "agent_1"]
    assert len(histories["agent_0"][0]["points"]) == 3
    assert histories["agent_0"][0]["kind"] == "physical"
    assert histories["agent_1"][1]["kind"] == "teleport"
    assert histories["agent_1"][1]["moved"] is True


def test_movement_history_falls_back_to_archived_endpoint_positions(tmp_path: Path) -> None:
    scenario, _ = _scenario_payload(tmp_path)
    start = scenario["agents"][0]["position"][:2]
    end = scenario["venues"][0]["region"]["center"]
    trajectory = [
        {
            "step": 0,
            "turns": {"agent_0": {"choice": 5}, "agent_1": {"choice": 0}},
            "info": {
                "positions_internal": {"agent_0": end, "agent_1": scenario["agents"][1]["position"][:2]},
                "actions": {"agent_0": {"result": "NAVIGATE_OK"}},
            },
        }
    ]

    history = movement_history(scenario, trajectory)["agent_0"][0]

    assert history["points"] == [tuple(start), tuple(end)]
    assert history["kind"] == "teleport"


def test_render_trajectory_minimap_uses_public_map_background(tmp_path: Path) -> None:
    scenario, coarse_map_path = _scenario_payload(tmp_path)
    trajectory = _sample_trajectory(scenario)
    (tmp_path / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2),
        encoding="utf-8",
    )

    outputs = render_trajectory_minimap(tmp_path, write_video=False)

    assert outputs == {"trajectory_minimap": tmp_path / "trajectory_minimap.png"}
    rendered_path = outputs["trajectory_minimap"]
    assert rendered_path.is_file()
    with Image.open(coarse_map_path) as base, Image.open(rendered_path) as rendered:
        base.load()
        rendered.load()
        assert rendered.size == base.size
        assert ImageChops.difference(base.convert("RGB"), rendered.convert("RGB")).getbbox() is not None
