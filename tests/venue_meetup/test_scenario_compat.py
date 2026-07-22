"""Backward-compatible Scenario JSON load/serialize tests."""

from __future__ import annotations

from dataclasses import replace

from benchmark.venue_meetup.layout import DistrictLayout, WalkEdge, WalkNode
from benchmark.venue_meetup.scenario import scenario_from_dict
from benchmark.venue_meetup.templates.central_square import build_fixed_scenario


def _tiny_layout() -> DistrictLayout:
    return DistrictLayout(
        layout_id="compat_layout",
        walk_nodes=(
            WalkNode(node_id="a", position=(0.0, 0.0), kind="spawn"),
            WalkNode(node_id="b", position=(10.0, 0.0), kind="frontage"),
        ),
        walk_edges=(WalkEdge(start_node_id="a", end_node_id="b", length_cm=10.0),),
    )


def test_old_scenario_json_without_layout_loads_with_none() -> None:
    scenario = build_fixed_scenario(seed=7)
    payload = scenario.compact(include_hidden=True)
    assert "layout" not in payload
    restored = scenario_from_dict(payload)
    assert restored.layout is None
    assert restored.scenario_id == scenario.scenario_id
    assert restored.map_template_id == "central_square_v0"
    assert [venue.venue_id for venue in restored.venues] == [venue.venue_id for venue in scenario.venues]


def test_scenario_with_layout_round_trips() -> None:
    scenario = replace(build_fixed_scenario(seed=7), layout=_tiny_layout())
    payload = scenario.compact(include_hidden=True)
    assert payload["layout"]["layout_id"] == "compat_layout"
    restored = scenario_from_dict(payload)
    assert restored.layout is not None
    assert restored.layout == scenario.layout
    assert restored.layout.shortest_path("a", "b") == ["a", "b"]


def test_public_compact_still_hides_venue_secrets_with_layout() -> None:
    scenario = replace(build_fixed_scenario(seed=7), layout=_tiny_layout())
    public = scenario.compact(include_hidden=False)
    assert "layout" in public
    assert "requirements" not in public
    assert "soft_weights" not in public
    for venue in public["venues"]:
        assert "properties" not in venue
        assert "entrances" not in venue
        assert "mask_color_rgb" not in venue
