"""Focused tests for DistrictLayout walk-graph helpers."""

from __future__ import annotations

import pytest

from benchmark.venue_meetup.layout import (
    Block,
    DistrictLayout,
    Frontage,
    Intersection,
    MeetingRegion,
    StreetSegment,
    WalkEdge,
    WalkNode,
)


def _node(node_id: str, x: float = 0.0, y: float = 0.0, kind: str = "intersection") -> WalkNode:
    return WalkNode(node_id=node_id, position=(x, y), kind=kind)  # type: ignore[arg-type]


def _edge(
    start: str,
    end: str,
    length_cm: float,
    *,
    enabled: bool = True,
    route_kind: str = "sidewalk",
) -> WalkEdge:
    return WalkEdge(
        start_node_id=start,
        end_node_id=end,
        length_cm=length_cm,
        enabled=enabled,
        route_kind=route_kind,  # type: ignore[arg-type]
    )


def _diamond_layout() -> DistrictLayout:
    """A --1--> B --1--> D
       |         ^
       1         1
       v         |
       C --------+
    """

    return DistrictLayout(
        layout_id="diamond",
        walk_nodes=(
            _node("A", 0.0, 0.0, "spawn"),
            _node("B", 1.0, 0.0),
            _node("C", 0.0, 1.0),
            _node("D", 1.0, 1.0, "frontage"),
        ),
        walk_edges=(
            _edge("A", "B", 1.0),
            _edge("A", "C", 1.0),
            _edge("B", "D", 1.0),
            _edge("C", "D", 1.0),
        ),
    )


def test_enabled_route_finds_shortest_path() -> None:
    layout = _diamond_layout()
    assert layout.reachable_nodes("A") == frozenset({"A", "B", "C", "D"})


def test_disabled_edge_is_ignored() -> None:
    layout = DistrictLayout(
        layout_id="alley_closed",
        walk_nodes=(_node("spawn"), _node("mid"), _node("venue"), _node("island")),
        walk_edges=(
            _edge("spawn", "mid", 10.0),
            _edge("mid", "venue", 10.0, enabled=False),
            _edge("spawn", "venue", 50.0),
            _edge("mid", "island", 5.0, enabled=False),
        ),
    )
    # Disabled mid->venue forces the longer direct sidewalk.
    assert layout.shortest_path("spawn", "venue") == ["spawn", "venue"]
    # Disabled mid->island keeps island unreachable from the connected component.
    assert layout.reachable_nodes("mid") == frozenset({"mid", "spawn", "venue"})
    assert layout.shortest_path("mid", "island") is None


def test_deterministic_tie_prefers_lexicographically_smaller_frontier() -> None:
    # Equal-length A->B->D and A->C->D; Dijkstra settles B before C, so path is A-B-D.
    layout = _diamond_layout()
    assert layout.shortest_path("A", "D") == ["A", "B", "D"]
    # Reverse node / edge insertion order must not change the tie outcome.
    layout_reordered = DistrictLayout(
        layout_id="diamond_reordered",
        walk_nodes=(
            _node("D", 1.0, 1.0, "frontage"),
            _node("C", 0.0, 1.0),
            _node("B", 1.0, 0.0),
            _node("A", 0.0, 0.0, "spawn"),
        ),
        walk_edges=(
            _edge("C", "D", 1.0),
            _edge("B", "D", 1.0),
            _edge("A", "C", 1.0),
            _edge("A", "B", 1.0),
        ),
    )
    assert layout_reordered.shortest_path("A", "D") == ["A", "B", "D"]


def test_unreachable_path_returns_none() -> None:
    layout = DistrictLayout(
        layout_id="split",
        walk_nodes=(_node("west"), _node("east"), _node("island")),
        walk_edges=(_edge("west", "east", 5.0),),
    )
    assert layout.shortest_path("west", "island") is None
    assert layout.reachable_nodes("west") == frozenset({"west", "east"})


def test_unknown_requested_nodes_raise_value_error() -> None:
    layout = _diamond_layout()
    with pytest.raises(ValueError, match="Unknown walk node_id"):
        layout.shortest_path("A", "missing")
    with pytest.raises(ValueError, match="Unknown walk node_id"):
        layout.shortest_path("missing", "A")
    with pytest.raises(ValueError, match="Unknown walk node_id"):
        layout.reachable_nodes("missing")
    with pytest.raises(ValueError, match="Unknown walk node_id"):
        layout.path_length("A", "missing")
    with pytest.raises(ValueError, match="Unknown walk node_id"):
        layout.path_length_cm("missing", "A")
    with pytest.raises(ValueError, match="Unknown walk node_id"):
        layout.node_by_id("missing")


def test_unknown_enabled_edge_endpoints_raise_value_error() -> None:
    bad_end = DistrictLayout(
        layout_id="bad_end",
        walk_nodes=(_node("A"), _node("B")),
        walk_edges=(_edge("A", "ghost", 1.0),),
    )
    with pytest.raises(ValueError, match="Unknown walk edge endpoint"):
        bad_end.shortest_path("A", "B")
    with pytest.raises(ValueError, match="Unknown walk edge endpoint"):
        bad_end.reachable_nodes("A")
    with pytest.raises(ValueError, match="Unknown walk edge endpoint"):
        bad_end.path_length("A", "B")

    bad_start = DistrictLayout(
        layout_id="bad_start",
        walk_nodes=(_node("A"), _node("B")),
        walk_edges=(_edge("ghost", "B", 1.0),),
    )
    with pytest.raises(ValueError, match="Unknown walk edge endpoint"):
        bad_start.path_length_cm("A", "B")


def test_non_positive_and_non_finite_enabled_weights_raise_value_error() -> None:
    nodes = (_node("A"), _node("B"))
    for bad_length in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        layout = DistrictLayout(
            layout_id=f"bad_weight_{bad_length!r}",
            walk_nodes=nodes,
            walk_edges=(_edge("A", "B", bad_length),),
        )
        with pytest.raises(ValueError, match="Invalid walk edge length_cm"):
            layout.shortest_path("A", "B")
        with pytest.raises(ValueError, match="Invalid walk edge length_cm"):
            layout.reachable_nodes("A")
        with pytest.raises(ValueError, match="Invalid walk edge length_cm"):
            layout.path_length("A", "B")


def test_malformed_disabled_edges_are_ignored() -> None:
    layout = DistrictLayout(
        layout_id="ignore_disabled",
        walk_nodes=(_node("A"), _node("B")),
        walk_edges=(
            _edge("A", "ghost", 1.0, enabled=False),
            _edge("missing", "B", -5.0, enabled=False),
            _edge("A", "B", 0.0, enabled=False),
            _edge("A", "B", float("nan"), enabled=False),
            _edge("A", "B", float("inf"), enabled=False),
            _edge("A", "B", 3.0),
        ),
    )
    assert layout.shortest_path("A", "B") == ["A", "B"]
    assert layout.path_length("A", "B") == 3.0
    assert layout.reachable_nodes("A") == frozenset({"A", "B"})


def test_duplicate_identifiers_raise_value_error() -> None:
    dup_nodes = DistrictLayout(
        layout_id="dup_nodes",
        walk_nodes=(_node("A"), _node("A", 1.0, 1.0)),
        walk_edges=(),
    )
    with pytest.raises(ValueError, match="Duplicate walk node_id"):
        dup_nodes.node_by_id("A")
    with pytest.raises(ValueError, match="Duplicate walk node_id"):
        dup_nodes.shortest_path("A", "A")

    dup_streets = DistrictLayout(
        layout_id="dup_streets",
        streets=(
            StreetSegment("main", (0.0, 0.0), (10.0, 0.0), 800.0, 200.0),
            StreetSegment("main", (0.0, 0.0), (0.0, 10.0), 800.0, 200.0),
        ),
    )
    with pytest.raises(ValueError, match="Duplicate street_id"):
        dup_streets.street_by_id("main")

    dup_blocks = DistrictLayout(
        layout_id="dup_blocks",
        blocks=(
            Block("b1", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))),
            Block("b1", ((2.0, 2.0), (3.0, 2.0), (3.0, 3.0))),
        ),
    )
    with pytest.raises(ValueError, match="Duplicate block_id"):
        dup_blocks.block_by_id("b1")


def test_json_round_trip_via_compact_and_from_dict() -> None:
    layout = DistrictLayout(
        layout_id="station_quarter_medium_v1",
        schema_version=1,
        streets=(
            StreetSegment(
                street_id="market",
                start=(0.0, 0.0),
                end=(10000.0, 0.0),
                width_cm=1200.0,
                sidewalk_width_cm=250.0,
            ),
        ),
        intersections=(
            Intersection(intersection_id="x_market_cross", position=(5000.0, 0.0), landmark_id="clock"),
        ),
        blocks=(
            Block(
                block_id="nw",
                footprint=((0.0, 0.0), (4000.0, 0.0), (4000.0, 4000.0), (0.0, 4000.0)),
                frontage_ids=("nw_cafe",),
            ),
        ),
        frontages=(
            Frontage(
                frontage_id="nw_cafe",
                block_id="nw",
                position=(2000.0, 100.0, 0.0),
                yaw_deg=90.0,
                entrance_point=(2000.0, 300.0, 0.0),
                meeting_region=MeetingRegion(center=(2000.0, 500.0), radius=150.0),
                venue_slot_id="slot_nw_cafe",
            ),
        ),
        walk_nodes=(
            _node("spawn_a", 0.0, 0.0, "spawn"),
            _node("cross_1", 5000.0, 0.0, "crossing"),
            _node("front_nw", 2000.0, 500.0, "frontage"),
        ),
        walk_edges=(
            _edge("spawn_a", "cross_1", 5000.0, route_kind="sidewalk"),
            _edge("cross_1", "front_nw", 3200.0, route_kind="crossing"),
            _edge("spawn_a", "front_nw", 99999.0, enabled=False, route_kind="alley"),
        ),
    )

    payload = layout.compact()
    restored = DistrictLayout.from_dict(payload)
    assert restored == layout
    assert restored.compact() == payload
    assert restored.shortest_path("spawn_a", "front_nw") == ["spawn_a", "cross_1", "front_nw"]
    assert restored.street_by_id("market").width_cm == 1200.0
    assert restored.frontage_by_id("nw_cafe").meeting_region.radius == 150.0
    assert restored.intersection_by_id("x_market_cross").landmark_id == "clock"
