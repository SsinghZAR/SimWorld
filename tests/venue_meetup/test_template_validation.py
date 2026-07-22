"""Focused tests for pure district layout validation."""

from __future__ import annotations

import pytest

from benchmark.venue_meetup.layout import Block, DistrictLayout, Frontage, MeetingRegion, WalkEdge, WalkNode
from benchmark.venue_meetup.template_validation import (
    LayoutValidationError,
    collect_layout_errors,
    validate_layout,
)


def _node(node_id: str, x: float = 0.0, y: float = 0.0, kind: str = "intersection") -> WalkNode:
    return WalkNode(node_id=node_id, position=(x, y), kind=kind)  # type: ignore[arg-type]


def _edge(start: str, end: str, length_cm: float, *, enabled: bool = True) -> WalkEdge:
    return WalkEdge(start_node_id=start, end_node_id=end, length_cm=length_cm, enabled=enabled)


def test_validation_reports_duplicate_ids_and_bad_references() -> None:
    layout = DistrictLayout(
        layout_id="bad",
        blocks=(
            Block(
                block_id="block_a",
                footprint=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
                frontage_ids=("front_a",),
            ),
            Block(
                block_id="block_a",
                footprint=((2.0, 2.0), (3.0, 2.0), (3.0, 3.0)),
                frontage_ids=("missing_front",),
            ),
        ),
        frontages=(
            Frontage(
                frontage_id="front_a",
                block_id="missing_block",
                position=(0.0, 0.0, 0.0),
                yaw_deg=0.0,
                entrance_point=(1.0, 0.0, 0.0),
                meeting_region=MeetingRegion(center=(1.0, 0.0), radius=100.0),
            ),
        ),
        walk_nodes=(_node("n0"), _node("n0", 1.0, 1.0)),
        walk_edges=(
            _edge("n0", "ghost", 10.0),
            _edge("n0", "n0", 0.0),
        ),
    )
    errors = collect_layout_errors(layout, required_paths=[("n0", "ghost")])
    joined = " | ".join(errors)
    assert "duplicate block_id: 'block_a'" in joined
    assert "duplicate walk node_id: 'n0'" in joined
    assert "missing block_id='missing_block'" in joined
    assert "missing frontage_id='missing_front'" in joined
    assert "end_node_id='ghost' is not a walk node" in joined
    assert "length_cm=0.0 must be finite and > 0" in joined
    with pytest.raises(LayoutValidationError):
        validate_layout(layout)


def test_validation_detects_unreachable_required_nodes() -> None:
    layout = DistrictLayout(
        layout_id="barrier",
        walk_nodes=(
            _node("spawn_n", 0.0, 100.0, "spawn"),
            _node("venue_s", 80.0, -80.0, "frontage"),
            _node("isolated", 500.0, 500.0, "spawn"),
        ),
        walk_edges=(_edge("spawn_n", "venue_s", 200.0),),
    )
    validate_layout(layout, required_paths=[("spawn_n", "venue_s")])
    with pytest.raises(LayoutValidationError) as excinfo:
        validate_layout(layout, required_paths=[("spawn_n", "isolated")])
    assert "required node 'isolated' is unreachable from 'spawn_n'" in str(excinfo.value)
