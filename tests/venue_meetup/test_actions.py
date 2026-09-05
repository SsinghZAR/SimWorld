"""Offline tests for single-action dispatch and inspection (no UE / network)."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark.venue_meetup._core.action_space import (VenueAction,
                                                       VenueAgentTurn)
from benchmark.venue_meetup.actions import (compute_inspection,
                                            count_mask_pixels,
                                            dispatch_single_action,
                                            resolve_inspect_target)
from benchmark.venue_meetup.scenario import Region, Venue, VenueProperties

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _venue(
    venue_id: str = "cafe_a",
    center: tuple[float, float] = (1000.0, 0.0),
    radius: float = 500.0,
    zone_id: str | None = None,
) -> Venue:
    return Venue(
        venue_id=venue_id,
        slot_id="slot_0",
        venue_type="cafe",
        asset_key="cafe_asset",
        asset_path="/Game/Cafe",
        position=(center[0], center[1], 0.0),
        yaw_deg=0.0,
        region=Region(center=center, radius=radius),
        mask_color_rgb=(255, 0, 0),
        properties=VenueProperties(
            open=True, reachable=True, capacity=20,
            accessible=True, shelter=True, food_drink=True,
            quiet_score=0.8, crowding_score=0.3,
        ),
        entrances=[],
        visual_summary="A cozy cafe with outdoor seating",
        zone_id=zone_id,
    )


def _pub(center: tuple[float, float] = (5000.0, 0.0), *, venue_id: str = "pub_b") -> Venue:
    return Venue(
        venue_id=venue_id,
        slot_id="slot_1",
        venue_type="pub",
        asset_key="pub_asset",
        asset_path="/Game/Pub",
        position=(center[0], center[1], 0.0),
        yaw_deg=0.0,
        region=Region(center=center, radius=500.0),
        mask_color_rgb=(0, 255, 0),
        properties=VenueProperties(
            open=True, reachable=True, capacity=30,
            accessible=True, shelter=False, food_drink=True,
            quiet_score=0.4, crowding_score=0.7,
        ),
        entrances=[],
        visual_summary="A lively pub on the corner",
        zone_id=None,
    )


# ---------------------------------------------------------------------------
# resolve_inspect_target
# ---------------------------------------------------------------------------

class TestResolveInspectTarget:
    def test_exact_id_match(self):
        venues = [_venue(), _pub()]
        result = resolve_inspect_target(venues, (0.0, 0.0), target_venue_id="pub_b")
        assert result is not None
        assert result.venue_id == "pub_b"

    def test_id_not_found_falls_through_to_proximity(self):
        venues = [_venue(center=(100.0, 0.0)), _pub(center=(9000.0, 0.0))]
        result = resolve_inspect_target(venues, (0.0, 0.0), target_venue_id="nonexistent")
        assert result is not None
        assert result.venue_id == "cafe_a"

    def test_description_match_by_venue_id(self):
        venues = [_venue(), _pub()]
        result = resolve_inspect_target(venues, (0.0, 0.0), target_description="pub")
        assert result is not None
        assert result.venue_id == "pub_b"

    def test_description_match_by_visual_summary(self):
        venues = [_venue(), _pub()]
        result = resolve_inspect_target(venues, (0.0, 0.0), target_description="outdoor seating")
        assert result is not None
        assert result.venue_id == "cafe_a"

    def test_description_match_by_venue_type(self):
        venues = [_venue(), _pub(venue_id="corner_b")]
        result = resolve_inspect_target(venues, (0.0, 0.0), target_description="pub")
        assert result is not None
        assert result.venue_id == "corner_b"

    def test_fallback_to_nearest(self):
        venues = [_venue(center=(3000.0, 0.0)), _pub(center=(500.0, 0.0))]
        result = resolve_inspect_target(venues, (0.0, 0.0))
        assert result is not None
        assert result.venue_id == "pub_b"

    def test_empty_venues_returns_none(self):
        result = resolve_inspect_target([], (0.0, 0.0), target_venue_id="cafe_a")
        assert result is None

    def test_id_takes_priority_over_description(self):
        venues = [_venue(), _pub()]
        result = resolve_inspect_target(
            venues, (0.0, 0.0),
            target_venue_id="cafe_a",
            target_description="pub",
        )
        assert result is not None
        assert result.venue_id == "cafe_a"


# ---------------------------------------------------------------------------
# compute_inspection
# ---------------------------------------------------------------------------

class TestComputeInspection:
    def _action(self, **kwargs: Any) -> VenueAgentTurn:
        return VenueAgentTurn(choice=VenueAction.INSPECT.value, **kwargs)

    def test_successful_inspection_at_venue(self):
        venue = _venue(center=(1000.0, 0.0), radius=500.0)
        result, valid = compute_inspection(
            venue, "agent_0", self._action(target_venue_id="cafe_a"),
            agent_xy=(1000.0, 0.0),
            mask_pixels=200,
            info_partition="none",
            agent_zone={"agent_0": None},
            venue_facts_fn=lambda v: {"open": True, "capacity": 20},
        )
        assert valid is True
        assert result["result"] == "INSPECT_OK"
        assert result["venue_id"] == "cafe_a"
        assert result["facts"] == {"open": True, "capacity": 20}
        assert "focused camera frame" in result["agent_visible_result"]

    def test_failed_inspection_not_at_venue(self):
        venue = _venue(center=(1000.0, 0.0), radius=500.0)
        result, valid = compute_inspection(
            venue, "agent_0", self._action(target_venue_id="cafe_a"),
            agent_xy=(0.0, 0.0),
            mask_pixels=0,
            info_partition="none",
            agent_zone={"agent_0": None},
            venue_facts_fn=lambda v: {"open": True},
        )
        assert valid is False
        assert result["result"] == "INSPECT_FAILED"
        assert "NAVIGATE" in result["agent_visible_result"]
        assert "facts" not in result

    def test_zone_blocked(self):
        venue = _venue(zone_id="zone_a")
        result, valid = compute_inspection(
            venue, "agent_0", self._action(),
            agent_xy=(1000.0, 0.0),
            mask_pixels=500,
            info_partition="spatial",
            agent_zone={"agent_0": "zone_b"},
            venue_facts_fn=lambda v: {"open": True},
        )
        assert valid is False
        assert result["result"] == "INSPECT_FAILED"
        assert "teammate" in result["agent_visible_result"]

    def test_zone_allowed_same_zone(self):
        venue = _venue(center=(1000.0, 0.0), radius=500.0, zone_id="zone_a")
        result, valid = compute_inspection(
            venue, "agent_0", self._action(target_venue_id="cafe_a"),
            agent_xy=(1000.0, 0.0),
            mask_pixels=100,
            info_partition="spatial",
            agent_zone={"agent_0": "zone_a"},
            venue_facts_fn=lambda v: {"open": True},
        )
        assert valid is True
        assert result["result"] == "INSPECT_OK"

    def test_diagnostic_fields_present(self):
        venue = _venue(center=(1000.0, 0.0), radius=500.0)
        result, _ = compute_inspection(
            venue, "agent_0", self._action(target_description="cozy"),
            agent_xy=(1000.0, 0.0),
            mask_pixels=42,
            info_partition="none",
            agent_zone={"agent_0": None},
            venue_facts_fn=lambda v: {"open": True},
        )
        assert result["target_description"] == "cozy"
        assert result["mask_pixels_internal"] == 42
        assert result["region_radius_internal"] == 500.0
        assert "distance_to_center_internal" in result


class TestMaskPixelCounting:
    def test_counts_rgb_and_bgr_pixels_with_tolerance(self):
        color = (100, 150, 200)
        frame = np.array([[[100, 150, 200], [200, 150, 100], [110, 150, 200]]], dtype=np.uint8)
        assert count_mask_pixels(frame, color) == 2

    def test_ignores_invalid_frames_and_distant_colors(self):
        assert count_mask_pixels(None, (1, 2, 3)) == 0
        frame = np.array([[[50, 50, 50]]], dtype=np.uint8)
        assert count_mask_pixels(frame, (1, 2, 3)) == 0


# ---------------------------------------------------------------------------
# dispatch_single_action
# ---------------------------------------------------------------------------

class TestDispatchSingleAction:
    def _calls(self) -> dict[str, list[Any]]:
        return {"step": [], "rotate": [], "inspect": [], "navigate": [], "stop": []}

    def test_step_forward(self):
        calls = self._calls()

        def step_fn(aid: str, act: VenueAgentTurn) -> dict[str, Any]:
            calls["step"].append(aid)
            return {"turn": act.compact(), "result": "STEP_FORWARD", "moved_cm": 100.0}

        action = VenueAgentTurn(choice=VenueAction.STEP_FORWARD.value, duration=0.25)
        result = dispatch_single_action(
            "agent_0", action,
            step_forward_fn=step_fn,
            rotate_fn=lambda *a: {},
            inspect_fn=lambda *a: {},
            navigate_fn=lambda *a: {},
            stop_fn=lambda a: None,
        )
        assert calls["step"] == ["agent_0"]
        assert result["result"] == "STEP_FORWARD"
        assert result["moved_cm"] == 100.0

    def test_turn_around(self):
        calls = self._calls()

        def rotate_fn(aid: str, angle: float, direction: str) -> dict[str, Any]:
            calls["rotate"].append((aid, angle, direction))
            return {"result": f"TURN_AROUND angle={angle} direction={direction}", "yaw_deg": 45.0}

        action = VenueAgentTurn(choice=VenueAction.TURN_AROUND.value, angle=90.0, clockwise=True)
        result = dispatch_single_action(
            "agent_0", action,
            step_forward_fn=lambda *a: {},
            rotate_fn=rotate_fn,
            inspect_fn=lambda *a: {},
            navigate_fn=lambda *a: {},
            stop_fn=lambda a: None,
        )
        assert calls["rotate"] == [("agent_0", 90.0, "right")]
        assert "turn" in result
        assert "TURN_AROUND" in result["result"]

    def test_turn_around_counterclockwise(self):
        calls = self._calls()

        def rotate_fn(aid: str, angle: float, direction: str) -> dict[str, Any]:
            calls["rotate"].append((aid, angle, direction))
            return {"result": "TURN_AROUND", "yaw_deg": 0.0}

        action = VenueAgentTurn(choice=VenueAction.TURN_AROUND.value, angle=45.0, clockwise=False)
        dispatch_single_action(
            "agent_0", action,
            step_forward_fn=lambda *a: {},
            rotate_fn=rotate_fn,
            inspect_fn=lambda *a: {},
            navigate_fn=lambda *a: {},
            stop_fn=lambda a: None,
        )
        assert calls["rotate"] == [("agent_0", 45.0, "left")]

    def test_inspect(self):
        calls = self._calls()

        def inspect_fn(aid: str, act: VenueAgentTurn) -> dict[str, Any]:
            calls["inspect"].append(aid)
            return {"result": "INSPECT_OK", "venue_id": "cafe_a"}

        action = VenueAgentTurn(choice=VenueAction.INSPECT.value, target_venue_id="cafe_a")
        result = dispatch_single_action(
            "agent_0", action,
            step_forward_fn=lambda *a: {},
            rotate_fn=lambda *a: {},
            inspect_fn=inspect_fn,
            navigate_fn=lambda *a: {},
            stop_fn=lambda a: None,
        )
        assert calls["inspect"] == ["agent_0"]
        assert result["result"] == "INSPECT_OK"
        assert "turn" in result

    def test_navigate(self):
        calls = self._calls()

        def navigate_fn(aid: str, act: VenueAgentTurn) -> dict[str, Any]:
            calls["navigate"].append(aid)
            return {"result": "NAVIGATE_OK", "venue_id": "pub_b"}

        action = VenueAgentTurn(choice=VenueAction.NAVIGATE.value, target_venue_id="pub_b")
        result = dispatch_single_action(
            "agent_0", action,
            step_forward_fn=lambda *a: {},
            rotate_fn=lambda *a: {},
            inspect_fn=lambda *a: {},
            navigate_fn=navigate_fn,
            stop_fn=lambda a: None,
        )
        assert calls["navigate"] == ["agent_0"]
        assert result["result"] == "NAVIGATE_OK"

    def test_communicate(self):
        action = VenueAgentTurn(choice=VenueAction.COMMUNICATE.value, message="hello team")
        result = dispatch_single_action(
            "agent_0", action,
            step_forward_fn=lambda *a: {},
            rotate_fn=lambda *a: {},
            inspect_fn=lambda *a: {},
            navigate_fn=lambda *a: {},
            stop_fn=lambda a: None,
        )
        assert result["result"] == "COMMUNICATE"
        assert result["message"] == "hello team"
        assert "turn" in result

    def test_wait_calls_stop(self):
        calls = self._calls()

        def stop_fn(aid: str) -> None:
            calls["stop"].append(aid)

        action = VenueAgentTurn(choice=VenueAction.WAIT.value)
        result = dispatch_single_action(
            "agent_0", action,
            step_forward_fn=lambda *a: {},
            rotate_fn=lambda *a: {},
            inspect_fn=lambda *a: {},
            navigate_fn=lambda *a: {},
            stop_fn=stop_fn,
        )
        assert calls["stop"] == ["agent_0"]
        assert result["result"] == "WAIT"
        assert "turn" in result

    def test_unknown_choice_treated_as_wait(self):
        calls = self._calls()

        def stop_fn(aid: str) -> None:
            calls["stop"].append(aid)

        action = VenueAgentTurn(choice=99)
        result = dispatch_single_action(
            "agent_0", action,
            step_forward_fn=lambda *a: {},
            rotate_fn=lambda *a: {},
            inspect_fn=lambda *a: {},
            navigate_fn=lambda *a: {},
            stop_fn=stop_fn,
        )
        assert calls["stop"] == ["agent_0"]
        assert result["result"] == "WAIT"

    def test_default_angle_when_none(self):
        calls = self._calls()

        def rotate_fn(aid: str, angle: float, direction: str) -> dict[str, Any]:
            calls["rotate"].append((aid, angle, direction))
            return {"result": "TURN_AROUND"}

        action = VenueAgentTurn(choice=VenueAction.TURN_AROUND.value, angle=None, clockwise=True)
        dispatch_single_action(
            "agent_0", action,
            step_forward_fn=lambda *a: {},
            rotate_fn=rotate_fn,
            inspect_fn=lambda *a: {},
            navigate_fn=lambda *a: {},
            stop_fn=lambda a: None,
        )
        assert calls["rotate"][0][1] == 45.0
