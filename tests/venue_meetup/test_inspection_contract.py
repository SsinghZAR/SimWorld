"""Focused offline checks for the public/evaluator inspection contract."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from benchmark.venue_meetup._core.action_space import VenueAction, VenueAgentTurn
from benchmark.venue_meetup.actions import complete_inspection, precheck_inspection
from benchmark.venue_meetup.inspection_evidence import DECISION_FACT_TRAITS, build_inspection_evidence
from benchmark.venue_meetup.observations import build_observations, public_action_result
from benchmark.venue_meetup.scenario import AgentSpec, Region, Requirement, Scenario, Venue, VenueProperties
from benchmark.venue_meetup.social_metrics import _observed_facts
from benchmark.venue_meetup.venue_env import VenueMeetupEnv
from simworld.utils.vector import Vector


def _venue(*, venue_id: str = "cafe_a", zone_id: str | None = None) -> Venue:
    return Venue(
        venue_id=venue_id,
        slot_id="slot_0",
        venue_type="cafe",
        asset_key="cafe_asset",
        asset_path="/Game/Cafe",
        position=(1000.0, 0.0, 0.0),
        yaw_deg=0.0,
        region=Region(center=(1000.0, 0.0), radius=500.0),
        mask_color_rgb=(255, 0, 0),
        properties=VenueProperties(
            open=True,
            reachable=True,
            capacity=20,
            accessible=True,
            shelter=True,
            food_drink=True,
            quiet_score=0.8,
            crowding_score=0.2,
            near_transit=True,
        ),
        entrances=[],
        visual_summary="A cozy cafe",
        zone_id=zone_id,
    )


def _scenario() -> Scenario:
    venue = _venue()
    agents = [
        AgentSpec(
            agent_id=f"agent_{index}",
            spawn_slot=f"spawn_{index}",
            position=(0.0, 0.0, 0.0),
            yaw_deg=0.0,
            private_constraint="find an open venue",
            private_requirement_keys=["open"],
        )
        for index in range(2)
    ]
    return Scenario(
        scenario_id="inspection_contract",
        map_template_id="test",
        seed=1,
        venues=[venue],
        landmarks=[],
        agents=agents,
        requirements=[Requirement(key="open", weight=1.0, hard=True)],
        soft_weights={"quiet_threshold": 0.65, "crowding_threshold": 0.5},
        coarse_map_text="test map",
    )


def _action() -> VenueAgentTurn:
    return VenueAgentTurn(choice=VenueAction.INSPECT.value, target_venue_id="cafe_a")


def test_evidence_builder_is_ordered_traceable_and_filters_unknown_traits():
    facts = {trait: True for trait in DECISION_FACT_TRAITS}
    facts["capacity"] = 20
    facts["internal_debug"] = "must not appear"
    evidence = build_inspection_evidence(facts)

    assert list(evidence.by_trait) == list(DECISION_FACT_TRAITS)
    assert list(evidence.sentences) == [evidence.by_trait[trait] for trait in DECISION_FACT_TRAITS]
    assert "internal_debug" not in evidence.by_trait
    assert evidence.public_evidence[-1] == "Visible seating or space suggests room for 20 people."


def test_precheck_requires_zone_region_and_range_without_camera_data():
    venue = _venue(zone_id="zone_a")
    action = _action()
    outside_zone = precheck_inspection(
        venue,
        "agent_0",
        action,
        agent_xy=(1000.0, 0.0),
        info_partition="spatial",
        agent_zone={"agent_0": "zone_b"},
        inspect_range=5000.0,
    )
    outside_region = precheck_inspection(
        venue,
        "agent_0",
        action,
        agent_xy=(0.0, 0.0),
        info_partition="none",
        agent_zone={},
        inspect_range=5000.0,
    )
    outside_range = precheck_inspection(
        venue,
        "agent_0",
        action,
        agent_xy=(1400.0, 0.0),
        info_partition="none",
        agent_zone={},
        inspect_range=100.0,
    )

    for failed in (outside_zone, outside_region, outside_range):
        assert failed.success is False
        assert "facts" not in failed.internal_record
        assert "evidence" not in failed.public_record
        assert "mask_pixels_internal" not in failed.internal_record

    assert outside_zone.public_record["reason"] == "outside your area"
    assert outside_region.public_record["reason"] == "not at this venue"
    assert outside_range.public_record["reason"] == "too far from this venue"


def test_visibility_threshold_controls_fact_reveal_and_success_evidence():
    venue = _venue()
    action = _action()
    precheck = precheck_inspection(
        venue,
        "agent_0",
        action,
        agent_xy=(1000.0, 0.0),
        info_partition="none",
        agent_zone={},
        inspect_range=5000.0,
    )
    failed = complete_inspection(
        venue,
        action,
        precheck,
        mask_pixels=4,
        inspect_min_mask_pixels=5,
        venue_facts_fn=lambda _: {"open": True, "capacity": 20},
    )
    succeeded = complete_inspection(
        venue,
        action,
        precheck,
        mask_pixels=5,
        inspect_min_mask_pixels=5,
        venue_facts_fn=lambda _: {"open": True, "capacity": 20},
    )

    assert failed.success is False
    assert "facts" not in failed.internal_record
    assert "evidence" not in failed.public_record
    assert succeeded.success is True
    assert succeeded.internal_record["facts"] == {"open": True, "capacity": 20}
    assert succeeded.public_record["evidence"] == [
        "The entrance appears open.",
        "Visible seating or space suggests room for 20 people.",
    ]


def test_observation_exposes_evidence_but_not_internal_records_or_candidate_metadata():
    scenario = _scenario()
    agent_ids = scenario.agent_ids()
    common = {
        "scenario": scenario,
        "agent_ids": agent_ids,
        "step_index": 1,
        "frames": {agent_id: np.zeros((2, 2, 3), dtype=np.uint8) for agent_id in agent_ids},
        "kinematic_states": {agent_id: (Vector(0.0, 0.0), 0.0) for agent_id in agent_ids},
        "inboxes": {agent_id: [] for agent_id in agent_ids},
        "last_actions": {
            "agent_0": {
                "result": "INSPECT_OK",
                "facts": {"open": True},
                "distance_to_center_internal": 0.0,
            }
        },
        "last_inspections_public": {
            "agent_0": {
                "result": "INSPECT_OK",
                "facts": {"open": True},
                "evidence": ["The entrance appears open."],
                "mask_pixels_internal": 100,
            }
        },
        "revealed_facts": {"agent_0": {"cafe_a": {"open": True}}, "agent_1": {}},
        "revealed_evidence": {"agent_0": {"cafe_a": ["The entrance appears open."]}, "agent_1": {}},
        "agent_zone": {agent_id: None for agent_id in agent_ids},
        "no_coarse_map": True,
        "full_shared_information": False,
        "shared_constraints": False,
        "info_partition": "none",
        "navigate_mode": "teleport",
        "venue_facts_fn": lambda _: {"open": True},
    }
    observations = build_observations(**common)
    public = observations["agent_0"]

    assert public["known_venue_evidence"] == {"cafe_a": ["The entrance appears open."]}
    assert "known_venue_facts" not in public
    assert "facts" not in (public["last_action"] or {})
    assert "distance_to_center_internal" not in (public["last_action"] or {})
    assert "facts" not in (public["last_inspect_result"] or {})
    assert "mask_pixels_internal" not in (public["last_inspect_result"] or {})
    assert set(public["candidate_venues"][0]) == {"venue_id", "venue_type", "slot_id", "visual_summary"}

    full = build_observations(**{**common, "full_shared_information": True})["agent_0"]
    assert full["known_venue_facts"]["cafe_a"]["open"] is True
    assert "properties" not in full["candidate_venues"][0]
    assert "region" not in full["candidate_venues"][0]
    assert "asset_path" not in full["candidate_venues"][0]


def test_env_checks_current_orientation_before_refocusing_and_only_refocuses_on_success():
    venue = _venue()
    env = object.__new__(VenueMeetupEnv)
    env.scenario = SimpleNamespace(venues=[venue])
    env.info_partition = "none"
    env._agent_zone = {"agent_0": None}
    env.inspect_range = 5000.0
    env.inspect_min_mask_pixels = 5
    env.camera_mode = "direct"
    env.last_inspections_internal = {}
    env.last_inspections_public = {}
    env.revealed_facts = {}
    env.revealed_evidence = {}
    env.inspected_venues = set()
    calls: list[str] = []
    state = SimpleNamespace(actor_name="actor", humanoid=SimpleNamespace(camera_id="camera"))
    env._resolve_inspect_target = lambda _agent_id, _action: venue
    env.get_agent_state = lambda _agent_id: state
    env.get_kinematic_state = lambda _agent_id: {"position": Vector(1000.0, 0.0), "yaw_deg": 90.0}
    env._count_mask_pixels = lambda _frame, _color: calls.append("mask") or 0
    env._venue_facts = lambda _venue: {"open": True}
    env._face_point = lambda *_args: calls.append("face")
    env._tick = lambda: calls.append("tick")
    env.communicator = SimpleNamespace(
        get_camera_observation=lambda *_args, **_kwargs: calls.append("capture") or object()
    )

    failed = env._inspect("agent_0", _action())
    assert failed["result"] == "INSPECT_FAILED"
    assert calls == ["capture", "mask"]
    assert "facts" not in env.last_inspections_internal["agent_0"]
    assert "evidence" not in env.last_inspections_public["agent_0"]

    calls.clear()
    env._count_mask_pixels = lambda _frame, _color: calls.append("mask") or 5
    succeeded = env._inspect("agent_0", _action())
    assert succeeded["result"] == "INSPECT_OK"
    assert calls == ["capture", "mask", "face", "tick"]
    assert env.last_inspections_internal["agent_0"]["facts"] == {"open": True}
    assert env.last_inspections_public["agent_0"]["evidence"] == ["The entrance appears open."]

    # Returned evaluator records and state records are independent snapshots.
    succeeded["facts"]["open"] = False
    assert env.revealed_facts["agent_0"]["cafe_a"]["open"] is True
    assert env.last_inspections_internal["agent_0"]["facts"]["open"] is True


def test_step_observation_contains_current_public_action_feedback(monkeypatch):
    """Current action feedback is sanitized before the returned observation."""

    from benchmark.venue_meetup._core.comms import MessageBus

    scenario = _scenario()
    env = object.__new__(VenueMeetupEnv)
    env.scenario = scenario
    env.agent_ids = scenario.agent_ids()
    env.no_communication = True
    env.bus = MessageBus(env.agent_ids)
    env.step_index = 0
    env.last_actions_internal = {}
    env.last_actions_public = {}
    env.last_inspections_internal = {}
    env.last_inspections_public = {}
    env.revealed_facts = {agent_id: {} for agent_id in env.agent_ids}
    env.revealed_evidence = {agent_id: {} for agent_id in env.agent_ids}
    env.inspected_venues = set()
    env._positions = lambda: {agent_id: (0.0, 0.0) for agent_id in env.agent_ids}
    env._converged = lambda: False
    env._tick = lambda: None

    def execute_action(_agent_id, action):
        return {
            "turn": action.compact(),
            "result": "WAIT",
            "facts": {"open": True},
            "distance_to_center_internal": 1.0,
        }

    env.execute_action = execute_action
    seen: dict[str, object] = {}

    def build_observations(*, inboxes=None):
        del inboxes
        seen["actions"] = deepcopy(env.last_actions_public)
        return {agent_id: {"last_action": env.last_actions_public[agent_id]} for agent_id in env.agent_ids}

    env._build_observations = build_observations
    monkeypatch.setattr(
        "benchmark.venue_meetup.venue_env.episode_score",
        lambda *args, **kwargs: {"episode_score": 0.0},
    )
    observations, _rewards, _done, info = env.step(
        {agent_id: {"choice": VenueAction.WAIT.value} for agent_id in env.agent_ids}
    )

    public = seen["actions"]["agent_0"]
    assert public["result"] == "WAIT"
    assert "facts" not in public
    assert "distance_to_center_internal" not in public
    assert observations["agent_0"]["last_action"]["result"] == "WAIT"

    # The evaluator action log is a separate deep copy from state/public views.
    info["actions"]["agent_0"]["facts"]["open"] = False
    assert env.last_actions_internal["agent_0"]["facts"]["open"] is True


def test_public_action_result_uses_ordered_allow_list_and_skips_malformed_evidence():
    result = public_action_result(
        {
            "turn": {"choice": 3, "shared_facts": [{"value": True}], "target_venue_id": "cafe_a"},
            "result": "INSPECT_OK",
            "evidence": {"open": "internal mapping should not pass"},
            "facts": {"open": True},
            "distance_to_center_internal": 0.0,
        }
    )
    assert result == {
        "turn": {"choice": 3, "target_venue_id": "cafe_a"},
        "result": "INSPECT_OK",
    }


def test_legacy_env_aliases_point_to_internal_stores_only():
    env = object.__new__(VenueMeetupEnv)
    env.last_actions = {"agent_0": {"facts": {"open": True}}}
    env.last_inspections = {"agent_0": {"facts": {"open": True}}}
    assert env.last_actions is env.last_actions_internal
    assert env.last_inspections is env.last_inspections_internal
    env.last_actions["agent_0"]["facts"]["open"] = False
    assert env.last_actions_internal["agent_0"]["facts"]["open"] is False


def test_social_metric_reconstruction_reads_first_hand_canonical_action_facts():
    trajectory = [
        {
            "info": {
                "actions": {
                    "agent_0": {
                        "result": "INSPECT_OK",
                        "venue_id": "cafe_a",
                        "facts": {"open": True, "accessible": True},
                    }
                }
            }
        }
    ]
    assert _observed_facts(trajectory) == {
        "agent_0": {"cafe_a": {"open": True, "accessible": True}}
    }
