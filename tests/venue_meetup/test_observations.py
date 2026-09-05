"""Offline tests for the observation assembly module (no UE / network)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.venue_meetup._core.action_space import (SharedFactClaim,
                                                       VenueAction)
from benchmark.venue_meetup._core.comms import Message, MessageBus
from benchmark.venue_meetup.observations import (ACTION_LEGEND,
                                                 build_observations,
                                                 can_inspect_zone,
                                                 compass_label, heading_cue,
                                                 normalize_angle,
                                                 observation_summary,
                                                 target_cue, turn_to_face,
                                                 vector_to_dict)
from benchmark.venue_meetup.scenario import (AgentSpec, Landmark, Region,
                                             Requirement, Scenario, Venue,
                                             VenueProperties)
from benchmark.venue_meetup.venue_env import VenueMeetupEnv


@dataclass
class Vector:
    """Lightweight stand-in for simworld.utils.vector.Vector."""
    x: float
    y: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_venue(venue_id: str = "cafe_a", slot_id: str = "slot_0", zone_id: str | None = None) -> Venue:
    return Venue(
        venue_id=venue_id,
        slot_id=slot_id,
        venue_type="cafe",
        asset_key="cafe_asset",
        asset_path="/Game/Cafe",
        position=(1000.0, 0.0, 0.0),
        yaw_deg=0.0,
        region=Region(center=(1000.0, 0.0), radius=500.0),
        mask_color_rgb=(255, 0, 0),
        properties=VenueProperties(
            open=True, reachable=True, capacity=20,
            accessible=True, shelter=True, food_drink=True,
            quiet_score=0.8, crowding_score=0.3,
        ),
        entrances=[],
        visual_summary="A cozy cafe",
        zone_id=zone_id,
    )


def _dummy_landmark(landmark_id: str = "clock_tower_1") -> Landmark:
    return Landmark(
        landmark_id=landmark_id,
        slot_id="lm_slot_0",
        landmark_type="clock_tower",
        asset_key="tower_asset",
        asset_path="/Game/Tower",
        position=(0.0, 2000.0, 0.0),
        yaw_deg=90.0,
        mask_color_rgb=(0, 255, 0),
        visual_summary="A tall clock tower",
    )


def _dummy_scenario(
    num_agents: int = 2,
    zone_ids: list[str | None] | None = None,
    no_landmarks: bool = False,
) -> Scenario:
    zones = zone_ids or [None] * num_agents
    agents = [
        AgentSpec(
            agent_id=f"agent_{i}",
            spawn_slot=f"spawn_{i}",
            position=(float(i * 500), 0.0, 0.0),
            yaw_deg=0.0,
            private_constraint=f"needs option {i}",
            private_requirement_keys=["open"],
            zone_id=zones[i],
        )
        for i in range(num_agents)
    ]
    venues = [_dummy_venue("cafe_a", "slot_0"), _dummy_venue("pub_b", "slot_1")]
    landmarks = [] if no_landmarks else [_dummy_landmark()]
    return Scenario(
        scenario_id="test_scenario",
        map_template_id="test_map",
        seed=42,
        venues=venues,
        landmarks=landmarks,
        agents=agents,
        requirements=[Requirement(key="open", weight=1.0, hard=True)],
        soft_weights={"quiet_threshold": 0.65, "crowding_threshold": 0.5},
        coarse_map_text="North: cafe_a | South: pub_b",
        max_steps=10,
    )


# ---------------------------------------------------------------------------
# normalize_angle
# ---------------------------------------------------------------------------

class TestNormalizeAngle:
    def test_within_range(self):
        assert normalize_angle(45.0) == 45.0
        assert normalize_angle(-90.0) == -90.0

    def test_wrap_positive(self):
        assert normalize_angle(270.0) == -90.0
        assert normalize_angle(360.0) == 0.0

    def test_wrap_negative(self):
        assert normalize_angle(-270.0) == 90.0
        assert normalize_angle(-360.0) == 0.0

    def test_boundary(self):
        assert normalize_angle(180.0) == 180.0
        assert normalize_angle(-180.0) == -180.0


# ---------------------------------------------------------------------------
# compass_label
# ---------------------------------------------------------------------------

class TestCompassLabel:
    def test_cardinal_directions(self):
        assert compass_label(0.0) == "east"
        assert compass_label(90.0) == "north"
        assert compass_label(180.0) == "west"
        assert compass_label(270.0) == "south"

    def test_intercardinal_directions(self):
        assert compass_label(45.0) == "north-east"
        assert compass_label(135.0) == "north-west"
        assert compass_label(225.0) == "south-west"
        assert compass_label(315.0) == "south-east"

    def test_wraps_around(self):
        assert compass_label(360.0) == "east"
        assert compass_label(-90.0) == "south"


# ---------------------------------------------------------------------------
# turn_to_face
# ---------------------------------------------------------------------------

class TestTurnToFace:
    def test_already_facing(self):
        result = turn_to_face(90.0, 95.0)
        assert result["needs_turn"] is False
        assert "already facing" in result["instruction"]

    def test_turn_left(self):
        result = turn_to_face(0.0, 90.0)
        assert result["needs_turn"] is True
        assert "LEFT" in result["instruction"]
        assert result["action"]["clockwise"] is False
        assert result["action"]["angle"] == 90

    def test_turn_right(self):
        result = turn_to_face(0.0, -90.0)
        assert result["needs_turn"] is True
        assert "RIGHT" in result["instruction"]
        assert result["action"]["clockwise"] is True
        assert result["action"]["angle"] == 90

    def test_custom_tolerance(self):
        result = turn_to_face(0.0, 15.0, tolerance=20.0)
        assert result["needs_turn"] is False


# ---------------------------------------------------------------------------
# vector_to_dict
# ---------------------------------------------------------------------------

class TestVectorToDict:
    def test_basic(self):
        v = Vector(3.5, -7.2)
        d = vector_to_dict(v)
        assert d == {"x": 3.5, "y": -7.2}


# ---------------------------------------------------------------------------
# target_cue
# ---------------------------------------------------------------------------

class TestTargetCue:
    def test_basic_bearing_and_distance(self):
        agent_pos = Vector(0.0, 0.0)
        target_pos = (100.0, 0.0, 0.0)
        cue = target_cue("v1", "venue", "cafe", target_pos, agent_pos, 0.0)
        assert cue["id"] == "v1"
        assert cue["kind"] == "venue"
        assert cue["type"] == "cafe"
        assert cue["bearing_deg"] == 0
        assert cue["distance_m"] == 1  # 100cm / 100
        assert cue["direction"] == "east"

    def test_arrived_when_in_region(self):
        agent_pos = Vector(1000.0, 0.0)
        region = Region(center=(1000.0, 0.0), radius=500.0)
        cue = target_cue("v1", "venue", "cafe", (1000.0, 0.0, 0.0), agent_pos, 0.0, region=region)
        assert cue.get("arrived") is True
        assert "INSPECT or WAIT" in cue["guidance"]
        assert "suggested_action" not in cue

    def test_not_arrived_suggests_action(self):
        agent_pos = Vector(0.0, 0.0)
        region = Region(center=(5000.0, 0.0), radius=500.0)
        cue = target_cue("v1", "venue", "cafe", (5000.0, 0.0, 0.0), agent_pos, 90.0, region=region)
        assert "arrived" not in cue
        assert "suggested_action" in cue

    def test_no_region_never_arrived(self):
        agent_pos = Vector(100.0, 0.0)
        cue = target_cue("lm1", "landmark", "clock_tower", (100.0, 0.0, 0.0), agent_pos, 0.0)
        assert "arrived" not in cue


# ---------------------------------------------------------------------------
# heading_cue
# ---------------------------------------------------------------------------

class TestHeadingCue:
    def test_no_coarse_map_returns_pose_only(self):
        scenario = _dummy_scenario()
        pos = Vector(0.0, 0.0)
        self_pose, nav = heading_cue(pos, 45.0, scenario, no_coarse_map=True, navigate_mode="teleport")
        assert self_pose["facing"] == "north-east"
        assert self_pose["heading_deg"] == 45
        assert nav is None

    def test_with_coarse_map_includes_targets(self):
        scenario = _dummy_scenario()
        pos = Vector(0.0, 0.0)
        self_pose, nav = heading_cue(pos, 0.0, scenario, no_coarse_map=False, navigate_mode="teleport")
        assert nav is not None
        assert "targets" in nav
        venue_ids = {t["id"] for t in nav["targets"] if t["kind"] == "venue"}
        assert "cafe_a" in venue_ids
        assert "pub_b" in venue_ids
        landmark_ids = {t["id"] for t in nav["targets"] if t["kind"] == "landmark"}
        assert "clock_tower_1" in landmark_ids

    def test_walk_mode_hint(self):
        scenario = _dummy_scenario(no_landmarks=True)
        pos = Vector(0.0, 0.0)
        _, nav = heading_cue(pos, 0.0, scenario, no_coarse_map=False, navigate_mode="walk")
        assert "plans a route around the buildings" in nav["hint"]

    def test_teleport_mode_hint(self):
        scenario = _dummy_scenario(no_landmarks=True)
        pos = Vector(0.0, 0.0)
        _, nav = heading_cue(pos, 0.0, scenario, no_coarse_map=False, navigate_mode="teleport")
        assert "in one action" in nav["hint"]


# ---------------------------------------------------------------------------
# can_inspect_zone
# ---------------------------------------------------------------------------

class TestCanInspectZone:
    def test_non_spatial_always_true(self):
        venue = _dummy_venue(zone_id="zone_a")
        assert can_inspect_zone("agent_0", venue, info_partition="none", agent_zone={"agent_0": "zone_b"})

    def test_spatial_same_zone(self):
        venue = _dummy_venue(zone_id="zone_a")
        assert can_inspect_zone("agent_0", venue, info_partition="spatial", agent_zone={"agent_0": "zone_a"})

    def test_spatial_different_zone(self):
        venue = _dummy_venue(zone_id="zone_a")
        assert not can_inspect_zone("agent_0", venue, info_partition="spatial", agent_zone={"agent_0": "zone_b"})

    def test_spatial_unzoned_venue_is_public(self):
        venue = _dummy_venue(zone_id=None)
        assert can_inspect_zone("agent_0", venue, info_partition="spatial", agent_zone={"agent_0": "zone_b"})

    def test_spatial_unzoned_agent_is_public(self):
        venue = _dummy_venue(zone_id="zone_a")
        assert can_inspect_zone("agent_0", venue, info_partition="spatial", agent_zone={"agent_0": None})


# ---------------------------------------------------------------------------
# build_observations  (full assembly)
# ---------------------------------------------------------------------------

class TestBuildObservations:
    def _call(self, **overrides):
        scenario = overrides.pop("scenario", _dummy_scenario())
        agent_ids = scenario.agent_ids()
        defaults = {
            "scenario": scenario,
            "agent_ids": agent_ids,
            "step_index": 3,
            "frames": {aid: np.zeros((64, 64, 3), dtype=np.uint8) for aid in agent_ids},
            "kinematic_states": {aid: (Vector(0.0, 0.0), 0.0) for aid in agent_ids},
            "inboxes": {aid: [] for aid in agent_ids},
            "last_actions": {},
            "last_inspections": {},
            "revealed_facts": {aid: {} for aid in agent_ids},
            "agent_zone": {aid: None for aid in agent_ids},
            "no_coarse_map": False,
            "full_shared_information": False,
            "shared_constraints": False,
            "info_partition": "none",
            "navigate_mode": "teleport",
            "venue_facts_fn": lambda v: {"open": True},
        }
        defaults.update(overrides)
        return build_observations(**defaults)

    def test_returns_dict_per_agent(self):
        obs = self._call()
        assert set(obs.keys()) == {"agent_0", "agent_1"}

    def test_observation_keys_present(self):
        obs = self._call()
        required_keys = {
            "agent_id", "step", "max_steps", "role", "objective",
            "private_constraint", "zone_id", "info_partition",
            "coarse_map_text", "coarse_map_path", "self_pose",
            "candidate_venues", "known_venue_evidence", "landmarks",
            "group_chat", "roster", "last_action", "last_inspect_result",
            "valid_actions", "ego_view",
        }
        for agent_obs in obs.values():
            assert required_keys.issubset(agent_obs.keys())

    def test_normal_candidate_summaries_hide_properties(self):
        obs = self._call()
        expected = {"venue_id", "venue_type", "slot_id", "visual_summary"}
        for agent_obs in obs.values():
            for venue in agent_obs["candidate_venues"]:
                assert set(venue) == expected
                assert "properties" not in venue

    def test_step_and_max_steps(self):
        obs = self._call(step_index=5)
        for agent_obs in obs.values():
            assert agent_obs["step"] == 5
            assert agent_obs["max_steps"] == 10

    def test_private_constraints_separate(self):
        obs = self._call(shared_constraints=False)
        assert obs["agent_0"]["private_constraint"] == "needs option 0"
        assert obs["agent_1"]["private_constraint"] == "needs option 1"

    def test_shared_constraints_merged(self):
        obs = self._call(shared_constraints=True)
        for agent_obs in obs.values():
            assert "needs option 0" in agent_obs["private_constraint"]
            assert "needs option 1" in agent_obs["private_constraint"]

    def test_no_coarse_map_nulls_map_fields(self):
        obs = self._call(no_coarse_map=True)
        for agent_obs in obs.values():
            assert agent_obs["coarse_map_text"] is None
            assert agent_obs["coarse_map_path"] is None
            assert "navigation" not in agent_obs

    def test_coarse_map_included(self):
        obs = self._call(no_coarse_map=False)
        for agent_obs in obs.values():
            assert agent_obs["coarse_map_text"] is not None
            assert "navigation" in agent_obs

    def test_full_shared_information_exposes_all_facts(self):
        obs = self._call(full_shared_information=True)
        for agent_obs in obs.values():
            assert "cafe_a" in agent_obs["known_venue_facts"]
            assert "pub_b" in agent_obs["known_venue_facts"]

    def test_partial_information_uses_readable_revealed_evidence(self):
        obs = self._call(
            full_shared_information=False,
            revealed_facts={"agent_0": {"cafe_a": {"open": True}}, "agent_1": {}},
            revealed_evidence={"agent_0": {"cafe_a": ["The entrance appears open."]}, "agent_1": {}},
        )
        assert obs["agent_0"]["known_venue_evidence"] == {"cafe_a": ["The entrance appears open."]}
        assert obs["agent_1"]["known_venue_evidence"] == {}
        assert "known_venue_facts" not in obs["agent_0"]
        assert "known_venue_facts" not in obs["agent_1"]

    def test_spatial_partition_adds_zone_info(self):
        scenario = _dummy_scenario(zone_ids=["zone_a", "zone_b"])
        scenario.venues[0] = _dummy_venue("cafe_a", "slot_0", zone_id="zone_a")
        scenario.venues[1] = _dummy_venue("pub_b", "slot_1", zone_id="zone_b")
        obs = self._call(
            scenario=scenario,
            info_partition="spatial",
            agent_zone={"agent_0": "zone_a", "agent_1": "zone_b"},
        )
        venues_0 = obs["agent_0"]["candidate_venues"]
        cafe = next(v for v in venues_0 if v["venue_id"] == "cafe_a")
        pub = next(v for v in venues_0 if v["venue_id"] == "pub_b")
        assert cafe["can_inspect"] is True
        assert pub["can_inspect"] is False

    def test_ego_view_present(self):
        obs = self._call()
        for agent_obs in obs.values():
            assert agent_obs["ego_view"] is not None
            assert hasattr(agent_obs["ego_view"], "shape")

    def test_valid_actions_complete(self):
        obs = self._call()
        for agent_obs in obs.values():
            assert agent_obs["valid_actions"] == ACTION_LEGEND

    def test_landmarks_in_observation(self):
        obs = self._call()
        for agent_obs in obs.values():
            assert len(agent_obs["landmarks"]) == 1
            assert agent_obs["landmarks"][0]["landmark_id"] == "clock_tower_1"

    def test_chat_serialization_excludes_evaluator_claims(self):
        message = Message(
            sender="agent_0",
            recipients=["agent_1"],
            content="Cafe is accessible.",
            claims=[SharedFactClaim(venue_id="cafe_a", trait="accessible", value=True)],
            step=2,
        )
        obs = self._call(inboxes={"agent_0": [], "agent_1": [message]})
        compact = obs["agent_1"]["group_chat"]
        assert compact == [{
            "sender": "agent_0",
            "recipients": ["agent_1"],
            "content": "Cafe is accessible.",
            "step": 2,
            "delivered_to": [],
        }]
        assert "claims" not in compact[0]

    def test_observation_summary_excludes_ego_view_only(self):
        summary = observation_summary({"agent_id": "agent_0", "ego_view": np.zeros((2, 2, 3)), "step": 1})
        assert summary == {"agent_id": "agent_0", "step": 1}


def test_no_communication_env_step_suppresses_delivery(monkeypatch):
    """Exercise the environment-level no_communication branch without UE."""

    scenario = _dummy_scenario()
    agent_ids = scenario.agent_ids()
    env = object.__new__(VenueMeetupEnv)
    env.scenario = scenario
    env.agent_ids = agent_ids
    env.no_communication = True
    env.bus = MessageBus(agent_ids)
    env.step_index = 0
    env.last_actions = {}
    env.last_inspections = {}
    env.inspected_venues = set()
    env._agent_zone = {agent_id: None for agent_id in agent_ids}

    positions = {agent_id: (0.0, 0.0) for agent_id in agent_ids}
    captured_inboxes = {}
    env._positions = lambda: positions
    env.execute_action = lambda agent_id, action: {"result": "WAIT"}
    env._tick = lambda: None
    env._converged = lambda: False

    def capture_observations(*, inboxes=None):
        captured_inboxes.update({agent_id: list(inboxes.get(agent_id, [])) for agent_id in agent_ids})
        return {agent_id: {} for agent_id in agent_ids}

    env._build_observations = capture_observations
    monkeypatch.setattr(
        "benchmark.venue_meetup.venue_env.episode_score",
        lambda *args, **kwargs: {"episode_score": 0.0},
    )

    observations, rewards, done, info = env.step(
        {
            "agent_0": {"choice": VenueAction.COMMUNICATE.value, "message": "do not deliver"},
            "agent_1": {"choice": VenueAction.WAIT.value},
        }
    )

    assert set(observations) == set(agent_ids)
    assert rewards == {agent_id: 0.0 for agent_id in agent_ids}
    assert done is False
    assert captured_inboxes == {agent_id: [] for agent_id in agent_ids}
    assert env.bus.transcript == []
    assert all(not inbox for inbox in env.bus.inboxes.values())
    assert info["comms"]["transcript"] == []
    assert info["comms"]["inboxes"] == {agent_id: [] for agent_id in agent_ids}
    assert info["movement_paths_internal"] == {
        agent_id: [(0.0, 0.0)] for agent_id in agent_ids
    }
    assert all("movement_paths_internal" not in observation for observation in observations.values())


class TestFrameEnhancementCompatibility:
    def test_environment_gamma_wrapper_still_maps_uint8_ego_frames(self):
        env = object.__new__(VenueMeetupEnv)
        env._frame_lut = ((np.arange(256) / 255.0) ** 0.5 * 255.0).astype(np.uint8)
        frame = np.array([[[64, 100, 255]]], dtype=np.uint8)
        enhanced = env._enhance_frame(frame)
        assert enhanced.dtype == np.uint8
        assert enhanced[0, 0, 0] > frame[0, 0, 0]
        assert enhanced[0, 0, 2] == 255
