"""Offline boundary/integration harness for the Venue Meetup POC conditions."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from benchmark.venue_meetup._core.action_space import SharedFactClaim, VenueAction, VenueAgentTurn
from benchmark.venue_meetup._core.comms import MessageBus, messages_from_turns
from benchmark.venue_meetup.ablations import resolve_condition
from benchmark.venue_meetup.generator import generate_scenario
from benchmark.venue_meetup.prompt import build_system_prompt
from benchmark.venue_meetup.scoring import score_venue, venue_decision_facts
from benchmark.venue_meetup.social_metrics import _observed_facts, compute_social_metrics
from benchmark.venue_meetup.venue_env import VenueMeetupEnv
from simworld.utils.vector import Vector


def _offline_env(scenario, *, info_partition: str = "spatial") -> VenueMeetupEnv:
    """Create the smallest environment harness needed by the public APIs.

    The live constructor/reset path is intentionally not used: this harness
    supplies camera, pose, and tick stubs while exercising ``VenueMeetupEnv``'s
    real inspection, observation, and step methods.
    """

    env = object.__new__(VenueMeetupEnv)
    env.scenario = scenario
    env.agent_ids = scenario.agent_ids()
    env.info_partition = info_partition
    env._agent_zone = {agent.agent_id: agent.zone_id for agent in scenario.agents}
    env.inspect_range = 5000.0
    env.inspect_min_mask_pixels = 5
    env._frame_lut = None
    env.camera_mode = "direct"
    env.navigate_mode = "teleport"
    env.no_coarse_map = False
    env.full_shared_information = False
    env.shared_constraints = False
    env.last_actions_internal = {}
    env.last_actions_public = {}
    env.last_inspections_internal = {}
    env.last_inspections_public = {}
    env.revealed_facts = {agent_id: {} for agent_id in env.agent_ids}
    env.revealed_evidence = {agent_id: {} for agent_id in env.agent_ids}
    env.inspected_venues = set()
    env.bus = MessageBus(env.agent_ids)
    env.step_index = 0

    by_id = {venue.venue_id: venue for venue in scenario.venues}
    states = {
        agent_id: SimpleNamespace(
            actor_name=f"stub_{agent_id}",
            humanoid=SimpleNamespace(camera_id=f"camera_{agent_id}"),
        )
        for agent_id in env.agent_ids
    }
    env._resolve_inspect_target = lambda _agent_id, action: by_id.get(action.target_venue_id)
    env.get_agent_state = lambda agent_id: states[agent_id]

    env._inspection_center = by_id[next(iter(by_id))].region.center

    def _kinematic_state(_agent_id: str) -> dict[str, object]:
        return {"position": Vector(*env._inspection_center), "yaw_deg": 0.0}

    env.get_kinematic_state = _kinematic_state
    env.communicator = SimpleNamespace(
        get_camera_observation=lambda *_args, **_kwargs: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    env._count_mask_pixels = lambda _frame, _color: 5
    env._face_point = lambda *_args: None
    env._tick = lambda: None
    env._capture_frames = lambda _viewmode=None: {
        agent_id: np.zeros((2, 2, 3), dtype=np.uint8) for agent_id in env.agent_ids
    }
    env._venue_facts = lambda venue: venue_decision_facts(venue, scenario.soft_weights)
    return env


def _step_only_env(scenario, *, no_communication: bool) -> VenueMeetupEnv:
    """Build an offline boundary harness for the actual ``VenueMeetupEnv.step`` path."""

    env = object.__new__(VenueMeetupEnv)
    env.scenario = scenario
    env.agent_ids = scenario.agent_ids()
    env.no_communication = no_communication
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

    def _execute(_agent_id, action):
        result = "COMMUNICATE" if action.choice == VenueAction.COMMUNICATE.value else "WAIT"
        return {"result": result, "message": action.message, "turn": action.compact()}

    env.execute_action = _execute
    captured: dict[str, list[object]] = {}

    def _observations(*, inboxes=None):
        captured.update({agent_id: list((inboxes or {}).get(agent_id, [])) for agent_id in env.agent_ids})
        return {agent_id: {} for agent_id in env.agent_ids}

    env._build_observations = _observations
    env._captured_inboxes = captured
    return env


def test_venue_meetup_poc_contract_offline() -> None:
    """Exercise the complete POC contract without UE, models, credentials, or network."""

    scenario = generate_scenario(
        seed=7,
        template_id="station_quarter_medium_v1",
        num_agents=2,
        randomize=False,
        hidden_profile=True,
    )
    main = resolve_condition("main")
    no_communication = resolve_condition("no_communication")
    full_information = resolve_condition("full_information")
    cooperative = resolve_condition("cooperative_scaffold")
    assert len(scenario.agents) == 2
    assert len([venue for venue in scenario.venues if not score_venue(venue, scenario).hard_failures]) == 1
    assert main.prompt_mode == "minimal"
    assert main.info_partition == "spatial"
    assert no_communication.no_communication
    assert full_information.full_shared_information and full_information.shared_constraints

    # A successful typed inspection produces readable evidence for the agent,
    # while its canonical values and mask/proximity diagnostics remain evaluator-only.
    env = _offline_env(scenario, info_partition=main.info_partition)
    agent = scenario.agents[0]
    own_venue = next(venue for venue in scenario.venues if venue.zone_id == agent.zone_id)
    env._inspection_center = own_venue.region.center
    inspect_turn = VenueAgentTurn(choice=VenueAction.INSPECT.value, target_venue_id=own_venue.venue_id)
    inspect_result = env._inspect(agent.agent_id, inspect_turn)
    canonical = venue_decision_facts(own_venue, scenario.soft_weights)
    assert inspect_result["result"] == "INSPECT_OK"
    assert inspect_result["facts"] == canonical
    public_inspect = env.last_inspections_public[agent.agent_id]
    assert public_inspect["result"] == "INSPECT_OK"
    assert public_inspect["evidence"] and all(isinstance(sentence, str) for sentence in public_inspect["evidence"])
    assert "facts" not in public_inspect
    assert "mask_pixels_internal" in env.last_inspections_internal[agent.agent_id]
    assert env.revealed_facts[agent.agent_id][own_venue.venue_id] == canonical
    assert env.revealed_evidence[agent.agent_id][own_venue.venue_id] == public_inspect["evidence"]

    observations = env._build_observations()
    public = observations[agent.agent_id]
    assert public["known_venue_evidence"][own_venue.venue_id] == public_inspect["evidence"]
    assert "known_venue_facts" not in public
    assert "properties" not in public["candidate_venues"][0]
    assert all(
        not forbidden
        for forbidden in ("region", "asset_path", "mask_color_rgb", "entrances")
        for candidate in public["candidate_venues"]
        if forbidden in candidate
    )
    assert "facts" not in (public["last_inspect_result"] or {})
    assert "mask_pixels_internal" not in (public["last_inspect_result"] or {})

    # Claims are extracted only from COMMUNICATE. A non-communication action
    # carrying the same fields must not create a message.
    claim_trait = next(key for key in agent.private_requirement_keys if key != "open")
    claim = SharedFactClaim(venue_id=own_venue.venue_id, trait=claim_trait, value=canonical[claim_trait])
    turns = {
        agent.agent_id: VenueAgentTurn(
            choice=VenueAction.COMMUNICATE.value,
            message="The inspected venue has a relevant report.",
            shared_facts=[claim],
        ),
        scenario.agents[1].agent_id: VenueAgentTurn(
            choice=VenueAction.WAIT.value,
            message="This text is not delivered.",
            shared_facts=[claim],
        ),
    }
    assert messages_from_turns(
        {agent.agent_id: VenueAgentTurn(choice=VenueAction.INSPECT.value, message="not sent", shared_facts=[claim])},
        step=0,
    ) == []
    messages = messages_from_turns(turns, step=1)
    assert len(messages) == 1

    # The communication-enabled environment step performs the same extraction
    # and delivery, and its public recipient snapshot omits evaluator claims.
    communication_env = _step_only_env(scenario, no_communication=False)
    _observations, _rewards, _done, communication_info = communication_env.step(turns)
    delivered = communication_env._captured_inboxes[scenario.agents[1].agent_id][0]
    assert delivered.content == "The inspected venue has a relevant report."
    assert delivered.claims == [claim]
    recipient = communication_info["comms"]["inboxes"][scenario.agents[1].agent_id][0]
    assert recipient["content"] == "The inspected venue has a relevant report."
    assert "claims" not in recipient
    logged = communication_info["comms"]["transcript"][0]
    assert logged["claims"] == [claim.compact()]

    # The no-communication condition reaches the real env step method but
    # suppresses extraction/delivery before the bus is touched.
    no_comm_env = _step_only_env(scenario, no_communication=True)
    _observations, _rewards, _done, no_comm_info = no_comm_env.step(
        {
            agent.agent_id: {
                "choice": VenueAction.COMMUNICATE.value,
                "message": "must not arrive",
                "shared_facts": [claim.compact()],
            },
            scenario.agents[1].agent_id: {"choice": VenueAction.WAIT.value},
        }
    )
    assert no_comm_info["actions"][agent.agent_id]["result"] == "COMMUNICATE"
    assert no_comm_env._captured_inboxes == {agent_id: [] for agent_id in scenario.agent_ids()}
    assert no_comm_info["comms"]["transcript"] == []
    assert no_comm_info["comms"]["inboxes"] == {agent_id: [] for agent_id in scenario.agent_ids()}

    # Full information is an upper bound over canonical decision facts and all
    # constraints; candidate records remain the same safe navigation summaries.
    env.full_shared_information = full_information.full_shared_information
    env.shared_constraints = full_information.shared_constraints
    full = env._build_observations()[agent.agent_id]
    assert full["known_venue_facts"] == {
        venue.venue_id: venue_decision_facts(venue, scenario.soft_weights) for venue in scenario.venues
    }
    assert all(agent_spec.private_constraint in full["private_constraint"] for agent_spec in scenario.agents)
    assert all(set(candidate) <= {"venue_id", "venue_type", "slot_id", "visual_summary", "zone_id", "can_inspect"} for candidate in full["candidate_venues"])
    assert all("properties" not in candidate and "region" not in candidate for candidate in full["candidate_venues"])

    # The cooperative scaffold changes the prompt only; its environment flags
    # are exactly the main condition's flags.
    assert cooperative.env_kwargs() == main.env_kwargs()
    assert cooperative.prompt_mode != main.prompt_mode
    assert build_system_prompt(cooperative.prompt_mode) != build_system_prompt(main.prompt_mode)
    assert "Cooperative strategy addendum" in build_system_prompt(cooperative.prompt_mode)
    assert "Cooperative strategy addendum" not in build_system_prompt(main.prompt_mode)

    # Social metrics reconstruct canonical first-hand facts from evaluator logs,
    # and exact claims are credited only against those facts.
    trajectory = [
        {"step": 0, "info": {"actions": {agent.agent_id: deepcopy(inspect_result)}, "comms": {"transcript": []}}},
        {"step": 1, "info": {"actions": communication_info["actions"], "comms": communication_info["comms"]}},
    ]
    assert _observed_facts(trajectory) == {agent.agent_id: {own_venue.venue_id: canonical}}
    exact = compute_social_metrics(scenario, trajectory)["exact_structured_claims"]["per_agent"][agent.agent_id]
    assert exact["first_hand_supported_claims"] == 1
    assert exact["unsupported_claims"] == 0
