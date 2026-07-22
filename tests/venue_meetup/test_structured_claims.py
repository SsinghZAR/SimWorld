"""Offline tests for structured fact claims (no UE / network)."""

from __future__ import annotations

from benchmark.venue_meetup._core.action_space import (
    SharedFactClaim,
    VenueAction,
    VenueAgentTurn,
    sanitize_turn,
)
from benchmark.venue_meetup._core.comms import BroadcastRouter, Message, MessageBus, messages_from_turns
from benchmark.venue_meetup.prompt import VENUE_MEETUP_SYSTEM_PROMPT, build_agent_prompt
from benchmark.venue_meetup.scenario import (
    AgentSpec,
    Region,
    Requirement,
    Scenario,
    Venue,
    VenueProperties,
)
from benchmark.venue_meetup.scoring import venue_decision_facts
from benchmark.venue_meetup.social_metrics import compute_social_metrics


def _venue(venue_id: str, *, zone_id: str, accessible: bool, food_drink: bool) -> Venue:
    return Venue(
        venue_id=venue_id,
        slot_id=f"slot_{venue_id}",
        venue_type="cafe",
        asset_key="BP_Building_05_C",
        asset_path="/Game/Fake",
        position=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        region=Region(center=(0.0, 0.0), radius=100.0),
        mask_color_rgb=(1, 2, 3),
        properties=VenueProperties(
            open=True,
            reachable=True,
            capacity=4,
            accessible=accessible,
            shelter=True,
            food_drink=food_drink,
            quiet_score=0.8,
            crowding_score=0.1,
            near_transit=False,
        ),
        entrances=[],
        zone_id=zone_id,
    )


def _mini_scenario() -> Scenario:
    return Scenario(
        scenario_id="structured_claims_mini",
        map_template_id="test_mini",
        seed=1,
        venues=[
            _venue("venue_west", zone_id="zone_a", accessible=True, food_drink=False),
            _venue("venue_east", zone_id="zone_b", accessible=False, food_drink=True),
        ],
        landmarks=[],
        agents=[
            AgentSpec(
                agent_id="agent_0",
                spawn_slot="a",
                position=(0.0, 0.0, 0.0),
                yaw_deg=0.0,
                private_constraint="need accessible",
                private_requirement_keys=["accessible"],
                zone_id="zone_a",
            ),
            AgentSpec(
                agent_id="agent_1",
                spawn_slot="b",
                position=(10.0, 0.0, 0.0),
                yaw_deg=0.0,
                private_constraint="need food",
                private_requirement_keys=["food_drink"],
                zone_id="zone_b",
            ),
        ],
        requirements=[
            Requirement(key="accessible", weight=1.0, hard=True),
            Requirement(key="food_drink", weight=1.0, hard=True),
        ],
        soft_weights={"quiet_threshold": 0.65, "crowding_threshold": 0.5},
        coarse_map_text="mini",
        max_steps=8,
    )


def _inspect_step(agent_id: str, venue: Venue, scenario: Scenario, *, step: int = 0) -> dict:
    facts = venue_decision_facts(venue, scenario.soft_weights)
    return {
        "info": {
            "actions": {
                agent_id: {
                    "result": "INSPECT_OK",
                    "venue_id": venue.venue_id,
                    "facts": facts,
                }
            },
            "comms": {"transcript": []},
        },
        "step": step,
    }


def _comms_step(transcript: list[dict], *, step: int = 1) -> dict:
    return {"info": {"actions": {}, "comms": {"transcript": transcript}}, "step": step}


def test_parse_schema_sanitize_shared_facts() -> None:
    payload = {
        "choice": VenueAction.COMMUNICATE.value,
        "message": "west cafe is accessible",
        "shared_facts": [
            {"venue_id": "venue_west", "trait": "accessible", "value": True},
            {"venue_id": "venue_west", "trait": "unknown_trait", "value": "x"},
            {"venue_id": "no_such_venue", "trait": "food_drink", "value": False},
            {"bad": "entry"},
            "skip-me",
        ],
        "reasoning": "report",
    }
    turn = VenueAgentTurn.from_json(payload)
    assert turn.message == "west cafe is accessible"
    assert len(turn.shared_facts) == 3
    assert turn.shared_facts[0] == SharedFactClaim(venue_id="venue_west", trait="accessible", value=True)
    assert turn.shared_facts[1].trait == "unknown_trait"
    assert turn.shared_facts[2].venue_id == "no_such_venue"

    schema = VenueAgentTurn.to_json_schema()
    assert "shared_facts" in schema["schema"]["properties"]
    assert schema["schema"]["required"] == ["choice"]
    assert "message" in schema["schema"]["properties"]

    sanitized = sanitize_turn(turn)
    assert sanitized.choice == VenueAction.COMMUNICATE.value
    assert sanitized.message == turn.message
    assert sanitized.shared_facts == turn.shared_facts

    for choice in (
        VenueAction.WAIT.value,
        VenueAction.STEP_FORWARD.value,
        VenueAction.TURN_AROUND.value,
        VenueAction.INSPECT.value,
        VenueAction.NAVIGATE.value,
    ):
        kept = sanitize_turn(
            VenueAgentTurn(
                choice=choice,
                duration=0.2,
                direction=0,
                angle=30.0,
                clockwise=True,
                target_venue_id="venue_west",
                message="hi",
                shared_facts=turn.shared_facts,
            )
        )
        assert kept.shared_facts == turn.shared_facts

    legacy = VenueAgentTurn.from_json({"choice": 0, "message": "hello only"})
    assert legacy.shared_facts == []
    assert legacy.message == "hello only"


def test_claims_only_delivery_appears_in_transcript_and_inbox() -> None:
    turn = VenueAgentTurn(
        choice=VenueAction.COMMUNICATE.value,
        message=None,
        shared_facts=[SharedFactClaim(venue_id="venue_west", trait="accessible", value=True)],
    )
    messages = messages_from_turns({"agent_0": turn}, step=3)
    assert len(messages) == 1
    assert messages[0].content == ""
    assert messages[0].claims[0].trait == "accessible"

    bus = MessageBus(["agent_0", "agent_1"], router=BroadcastRouter())
    inboxes = bus.deliver(messages)
    assert len(bus.transcript) == 1
    compact = bus.transcript[0].compact()
    assert compact["content"] == ""
    assert compact["claims"] == [{"venue_id": "venue_west", "trait": "accessible", "value": True}]
    assert inboxes["agent_1"][0].claims[0].venue_id == "venue_west"
    assert bus.snapshot()["inboxes"]["agent_1"][0]["claims"]

    empty = Message(sender="agent_0", content="   ", step=4, claims=[])
    cleaned, transcript, _ = BroadcastRouter().deliver([empty], ["agent_0", "agent_1"])
    assert transcript == []
    assert cleaned["agent_1"] == []


def test_exact_supported_partner_relevant_claim() -> None:
    scenario = _mini_scenario()
    west = scenario.venues[0]
    facts = venue_decision_facts(west, scenario.soft_weights)
    trajectory = [
        _inspect_step("agent_0", west, scenario, step=0),
        _comms_step(
            [
                {
                    "sender": "agent_0",
                    "content": "",
                    "claims": [{"venue_id": "venue_west", "trait": "food_drink", "value": facts["food_drink"]}],
                    "step": 1,
                }
            ],
            step=1,
        ),
    ]
    metrics = compute_social_metrics(scenario, trajectory)
    exact = metrics["exact_structured_claims"]
    agent0 = exact["per_agent"]["agent_0"]
    assert agent0["first_hand_supported_claims"] == 1
    assert agent0["partner_relevant_claims"] == 1
    assert agent0["unsupported_claims"] == 0
    assert agent0["contradictory_claims"] == 0
    # Relevant traits are accessible + food_drink; only food_drink was claimed.
    assert agent0["observed_relevant_facts"] == 2
    assert agent0["exact_shared_relevant_facts"] == 1
    assert agent0["exact_sharing_completeness"] == 0.5
    assert exact["aggregate"]["partner_relevant_claims"] == 1


def test_exact_duplicate_redundant_claim() -> None:
    scenario = _mini_scenario()
    west = scenario.venues[0]
    facts = venue_decision_facts(west, scenario.soft_weights)
    claim = {"venue_id": "venue_west", "trait": "accessible", "value": facts["accessible"]}
    trajectory = [
        _inspect_step("agent_0", west, scenario, step=0),
        _comms_step(
            [
                {"sender": "agent_0", "content": "", "claims": [claim], "step": 1},
                {"sender": "agent_0", "content": "", "claims": [claim], "step": 2},
            ],
            step=2,
        ),
    ]
    exact = compute_social_metrics(scenario, trajectory)["exact_structured_claims"]["per_agent"]["agent_0"]
    assert exact["first_hand_supported_claims"] == 1
    assert exact["duplicate_redundant_claims"] == 1


def test_exact_unsupported_claim() -> None:
    scenario = _mini_scenario()
    trajectory = [
        _comms_step(
            [
                {
                    "sender": "agent_0",
                    "content": "I guess venue_east has food",
                    "claims": [{"venue_id": "venue_east", "trait": "food_drink", "value": True}],
                    "step": 1,
                }
            ]
        )
    ]
    exact = compute_social_metrics(scenario, trajectory)["exact_structured_claims"]["per_agent"]["agent_0"]
    assert exact["unsupported_claims"] == 1
    assert exact["first_hand_supported_claims"] == 0
    # Free-text co-mention must not create an exact share.
    assert exact["exact_sharing_completeness"] is None


def test_exact_contradictory_claim() -> None:
    scenario = _mini_scenario()
    west = scenario.venues[0]
    facts = venue_decision_facts(west, scenario.soft_weights)
    trajectory = [
        _inspect_step("agent_0", west, scenario, step=0),
        _comms_step(
            [
                {
                    "sender": "agent_0",
                    "content": "accessible!",
                    "claims": [{"venue_id": "venue_west", "trait": "accessible", "value": not facts["accessible"]}],
                    "step": 1,
                }
            ]
        ),
    ]
    exact = compute_social_metrics(scenario, trajectory)["exact_structured_claims"]["per_agent"]["agent_0"]
    assert exact["contradictory_claims"] == 1
    assert exact["first_hand_supported_claims"] == 0


def test_legacy_free_text_metrics_still_present() -> None:
    scenario = _mini_scenario()
    west = scenario.venues[0]
    trajectory = [
        _inspect_step("agent_0", west, scenario, step=0),
        _comms_step(
            [
                {
                    "sender": "agent_0",
                    "content": "venue_west is accessible and has no food",
                    "step": 1,
                }
            ]
        ),
    ]
    metrics = compute_social_metrics(scenario, trajectory)
    assert "sharing_completeness" in metrics["per_agent"]["agent_0"]
    assert "other_regarding_ratio" in metrics["per_agent"]["agent_0"]
    assert "redundant_messages" in metrics["per_agent"]["agent_0"]
    assert "must_pool" in metrics["per_agent"]["agent_0"]
    assert "optimum_venue_id" in metrics
    assert "notes" in metrics
    assert metrics["per_agent"]["agent_0"]["messages_sent"] == 1
    # Heuristic path can credit free text; exact path must not without claims.
    assert metrics["exact_structured_claims"]["aggregate"]["first_hand_supported_claims"] == 0
    assert metrics["exact_structured_claims"]["aggregate"]["claims_emitted"] == 0


def test_prompt_mentions_structured_claims_for_inspected_facts_only() -> None:
    assert "shared_facts" in VENUE_MEETUP_SYSTEM_PROMPT
    assert "directly inspected" in VENUE_MEETUP_SYSTEM_PROMPT
    prompt = build_agent_prompt({"task": "x", "ego_view": "drop-me"})
    assert "shared_facts" in prompt
    assert "personally inspected" in prompt
    assert "ego_view" not in prompt


def test_no_ue_or_network_imports_in_claim_modules() -> None:
    import benchmark.venue_meetup._core.action_space as action_space
    import benchmark.venue_meetup._core.comms as comms
    import benchmark.venue_meetup.social_metrics as social_metrics

    for module in (action_space, comms, social_metrics):
        source = open(module.__file__, encoding="utf-8").read()
        assert "unreal" not in source.lower()
        assert "requests" not in source
        assert "socket" not in source
