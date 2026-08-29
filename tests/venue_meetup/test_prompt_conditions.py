"""Offline tests for the prompt/condition contract."""

from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest

from benchmark.venue_meetup._core.action_space import VenueAgentTurn
from benchmark.venue_meetup.ablations import (
    ABLATIONS,
    all_condition_names,
    minimal_ablation_names,
    poc_condition_names,
    resolve_condition,
)
from benchmark.venue_meetup.prompt import (
    VENUE_MEETUP_SYSTEM_PROMPT,
    build_agent_prompt,
    build_system_prompt,
)
from benchmark.venue_meetup._core.policy import VenueMeetupPolicy
from benchmark.venue_meetup.run_venue_eval import (
    build_parser,
    build_run_manifest,
    sanitize_run_args,
    scenarios_from_args,
)


def test_minimal_is_default_and_does_not_scaffold_strategy() -> None:
    prompt = build_system_prompt()
    assert prompt == VENUE_MEETUP_SYSTEM_PROMPT
    assert "known_venue_evidence" in prompt
    assert "known_venue_facts" not in prompt
    assert "only choice=4 (communicate) sends" in prompt.lower()
    for forbidden in ("disclose your private need", "useful to teammates", "pool observations", "coordinate before convergence"):
        assert forbidden not in prompt


def test_cooperative_addendum_and_full_information_note() -> None:
    cooperative = build_system_prompt("cooperative")
    assert "disclose" in cooperative
    assert "teammates" in cooperative
    assert "pool observations" in cooperative
    assert "coordinate before convergence" in cooperative

    full_prompt = build_agent_prompt({"ego_view": [], "known_venue_facts": {}})
    assert "known_venue_facts" in full_prompt
    assert "known_venue_evidence" in full_prompt
    assert "all decision facts and all group constraints are intentionally exposed" in full_prompt.lower()
    system = build_system_prompt()
    assert "readable evidence from your successful inspections" not in system
    assert "the acting agent's hard requirement" not in system


def test_condition_resolution_and_order_are_deterministic() -> None:
    expected_poc = ["main", "no_communication", "full_information", "cooperative_scaffold"]
    expected_all = expected_poc + ["no_coarse_map", "shared_constraints", "full_shared_information"]
    assert all_condition_names() == expected_all
    assert poc_condition_names() == expected_poc

    main = resolve_condition("main")
    no_communication = resolve_condition("no_communication")
    full = resolve_condition("full_information")
    cooperative = resolve_condition("cooperative_scaffold")
    assert main.prompt_mode == "minimal"
    assert main.info_partition == "spatial"
    assert no_communication.no_communication
    assert full.full_shared_information and full.shared_constraints
    assert cooperative.prompt_mode == "cooperative"
    assert cooperative.env_kwargs() == main.env_kwargs()
    assert json.dumps(full.compact(), sort_keys=True)


def test_action_schema_documents_delivery_boundary_and_evaluator_claims() -> None:
    schema = VenueAgentTurn.to_json_schema()["schema"]
    descriptions = " ".join(
        str(value.get("description", "")) for value in schema["properties"].values() if isinstance(value, dict)
    ).lower()
    assert "only choice=4" in descriptions
    assert "evaluator-only" in descriptions
    assert "recipient-visible" in descriptions
    assert "proximity" in descriptions
    assert "readable evidence" in descriptions


def test_manifest_records_resolved_conditions_and_prompt_modes() -> None:
    args = Namespace(info_partition=None, prompt_mode=None, walk=False)
    manifest = build_run_manifest(
        args,
        scenarios=[],
        ablations=poc_condition_names(),
        created_at="2026-08-29T00:00:00+00:00",
        git_commit=None,
        runtime_versions={"python": "test"},
    )
    assert [item["condition_id"] for item in manifest["conditions"]] == poc_condition_names()
    assert manifest["prompt_modes"] == ["minimal", "minimal", "minimal", "cooperative"]


def test_policy_uses_selected_prompt_mode_without_network(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeLLM:
        def __init__(self, **_: object) -> None:
            pass

        def generate_instructions(self, system_prompt, user_prompt, **_: object):
            calls.append((system_prompt, user_prompt))
            return {"choice": 0}, 0.0

    monkeypatch.setattr("benchmark.venue_meetup._core.policy.A2ALLM", FakeLLM)
    policy = VenueMeetupPolicy(prompt_mode="cooperative")
    policy.act({"agent_id": "agent_0", "ego_view": np.zeros((2, 2, 3), dtype=np.uint8)})
    assert policy.prompt_mode == "cooperative"
    assert len(calls) == 1
    combined = "\n".join(calls[0])
    assert combined.count("Cooperative strategy addendum") == 1
    assert combined.count("You are one visitor agent") == 1
    assert "disclose" in calls[0][0].lower()


def test_recursive_argument_sanitization_and_small_eval_guard() -> None:
    sanitized = sanitize_run_args(
        {
            "max_tokens": 2048,
            "nested": {
                "api_key": "sk-nested-secret",
                "safe": "ok",
                "list": [
                    {"Authorization": "Bearer nested-secret", "keep": 3},
                    "token: hidden",
                    "ordinary",
                ],
            },
        }
    )
    assert sanitized == {"max_tokens": 2048, "nested": {"safe": "ok", "list": [{"keep": 3}, "ordinary"]}}

    args = build_parser().parse_args(["--small-eval", "--hidden-profile"])
    with pytest.raises(ValueError, match="cannot be combined"):
        scenarios_from_args(args)


def test_ablation_registry_matches_resolved_environment() -> None:
    for name in all_condition_names():
        assert ABLATIONS[name] == resolve_condition(name).env_kwargs()
    assert minimal_ablation_names() == [
        "main",
        "no_communication",
        "no_coarse_map",
        "shared_constraints",
        "full_shared_information",
    ]
