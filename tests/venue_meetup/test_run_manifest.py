"""Offline tests for Venue Meetup run_manifest construction (no UE / network)."""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.venue_meetup.run_venue_eval import (
    RUN_MANIFEST_SCHEMA_VERSION,
    build_run_manifest,
    is_secret_arg_key,
    looks_like_secret_value,
    sanitize_run_args,
)


@dataclass(frozen=True)
class _FakeAgent:
    agent_id: str


@dataclass(frozen=True)
class _FakeScenario:
    map_template_id: str
    scenario_id: str
    seed: int
    agents: tuple[_FakeAgent, ...]


REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "created_at",
    "git_commit",
    "args",
    "runtime_versions",
    "template_ids",
    "scenario_ids",
    "seeds",
    "agent_counts",
    "ablations",
    "navigation_mode",
}

LIKELY_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "password",
    "passwd",
    "secret",
    "credential",
    "private_key",
    "client_secret",
)


def _sample_args(**overrides: Any) -> Namespace:
    payload = {
        "ip": "127.0.0.1",
        "port": 9000,
        "dry_run": True,
        "template_id": "central_square_v0",
        "seeds": "7,11",
        "num_agents": "2",
        "hidden_profile": True,
        "info_partition": "spatial",
        "policy": "scripted",
        "provider": "minimax",
        "model": "MiniMax-M3",
        "base_url": "https://api.example.test/v1",
        "max_tokens": 2048,
        "ablation": "main",
        "walk": False,
        "output_dir": Path("runs/venue_meetup"),
        "api_key": "sk-should-never-appear",
        "access_token": "bearer-should-never-appear",
        "password": "hunter2",
        "client_secret": "super-secret",
    }
    payload.update(overrides)
    return Namespace(**payload)


def _sample_scenarios() -> list[_FakeScenario]:
    return [
        _FakeScenario(
            map_template_id="central_square_v0",
            scenario_id="central_square_v0_hp_seed_7_n2",
            seed=7,
            agents=(_FakeAgent("agent_0"), _FakeAgent("agent_1")),
        ),
        _FakeScenario(
            map_template_id="station_quarter_medium_v1",
            scenario_id="station_quarter_medium_v1_hp_seed_11_n2",
            seed=11,
            agents=(_FakeAgent("agent_0"), _FakeAgent("agent_1")),
        ),
    ]


def test_build_run_manifest_required_fields_and_selection() -> None:
    args = _sample_args(walk=True)
    scenarios = _sample_scenarios()
    manifest = build_run_manifest(
        args,
        scenarios=scenarios,
        ablations=["main", "no_communication"],
        created_at="2026-07-22T12:00:00+00:00",
        git_commit="abc123deadbeef",
        runtime_versions={"python": "3.11.0", "simworld": "0.0.0-test"},
    )

    assert set(manifest) >= REQUIRED_MANIFEST_KEYS
    assert manifest["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert manifest["created_at"] == "2026-07-22T12:00:00+00:00"
    assert manifest["git_commit"] == "abc123deadbeef"
    assert manifest["template_ids"] == ["central_square_v0", "station_quarter_medium_v1"]
    assert manifest["scenario_ids"] == [
        "central_square_v0_hp_seed_7_n2",
        "station_quarter_medium_v1_hp_seed_11_n2",
    ]
    assert manifest["seeds"] == [7, 11]
    assert manifest["agent_counts"] == [2]
    assert manifest["ablations"] == ["main", "no_communication"]
    assert manifest["navigation_mode"] == "walk"
    assert manifest["runtime_versions"]["python"] == "3.11.0"
    assert manifest["args"]["template_id"] == "central_square_v0"
    assert manifest["args"]["max_tokens"] == 2048
    assert manifest["args"]["output_dir"] == "runs/venue_meetup"


def test_build_run_manifest_null_git_commit_and_teleport_mode() -> None:
    manifest = build_run_manifest(
        _sample_args(walk=False),
        scenarios=_sample_scenarios()[:1],
        ablations=["main"],
        created_at="2026-07-22T12:00:00+00:00",
        git_commit=None,
        runtime_versions={"python": "3.11.0"},
    )
    assert manifest["git_commit"] is None
    assert manifest["navigation_mode"] == "teleport"


def test_sanitize_run_args_drops_secret_keys_and_values() -> None:
    args = {
        "model": "MiniMax-M3",
        "max_tokens": 1024,
        "api_key": "sk-live-secret",
        "Authorization": "Bearer abc",
        "client_secret": "x",
        "note": "plain text",
        "callback": "https://example.test?api_key=leaked",
        "config": Path("configs/demo.yaml"),
    }
    sanitized = sanitize_run_args(args)
    assert sanitized == {
        "model": "MiniMax-M3",
        "max_tokens": 1024,
        "note": "plain text",
        "config": "configs/demo.yaml",
    }
    assert "api_key" not in sanitized
    assert "Authorization" not in sanitized
    assert "client_secret" not in sanitized
    assert "callback" not in sanitized


def test_manifest_json_has_no_likely_secret_leak() -> None:
    args = _sample_args(
        walk=False,
        openai_api_key="sk-openai-should-not-leak",
        auth_token="tok_secret",
        weird_value="Bearer should-not-survive",
    )
    manifest = build_run_manifest(
        args,
        scenarios=_sample_scenarios(),
        ablations=["main"],
        created_at="2026-07-22T12:00:00+00:00",
        git_commit="deadbeef",
        runtime_versions={"python": "3.11.0"},
    )
    blob = json.dumps(manifest, sort_keys=True)

    for fragment in LIKELY_SECRET_KEY_FRAGMENTS:
        assert fragment not in blob.lower()

    for banned in (
        "sk-should-never-appear",
        "bearer-should-never-appear",
        "hunter2",
        "super-secret",
        "sk-openai-should-not-leak",
        "tok_secret",
        "Bearer should-not-survive",
    ):
        assert banned not in blob

    for key in manifest["args"]:
        assert not is_secret_arg_key(key)
        assert not looks_like_secret_value(manifest["args"][key])


def test_secret_key_helpers_keep_max_tokens() -> None:
    assert is_secret_arg_key("api_key")
    assert is_secret_arg_key("client_secret")
    assert not is_secret_arg_key("max_tokens")
    assert looks_like_secret_value("sk-abc123")
    assert not looks_like_secret_value("https://api.example.test/v1")
