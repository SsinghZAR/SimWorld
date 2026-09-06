#!/usr/bin/env python3
"""Run UE-grounded Venue Meetup evaluations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import socket
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - dry-run environments may not install OpenCV.
    cv2 = None

from benchmark.venue_meetup._core.policy import (
    ScriptedVenueNavPolicy,
    ScriptedVenueSmokePolicy,
    VenueMeetupPolicy,
)
from benchmark.venue_meetup.ablations import (
    ConditionSpec,
    all_condition_names,
    minimal_ablation_names,
    poc_condition_names,
    resolve_condition,
)
from benchmark.venue_meetup.closing_clock import (
    DEFAULT_ACTION_MINUTES,
    DEFAULT_SHOPS_CLOSE_AT,
    ClosingClock,
)
from benchmark.venue_meetup.coarse_map import with_rendered_coarse_map
from benchmark.venue_meetup.episode_report import write_chat_log
from benchmark.venue_meetup.generator import evaluation_matrix, generate_scenario
from benchmark.venue_meetup.rosebank_grid import ROSEBANK_GRID_TEMPLATE_IDS
from benchmark.venue_meetup.scoring import episode_score
from benchmark.venue_meetup.social_metrics import compute_social_metrics
from benchmark.venue_meetup.targeted_env import TargetedVenueEnv
from benchmark.venue_meetup.timing import TimingConfig
from benchmark.venue_meetup.varied_profiles import varied_profile
from benchmark.venue_meetup.venue_env import VenueMeetupEnv
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from simworld.config import Config

RUN_MANIFEST_SCHEMA_VERSION = 2

# Key fragments treated as secrets when serializing CLI/config into the manifest.
_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "password",
    "passwd",
    "secret",
    "credential",
    "private_key",
    "client_secret",
    "token",
)

_SECRET_VALUE_MARKERS = (
    "sk-",
    "bearer ",
    "basic ",
    "api_key=",
    "api_key:",
    "api-key=",
    "api-key:",
    "api-token=",
    "api-token:",
    "api token=",
    "api token:",
    "apikey=",
    "apikey:",
    "access_token=",
    "access_token:",
    "access-token=",
    "access-token:",
    "auth_token=",
    "auth_token:",
    "auth-token=",
    "auth-token:",
    "authorization=",
    "authorization:",
    "password=",
    "password:",
    "passwd=",
    "passwd:",
    "secret=",
    "secret:",
    "client_secret=",
    "client_secret:",
    "private_key=",
    "private_key:",
    "token=",
    "token:",
    "x-api-key",
)

_RUNTIME_PACKAGE_CANDIDATES = (
    "simworld",
    "numpy",
    "pydantic",
    "pillow",
    "opencv-python",
    "opencv-python-headless",
)

_UNSET = object()


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT."""

    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def parse_csv_ints(value: str) -> list[int]:
    """Parse comma-separated integer values."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    """Parse comma-separated strings."""

    return [item.strip() for item in value.split(",") if item.strip()]


def backend_reachable(ip: str, port: int, timeout_seconds: float = 2.0) -> bool:
    """Return whether UnrealCV appears reachable."""

    try:
        with socket.create_connection((ip, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _normalize_arg_key(key: str) -> str:
    return str(key).lower().replace("-", "_")


def is_secret_arg_key(key: str) -> bool:
    """Return whether a CLI/config key name looks like a secret."""

    normalized = _normalize_arg_key(key)
    # Keep benign token-budget knobs (e.g. max_tokens) out of the secret filter.
    if "max_token" in normalized:
        return False
    return any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS)


def looks_like_secret_value(value: Any) -> bool:
    """Return whether a string value looks like an embedded credential."""

    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _SECRET_VALUE_MARKERS)


def _jsonable_arg_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_jsonable_arg_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_arg_value(item) for key, item in value.items()}
    return value


_DROP_SECRET = object()


def _sanitize_arg_value(value: Any) -> Any:
    """Recursively serialize an argument while dropping credential material."""

    if isinstance(value, Path):
        converted: Any = value.as_posix()
    elif isinstance(value, Mapping):
        converted = {}
        for key, nested_value in value.items():
            key_text = str(key)
            if is_secret_arg_key(key_text):
                continue
            nested = _sanitize_arg_value(nested_value)
            if nested is _DROP_SECRET:
                continue
            converted[key_text] = nested
    elif isinstance(value, (list, tuple)):
        converted = []
        for nested_value in value:
            nested = _sanitize_arg_value(nested_value)
            if nested is not _DROP_SECRET:
                converted.append(nested)
    else:
        converted = _jsonable_arg_value(value)

    if looks_like_secret_value(converted):
        return _DROP_SECRET
    return converted


def sanitize_run_args(args: Any) -> dict[str, Any]:
    """Serialize CLI/config args with nested secret keys/values removed."""

    if hasattr(args, "__dict__"):
        raw = vars(args)
    elif isinstance(args, Mapping):
        raw = dict(args)
    else:
        raise TypeError(f"args must be a Namespace or mapping, got {type(args)!r}")

    sanitized = _sanitize_arg_value(raw)
    if sanitized is _DROP_SECRET or not isinstance(sanitized, dict):
        return {}
    return sanitized


def discover_git_commit() -> str | None:
    """Return the current HEAD commit hash when git is available locally."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = (completed.stdout or "").strip()
    return commit or None


def discover_runtime_versions() -> dict[str, str]:
    """Return safely discoverable Python/package versions (missing pkgs omitted)."""

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package_name in _RUNTIME_PACKAGE_CANDIDATES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def build_run_manifest(
    args: Any,
    *,
    scenarios: Sequence[Any],
    ablations: Sequence[str],
    conditions: Sequence[str | ConditionSpec] | None = None,
    created_at: str | None = None,
    git_commit: Any = _UNSET,
    runtime_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a reproducible run manifest (pure when overrides are supplied).

    Pass ``git_commit`` / ``runtime_versions`` explicitly in tests to avoid
    subprocess and packaging side effects. When ``git_commit`` is left unset, the
    helper discovers HEAD locally (or returns null).
    """

    resolved_commit = discover_git_commit() if git_commit is _UNSET else git_commit

    template_ids: list[str] = []
    scenario_ids: list[str] = []
    seeds: list[int] = []
    agent_counts: list[int] = []
    for scenario in scenarios:
        template_id = getattr(scenario, "map_template_id", None)
        scenario_id = getattr(scenario, "scenario_id", None)
        seed = getattr(scenario, "seed", None)
        agents = getattr(scenario, "agents", None)
        if template_id is not None and template_id not in template_ids:
            template_ids.append(str(template_id))
        if scenario_id is not None and scenario_id not in scenario_ids:
            scenario_ids.append(str(scenario_id))
        if seed is not None and int(seed) not in seeds:
            seeds.append(int(seed))
        if agents is not None:
            count = len(agents)
            if count not in agent_counts:
                agent_counts.append(count)

    if isinstance(args, Mapping):
        arg_prompt_mode = args.get("prompt_mode")
        arg_info_partition = args.get("info_partition")
        walk = bool(args.get("walk", False))
    else:
        arg_prompt_mode = getattr(args, "prompt_mode", None)
        arg_info_partition = getattr(args, "info_partition", None)
        walk = bool(getattr(args, "walk", False))

    resolved_conditions: list[ConditionSpec] = []
    condition_inputs: Sequence[str | ConditionSpec] = conditions if conditions is not None else ablations
    for condition in condition_inputs:
        resolved = resolve_condition(
            condition,
            prompt_mode=arg_prompt_mode,
            info_partition=arg_info_partition,
        )
        if resolved not in resolved_conditions:
            resolved_conditions.append(resolved)

    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": resolved_commit,
        "args": sanitize_run_args(args),
        "runtime_versions": dict(runtime_versions) if runtime_versions is not None else discover_runtime_versions(),
        "template_ids": template_ids,
        "scenario_ids": scenario_ids,
        "seeds": seeds,
        "agent_counts": agent_counts,
        "ablations": [str(name) for name in ablations],
        "conditions": [condition.compact() for condition in resolved_conditions],
        "prompt_modes": [condition.prompt_mode for condition in resolved_conditions],
        "navigation_mode": "walk" if walk else "teleport",
    }


def save_video(frames: list[Any], path: Path, fps: float) -> None:
    """Save BGR frames to MP4."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required for --save-video.")
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
    finally:
        writer.release()


def observation_log(env: VenueMeetupEnv, observations: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Drop image arrays from all observations."""

    return {agent_id: env.observation_summary(observation) for agent_id, observation in observations.items()}


def capture_frames(observations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Extract image frames from observations."""

    return {agent_id: observation["ego_view"] for agent_id, observation in observations.items()}


def make_policy(args: argparse.Namespace, *, prompt_mode: str | None = None):
    """Construct the requested policy."""

    if args.policy == "scripted":
        if getattr(args, "protocol", "legacy") == "targeted":
            from benchmark.venue_meetup.targeted_scripted import TargetedScriptedPolicy
            return TargetedScriptedPolicy()
        return ScriptedVenueSmokePolicy()
    if args.policy == "nav_smoke":
        return ScriptedVenueNavPolicy()
    return VenueMeetupPolicy(
        model_name=args.model,
        provider=args.provider,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        vision_max_width=args.vision_max_width,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning=args.reasoning,
        prompt_mode=prompt_mode or getattr(args, "prompt_mode", None) or "minimal",
    )


def token_count(records: list[dict[str, Any]]) -> int:
    """Approximate prompt/response token count for diagnostics."""

    total = 0
    for record in records:
        total += len(str(record.get("prompt", "")).split())
        total += len(str(record.get("raw_response", "")).split())
    return total


def run_case(
    args: argparse.Namespace,
    scenario,
    case_dir: Path,
    *,
    ablation: str,
    condition: ConditionSpec | None = None,
) -> dict[str, Any]:
    """Run one live venue-meetup episode and write artifacts."""

    # Resolve once at the case boundary so the environment, policy, metadata,
    # and dry-run path all share exactly the same immutable configuration.
    condition = condition or resolve_condition(
        ablation,
        prompt_mode=getattr(args, "prompt_mode", None),
        info_partition=getattr(args, "info_partition", None),
    )
    scenario = with_rendered_coarse_map(scenario, case_dir)
    scenario = replace(scenario, max_steps=getattr(args, "max_steps", None) or scenario.max_steps)
    targeted = getattr(args, "protocol", "legacy") == "targeted"
    timing = (TimingConfig(starts_at=getattr(args, "starts_at", "17:30"),
                           shops_close_at=getattr(args, "shops_close_at", DEFAULT_SHOPS_CLOSE_AT))
              if targeted else None)
    closing_clock = timing if targeted else ClosingClock(
        max_turns=scenario.max_steps,
        shops_close_at=getattr(args, "shops_close_at", DEFAULT_SHOPS_CLOSE_AT),
        action_minutes=getattr(args, "action_minutes", DEFAULT_ACTION_MINUTES),
    )
    write_json(case_dir / "scenario_hidden.json", scenario.compact(include_hidden=True))
    write_json(case_dir / "scenario_public.json", scenario.compact(include_hidden=False))

    if args.dry_run:
        fake_positions = {agent.agent_id: scenario.venues[0].region.center for agent in scenario.agents}
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": True,
            "scenario_id": scenario.scenario_id,
            "map_template_id": scenario.map_template_id,
            "num_agents": len(scenario.agents),
            "policy": args.policy,
            "ablation": ablation,
            "condition_id": condition.condition_id,
            "condition": condition.compact(),
            "prompt_mode": condition.prompt_mode,
            "closing_clock": {
                "initial": closing_clock.snapshot(0),
                "final": closing_clock.snapshot(0),
            },
            "scores": episode_score(scenario, fake_positions),
            "args": sanitize_run_args(args),
            "artifacts": {"scenario_hidden": str(case_dir / "scenario_hidden.json"), "coarse_map": scenario.coarse_map_path},
        }
        write_json(case_dir / "metadata.json", metadata)
        return metadata

    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=args.resolution)
    communicator = Communicator(unrealcv)
    # Recording config. Camera grabs are slow (~0.4 s each), so at the benchmark
    # walk speed an agent covers several metres between capturable frames and the
    # video looks like skating. --cinematic trades step-efficiency for a smooth
    # clip: it slows the walk and lengthens strides so consecutive frames are
    # ~0.6 m apart, captures as fast as the engine allows, and plays back near
    # real time. Scoring/locomotion semantics are unchanged otherwise.
    move_speed = args.speed
    camera_mode = args.camera_mode
    playback_fps = args.fps
    motion_fps = max(args.fps, 12.0)
    stride_kwargs: dict[str, Any] = {}
    if args.cinematic:
        args.save_video = True
        move_speed = args.cinematic_speed
        camera_mode = "fast"
        playback_fps = args.cinematic_fps
        motion_fps = 1000.0  # capture back-to-back during the walk window.
        stride_kwargs = {"step_duration": 1.1, "min_step_duration": 1.0, "max_step_duration": 1.4}
    env_class = TargetedVenueEnv if targeted else VenueMeetupEnv
    env = env_class(
        communicator,
        scenario,
        config=Config(str(args.config)) if args.config else Config(),
        resolution=args.resolution,
        viewmode=args.viewmode,
        frame_gamma=args.frame_gamma,
        camera_mode=camera_mode,
        tick_interval=args.tick_interval,
        tick_count=args.tick_count,
        speed=move_speed,
        record_motion=args.save_video,
        motion_fps=motion_fps,
        navigate_mode="walk" if args.walk else "teleport",
        shops_close_at=closing_clock.shops_close_at,
        action_minutes=getattr(args, "action_minutes", DEFAULT_ACTION_MINUTES),
        **({"timing": timing} if targeted else {}),
        **stride_kwargs,
        **condition.env_kwargs(),
    )
    policy = make_policy(args, prompt_mode=condition.prompt_mode)
    trajectory: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    videos: dict[str, list[Any]] = {}
    try:
        observations = env.reset()
        if args.save_video:
            videos = {agent_id: [frame] for agent_id, frame in capture_frames(observations).items()}

        done = False
        final_info: dict[str, Any] = {}
        decision_rounds = 0
        while not done:
            step_index = env.step_index
            ready_observations = ({agent: observation for agent, observation in observations.items()
                                   if agent in env.ready_agent_ids} if targeted else observations)
            if ready_observations and decision_rounds >= scenario.max_steps:
                raise RuntimeError("Decision safety cap reached before closing; run is incomplete, not an agent failure")
            turns, records = policy.act_all(ready_observations) if ready_observations else ({}, [])
            decision_rounds += bool(ready_observations)
            next_observations, rewards, done, info = env.step(turns)
            model_records.extend({"step": step_index, **record} for record in records)
            trajectory.append(
                {
                    "step": step_index,
                    "observations": observation_log(env, observations),
                    "turns": {agent_id: turn.compact() for agent_id, turn in turns.items()},
                    "rewards": rewards,
                    "done": done,
                    "info": info,
                }
            )
            observations = next_observations
            final_info = info
            # Keep durable partial artifacts if the PC/backend stops mid-run.
            write_json(case_dir / "trajectory.json", trajectory)
            write_jsonl(case_dir / "model_responses.jsonl", model_records)
            if targeted:
                progress = {"tick": env.step_index, "clock": info["closing_clock"],
                            "decision_rounds": decision_rounds, "done": done}
                write_json(case_dir / "progress.json", progress)
                print(f"  tick={env.step_index} time={info['closing_clock']['current_time']} decisions={len(records)}", flush=True)
                if cv2 is not None:
                    for agent, observation in observations.items():
                        if observation.get("nearby_interactables") and (info.get("actions", {}).get(agent) or step_index == 0):
                            cv2.imwrite(str(case_dir / f"interaction_{agent}_tick_{env.step_index:03d}.png"), observation["ego_view"])
            if args.save_video:
                for sample in env.drain_motion_frames():
                    for agent_id, frame in sample.items():
                        videos.setdefault(agent_id, []).append(frame)
                for agent_id, frame in capture_frames(observations).items():
                    videos.setdefault(agent_id, []).append(frame)

        write_json(case_dir / "trajectory.json", trajectory)
        write_jsonl(case_dir / "model_responses.jsonl", model_records)
        from benchmark.venue_meetup.trajectory_minimap import render_trajectory_minimap

        chat_log = write_chat_log(case_dir, trajectory)

        trajectory_artifacts: dict[str, Path] = {}
        trajectory_render_error: str | None = None
        try:
            trajectory_artifacts = render_trajectory_minimap(case_dir)
        except Exception as exc:  # noqa: BLE001 - preserve an expensive completed episode.
            trajectory_render_error = f"{type(exc).__name__}: {exc}"
            print(
                f"Warning: trajectory minimap rendering failed: {trajectory_render_error}",
                file=sys.stderr,
                flush=True,
            )
        if args.save_video:
            for agent_id, frames in videos.items():
                save_video(frames, case_dir / f"{agent_id}.mp4", playback_fps)

        scores = final_info.get("scores", {})
        scores["token_count"] = token_count(model_records)
        social = compute_social_metrics(scenario, trajectory)
        write_json(case_dir / "social_metrics.json", social)
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scenario_id": scenario.scenario_id,
            "map_template_id": scenario.map_template_id,
            "num_agents": len(scenario.agents),
            "policy": args.policy,
            "provider": args.provider,
            "model": args.model,
            "ablation": ablation,
            "condition_id": condition.condition_id,
            "condition": condition.compact(),
            "prompt_mode": condition.prompt_mode,
            "closing_clock": {
                "initial": closing_clock.snapshot(0),
                "final": final_info.get(
                    "closing_clock",
                    closing_clock.snapshot(len(trajectory)),
                ),
            },
            "success": final_info.get("success", False),
            "steps_run": len(trajectory),
            "decision_rounds": decision_rounds,
            "protocol": "targeted_v1" if targeted else "legacy",
            "scores": scores,
            "social_metrics": social,
            "args": sanitize_run_args(args),
            "artifact_errors": (
                {"trajectory_minimap": trajectory_render_error}
                if trajectory_render_error
                else {}
            ),
            "artifacts": {
                "trajectory": str(case_dir / "trajectory.json"),
                "chat_log": str(chat_log),
                "model_responses": str(case_dir / "model_responses.jsonl"),
                "metadata": str(case_dir / "metadata.json"),
                "social_metrics": str(case_dir / "social_metrics.json"),
                "scenario_hidden": str(case_dir / "scenario_hidden.json"),
                "coarse_map": scenario.coarse_map_path,
                **{name: str(path) for name, path in trajectory_artifacts.items()},
            },
        }
        write_json(case_dir / "metadata.json", metadata)
        return metadata
    finally:
        env.disconnect()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--connect-timeout", type=float, default=2.0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--small-eval",
        action="store_true",
        help="Run the 3x3, 5x5, and 7x7 scale matrix across seeds and agent counts.",
    )
    parser.add_argument("--ablation-matrix", action="store_true", help="Run the four canonical POC conditions in order.")
    parser.add_argument("--template-id", default="central_square_v0")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--num-agents", default="2")
    parser.add_argument("--hidden-profile", action="store_true", help="Overlay a hidden-profile information structure (asymmetric per-agent needs + partition zones; see notes.md).")
    parser.add_argument("--info-partition", choices=["none", "spatial"], default=None, help="Optional inspection-partition override; otherwise use the selected condition's default.")
    parser.add_argument("--prompt-mode", choices=["minimal", "cooperative"], default=None, help="Optional prompt-mode override; otherwise use the selected condition's default.")
    parser.add_argument("--policy", choices=["scripted", "nav_smoke", "minimax"], default="scripted")
    parser.add_argument("--provider", default="minimax")
    parser.add_argument("--model", default="MiniMax-M3")
    parser.add_argument("--base-url")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning", choices=["enabled", "adaptive", "disabled"], default="disabled", help="MiniMax-M3 thinking mode. 'disabled' skips chain-of-thought (fewer output tokens, lower latency).")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--vision-max-width", type=int, default=512)
    condition_cli_names = list(dict.fromkeys(all_condition_names() + minimal_ablation_names()))
    parser.add_argument("--ablation", choices=condition_cli_names, default="main")
    parser.add_argument("--max-steps", type=int, help="Targeted: decision-round safety cap, not the closing deadline. Legacy: turn budget.")
    parser.add_argument("--protocol", choices=["targeted", "legacy"], default="targeted",
                        help="Targeted sources, varied requirements and independent timed actions; legacy explicitly reproduces old episodes.")
    parser.add_argument("--starts-at", default="17:30", help="Targeted-protocol simulated start time, independent of --max-steps safety cap.")
    parser.add_argument(
        "--shops-close-at",
        default=DEFAULT_SHOPS_CLOSE_AT,
        help="Simulated 24-hour closing time shown to agents (HH:MM).",
    )
    parser.add_argument(
        "--action-minutes",
        type=int,
        default=DEFAULT_ACTION_MINUTES,
        help="Legacy protocol only: fixed simulated minutes per synchronized action turn.",
    )
    parser.add_argument(
        "--walk",
        action="store_true",
        help=(
            "Walk-mode NAVIGATE: physically traverse graph-backed layout routes "
            "(sidewalks/crossings/bridges) when a district layout is available; "
            "otherwise fall back to the legacy obstacle-aware free-space planner. "
            "Default without this flag is teleport navigation."
        ),
    )
    parser.add_argument("--speed", type=float, default=1000.0, help="Engine MaxWalkSpeed (cm/s) for collision-aware locomotion.")
    parser.add_argument("--resolution", type=parse_resolution, default=parse_resolution("640x360"))
    parser.add_argument("--viewmode", choices=["lit", "depth", "object_mask"], default="lit")
    parser.add_argument("--frame-gamma", type=float, default=0.5, help="Brighten lit ego frames (gamma<1 brightens; 1.0 disables).")
    parser.add_argument("--camera-mode", choices=["direct", "fast", "file"], default="direct")
    parser.add_argument("--tick-interval", type=float, default=0.05)
    parser.add_argument("--tick-count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/venue_meetup"))
    parser.add_argument("--run-name")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fps", type=float, default=10.0, help="Saved-video fps; also the in-motion frame sampling target when recording.")
    parser.add_argument("--cinematic", action="store_true", help="Record a smooth walking clip (slower walk, longer strides, dense capture, real-time playback).")
    parser.add_argument("--cinematic-speed", type=float, default=180.0, help="Walk speed (cm/s) used under --cinematic.")
    parser.add_argument("--cinematic-fps", type=float, default=4.0, help="Playback fps used under --cinematic (matches the ~real capture cadence).")
    return parser


def scenarios_from_args(args: argparse.Namespace):
    """Build scenarios requested by CLI args."""

    if args.small_eval:
        agent_counts = tuple(parse_csv_ints(args.num_agents))
        if getattr(args, "hidden_profile", False) and agent_counts != (2,):
            raise ValueError(
                "--small-eval --hidden-profile currently requires "
                "--num-agents 2"
            )
        templates = [
            ROSEBANK_GRID_TEMPLATE_IDS[grid_size] for grid_size in (3, 5, 7)
        ]
        scenarios = evaluation_matrix(
            templates=templates,
            seeds=parse_csv_ints(args.seeds),
            agent_counts=agent_counts,
            hidden_profile=args.hidden_profile or getattr(args, "protocol", "legacy") == "targeted",
        )
        return [varied_profile(scenario) for scenario in scenarios] if getattr(args, "protocol", "legacy") == "targeted" else scenarios

    scenarios = []
    for seed in parse_csv_ints(args.seeds):
        for num_agents in parse_csv_ints(args.num_agents):
            scenarios.append(
                generate_scenario(
                    seed=seed,
                    template_id=args.template_id,
                    num_agents=num_agents,
                    randomize=False,
                    hidden_profile=args.hidden_profile or getattr(args, "protocol", "legacy") == "targeted",
                )
            )
    return [varied_profile(scenario) for scenario in scenarios] if getattr(args, "protocol", "legacy") == "targeted" else scenarios


def main() -> int:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    if not args.dry_run and not backend_reachable(args.ip, args.port, args.connect_timeout):
        print(f"UnrealCV backend is not reachable at {args.ip}:{args.port}", file=sys.stderr)
        return 1

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    ablations = poc_condition_names() if args.ablation_matrix else [args.ablation]
    try:
        scenarios = scenarios_from_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    conditions = [
        resolve_condition(
            name,
            prompt_mode=args.prompt_mode,
            info_partition=args.info_partition,
        )
        for name in ablations
    ]
    manifest = build_run_manifest(args, scenarios=scenarios, ablations=ablations, conditions=conditions)
    write_json(run_dir / "run_manifest.json", manifest)
    print(f"Wrote run manifest to {run_dir / 'run_manifest.json'}", flush=True)

    summaries = []
    for scenario in scenarios:
        for ablation, condition in zip(ablations, conditions):
            case_dir = run_dir / scenario.map_template_id / scenario.scenario_id / ablation
            print(f"Running venue meetup: scenario={scenario.scenario_id} ablation={ablation} output={case_dir}", flush=True)
            metadata = run_case(args, scenario, case_dir, ablation=ablation, condition=condition)
            summaries.append(metadata)
            print(
                f"  success={metadata.get('success')} steps={metadata.get('steps_run')} "
                f"score={metadata.get('scores', {}).get('episode_score')}",
                flush=True,
            )

    write_json(run_dir / "summary.json", {"created_at": datetime.now().isoformat(timespec="seconds"), "cases": summaries})
    print(f"Wrote run summary to {run_dir / 'summary.json'}")
    return 0 if args.dry_run or all(item.get("success") for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
