#!/usr/bin/env python3
"""Run UE-grounded Venue Meetup evaluations."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - dry-run environments may not install OpenCV.
    cv2 = None

from benchmark.venue_meetup._core.policy import ScriptedVenueSmokePolicy, VenueMeetupPolicy
from benchmark.venue_meetup.ablations import ablation_kwargs, minimal_ablation_names
from benchmark.venue_meetup.coarse_map import with_rendered_coarse_map
from benchmark.venue_meetup.generator import evaluation_matrix, generate_scenario
from benchmark.venue_meetup.social_metrics import compute_social_metrics
from benchmark.venue_meetup.scoring import episode_score
from benchmark.venue_meetup.venue_env import VenueMeetupEnv
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from simworld.config import Config


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


def make_policy(args: argparse.Namespace):
    """Construct the requested policy."""

    if args.policy == "scripted":
        return ScriptedVenueSmokePolicy()
    return VenueMeetupPolicy(
        model_name=args.model,
        provider=args.provider,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        vision_max_width=args.vision_max_width,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning=args.reasoning,
    )


def token_count(records: list[dict[str, Any]]) -> int:
    """Approximate prompt/response token count for diagnostics."""

    total = 0
    for record in records:
        total += len(str(record.get("prompt", "")).split())
        total += len(str(record.get("raw_response", "")).split())
    return total


def run_case(args: argparse.Namespace, scenario, case_dir: Path, *, ablation: str) -> dict[str, Any]:
    """Run one live venue-meetup episode and write artifacts."""

    scenario = with_rendered_coarse_map(scenario, case_dir)
    scenario = replace(scenario, max_steps=args.max_steps or scenario.max_steps)
    write_json(case_dir / "scenario_hidden.json", scenario.compact(include_hidden=True))
    write_json(case_dir / "scenario_public.json", scenario.compact(include_hidden=False))

    if args.dry_run:
        fake_positions = {agent.agent_id: scenario.venues[0].region.center for agent in scenario.agents}
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": True,
            "scenario_id": scenario.scenario_id,
            "ablation": ablation,
            "scores": episode_score(scenario, fake_positions),
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
    env = VenueMeetupEnv(
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
        info_partition=args.info_partition,
        **stride_kwargs,
        **ablation_kwargs(ablation),
    )
    policy = make_policy(args)
    trajectory: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    videos: dict[str, list[Any]] = {}
    try:
        observations = env.reset()
        if args.save_video:
            videos = {agent_id: [frame] for agent_id, frame in capture_frames(observations).items()}

        done = False
        final_info: dict[str, Any] = {}
        while not done and env.step_index < scenario.max_steps:
            step_index = env.step_index
            turns, records = policy.act_all(observations)
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
            if args.save_video:
                for sample in env.drain_motion_frames():
                    for agent_id, frame in sample.items():
                        videos.setdefault(agent_id, []).append(frame)
                for agent_id, frame in capture_frames(observations).items():
                    videos.setdefault(agent_id, []).append(frame)

        write_json(case_dir / "trajectory.json", trajectory)
        write_jsonl(case_dir / "model_responses.jsonl", model_records)
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
            "success": final_info.get("success", False),
            "steps_run": len(trajectory),
            "scores": scores,
            "social_metrics": social,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "artifacts": {
                "trajectory": str(case_dir / "trajectory.json"),
                "model_responses": str(case_dir / "model_responses.jsonl"),
                "metadata": str(case_dir / "metadata.json"),
                "social_metrics": str(case_dir / "social_metrics.json"),
                "scenario_hidden": str(case_dir / "scenario_hidden.json"),
                "coarse_map": scenario.coarse_map_path,
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
    parser.add_argument("--small-eval", action="store_true", help="Run the 3-template x seeds x N matrix.")
    parser.add_argument("--ablation-matrix", action="store_true", help="Run main + four V0 ablations.")
    parser.add_argument("--template-id", default="central_square_v0")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--num-agents", default="2")
    parser.add_argument("--hidden-profile", action="store_true", help="Overlay a hidden-profile information structure (asymmetric per-agent needs + partition zones; see notes.md).")
    parser.add_argument("--info-partition", choices=["none", "spatial"], default="none", help="Inspection partition: 'spatial' restricts each agent to inspecting venues in its own zone.")
    parser.add_argument("--policy", choices=["scripted", "minimax"], default="scripted")
    parser.add_argument("--provider", default="minimax")
    parser.add_argument("--model", default="MiniMax-M3")
    parser.add_argument("--base-url")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning", choices=["enabled", "adaptive", "disabled"], default="disabled", help="MiniMax-M3 thinking mode. 'disabled' skips chain-of-thought (fewer output tokens, lower latency).")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--vision-max-width", type=int, default=512)
    parser.add_argument("--ablation", choices=minimal_ablation_names(), default="main")
    parser.add_argument("--max-steps", type=int)
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
        return evaluation_matrix(seeds=parse_csv_ints(args.seeds), agent_counts=tuple(parse_csv_ints(args.num_agents)))

    scenarios = []
    for seed in parse_csv_ints(args.seeds):
        for num_agents in parse_csv_ints(args.num_agents):
            scenarios.append(
                generate_scenario(
                    seed=seed,
                    template_id=args.template_id,
                    num_agents=num_agents,
                    randomize=False,
                    hidden_profile=args.hidden_profile,
                )
            )
    return scenarios


def main() -> int:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    if not args.dry_run and not backend_reachable(args.ip, args.port, args.connect_timeout):
        print(f"UnrealCV backend is not reachable at {args.ip}:{args.port}", file=sys.stderr)
        return 1

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    ablations = minimal_ablation_names() if args.ablation_matrix else [args.ablation]
    summaries = []
    for scenario in scenarios_from_args(args):
        for ablation in ablations:
            case_dir = run_dir / scenario.map_template_id / scenario.scenario_id / ablation
            print(f"Running venue meetup: scenario={scenario.scenario_id} ablation={ablation} output={case_dir}", flush=True)
            metadata = run_case(args, scenario, case_dir, ablation=ablation)
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
