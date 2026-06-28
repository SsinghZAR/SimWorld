#!/usr/bin/env python3
"""Run UE-grounded N-agent rendezvous experiments with compulsory M3 vision observations."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimentations.multiagent.baselines import CommunicatingBaselinePolicy, SilentBaselinePolicy  # noqa: E402
from experimentations.multiagent.policy import MiniMaxRendezvousPolicy  # noqa: E402
from experimentations.multiagent.rendezvous_env import RendezvousEnv  # noqa: E402
from simworld.communicator.communicator import Communicator  # noqa: E402
from simworld.communicator.unrealcv import UnrealCV  # noqa: E402
from simworld.config import Config  # noqa: E402


def parse_csv_ints(value: str) -> list[int]:
    """Parse comma-separated integer values."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT."""

    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def parse_triplet(value: str) -> tuple[float, float, float]:
    """Parse x,y,z."""

    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return parts[0], parts[1], parts[2]


def parse_pair(value: str) -> tuple[float, float]:
    """Parse x,y."""

    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected x,y")
    return parts[0], parts[1]


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


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return JSON-friendly CLI args."""

    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def make_policy(args: argparse.Namespace):
    """Construct the requested policy."""

    if args.policy == "silent":
        return SilentBaselinePolicy()
    if args.policy == "communicating":
        return CommunicatingBaselinePolicy()
    return MiniMaxRendezvousPolicy(
        model_name=args.model,
        provider=args.provider,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        vision_max_width=args.vision_max_width,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def observation_log(env: RendezvousEnv, observations: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Drop image arrays from all observations."""

    return {agent_id: env.observation_summary(observation) for agent_id, observation in observations.items()}


def capture_frames(observations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Extract image frames from observations."""

    return {agent_id: observation["ego_view"] for agent_id, observation in observations.items()}


def run_case(args: argparse.Namespace, num_agents: int, case_dir: Path) -> dict[str, Any]:
    """Run one N-agent episode and write artifacts."""

    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=args.resolution)
    communicator = Communicator(unrealcv)
    env = RendezvousEnv(
        communicator,
        num_agents=num_agents,
        config=Config(str(args.config)) if args.config else Config(),
        radius=args.radius,
        max_steps=args.max_steps,
        meeting_mode=args.meeting_mode,
        meeting_point=args.meeting_point,
        observe_others=args.observe_others,
        reveal_target_to_all=args.reveal_target_to_all,
        spawn_origin=args.spawn_origin,
        spawn_spacing=args.spawn_spacing,
        speed=args.speed,
        resolution=args.resolution,
        viewmode=args.viewmode,
        camera_mode=args.camera_mode,
        tick_interval=args.tick_interval,
        tick_count=args.tick_count,
        clear_on_first_reset=not args.no_clear_env,
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
        while not done and env.step_index < args.max_steps:
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
                for agent_id, frame in capture_frames(observations).items():
                    videos.setdefault(agent_id, []).append(frame)

        write_json(case_dir / "trajectory.json", trajectory)
        write_jsonl(case_dir / "model_responses.jsonl", model_records)
        if args.save_video:
            for agent_id, frames in videos.items():
                save_video(frames, case_dir / f"{agent_id}.mp4", args.fps)

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_agents": num_agents,
            "policy": args.policy,
            "provider": args.provider,
            "model": args.model,
            "success": final_info.get("success", False),
            "steps_run": len(trajectory),
            "final_distances": final_info.get("distances", {}),
            "final_spread": final_info.get("spread"),
            "message_count": len(final_info.get("comms", {}).get("transcript", [])),
            "comms_error_count": len(final_info.get("comms", {}).get("errors", [])),
            "args": serializable_args(args),
            "artifacts": {
                "trajectory": str(case_dir / "trajectory.json"),
                "model_responses": str(case_dir / "model_responses.jsonl"),
                "metadata": str(case_dir / "metadata.json"),
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
    parser.add_argument("--num-agents", default="2,3,4")
    parser.add_argument("--policy", choices=["minimax", "silent", "communicating"], default="minimax")
    parser.add_argument("--provider", default="minimax")
    parser.add_argument("--model", default="MiniMax-M3")
    parser.add_argument("--base-url")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--vision-max-width", type=int, default=512)
    parser.add_argument("--meeting-mode", choices=["fixed_known", "anchor_known"], default="fixed_known")
    parser.add_argument("--meeting-point", type=parse_pair, default=parse_pair("1000,0"))
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--observe-others", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reveal-target-to-all", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--spawn-origin", type=parse_triplet, default=parse_triplet("0,0,600"))
    parser.add_argument("--spawn-spacing", type=float, default=250.0)
    parser.add_argument("--speed", type=float, default=200.0)
    parser.add_argument("--resolution", type=parse_resolution, default=parse_resolution("640x360"))
    parser.add_argument("--viewmode", choices=["lit", "depth", "object_mask"], default="lit")
    parser.add_argument("--camera-mode", choices=["direct", "fast", "file"], default="direct")
    parser.add_argument("--tick-interval", type=float, default=0.05)
    parser.add_argument("--tick-count", type=int, default=1)
    parser.add_argument("--no-clear-env", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/multiagent"))
    parser.add_argument("--run-name")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fps", type=float, default=5.0)
    return parser


def main() -> int:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    if not backend_reachable(args.ip, args.port, args.connect_timeout):
        print(f"UnrealCV backend is not reachable at {args.ip}:{args.port}", file=sys.stderr)
        return 1

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    summaries = []
    for num_agents in parse_csv_ints(args.num_agents):
        case_dir = run_dir / f"n_{num_agents}_{args.policy}"
        print(f"Running rendezvous case: N={num_agents}, policy={args.policy}, output={case_dir}", flush=True)
        metadata = run_case(args, num_agents, case_dir)
        summaries.append(metadata)
        print(
            f"  success={metadata['success']} steps={metadata['steps_run']} "
            f"messages={metadata['message_count']} spread={metadata['final_spread']}",
            flush=True,
        )

    write_json(run_dir / "summary.json", {"created_at": datetime.now().isoformat(timespec="seconds"), "cases": summaries})
    print(f"Wrote run summary to {run_dir / 'summary.json'}")
    return 0 if all(item.get("success") for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
