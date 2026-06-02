#!/usr/bin/env python3
"""Compare stock blocking humanoid wrapper against non-blocking variants.

Launch the SimWorld UE backend first, for example:

    /home/ssingh/simworld_ue/Linux/SimWorld.sh /Game/Maps/demo_1.umap -nullrhi -nosplash -unattended -NoSound

Then run this from the repo root:

    .venv/bin/python experimentations/compare_humanoid_wrappers.py --num-agents 1,2,4
"""

from __future__ import annotations

import argparse
import csv
import math
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.profile_unrealcv_loop import spawn_positions  # noqa: E402
from simworld.agent.humanoid import Humanoid  # noqa: E402
from simworld.communicator.communicator import Communicator  # noqa: E402
from simworld.communicator.unrealcv import UnrealCV  # noqa: E402
from simworld.utils.profiling import summarize_durations  # noqa: E402
from simworld.utils.vector import Vector  # noqa: E402


AGENT_BP = "/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C"


def parse_csv_ints(value: str) -> list[int]:
    """Parse a comma-separated integer list."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def backend_reachable(ip: str, port: int, timeout_seconds: float = 2.0) -> bool:
    """Return whether UnrealCV appears reachable."""

    try:
        with socket.create_connection((ip, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def time_loop(steps: int, warmup_steps: int, fn: Callable[[], None]) -> list[float]:
    """Run warmup steps and return measured durations."""

    for _ in range(warmup_steps):
        fn()

    durations = []
    for _ in range(steps):
        started = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - started)
    return durations


def spawn_humanoids(
    communicator: Communicator,
    num_agents: int,
    spawn_origin: tuple[float, float, float],
    spawn_spacing: float,
) -> tuple[list[Humanoid], list[str]]:
    """Spawn a grid of humanoids and return agents plus actor names."""

    agents = []
    actor_names = []
    for position in spawn_positions(num_agents, spawn_origin, spawn_spacing):
        agent = Humanoid(
            position=Vector(position[0], position[1]),
            direction=Vector(1, 0),
            communicator=communicator,
        )
        communicator.spawn_agent(
            agent,
            name=None,
            position=position,
            model_path=AGENT_BP,
            type="humanoid",
        )
        communicator.humanoid_set_speed(agent.id, 200)
        agents.append(agent)
        actor_names.append(communicator.get_humanoid_name(agent.id))
    return agents, actor_names


def collect_state(unrealcv: UnrealCV, actor_names: list[str], state_mode: str) -> None:
    """Collect structured state using either no state, individual requests, or batch."""

    if state_mode == "none":
        return
    if state_mode == "individual":
        for name in actor_names:
            unrealcv.get_location(name)
            unrealcv.get_orientation(name)
        return
    if state_mode == "batch":
        unrealcv.get_location_batch(actor_names)
        unrealcv.get_orientation_batch(actor_names)
        return
    raise ValueError(f"Unknown state mode: {state_mode}")


def raw_step_forward_no_sleep(unrealcv: UnrealCV, actor_name: str, duration: float, direction: int) -> None:
    """Send the StepForward blueprint command without Python-side sleep."""

    command = f"vbp {actor_name} StepForward {duration} {direction}"
    with unrealcv.lock:
        unrealcv.client.request(command)


def raw_step_forward_batch_no_sleep(unrealcv: UnrealCV, actor_names: list[str], duration: float, direction: int) -> None:
    """Send StepForward commands through UnrealCV request_batch without sleeping."""

    commands = [f"vbp {actor_name} StepForward {duration} {direction}" for actor_name in actor_names]
    with unrealcv.lock:
        unrealcv.client.request_batch(commands)


def build_step_fn(
    variant: str,
    communicator: Communicator,
    agents: list[Humanoid],
    actor_names: list[str],
    *,
    action_duration: float,
    tick_count: int,
    state_mode: str,
) -> Callable[[], None]:
    """Build one comparison step function."""

    unrealcv = communicator.unrealcv

    def maybe_tick() -> None:
        for _ in range(tick_count):
            unrealcv.tick()

    def stock_blocking() -> None:
        for agent in agents:
            communicator.humanoid_step_forward(agent.id, action_duration, direction=0)
        maybe_tick()
        collect_state(unrealcv, actor_names, state_mode)

    def raw_sequential() -> None:
        for name in actor_names:
            raw_step_forward_no_sleep(unrealcv, name, action_duration, direction=0)
        maybe_tick()
        collect_state(unrealcv, actor_names, state_mode)

    def raw_threaded() -> None:
        with ThreadPoolExecutor(max_workers=len(actor_names)) as executor:
            futures = [
                executor.submit(raw_step_forward_no_sleep, unrealcv, name, action_duration, 0)
                for name in actor_names
            ]
            for future in futures:
                future.result()
        maybe_tick()
        collect_state(unrealcv, actor_names, state_mode)

    def raw_batch() -> None:
        raw_step_forward_batch_no_sleep(unrealcv, actor_names, action_duration, direction=0)
        maybe_tick()
        collect_state(unrealcv, actor_names, state_mode)

    def state_tick_only() -> None:
        maybe_tick()
        collect_state(unrealcv, actor_names, state_mode)

    variants = {
        "stock_blocking": stock_blocking,
        "raw_sequential": raw_sequential,
        "raw_threaded": raw_threaded,
        "raw_batch": raw_batch,
        "state_tick_only": state_tick_only,
    }
    return variants[variant]


def run_case(args: argparse.Namespace, num_agents: int, variant: str) -> dict[str, Any]:
    """Run one variant and return a flat result row."""

    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=(args.width, args.height))
    communicator = Communicator(unrealcv)
    agents: list[Humanoid] = []
    actor_names: list[str] = []
    try:
        unrealcv.set_mode("sync", args.tick_interval)
        agents, actor_names = spawn_humanoids(
            communicator,
            num_agents,
            (args.spawn_x, args.spawn_y, args.spawn_z),
            args.spawn_spacing,
        )
        step_fn = build_step_fn(
            variant,
            communicator,
            agents,
            actor_names,
            action_duration=args.action_duration,
            tick_count=args.tick_count,
            state_mode=args.state_mode,
        )
        durations = time_loop(args.steps, args.warmup_steps, step_fn)
        summary = summarize_durations(durations)
        return {
            "variant": variant,
            "num_agents": num_agents,
            "steps": args.steps,
            "action_duration": args.action_duration,
            "tick_count": args.tick_count,
            "state_mode": args.state_mode,
            "mean_seconds": summary["mean_seconds"],
            "p95_seconds": summary["p95_seconds"],
            "p99_seconds": summary["p99_seconds"],
            "steps_per_second": summary["steps_per_second"],
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            "variant": variant,
            "num_agents": num_agents,
            "steps": args.steps,
            "action_duration": args.action_duration,
            "tick_count": args.tick_count,
            "state_mode": args.state_mode,
            "mean_seconds": "",
            "p95_seconds": "",
            "p99_seconds": "",
            "steps_per_second": "",
            "status": "error",
            "error": str(exc),
        }
    finally:
        for actor_name in actor_names:
            try:
                unrealcv.destroy(actor_name)
            except Exception:
                pass
        try:
            unrealcv.clean_garbage()
        except Exception:
            pass
        unrealcv.disconnect()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write result rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--num-agents", default="1,2,4")
    parser.add_argument(
        "--variants",
        default="state_tick_only,stock_blocking,raw_sequential,raw_threaded,raw_batch",
        help="Comma-separated variants: state_tick_only,stock_blocking,raw_sequential,raw_threaded,raw_batch",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--action-duration", type=float, default=0.05)
    parser.add_argument("--tick-count", type=int, default=1)
    parser.add_argument("--tick-interval", type=float, default=0.05)
    parser.add_argument("--state-mode", choices=["none", "individual", "batch"], default="individual")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=600.0)
    parser.add_argument("--spawn-spacing", type=float, default=250.0)
    parser.add_argument("--output", default="runs/profiles/wrapper_comparison.csv")
    return parser


def main() -> int:
    """Run all requested comparisons."""

    args = build_parser().parse_args()
    if not backend_reachable(args.ip, args.port):
        print(f"UnrealCV backend is not reachable at {args.ip}:{args.port}", file=sys.stderr)
        return 1

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    valid_variants = {"state_tick_only", "stock_blocking", "raw_sequential", "raw_threaded", "raw_batch"}
    unknown = sorted(set(variants) - valid_variants)
    if unknown:
        print(f"Unknown variants: {', '.join(unknown)}", file=sys.stderr)
        return 2

    rows = []
    for num_agents in parse_csv_ints(args.num_agents):
        for variant in variants:
            print(f"Running variant={variant} num_agents={num_agents}")
            row = run_case(args, num_agents, variant)
            rows.append(row)
            if row["status"] == "ok":
                print(
                    f"  {float(row['steps_per_second']):.2f} steps/sec "
                    f"(mean {float(row['mean_seconds']) * 1000:.2f} ms)"
                )
            else:
                print(f"  error: {row['error']}")

    write_rows(Path(args.output), rows)
    print(f"Wrote {args.output}")
    return 1 if any(row["status"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
