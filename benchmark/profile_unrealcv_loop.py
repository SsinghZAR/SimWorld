#!/usr/bin/env python3
"""Profile SimWorld UnrealCV+/Gym-like step throughput.

The script is intentionally small and explicit: launch the UE backend yourself,
then point this profiler at the running UnrealCV port.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROFILING_PATH = REPO_ROOT / "simworld" / "utils" / "profiling.py"
profiling_spec = importlib.util.spec_from_file_location("simworld_benchmark_profiling", PROFILING_PATH)
if profiling_spec is None or profiling_spec.loader is None:
    raise RuntimeError(f"Unable to load profiling helpers from {PROFILING_PATH}")
profiling = importlib.util.module_from_spec(profiling_spec)
sys.modules[profiling_spec.name] = profiling
profiling_spec.loader.exec_module(profiling)

BenchmarkCase = profiling.BenchmarkCase
benchmark_record = profiling.benchmark_record
nvidia_smi_metadata = profiling.nvidia_smi_metadata
parse_csv = profiling.parse_csv
parse_int_csv = profiling.parse_int_csv
parse_resolutions = profiling.parse_resolutions
resolution_label = profiling.resolution_label
write_csv_summary = profiling.write_csv_summary
write_jsonl = profiling.write_jsonl


OBSERVATION_PROFILES = {
    "structured_only",
    "rgb",
    "depth",
    "object_mask",
    "rgb_depth",
    "shared_rgb",
}
STEP_KINDS = {"sensor_tick", "gym_wrapper", "command_latency", "robot"}
MODES = {"sync", "async"}


PRESETS = {
    "quick": {
        "num_agents": "1",
        "observation_profiles": "structured_only",
        "resolutions": "640x480",
        "modes": "sync",
        "step_kinds": "sensor_tick",
        "steps": 10,
        "warmup_steps": 2,
    },
    "rl": {
        "num_agents": "1,2,4,8",
        "observation_profiles": "structured_only,rgb",
        "resolutions": "640x480",
        "modes": "sync",
        "step_kinds": "sensor_tick,gym_wrapper",
        "steps": 100,
        "warmup_steps": 10,
    },
    "full": {
        "num_agents": "1,2,4,8,16",
        "observation_profiles": "structured_only,rgb,depth,object_mask",
        "resolutions": "320x240,640x480,1280x720",
        "modes": "sync,async",
        "step_kinds": "sensor_tick",
        "steps": 100,
        "warmup_steps": 10,
    },
}


def parse_origin(value: str) -> tuple[float, float, float]:
    """Parse ``x,y,z`` origin coordinates."""

    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected origin as x,y,z")
    return parts[0], parts[1], parts[2]


def ensure_choices(values: list[str], allowed: set[str], label: str) -> None:
    """Validate a CLI matrix list."""

    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"Invalid {label}: {', '.join(invalid)}")


def backend_reachable(ip: str, port: int, timeout_seconds: float) -> bool:
    """Check UnrealCV TCP reachability before constructing the looping client."""

    try:
        with socket.create_connection((ip, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(
        description="Profile SimWorld UnrealCV+/Gym-like step throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--dry-run", action="store_true", help="Expand and write cases without connecting to UE.")

    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--connect-timeout", type=float, default=2.0)

    parser.add_argument("--scenario-name", default="ad_hoc")
    parser.add_argument("--map-uri", default=None)
    parser.add_argument("--roads-json", default=None)
    parser.add_argument("--world-json", default=None)
    parser.add_argument("--ue-launch-profile", default="rendered")
    parser.add_argument("--ue-launch-command", default=None)
    parser.add_argument("--traffic-profile", default=None)
    parser.add_argument("--navigation-graph-available", choices=["true", "false", "unknown"], default="unknown")

    parser.add_argument("--num-agents", default=None, help="Comma-separated agent counts, e.g. 1,2,4,8.")
    parser.add_argument("--observation-profiles", default=None, help="Comma-separated observation profiles.")
    parser.add_argument("--resolutions", default=None, help="Comma-separated resolutions, e.g. 320x240,640x480.")
    parser.add_argument("--modes", default=None, help="Comma-separated simulator modes: sync,async.")
    parser.add_argument("--step-kinds", default=None, help="Comma-separated step kinds.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--tick-interval", type=float, default=0.05)
    parser.add_argument("--action-duration", type=float, default=0.05)

    parser.add_argument("--agent-blueprint", default="/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C")
    parser.add_argument("--spawn-origin", type=parse_origin, default=(0.0, 0.0, 600.0))
    parser.add_argument("--spawn-spacing", type=float, default=250.0)
    parser.add_argument("--destroy-on-exit", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--robot-asset", default="/Game/Robot_Dog/Blueprint/BP_SpotRobot.BP_SpotRobot_C")
    parser.add_argument("--robot-camera-ids", default="1", help="Comma-separated camera IDs for robot image profiles.")

    parser.add_argument("--gpu-verified", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-name", default=None)
    parser.add_argument("--rhi", default=None)

    parser.add_argument("--output-dir", default="runs/profiles")
    parser.add_argument("--output-prefix", default=None)
    return parser


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unspecified matrix args from the selected preset."""

    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    """Expand CLI args into benchmark cases."""

    args = apply_preset(args)
    num_agents = parse_int_csv(args.num_agents)
    observation_profiles = parse_csv(args.observation_profiles)
    resolutions = parse_resolutions(args.resolutions)
    modes = parse_csv(args.modes)
    step_kinds = parse_csv(args.step_kinds)

    ensure_choices(observation_profiles, OBSERVATION_PROFILES, "observation profiles")
    ensure_choices(modes, MODES, "modes")
    ensure_choices(step_kinds, STEP_KINDS, "step kinds")

    metadata = {
        "ip": args.ip,
        "port": args.port,
        "roads_json": args.roads_json,
        "world_json": args.world_json,
        "ue_launch_profile": args.ue_launch_profile,
        "ue_launch_command": args.ue_launch_command,
        "traffic_profile": args.traffic_profile,
        "navigation_graph_available": args.navigation_graph_available,
        "gpu_verified": args.gpu_verified,
        "gpu_name": args.gpu_name,
        "rhi": args.rhi,
        "preset": args.preset,
    }

    cases = []
    for agent_count, obs_profile, resolution, mode, step_kind in itertools.product(
        num_agents,
        observation_profiles,
        resolutions,
        modes,
        step_kinds,
    ):
        if obs_profile != "structured_only" and args.ue_launch_profile.lower() in {"nullrhi", "no_render", "no-render"}:
            continue
        cases.append(
            BenchmarkCase(
                scenario_name=args.scenario_name,
                map_uri=args.map_uri,
                num_agents=agent_count,
                observation_profile=obs_profile,
                resolution=resolution,
                mode=mode,
                step_kind=step_kind,
                tick_interval=args.tick_interval,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                metadata=metadata,
            )
        )
    return cases


def spawn_positions(count: int, origin: tuple[float, float, float], spacing: float) -> list[tuple[float, float, float]]:
    """Generate simple grid spawn positions."""

    positions = []
    width = max(1, min(4, count))
    for index in range(count):
        x = origin[0] + (index % width) * spacing
        y = origin[1] + (index // width) * spacing
        positions.append((x, y, origin[2]))
    return positions


def import_live_api() -> dict[str, Any]:
    """Import SimWorld live classes lazily so dry-run has fewer requirements."""

    from simworld.agent.humanoid import Humanoid
    from simworld.communicator.communicator import Communicator
    from simworld.communicator.unrealcv import UnrealCV
    from simworld.utils.vector import Vector

    return {
        "Humanoid": Humanoid,
        "Communicator": Communicator,
        "UnrealCV": UnrealCV,
        "Vector": Vector,
    }


def setup_humanoids(args: argparse.Namespace, case: BenchmarkCase, communicator: Any) -> tuple[list[Any], list[str]]:
    """Spawn humanoids for one live benchmark case."""

    api = import_live_api()
    Humanoid = api["Humanoid"]
    Vector = api["Vector"]

    agents = []
    actor_names = []
    positions = spawn_positions(case.num_agents, args.spawn_origin, args.spawn_spacing)
    for position in positions:
        agent = Humanoid(position=Vector(position[0], position[1]), direction=Vector(1, 0), communicator=communicator)
        communicator.spawn_agent(
            agent,
            name=None,
            position=position,
            model_path=args.agent_blueprint,
            type="humanoid",
        )
        communicator.humanoid_set_speed(agent.id, 200)
        agents.append(agent)
        actor_names.append(communicator.get_humanoid_name(agent.id))

    if case.observation_profile != "structured_only":
        for agent in agents:
            communicator.unrealcv.set_camera_resolution(agent.camera_id, case.resolution)
    return agents, actor_names


def setup_robots(args: argparse.Namespace, case: BenchmarkCase, unrealcv: Any) -> list[str]:
    """Spawn robot actors for robot profiling."""

    names = []
    for index, position in enumerate(spawn_positions(case.num_agents, args.spawn_origin, args.spawn_spacing)):
        name = f"BENCH_Robot_{index}"
        unrealcv.spawn_bp_asset(args.robot_asset, name)
        unrealcv.set_location(position, name)
        unrealcv.enable_controller(name, True)
        names.append(name)
    return names


def collect_observation(case: BenchmarkCase, communicator: Any, agents: list[Any], actor_names: list[str]) -> None:
    """Fetch one observation payload and discard it."""

    if actor_names:
        communicator.unrealcv.get_location_batch(actor_names)
        communicator.unrealcv.get_orientation_batch(actor_names)

    profile = case.observation_profile
    if profile == "structured_only":
        return

    cameras = [agent.camera_id for agent in agents]
    if profile == "shared_rgb":
        cameras = cameras[:1]

    for camera_id in cameras:
        if profile in {"rgb", "shared_rgb"}:
            communicator.get_camera_observation(camera_id, "lit", mode="direct")
        elif profile == "depth":
            communicator.get_camera_observation(camera_id, "depth", mode="direct")
        elif profile == "object_mask":
            communicator.get_camera_observation(camera_id, "object_mask", mode="direct")
        elif profile == "rgb_depth":
            communicator.get_camera_observation(camera_id, "lit", mode="direct")
            communicator.get_camera_observation(camera_id, "depth", mode="direct")


def collect_robot_observation(case: BenchmarkCase, communicator: Any, robot_camera_ids: list[int]) -> None:
    """Fetch robot camera observations when requested."""

    if case.observation_profile == "structured_only":
        return
    cameras = robot_camera_ids[:1] if case.observation_profile == "shared_rgb" else robot_camera_ids
    for camera_id in cameras:
        communicator.unrealcv.set_camera_resolution(camera_id, case.resolution)
        if case.observation_profile in {"rgb", "shared_rgb"}:
            communicator.get_camera_observation(camera_id, "lit", mode="direct")
        elif case.observation_profile == "depth":
            communicator.get_camera_observation(camera_id, "depth", mode="direct")
        elif case.observation_profile == "object_mask":
            communicator.get_camera_observation(camera_id, "object_mask", mode="direct")
        elif case.observation_profile == "rgb_depth":
            communicator.get_camera_observation(camera_id, "lit", mode="direct")
            communicator.get_camera_observation(camera_id, "depth", mode="direct")


def time_loop(steps: int, warmup_steps: int, step_fn: Callable[[], None]) -> list[float]:
    """Run warmup iterations, then return measured durations."""

    for _ in range(warmup_steps):
        step_fn()

    durations = []
    for _ in range(steps):
        started = time.perf_counter()
        step_fn()
        durations.append(time.perf_counter() - started)
    return durations


def run_humanoid_case(args: argparse.Namespace, case: BenchmarkCase) -> list[dict[str, Any]]:
    """Run a single live humanoid benchmark case."""

    if not backend_reachable(args.ip, args.port, args.connect_timeout):
        raise RuntimeError(f"UnrealCV backend is not reachable at {args.ip}:{args.port}")

    api = import_live_api()
    UnrealCV = api["UnrealCV"]
    Communicator = api["Communicator"]

    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=case.resolution)
    communicator = Communicator(unrealcv)
    agents: list[Any] = []
    actor_names: list[str] = []
    try:
        unrealcv.set_mode(case.mode, case.tick_interval)
        agents, actor_names = setup_humanoids(args, case, communicator)

        if case.step_kind == "command_latency":
            return run_command_latency_case(case, communicator, agents, actor_names)

        def step_once() -> None:
            if case.step_kind == "sensor_tick":
                collect_observation(case, communicator, agents, actor_names)
                if case.mode == "sync":
                    unrealcv.tick()
            elif case.step_kind == "gym_wrapper":
                for agent in agents:
                    communicator.humanoid_step_forward(agent.id, args.action_duration, direction=0)
                collect_observation(case, communicator, agents, actor_names)
            else:
                raise ValueError(f"Unsupported humanoid step kind: {case.step_kind}")

        durations = time_loop(case.steps, case.warmup_steps, step_once)
        return [benchmark_record(case, status="ok", durations=durations, extra={"gpu": nvidia_smi_metadata()})]
    finally:
        if args.destroy_on_exit:
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


def run_robot_case(args: argparse.Namespace, case: BenchmarkCase) -> list[dict[str, Any]]:
    """Run a single live robot benchmark case."""

    if not backend_reachable(args.ip, args.port, args.connect_timeout):
        raise RuntimeError(f"UnrealCV backend is not reachable at {args.ip}:{args.port}")

    api = import_live_api()
    UnrealCV = api["UnrealCV"]
    Communicator = api["Communicator"]
    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=case.resolution)
    communicator = Communicator(unrealcv)
    robot_names: list[str] = []
    camera_ids = parse_int_csv(args.robot_camera_ids)
    try:
        unrealcv.set_mode(case.mode, case.tick_interval)
        robot_names = setup_robots(args, case, unrealcv)

        def step_once() -> None:
            for robot_name in robot_names:
                unrealcv.dog_move(robot_name, [200, args.action_duration, 0])
            collect_robot_observation(case, communicator, camera_ids)
            if case.mode == "sync":
                unrealcv.tick()

        durations = time_loop(case.steps, case.warmup_steps, step_once)
        return [benchmark_record(case, status="ok", durations=durations, extra={"gpu": nvidia_smi_metadata()})]
    finally:
        if args.destroy_on_exit:
            for robot_name in robot_names:
                try:
                    unrealcv.destroy(robot_name)
                except Exception:
                    pass
            try:
                unrealcv.clean_garbage()
            except Exception:
                pass
        unrealcv.disconnect()


def run_command_latency_case(
    case: BenchmarkCase,
    communicator: Any,
    agents: list[Any],
    actor_names: list[str],
) -> list[dict[str, Any]]:
    """Measure individual command latencies for a live case."""

    probes: list[tuple[str, Callable[[], None]]] = []
    first_actor = actor_names[0]
    probes.append(("get_location", lambda: communicator.unrealcv.get_location(first_actor)))
    probes.append(("get_orientation", lambda: communicator.unrealcv.get_orientation(first_actor)))
    probes.append(("tick", lambda: communicator.unrealcv.tick()))
    if agents and case.observation_profile != "structured_only":
        first_camera = agents[0].camera_id
        probes.append(("get_image", lambda: communicator.get_camera_observation(first_camera, "lit", mode="direct")))

    records = []
    for probe_name, probe in probes:
        durations = time_loop(case.steps, case.warmup_steps, probe)
        probe_case = BenchmarkCase(
            **{
                **case.__dict__,
                "step_kind": f"command_latency:{probe_name}",
            }
        )
        records.append(benchmark_record(probe_case, status="ok", durations=durations, extra={"gpu": nvidia_smi_metadata()}))
    return records


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Build output paths for JSONL and CSV results."""

    prefix = args.output_prefix or args.scenario_name
    output_dir = Path(args.output_dir)
    return output_dir / f"{prefix}.jsonl", output_dir / f"{prefix}.csv"


def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        cases = build_cases(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    jsonl_path, csv_path = output_paths(args)
    all_records: list[dict[str, Any]] = []
    print(f"Prepared {len(cases)} benchmark case(s).")

    for index, case in enumerate(cases, start=1):
        label = (
            f"{case.scenario_name} agents={case.num_agents} obs={case.observation_profile} "
            f"res={resolution_label(case.resolution)} mode={case.mode} step={case.step_kind}"
        )
        print(f"[{index}/{len(cases)}] {label}")

        if args.dry_run:
            records = [benchmark_record(case, status="dry_run", extra={"gpu": nvidia_smi_metadata()})]
        else:
            try:
                if case.step_kind == "robot":
                    records = run_robot_case(args, case)
                else:
                    records = run_humanoid_case(args, case)
            except Exception as exc:
                records = [benchmark_record(case, status="error", error=str(exc), extra={"gpu": nvidia_smi_metadata()})]
                print(f"  error: {exc}", file=sys.stderr)

        write_jsonl(jsonl_path, records)
        write_csv_summary(csv_path, records)
        all_records.extend(records)

    print(f"Wrote JSONL: {jsonl_path}")
    print(f"Wrote CSV: {csv_path}")
    errors = sum(1 for record in all_records if record.get("status") == "error")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
