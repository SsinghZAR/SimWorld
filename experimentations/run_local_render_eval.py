#!/usr/bin/env python3
"""Run a short local-model SimWorld task and save UE-rendered artifacts.

Prerequisites:

    1. Start a local OpenAI-compatible model server, for example Ollama.
    2. Start the external SimWorld UE5 server on a map such as /Game/Maps/demo_1.umap.

Examples:

    python experimentations/run_local_render_eval.py \
      --agent-mode text \
      --base-url http://127.0.0.1:11434/v1 \
      --model qwen3:8b \
      --steps 5 \
      --resolution 640x360

    python experimentations/run_local_render_eval.py \
      --agent-mode vision \
      --base-url http://127.0.0.1:11434/v1 \
      --model qwen3-vl:8b \
      --steps 3 \
      --resolution 640x360 \
      --vision-max-width 640
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simworld.agent.humanoid import Humanoid  # noqa: E402
from simworld.communicator.communicator import Communicator  # noqa: E402
from simworld.communicator.unrealcv import UnrealCV  # noqa: E402
from simworld.config import Config  # noqa: E402
from simworld.llm.a2a_llm import A2ALLM  # noqa: E402
from simworld.llm.base_llm import BaseLLM  # noqa: E402
from simworld.local_planner.action_space import LowLevelAction, LowLevelActionSpace  # noqa: E402
from simworld.utils.vector import Vector  # noqa: E402


SYSTEM_PROMPT = """You are a SimWorld low-level action planner.
Return ONLY one valid JSON object, no markdown and no prose.
Use minimal thinking. Do not write step-by-step reasoning; emit the final JSON action immediately. /no_think
Schema: {"choice": integer, "duration": number|null, "direction": integer|null, "angle": number|null, "clockwise": boolean|null, "reasoning": string}.
Valid choices: 0=DO_NOTHING, 1=STEP_FORWARD, 2=TURN_AROUND.
Use STEP_FORWARD only when the target is roughly ahead and the route looks clear.
Use TURN_AROUND when the target is not ahead; clockwise=true means turn right, clockwise=false means turn left.
Use DO_NOTHING if the target is reached or if the observation is unusable."""


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT CLI values."""

    try:
        width, height = value.lower().split("x", 1)
        parsed = (int(width), int(height))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must look like 640x360") from exc

    if parsed[0] <= 0 or parsed[1] <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return parsed


def parse_triplet(value: str) -> tuple[float, float, float]:
    """Parse comma-separated x,y,z coordinates."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("position must look like x,y,z")
    return tuple(float(part) for part in parts)


def parse_pair(value: str) -> tuple[float, float]:
    """Parse comma-separated x,y coordinates."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("value must look like x,y")
    return tuple(float(part) for part in parts)


def vector_to_dict(vector: Vector) -> dict[str, float]:
    """Serialize a SimWorld vector."""

    return {"x": float(vector.x), "y": float(vector.y)}


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-180, 180]."""

    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def action_to_dict(action: LowLevelActionSpace) -> dict[str, Any]:
    """Return a pydantic action as a JSON-serializable dictionary."""

    if hasattr(action, "model_dump"):
        return action.model_dump(mode="json")
    return action.dict()


def extract_json_object(text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Extract a JSON object from a model response."""

    if not text:
        return None, "empty response"

    try:
        parsed = json.loads(text)
        return parsed, None if isinstance(parsed, dict) else "response is not a JSON object"
    except json.JSONDecodeError as first_error:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, f"no JSON object found: {first_error}"
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed, None if isinstance(parsed, dict) else "extracted JSON is not an object"
        except json.JSONDecodeError as second_error:
            return None, f"invalid extracted JSON: {second_error}"


def parse_action_response(response: Any) -> tuple[LowLevelActionSpace, dict[str, Any] | None, str | None]:
    """Parse a raw model response into a low-level action."""

    if isinstance(response, dict):
        obj, json_error = response, None
    else:
        obj, json_error = extract_json_object(response)

    action = LowLevelActionSpace.from_json(obj) if obj is not None else LowLevelActionSpace()
    return action, obj, json_error


def sanitize_action(
    action: LowLevelActionSpace,
    *,
    default_step_duration: float,
    max_step_duration: float,
    default_turn_angle: float,
    max_turn_angle: float,
    relative_angle: float,
) -> LowLevelActionSpace:
    """Clamp model output to a small safe action envelope."""

    if action.choice == LowLevelAction.STEP_FORWARD:
        duration = action.duration if action.duration and action.duration > 0 else default_step_duration
        duration = max(0.05, min(float(duration), max_step_duration))
        direction = int(action.direction) if action.direction in (0, 1) else 0
        return LowLevelActionSpace(
            choice=LowLevelAction.STEP_FORWARD,
            duration=duration,
            direction=direction,
            angle=None,
            clockwise=None,
            reasoning=action.reasoning,
        )

    if action.choice == LowLevelAction.TURN_AROUND:
        angle = action.angle if action.angle and action.angle > 0 else min(abs(relative_angle), default_turn_angle)
        angle = max(1.0, min(float(angle), max_turn_angle))
        clockwise = bool(action.clockwise) if action.clockwise is not None else relative_angle > 0
        return LowLevelActionSpace(
            choice=LowLevelAction.TURN_AROUND,
            duration=None,
            direction=None,
            angle=angle,
            clockwise=clockwise,
            reasoning=action.reasoning,
        )

    return LowLevelActionSpace(choice=LowLevelAction.DO_NOTHING, reasoning=action.reasoning)


def fallback_action_from_state(
    state: dict[str, Any],
    *,
    default_step_duration: float,
    default_turn_angle: float,
) -> LowLevelActionSpace:
    """Choose a small deterministic action when a model response is unusable."""

    relative_angle = float(state["relative_angle_deg"])
    if abs(relative_angle) > 15:
        return LowLevelActionSpace(
            choice=LowLevelAction.TURN_AROUND,
            angle=min(abs(relative_angle), default_turn_angle),
            clockwise=relative_angle > 0,
            reasoning="fallback: target is not ahead and model returned no valid action",
        )

    return LowLevelActionSpace(
        choice=LowLevelAction.STEP_FORWARD,
        duration=default_step_duration,
        direction=0,
        reasoning="fallback: target is ahead and model returned no valid action",
    )


def state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    """Measure position and yaw change between two state snapshots."""

    moved_cm = before["position"].distance(after["position"])
    yaw_delta = normalize_angle(float(after["yaw_deg"]) - float(before["yaw_deg"]))
    distance_delta = float(before["distance_to_target"]) - float(after["distance_to_target"])
    return {
        "moved_cm": moved_cm,
        "yaw_delta_deg": yaw_delta,
        "distance_delta_cm": distance_delta,
    }


def resize_frame(frame: Any, max_width: int) -> Any:
    """Resize a frame while preserving aspect ratio."""

    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame

    scale = max_width / frame.shape[1]
    size = (max_width, max(1, int(frame.shape[0] * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def capture_frame(communicator: Communicator, camera_id: int, viewmode: str, mode: str) -> Any:
    """Capture one camera frame from UE."""

    frame = communicator.get_camera_observation(camera_id, viewmode, mode=mode)
    if frame is None or getattr(frame, "size", 0) == 0:
        raise RuntimeError(f"Failed to capture camera {camera_id} frame")
    return frame


def frame_for_vision(frame_bgr: Any, max_width: int) -> Any:
    """Convert an UnrealCV BGR frame to a small RGB image for VLM input."""

    frame_bgr = resize_frame(frame_bgr, max_width)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def ollama_native_base_url(base_url: str) -> str:
    """Convert an OpenAI-compatible Ollama URL to the native Ollama API root."""

    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunparse(parsed._replace(path=path, params="", query="", fragment="")).rstrip("/")


def image_to_base64_jpeg(image_rgb: Any) -> str:
    """Encode an RGB image as raw base64 JPEG for Ollama's native API."""

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("Failed to encode image for Ollama")
    return base64.b64encode(buffer).decode("ascii")


def get_state(communicator: Communicator, agent: Humanoid, actor_name: str, target: Vector) -> dict[str, Any]:
    """Read actor state and derive navigation features."""

    location = communicator.unrealcv.get_location(actor_name)
    orientation = communicator.unrealcv.get_orientation(actor_name)
    position = Vector(float(location[0]), float(location[1]))
    yaw = float(orientation[1])
    direction = Vector(math.cos(math.radians(yaw)), math.sin(math.radians(yaw))).normalize()
    target_vector = target - position
    target_yaw = math.degrees(math.atan2(target_vector.y, target_vector.x))
    relative_angle = normalize_angle(target_yaw - yaw)
    distance = position.distance(target)

    agent.position = position
    agent.direction = yaw

    return {
        "position": position,
        "direction": direction,
        "yaw_deg": yaw,
        "target": target,
        "distance_to_target": distance,
        "relative_angle_deg": relative_angle,
    }


def state_to_prompt(state: dict[str, Any], last_action: str | None, success_distance: float) -> str:
    """Build the structured text prompt for the local model."""

    last_action_text = last_action or "none"
    return f"""Task: navigate the humanoid to the target.
Coordinates are in Unreal centimeters.
Current position: {vector_to_dict(state["position"])}
Current heading vector: {vector_to_dict(state["direction"])}
Current yaw degrees: {state["yaw_deg"]:.1f}
Target position: {vector_to_dict(state["target"])}
Distance to target: {state["distance_to_target"]:.1f}
Success distance: {success_distance:.1f}
Relative target angle degrees: {state["relative_angle_deg"]:.1f}
Positive relative angles mean the target is to the left; negative means it is to the right.
Previous action result: {last_action_text}
Choose one small next action."""


def sample_gpu() -> dict[str, Any] | None:
    """Collect a lightweight NVIDIA GPU memory snapshot when available."""

    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        timestamp, name, memory_total, memory_used, utilization = parts
        rows.append(
            {
                "timestamp": timestamp,
                "name": name,
                "memory_total_mb": int(memory_total),
                "memory_used_mb": int(memory_used),
                "utilization_gpu_percent": int(utilization),
            }
        )
    return {"gpus": rows} if rows else None


def safe_collision_snapshot(communicator: Communicator, agent: Humanoid) -> dict[str, int] | None:
    """Read humanoid collision counters if the backend supports them."""

    try:
        human, obj, building, vehicle = communicator.get_collision_number(agent.id)
    except Exception:
        return None
    return {
        "human": human,
        "object": obj,
        "building": building,
        "vehicle": vehicle,
    }


def generate_text_action(llm: BaseLLM, prompt: str, max_tokens: int, timeout: float) -> tuple[Any, float]:
    """Generate an action from a text-only local model."""

    response, elapsed = llm.generate_text(
        SYSTEM_PROMPT,
        prompt,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        timeout=timeout,
        response_format={"type": "json_object"},
    )
    if response is not None:
        return response, elapsed

    return llm.generate_text(
        SYSTEM_PROMPT,
        prompt,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        timeout=timeout,
    )


def generate_vision_action(
    llm: A2ALLM,
    prompt: str,
    frame_bgr: Any,
    vision_max_width: int,
    max_tokens: int,
) -> tuple[Any, float]:
    """Generate an action from a local VLM and one camera frame."""

    image = frame_for_vision(frame_bgr, vision_max_width)
    return llm.generate_instructions(
        SYSTEM_PROMPT,
        prompt,
        images=[image],
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        response_format=LowLevelActionSpace,
    )


def generate_ollama_native_action(
    model_name: str,
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    *,
    frame_bgr: Any | None = None,
    vision_max_width: int = 0,
) -> tuple[Any, float]:
    """Generate an action through Ollama's native API with thinking disabled."""

    started = time.time()
    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    if frame_bgr is not None:
        user_message["images"] = [image_to_base64_jpeg(frame_for_vision(frame_bgr, vision_max_width))]

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": max_tokens,
        },
    }
    request = Request(
        f"{ollama_native_base_url(base_url)}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    content = data.get("message", {}).get("content")
    return content, time.time() - started


def execute_action(
    communicator: Communicator,
    agent: Humanoid,
    action: LowLevelActionSpace,
    tick_count: int,
) -> str:
    """Apply one low-level action in UE and return a compact description."""

    if action.choice == LowLevelAction.STEP_FORWARD:
        duration = float(action.duration or 0.2)
        direction = int(action.direction or 0)
        communicator.humanoid_step_forward(agent.id, duration, direction=direction)
        description = f"STEP_FORWARD duration={duration:.2f} direction={direction}"
    elif action.choice == LowLevelAction.TURN_AROUND:
        angle = float(action.angle or 45)
        direction = "right" if action.clockwise else "left"
        communicator.humanoid_rotate(agent.id, angle, direction)
        description = f"TURN_AROUND angle={angle:.1f} direction={direction}"
    else:
        communicator.humanoid_stop(agent.id)
        description = "DO_NOTHING"

    for _ in range(tick_count):
        communicator.unrealcv.tick()
    return description


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


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON with a fallback for non-standard scalar types."""

    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON Lines."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse values to JSON-friendly metadata."""

    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-mode", choices=["text", "vision"], default="text")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, help="Optional SimWorld YAML config.")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--resolution", type=parse_resolution, default=parse_resolution("640x360"))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evals"))
    parser.add_argument("--run-name", help="Optional output subdirectory name.")
    parser.add_argument("--spawn-position", type=parse_triplet, default=parse_triplet("0,0,600"))
    parser.add_argument("--direction", type=parse_pair, default=parse_pair("1,0"))
    parser.add_argument("--target-x", type=float, default=1700)
    parser.add_argument("--target-y", type=float, default=-1700)
    parser.add_argument("--success-distance", type=float, default=200)
    parser.add_argument("--agent-blueprint", help="Override humanoid UE blueprint path.")
    parser.add_argument("--speed", type=float, default=200)
    parser.add_argument("--viewmode", default="lit", choices=["lit", "depth", "object_mask"])
    parser.add_argument("--camera-mode", default="direct", choices=["direct", "fast", "file"])
    parser.add_argument("--vision-max-width", type=int, default=640)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--model-timeout", type=float, default=30.0, help="Seconds to wait for each text model call.")
    parser.add_argument("--ollama-native", action="store_true", help="Use Ollama's native /api/chat with think=false.")
    parser.add_argument("--default-step-duration", type=float, default=0.2)
    parser.add_argument("--max-step-duration", type=float, default=0.6)
    parser.add_argument("--default-turn-angle", type=float, default=45)
    parser.add_argument("--max-turn-angle", type=float, default=180)
    parser.add_argument("--tick-count", type=int, default=1)
    parser.add_argument(
        "--no-fallback-on-invalid",
        action="store_true",
        help="Do not apply a deterministic movement fallback when the model returns invalid JSON.",
    )
    parser.add_argument("--no-clear-env", action="store_true", help="Do not clear generated actors before spawning.")
    return parser


def run(args: argparse.Namespace) -> Path:
    """Run the eval and return the artifact directory."""

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = args.output_dir / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config = Config(str(args.config)) if args.config else Config()
    agent_blueprint = args.agent_blueprint or config.get("user.model_path")

    gpu_samples: list[dict[str, Any]] = []
    first_gpu_sample = sample_gpu()
    if first_gpu_sample:
        gpu_samples.append({"phase": "before_connect", **first_gpu_sample})

    unrealcv = UnrealCV(port=args.port, ip=args.ip, resolution=args.resolution)
    atexit.register(unrealcv.disconnect)
    communicator = Communicator(unrealcv)

    connected_gpu_sample = sample_gpu()
    if connected_gpu_sample:
        gpu_samples.append({"phase": "after_ue_connect", **connected_gpu_sample})

    if not args.no_clear_env:
        communicator.clear_env(keep_roads=True)

    direction = Vector(*args.direction).normalize()
    agent = Humanoid(
        position=Vector(args.spawn_position[0], args.spawn_position[1]),
        direction=direction,
        communicator=communicator,
        config=config,
    )
    communicator.spawn_agent(
        agent,
        name=None,
        position=args.spawn_position,
        model_path=agent_blueprint,
        type="humanoid",
    )
    communicator.humanoid_set_speed(agent.id, args.speed)
    communicator.unrealcv.set_camera_resolution(agent.camera_id, args.resolution)

    actor_name = communicator.get_humanoid_name(agent.id)
    target = Vector(args.target_x, args.target_y)
    llm: BaseLLM | A2ALLM | None
    if args.ollama_native:
        llm = None
    elif args.agent_mode == "vision":
        llm = A2ALLM(model_name=args.model, url=args.base_url, provider="local")
    else:
        llm = BaseLLM(model_name=args.model, url=args.base_url, provider="local")

    model_gpu_sample = sample_gpu()
    if model_gpu_sample:
        gpu_samples.append({"phase": "after_model_init", **model_gpu_sample})

    frames = []
    trajectory = []
    model_responses = []
    last_action_result = None
    success = False

    initial_frame = capture_frame(communicator, agent.camera_id, args.viewmode, args.camera_mode)
    frames.append(initial_frame)
    print(f"Connected to UE. Spawned {actor_name}; writing artifacts to {artifact_dir}", flush=True)

    for step_index in range(args.steps):
        state_before = get_state(communicator, agent, actor_name, target)
        print(
            f"Step {step_index + 1}/{args.steps}: "
            f"distance={state_before['distance_to_target']:.1f}, "
            f"relative_angle={state_before['relative_angle_deg']:.1f}. "
            "Waiting for local model...",
            flush=True,
        )
        if state_before["distance_to_target"] <= args.success_distance:
            success = True
            break

        prompt = state_to_prompt(state_before, last_action_result, args.success_distance)
        decision_frame = frames[-1]
        started = time.perf_counter()
        if args.ollama_native:
            raw_response, model_elapsed = generate_ollama_native_action(
                args.model,
                args.base_url,
                prompt,
                args.max_tokens,
                args.model_timeout,
                frame_bgr=decision_frame if args.agent_mode == "vision" else None,
                vision_max_width=args.vision_max_width,
            )
        elif args.agent_mode == "vision":
            raw_response, model_elapsed = generate_vision_action(
                llm,
                prompt,
                decision_frame,
                args.vision_max_width,
                args.max_tokens,
            )
        else:
            raw_response, model_elapsed = generate_text_action(llm, prompt, args.max_tokens, args.model_timeout)
        decision_elapsed = time.perf_counter() - started

        parsed_action, raw_json, json_error = parse_action_response(raw_response)
        action = sanitize_action(
            parsed_action,
            default_step_duration=args.default_step_duration,
            max_step_duration=args.max_step_duration,
            default_turn_angle=args.default_turn_angle,
            max_turn_angle=args.max_turn_angle,
            relative_angle=state_before["relative_angle_deg"],
        )
        used_fallback = False
        if json_error and not args.no_fallback_on_invalid:
            action = fallback_action_from_state(
                state_before,
                default_step_duration=args.default_step_duration,
                default_turn_angle=args.default_turn_angle,
            )
            used_fallback = True

        source = "fallback" if used_fallback else "model"
        print(f"Step {step_index + 1}: applying {source} action {action}", flush=True)

        action_result = execute_action(communicator, agent, action, args.tick_count)
        last_action_result = action_result
        state_after = get_state(communicator, agent, actor_name, target)
        delta = state_delta(state_before, state_after)
        print(
            f"Step {step_index + 1}: moved={delta['moved_cm']:.1f}cm, "
            f"yaw_delta={delta['yaw_delta_deg']:.1f}deg, "
            f"distance_delta={delta['distance_delta_cm']:.1f}cm",
            flush=True,
        )
        collisions = safe_collision_snapshot(communicator, agent)
        frame_after = capture_frame(communicator, agent.camera_id, args.viewmode, args.camera_mode)
        frames.append(frame_after)

        step_record = {
            "step": step_index,
            "state_before": {
                "position": vector_to_dict(state_before["position"]),
                "direction": vector_to_dict(state_before["direction"]),
                "yaw_deg": state_before["yaw_deg"],
                "distance_to_target": state_before["distance_to_target"],
                "relative_angle_deg": state_before["relative_angle_deg"],
            },
            "action": action_to_dict(action),
            "action_source": source,
            "action_result": action_result,
            "movement": delta,
            "state_after": {
                "position": vector_to_dict(state_after["position"]),
                "direction": vector_to_dict(state_after["direction"]),
                "yaw_deg": state_after["yaw_deg"],
                "distance_to_target": state_after["distance_to_target"],
                "relative_angle_deg": state_after["relative_angle_deg"],
            },
            "collisions": collisions,
            "success": state_after["distance_to_target"] <= args.success_distance,
            "model_elapsed_sec": round(float(model_elapsed), 3),
            "decision_elapsed_sec": round(decision_elapsed, 3),
            "json_error": json_error,
        }
        trajectory.append(step_record)
        model_responses.append(
            {
                "step": step_index,
                "prompt": prompt,
                "raw_response": raw_response,
                "raw_json": raw_json,
                "parsed_action": action_to_dict(parsed_action),
                "sanitized_action": action_to_dict(action),
                "action_source": source,
                "json_error": json_error,
            }
        )

        active_gpu_sample = sample_gpu()
        if active_gpu_sample:
            gpu_samples.append({"phase": f"after_step_{step_index}", **active_gpu_sample})

        if step_record["success"]:
            success = True
            break

    video_path = artifact_dir / "video.mp4"
    save_video(frames, video_path, args.fps)
    write_json(artifact_dir / "trajectory.json", trajectory)
    write_jsonl(artifact_dir / "model_responses.jsonl", model_responses)

    final_state = get_state(communicator, agent, actor_name, target)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "success": success,
        "steps_requested": args.steps,
        "steps_run": len(trajectory),
        "final_distance_to_target": final_state["distance_to_target"],
        "target": vector_to_dict(target),
        "agent_mode": args.agent_mode,
        "model": args.model,
        "base_url": args.base_url,
        "resolution": {"width": args.resolution[0], "height": args.resolution[1]},
        "artifacts": {
            "video": str(video_path),
            "trajectory": str(artifact_dir / "trajectory.json"),
            "model_responses": str(artifact_dir / "model_responses.jsonl"),
            "metadata": str(artifact_dir / "metadata.json"),
        },
        "gpu_samples": gpu_samples,
        "args": serializable_args(args),
    }
    write_json(artifact_dir / "metadata.json", metadata)

    print(f"Saved eval artifacts to {artifact_dir}")
    print(f"Success: {success}; final distance: {final_state['distance_to_target']:.1f}")
    return artifact_dir


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
