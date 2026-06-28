#!/usr/bin/env python3
"""Smoke-test local OpenAI-compatible LLMs for structured SimWorld actions.

Examples:

    ollama serve

    .venv/bin/python experimentations/test_local_llm_actions.py \
      --base-url http://127.0.0.1:11434/v1 \
      --models phi3:mini,llama3:8b

    .venv/bin/python experimentations/test_local_llm_actions.py \
      --base-url http://127.0.0.1:11434/v1 \
      --skip-text \
      --vision-models qwen3-vl:8b

    .venv/bin/python experimentations/test_local_llm_actions.py \
      --provider minimax \
      --models MiniMax-M3 \
      --vision-models MiniMax-M3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simworld.llm.a2a_llm import A2ALLM  # noqa: E402
from simworld.llm.base_llm import BaseLLM  # noqa: E402
from simworld.local_planner.action_space import LowLevelActionSpace  # noqa: E402


DEFAULT_SYSTEM_PROMPT = """You are a SimWorld action planner.
Return ONLY valid JSON, no markdown and no prose.
Schema: {"choice": integer, "duration": number|null, "direction": integer|null, "angle": number|null, "clockwise": boolean|null, "reasoning": string}.
Valid choices: 0=DO_NOTHING, 1=STEP_FORWARD, 2=TURN_AROUND.
If moving toward a visible goal ahead choose STEP_FORWARD with duration 0.2 and direction 0.
If blocked and target is not ahead choose TURN_AROUND with angle 90 or 180 and clockwise true/false.
If already at target choose DO_NOTHING.
"""

DEFAULT_SCENARIOS = [
    ("goal_ahead", "Observation: goal is straight ahead 5 meters, no obstacle. Choose next action."),
    ("wall_left_goal", "Observation: wall directly ahead, goal is to the left. Choose next action."),
    ("goal_behind", "Observation: goal is behind the agent. Choose next action."),
    ("at_target", "Observation: already at target location. Choose next action."),
    ("back_clear", "Observation: obstacle ahead but clear path backward. Choose next action."),
]

DEFAULT_VISION_PROMPT = """Observation: the image shows a simple test scene with a green goal marker ahead.
The route is clear. Choose the next SimWorld low-level action."""


def strip_think_blocks(text: str | None) -> str | None:
    """Remove thinking blocks emitted by reasoning models before JSON parsing."""

    if text is None:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*<think>.*?(?=\{)", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def parse_csv(value: str) -> list[str]:
    """Parse a comma-separated string list."""

    return [item.strip() for item in value.split(",") if item.strip()]


def extract_json_object(text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Extract a JSON object from a model response."""

    text = strip_think_blocks(text)
    if not text:
        return None, "empty response"

    try:
        parsed = json.loads(text)
        return parsed, None if isinstance(parsed, dict) else "response is not a JSON object"
    except json.JSONDecodeError as first_error:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return None, f"no JSON object found: {first_error}"
        try:
            parsed = json.loads(match.group(0))
            return parsed, None if isinstance(parsed, dict) else "extracted JSON is not an object"
        except json.JSONDecodeError as second_error:
            return None, f"invalid extracted JSON: {second_error}"


def action_to_dict(action: LowLevelActionSpace) -> dict[str, Any]:
    """Return a pydantic action as a JSON-serializable dictionary."""

    if hasattr(action, "model_dump"):
        return action.model_dump(mode="json")
    return action.dict()


def validate_action(obj: dict[str, Any] | None) -> tuple[bool, dict[str, Any] | None]:
    """Validate an action using SimWorld's low-level action model."""

    if obj is None or "choice" not in obj:
        return False, None

    parsed = LowLevelActionSpace.from_json(obj)
    valid = parsed.choice.value in {0, 1, 2}
    return valid, action_to_dict(parsed)


def parse_action_response(response: Any) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Parse a model response into a validated action payload."""

    if isinstance(response, dict):
        obj, json_error = response, None
    else:
        obj, json_error = extract_json_object(response)
    valid, parsed_action = validate_action(obj)
    return valid, parsed_action, json_error


def resize_image(image: Any, max_width: int) -> Any:
    """Resize an RGB image while preserving aspect ratio."""

    if max_width <= 0 or image.shape[1] <= max_width:
        return image

    import cv2

    scale = max_width / image.shape[1]
    size = (max_width, max(1, int(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def build_vision_test_image(max_width: int) -> Any:
    """Create a tiny synthetic RGB scene for VLM endpoint smoke tests."""

    import cv2
    import numpy as np

    image = np.full((240, 320, 3), 42, dtype=np.uint8)
    cv2.rectangle(image, (120, 80), (200, 220), (80, 80, 80), -1)
    cv2.circle(image, (160, 68), 18, (0, 200, 0), -1)
    cv2.arrowedLine(image, (160, 210), (160, 95), (220, 220, 220), 4)
    cv2.putText(image, "GOAL", (124, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    return resize_image(image, max_width)


def load_vision_image(path: Path, max_width: int) -> Any:
    """Load a user-provided image as RGB for VLM endpoint smoke tests."""

    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read vision image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return resize_image(image, max_width)


def run_model(
    model_name: str,
    base_url: str,
    provider: str,
    scenarios: list[tuple[str, str]],
    json_mode: bool,
    max_tokens: int,
) -> dict[str, Any]:
    """Run a model over constrained structured-action scenarios."""

    llm = BaseLLM(model_name=model_name, url=base_url, provider=provider)
    results = []

    for case_name, user_prompt in scenarios:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response, elapsed = llm.generate_text(
            DEFAULT_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            **kwargs,
        )
        obj, json_error = extract_json_object(response)
        valid, parsed_action = validate_action(obj)
        results.append(
            {
                "case": case_name,
                "valid": valid,
                "elapsed_sec": round(elapsed, 3),
                "json_error": json_error,
                "json": obj,
                "parsed_action": parsed_action,
                "raw": response,
            }
        )

    return {
        "model": model_name,
        "base_url": base_url,
        "provider": provider,
        "json_mode": json_mode,
        "valid_actions": sum(1 for result in results if result["valid"]),
        "total": len(results),
        "results": results,
    }


def run_vision_model(
    model_name: str,
    base_url: str,
    provider: str,
    image: Any,
    max_tokens: int,
) -> dict[str, Any]:
    """Run a VLM over one constrained structured-action scenario."""

    llm = A2ALLM(model_name=model_name, url=base_url, provider=provider)
    response, elapsed = llm.generate_instructions(
        DEFAULT_SYSTEM_PROMPT,
        DEFAULT_VISION_PROMPT,
        images=[image],
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        response_format=LowLevelActionSpace,
    )
    valid, parsed_action, json_error = parse_action_response(response)
    return {
        "model": model_name,
        "base_url": base_url,
        "provider": provider,
        "valid": valid,
        "elapsed_sec": round(elapsed, 3),
        "json_error": json_error,
        "parsed_action": parsed_action,
        "raw": response,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["local", "minimax"], default="local")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to local Ollama or MiniMax.")
    parser.add_argument("--models", help="Comma-separated model names.")
    parser.add_argument("--vision-models", help="Comma-separated vision model names to test.")
    parser.add_argument("--vision-image", type=Path, help="Optional image for vision model preflight.")
    parser.add_argument("--vision-max-width", type=int, default=640, help="Resize preflight image to this width.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--no-json-mode", action="store_true", help="Do not request JSON mode.")
    parser.add_argument("--skip-text", action="store_true", help="Only run vision model preflight.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.base_url is None:
        args.base_url = "https://api.minimax.io/v1" if args.provider == "minimax" else "http://127.0.0.1:11434/v1"
    if args.models is None:
        args.models = "MiniMax-M3" if args.provider == "minimax" else "phi3:mini"

    vision_image = None
    if args.vision_models:
        vision_image = (
            load_vision_image(args.vision_image, args.vision_max_width)
            if args.vision_image
            else build_vision_test_image(args.vision_max_width)
        )

    report = {
        "provider": args.provider,
        "text_models": [] if args.skip_text else [
            run_model(
                model_name=model_name,
                base_url=args.base_url,
                provider=args.provider,
                scenarios=DEFAULT_SCENARIOS,
                json_mode=not args.no_json_mode,
                max_tokens=args.max_tokens,
            )
            for model_name in parse_csv(args.models)
        ],
        "vision_models": [
            run_vision_model(
                model_name=model_name,
                base_url=args.base_url,
                provider=args.provider,
                image=vision_image,
                max_tokens=args.max_tokens,
            )
            for model_name in parse_csv(args.vision_models or "")
        ],
    }

    text = json.dumps(report, indent=2)
    print(text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
