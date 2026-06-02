#!/usr/bin/env python3
"""Smoke-test local OpenAI-compatible LLMs for structured SimWorld actions.

Examples:

    ollama serve

    .venv/bin/python experimentations/test_local_llm_actions.py \
      --base-url http://127.0.0.1:11434/v1 \
      --models phi3:mini,llama3:8b
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


def parse_csv(value: str) -> list[str]:
    """Parse a comma-separated string list."""

    return [item.strip() for item in value.split(",") if item.strip()]


def extract_json_object(text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Extract a JSON object from a model response."""

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


def validate_action(obj: dict[str, Any] | None) -> tuple[bool, dict[str, Any] | None]:
    """Validate an action using SimWorld's low-level action model."""

    if obj is None or "choice" not in obj:
        return False, None

    parsed = LowLevelActionSpace.from_json(obj)
    valid = parsed.choice.value in {0, 1, 2}
    return valid, parsed.model_dump(mode="json")


def run_model(
    model_name: str,
    base_url: str,
    scenarios: list[tuple[str, str]],
    json_mode: bool,
    max_tokens: int,
) -> dict[str, Any]:
    """Run a model over constrained structured-action scenarios."""

    llm = BaseLLM(model_name=model_name, url=base_url, provider="local")
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
        "json_mode": json_mode,
        "valid_actions": sum(1 for result in results if result["valid"]),
        "total": len(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--models", default="phi3:mini", help="Comma-separated model names.")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--no-json-mode", action="store_true", help="Do not request JSON mode.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    report = {
        "provider": "local",
        "models": [
            run_model(
                model_name=model_name,
                base_url=args.base_url,
                scenarios=DEFAULT_SCENARIOS,
                json_mode=not args.no_json_mode,
                max_tokens=args.max_tokens,
            )
            for model_name in parse_csv(args.models)
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
