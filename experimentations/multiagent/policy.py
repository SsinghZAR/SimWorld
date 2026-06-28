"""MiniMax-M3 policy wrappers for the multi-agent rendezvous experiments."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import cv2

from simworld.llm.a2a_llm import A2ALLM

from .action_space import MultiAgentTurn


RENDEZVOUS_SYSTEM_PROMPT = """You are one humanoid agent in a shared SimWorld rendezvous task.
You receive structured state, a teammate message inbox, and your current first-person camera image.
Choose one small movement action and optionally one short broadcast message.
Return ONLY one valid JSON object, no markdown and no prose.
Movement choices: 0=DO_NOTHING, 1=STEP_FORWARD, 2=TURN_AROUND.
Use STEP_FORWARD only when the route ahead appears clear. Use TURN_AROUND when the target or teammate direction is not ahead.
If you know the meeting target, help teammates by broadcasting concise coordinates or guidance when useful.
If you do not know the target, use your inbox and visible scene to infer where to go."""


def strip_frame(observation: dict[str, Any]) -> dict[str, Any]:
    """Return an observation copy without image arrays for prompts/logs."""

    return {key: value for key, value in observation.items() if key != "ego_view"}


def resize_frame(frame: Any, max_width: int) -> Any:
    """Resize an image frame while preserving aspect ratio."""

    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    size = (max_width, max(1, int(frame.shape[0] * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def frame_for_model(frame_bgr: Any, max_width: int) -> Any:
    """Convert an UnrealCV BGR frame to RGB for VLM input."""

    frame_bgr = resize_frame(frame_bgr, max_width)
    if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_bgr


class MiniMaxRendezvousPolicy:
    """LLM/VLM policy for one or more rendezvous agents."""

    def __init__(
        self,
        *,
        model_name: str = "MiniMax-M3",
        provider: str = "minimax",
        base_url: str | None = None,
        max_tokens: int = 2048,
        vision_max_width: int = 512,
        temperature: float = 0,
        top_p: float = 1.0,
    ):
        self.llm = A2ALLM(model_name=model_name, url=base_url, provider=provider)
        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.vision_max_width = vision_max_width
        self.temperature = temperature
        self.top_p = top_p

    def build_prompt(self, observation: dict[str, Any]) -> str:
        """Build the structured prompt for one agent."""

        prompt_payload = strip_frame(observation)
        return (
            "Task: all agents must rendezvous within the target radius.\n"
            "Coordinates are Unreal centimeters.\n"
            "Your observation JSON follows. Use the ego image to avoid obvious obstacles and orient yourself.\n"
            f"{json.dumps(prompt_payload, indent=2, default=str)}\n"
            "Return JSON keys: choice, duration, direction, angle, clockwise, message, reasoning.\n"
            "Keep message null unless it helps teammates coordinate."
        )

    def act(self, observation: dict[str, Any]) -> tuple[MultiAgentTurn, dict[str, Any]]:
        """Generate one agent turn plus a log record."""

        prompt = self.build_prompt(observation)
        started = time.perf_counter()
        response, model_elapsed = self.llm.generate_instructions(
            RENDEZVOUS_SYSTEM_PROMPT,
            prompt,
            images=[frame_for_model(observation["ego_view"], self.vision_max_width)],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            response_format=MultiAgentTurn,
        )
        decision_elapsed = time.perf_counter() - started
        turn = MultiAgentTurn.from_json(response)
        record = {
            "agent_id": observation.get("agent_id"),
            "provider": self.provider,
            "model": self.model_name,
            "prompt": prompt,
            "raw_response": response,
            "parsed_turn": turn.compact(),
            "model_elapsed_sec": round(float(model_elapsed), 3),
            "decision_elapsed_sec": round(decision_elapsed, 3),
        }
        return turn, record

    def act_all(
        self,
        observations: dict[str, dict[str, Any]],
        *,
        max_workers: int | None = None,
    ) -> tuple[dict[str, MultiAgentTurn], list[dict[str, Any]]]:
        """Generate turns for all agents concurrently."""

        max_workers = max_workers or len(observations)
        turns: dict[str, MultiAgentTurn] = {}
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.act, observation): agent_id
                for agent_id, observation in observations.items()
            }
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    turn, record = future.result()
                except Exception as exc:
                    turn = MultiAgentTurn(reasoning=f"policy error: {exc}")
                    record = {
                        "agent_id": agent_id,
                        "provider": self.provider,
                        "model": self.model_name,
                        "error": str(exc),
                        "parsed_turn": turn.compact(),
                    }
                turns[agent_id] = turn
                records.append(record)
        return turns, sorted(records, key=lambda item: item.get("agent_id", ""))
