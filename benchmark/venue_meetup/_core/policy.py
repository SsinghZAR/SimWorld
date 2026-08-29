"""VLM and scripted policies for Venue Meetup."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised when only dry-run tooling is installed.
    cv2 = None

from benchmark.venue_meetup._core.action_space import VenueAction, VenueAgentTurn
from benchmark.venue_meetup.prompt import PromptMode, build_agent_prompt, build_system_prompt, normalize_prompt_mode
from simworld.llm.a2a_llm import A2ALLM


def resize_frame(frame: Any, max_width: int) -> Any:
    """Resize an image frame while preserving aspect ratio."""

    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    if cv2 is None:
        return frame
    scale = max_width / frame.shape[1]
    size = (max_width, max(1, int(frame.shape[0] * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def frame_for_model(frame_bgr: Any, max_width: int) -> Any:
    """Convert an UnrealCV BGR frame to RGB for VLM input."""

    frame_bgr = resize_frame(frame_bgr, max_width)
    if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
        if cv2 is not None:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_bgr[:, :, ::-1]
    return frame_bgr


class VenueMeetupPolicy:
    """VLM policy for venue-meetup agents."""

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
        reasoning: str | None = None,
        prompt_mode: PromptMode | str = "minimal",
    ):
        self.llm = A2ALLM(model_name=model_name, url=base_url, provider=provider)
        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.vision_max_width = vision_max_width
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning = reasoning
        self.prompt_mode = normalize_prompt_mode(prompt_mode)
        self.system_prompt = build_system_prompt(self.prompt_mode)

    def act(self, observation: dict[str, Any]) -> tuple[VenueAgentTurn, dict[str, Any]]:
        """Generate one agent turn plus a log record."""

        prompt = build_agent_prompt(observation, prompt_mode=self.prompt_mode)
        started = time.perf_counter()
        response, model_elapsed = self.llm.generate_instructions(
            self.system_prompt,
            prompt,
            images=[frame_for_model(observation["ego_view"], self.vision_max_width)],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            response_format=VenueAgentTurn,
            reasoning=self.reasoning,
        )
        decision_elapsed = time.perf_counter() - started
        turn = VenueAgentTurn.from_json(response)
        record = {
            "agent_id": observation.get("agent_id"),
            "provider": self.provider,
            "model": self.model_name,
            "prompt": prompt,
            "raw_response": response,
            "parsed_turn": turn.compact(),
            "reasoning": self.reasoning,
            "prompt_mode": self.prompt_mode,
            "model_elapsed_sec": round(float(model_elapsed), 3),
            "decision_elapsed_sec": round(decision_elapsed, 3),
        }
        return turn, record

    def act_all(
        self,
        observations: dict[str, dict[str, Any]],
        *,
        max_workers: int | None = None,
    ) -> tuple[dict[str, VenueAgentTurn], list[dict[str, Any]]]:
        """Generate turns for all agents concurrently."""

        max_workers = max_workers or len(observations)
        turns: dict[str, VenueAgentTurn] = {}
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.act, observation): agent_id for agent_id, observation in observations.items()}
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    turn, record = future.result()
                except Exception as exc:
                    turn = VenueAgentTurn(reasoning=f"policy error: {exc}")
                    record = {
                        "agent_id": agent_id,
                        "provider": self.provider,
                        "model": self.model_name,
                        "prompt_mode": self.prompt_mode,
                        "error": str(exc),
                        "parsed_turn": turn.compact(),
                    }
                turns[agent_id] = turn
                records.append(record)
        return turns, sorted(records, key=lambda item: item.get("agent_id", ""))


class ScriptedVenueSmokePolicy:
    """Deterministic smoke policy that drives both agents to a shared venue.

    It does not look at coordinates (agents never receive them); instead it
    leans on ``INSPECT`` to re-face the chosen venue and ``STEP_FORWARD`` to
    approach it, so a live run exercises inspect, communication, movement,
    convergence, and scoring end-to-end without any model calls.
    """

    def __init__(self, *, target_hint: str = "red_awning"):
        self._step = 0
        self.target_hint = target_hint

    def _target_for(self, observation: dict[str, Any]) -> str | None:
        """Pick a shared target venue id from the public candidate list."""

        candidates = [venue.get("venue_id") for venue in observation.get("candidate_venues", []) if venue.get("venue_id")]
        if not candidates:
            return None
        preferred = [venue_id for venue_id in candidates if self.target_hint in venue_id]
        return (preferred or candidates)[0]

    def act_all(self, observations: dict[str, dict[str, Any]], **_: Any) -> tuple[dict[str, VenueAgentTurn], list[dict[str, Any]]]:
        """Inspect, announce, then re-face and approach the shared venue."""

        turns: dict[str, VenueAgentTurn] = {}
        records: list[dict[str, Any]] = []
        for agent_id, observation in observations.items():
            target = self._target_for(observation)
            if self._step == 0:
                turn = VenueAgentTurn(choice=VenueAction.INSPECT.value, target_venue_id=target, message=f"{agent_id} inspecting {target}.")
            elif self._step == 1:
                turn = VenueAgentTurn(choice=VenueAction.COMMUNICATE.value, message=f"{agent_id}: let's meet at {target}.")
            elif self._step % 4 == 0:
                turn = VenueAgentTurn(choice=VenueAction.INSPECT.value, target_venue_id=target, reasoning="scripted re-face target")
            else:
                turn = VenueAgentTurn(choice=VenueAction.STEP_FORWARD.value, duration=0.5, direction=0, reasoning="scripted approach")
            turns[agent_id] = turn
            records.append({"agent_id": agent_id, "baseline": "scripted_smoke", "parsed_turn": turn.compact()})
        self._step += 1
        return turns, records


class ScriptedVenueNavPolicy:
    """Deterministic walk-mode demo policy: every agent NAVIGATEs to one venue.

    Used to exercise and demonstrate walk-mode locomotion (real route planning +
    physical walking around buildings) end-to-end without any model calls: both
    agents NAVIGATE to a shared venue each step, physically travel there, and
    converge. After arrival the repeated NAVIGATE is a no-op (already in region),
    then they INSPECT once to close the loop.
    """

    def __init__(self, *, target_hint: str = "red_awning"):
        self._step = 0
        self.target_hint = target_hint

    def _target_for(self, observation: dict[str, Any]) -> str | None:
        candidates = [venue.get("venue_id") for venue in observation.get("candidate_venues", []) if venue.get("venue_id")]
        if not candidates:
            return None
        preferred = [venue_id for venue_id in candidates if self.target_hint in venue_id]
        return (preferred or candidates)[0]

    def act_all(self, observations: dict[str, dict[str, Any]], **_: Any) -> tuple[dict[str, VenueAgentTurn], list[dict[str, Any]]]:
        """NAVIGATE (walk) to the shared venue, then INSPECT it on arrival."""

        turns: dict[str, VenueAgentTurn] = {}
        records: list[dict[str, Any]] = []
        for agent_id, observation in observations.items():
            target = self._target_for(observation)
            last = observation.get("last_action") or {}
            arrived = bool(last.get("arrived")) and last.get("venue_id") == target
            if arrived:
                turn = VenueAgentTurn(choice=VenueAction.INSPECT.value, target_venue_id=target, message=f"{agent_id} at {target}.")
            else:
                turn = VenueAgentTurn(choice=VenueAction.NAVIGATE.value, target_venue_id=target, message=f"{agent_id} walking to {target}.")
            turns[agent_id] = turn
            records.append({"agent_id": agent_id, "baseline": "scripted_nav", "parsed_turn": turn.compact()})
        self._step += 1
        return turns, records
