"""Pure shared-clock scheduling; no engine time, model time, or reward shaping."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from benchmark.venue_meetup.closing_clock import parse_clock_time


@dataclass(frozen=True)
class TimingConfig:
    starts_at: str = "17:30"
    shops_close_at: str = "18:00"
    tick_seconds: int = 30
    travel_metres_per_tick: float = 40.0

    def __post_init__(self) -> None:
        if not isinstance(self.tick_seconds, int) or isinstance(self.tick_seconds, bool):
            raise ValueError("Tick duration must be a whole number of seconds")
        if self.tick_seconds <= 0 or not math.isfinite(self.travel_metres_per_tick) or self.travel_metres_per_tick <= 0:
            raise ValueError("Tick duration and travel distance must be positive and finite")
        if self.budget_seconds <= 0 or self.budget_seconds % self.tick_seconds:
            raise ValueError("Closing must follow start on the same day by a whole number of ticks")

    @property
    def budget_seconds(self) -> int:
        return 60 * (parse_clock_time(self.shops_close_at) - parse_clock_time(self.starts_at))

    @property
    def max_ticks(self) -> int:
        return self.budget_seconds // self.tick_seconds

    def travel_ticks(self, distance_cm: float) -> int:
        if not math.isfinite(distance_cm) or distance_cm < 0:
            raise ValueError("Travel distance must be finite and nonnegative")
        return max(1, math.ceil(distance_cm / (100 * self.travel_metres_per_tick)))

    def snapshot(self, completed_ticks: int) -> dict[str, Any]:
        completed = max(0, min(completed_ticks, self.max_ticks))
        current = parse_clock_time(self.starts_at) * 60 + completed * self.tick_seconds
        return {
            "starts_at": self.starts_at,
            "current_time": f"{current // 3600:02d}:{current % 3600 // 60:02d}:{current % 60:02d}",
            "shops_close_at": self.shops_close_at,
            "minutes_remaining": (self.budget_seconds - completed * self.tick_seconds) / 60,
            "turns_remaining": self.max_ticks - completed,
            "ticks_remaining": self.max_ticks - completed,
            "tick_seconds": self.tick_seconds,
            "status": "closed" if completed == self.max_ticks else "open",
            "rule": "Independent actions advance on the same clock. Results arrive at completion. Completion at closing is allowed.",
        }

    def expired(self, completed_ticks: int) -> bool:
        return completed_ticks >= self.max_ticks


@dataclass
class PendingAction:
    turn: Any
    started_tick: int
    duration_ticks: int
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def completes_tick(self) -> int:
        return self.started_tick + self.duration_ticks

    def public(self, tick: int) -> dict[str, Any]:
        return {
            "status": "busy",
            "choice": self.turn.choice,
            "target_venue_id": self.turn.target_venue_id,
            "target_interactable_id": self.turn.target_interactable_id,
            "started_tick": self.started_tick,
            "duration_ticks": self.duration_ticks,
            "ticks_remaining": max(0, self.completes_tick - tick),
        }


class ActionScheduler:
    """Only idle agents accept actions; advance returns simultaneous completions."""

    def __init__(self, agent_ids: list[str], timing: TimingConfig):
        self.agent_ids = tuple(agent_ids)
        self.timing = timing
        self.tick = 0
        self.pending: dict[str, PendingAction] = {}

    @property
    def ready(self) -> tuple[str, ...]:
        if self.timing.expired(self.tick):
            return ()
        return tuple(agent for agent in self.agent_ids if agent not in self.pending)

    def start(self, agent: str, turn: Any, duration: int, payload: dict[str, Any] | None = None) -> None:
        if agent not in self.ready:
            raise ValueError(f"Agent {agent} cannot start an action now")
        if not isinstance(duration, int) or duration < 1:
            raise ValueError("Actions must occupy at least one whole tick")
        self.pending[agent] = PendingAction(turn, self.tick, duration, payload or {})

    def advance(self) -> dict[str, PendingAction]:
        if self.timing.expired(self.tick):
            raise RuntimeError("Cannot advance past closing")
        self.tick += 1
        completed = {agent: action for agent, action in self.pending.items() if action.completes_tick <= self.tick}
        for agent in completed:
            del self.pending[agent]
        return completed

    def activity(self, agent: str) -> dict[str, Any]:
        action = self.pending.get(agent)
        return action.public(self.tick) if action else {"status": "ready", "ticks_remaining": 0}
