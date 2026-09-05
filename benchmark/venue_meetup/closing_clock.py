"""Deterministic shop-closing clock for Venue Meetup episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_SHOPS_CLOSE_AT = "18:00"
DEFAULT_ACTION_MINUTES = 1


def parse_clock_time(value: str) -> int:
    """Parse a same-day ``HH:MM`` clock time into minutes after midnight."""

    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"shops_close_at must use 24-hour HH:MM format, got {value!r}")
    hour, minute = (int(part) for part in parts)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"shops_close_at must be a valid 24-hour time, got {value!r}")
    return hour * 60 + minute


def format_clock_time(minutes_after_midnight: int) -> str:
    """Format same-day minutes after midnight as ``HH:MM``."""

    hour, minute = divmod(int(minutes_after_midnight), 60)
    return f"{hour:02d}:{minute:02d}"


@dataclass(frozen=True)
class ClosingClock:
    """A shared clock where every synchronized action turn has a fixed cost."""

    max_turns: int
    shops_close_at: str = DEFAULT_SHOPS_CLOSE_AT
    action_minutes: int = DEFAULT_ACTION_MINUTES

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {self.max_turns}")
        if self.action_minutes <= 0:
            raise ValueError(f"action_minutes must be positive, got {self.action_minutes}")
        closing_minute = parse_clock_time(self.shops_close_at)
        if self.max_turns * self.action_minutes > closing_minute:
            raise ValueError(
                "the configured action budget starts before midnight; reduce max_turns/action_minutes "
                f"or use a later shops_close_at (got {self.max_turns} x {self.action_minutes} minutes "
                f"before {self.shops_close_at})"
            )

    @property
    def closing_minute(self) -> int:
        """Return closing time as minutes after midnight."""

        return parse_clock_time(self.shops_close_at)

    @property
    def starting_minute(self) -> int:
        """Return the deterministic episode start time."""

        return self.closing_minute - self.max_turns * self.action_minutes

    def snapshot(self, completed_turns: int) -> dict[str, Any]:
        """Return the public clock state after ``completed_turns`` actions."""

        completed = max(0, min(int(completed_turns), self.max_turns))
        current_minute = min(
            self.closing_minute,
            self.starting_minute + completed * self.action_minutes,
        )
        minutes_remaining = max(0, self.closing_minute - current_minute)
        turns_remaining = max(0, self.max_turns - completed)
        if minutes_remaining == 0:
            status = "closed"
        elif turns_remaining <= 10:
            status = "closing_soon"
        else:
            status = "open"
        time_unit = "minute" if self.action_minutes == 1 else "minutes"
        return {
            "current_time": format_clock_time(current_minute),
            "shops_close_at": format_clock_time(self.closing_minute),
            "minutes_remaining": minutes_remaining,
            "turns_remaining": turns_remaining,
            "action_cost_minutes": self.action_minutes,
            "status": status,
            "rule": (
                "Each agent chooses one action per synchronized turn; all actions consume "
                f"{self.action_minutes} simulated {time_unit}, and the shops close when the timer reaches zero."
            ),
        }

    def expired(self, completed_turns: int) -> bool:
        """Return whether the shared shop-closing deadline has been reached."""

        return int(completed_turns) >= self.max_turns
