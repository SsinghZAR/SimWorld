"""Tests for the deterministic Venue Meetup shop-closing clock."""

from __future__ import annotations

import pytest

from benchmark.venue_meetup.closing_clock import (ClosingClock,
                                                  format_clock_time,
                                                  parse_clock_time)


def test_100_turn_clock_runs_from_1620_to_1800() -> None:
    clock = ClosingClock(max_turns=100)

    assert clock.snapshot(0) == {
        "current_time": "16:20",
        "shops_close_at": "18:00",
        "minutes_remaining": 100,
        "turns_remaining": 100,
        "action_cost_minutes": 1,
        "status": "open",
        "rule": (
            "Each agent chooses one action per synchronized turn; all actions consume "
            "1 simulated minute, and the shops close when the timer reaches zero."
        ),
    }
    assert clock.snapshot(90)["status"] == "closing_soon"
    assert clock.snapshot(99)["current_time"] == "17:59"
    assert clock.snapshot(99)["minutes_remaining"] == 1
    assert clock.snapshot(100)["current_time"] == "18:00"
    assert clock.snapshot(100)["status"] == "closed"
    assert clock.expired(100)


def test_custom_action_duration_preserves_fixed_deadline() -> None:
    clock = ClosingClock(max_turns=12, shops_close_at="17:30", action_minutes=5)

    assert clock.snapshot(0)["current_time"] == "16:30"
    assert clock.snapshot(4)["current_time"] == "16:50"
    assert clock.snapshot(4)["minutes_remaining"] == 40
    assert clock.snapshot(12)["current_time"] == "17:30"


@pytest.mark.parametrize("value", ["18", "18:75", "24:00", "noon"])
def test_invalid_clock_times_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="24-hour"):
        parse_clock_time(value)


def test_clock_configuration_rejects_nonpositive_or_previous_day_budget() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        ClosingClock(max_turns=0)
    with pytest.raises(ValueError, match="action_minutes"):
        ClosingClock(max_turns=10, action_minutes=0)
    with pytest.raises(ValueError, match="before midnight"):
        ClosingClock(max_turns=200, shops_close_at="01:00")


def test_clock_time_round_trip() -> None:
    assert parse_clock_time("07:05") == 425
    assert format_clock_time(425) == "07:05"
