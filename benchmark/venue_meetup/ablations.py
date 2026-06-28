"""Ablation configuration for Venue Meetup."""

from __future__ import annotations

ABLATIONS = {
    "main": {},
    "no_communication": {"no_communication": True},
    "no_coarse_map": {"no_coarse_map": True},
    "shared_constraints": {"shared_constraints": True},
    "full_shared_information": {"full_shared_information": True},
}


def ablation_kwargs(name: str) -> dict[str, bool]:
    """Return VenueMeetupEnv keyword overrides for an ablation."""

    if name not in ABLATIONS:
        raise KeyError(f"Unknown ablation {name!r}; expected one of {sorted(ABLATIONS)}")
    return dict(ABLATIONS[name])


def minimal_ablation_names() -> list[str]:
    """Return the V0 ablation order."""

    return ["main", "no_communication", "no_coarse_map", "shared_constraints", "full_shared_information"]
