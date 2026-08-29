"""Immutable prompt/environment conditions for Venue Meetup evaluations.

The four POC conditions are intentionally explicit.  Legacy ablation names
remain available for archived commands, but the runner's condition matrix uses
only :func:`poc_condition_names`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from benchmark.venue_meetup.prompt import PromptMode, normalize_prompt_mode

InfoPartition = Literal["none", "spatial"]

POC_CONDITION_NAMES: tuple[str, str, str, str] = (
    "main",
    "no_communication",
    "full_information",
    "cooperative_scaffold",
)

# These names are retained for compatibility with the original V0 CLI and
# output artifacts.  Their old Boolean semantics are kept where possible.
LEGACY_ABLATION_NAMES: tuple[str, str, str, str, str] = (
    "main",
    "no_communication",
    "no_coarse_map",
    "shared_constraints",
    "full_shared_information",
)

ALL_CONDITION_NAMES: tuple[str, ...] = tuple(dict.fromkeys((*POC_CONDITION_NAMES, *LEGACY_ABLATION_NAMES)))


@dataclass(frozen=True)
class ConditionSpec:
    """Deterministic condition configuration resolved for one evaluation case."""

    condition_id: str
    name: str | None = None
    prompt_mode: PromptMode | str = "minimal"
    info_partition: InfoPartition | str = "spatial"
    no_communication: bool = False
    no_coarse_map: bool = False
    full_shared_information: bool = False
    shared_constraints: bool = False

    def __post_init__(self) -> None:
        condition_id = str(self.condition_id).strip()
        if not condition_id:
            raise ValueError("condition_id must be non-empty")
        object.__setattr__(self, "condition_id", condition_id)
        if self.name is None:
            object.__setattr__(self, "name", condition_id)
        else:
            name = str(self.name).strip()
            if not name:
                raise ValueError("name must be non-empty when provided")
            object.__setattr__(self, "name", name)
        object.__setattr__(self, "prompt_mode", normalize_prompt_mode(self.prompt_mode))
        info_partition = str(self.info_partition).strip().lower()
        if info_partition not in ("none", "spatial"):
            raise ValueError(f"info_partition must be 'none' or 'spatial', got {self.info_partition!r}")
        object.__setattr__(self, "info_partition", info_partition)
        for field_name in (
            "no_communication",
            "no_coarse_map",
            "full_shared_information",
            "shared_constraints",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))

    def env_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by :class:`VenueMeetupEnv`."""

        return {
            "info_partition": self.info_partition,
            "no_communication": self.no_communication,
            "no_coarse_map": self.no_coarse_map,
            "full_shared_information": self.full_shared_information,
            "shared_constraints": self.shared_constraints,
        }

    def compact(self) -> dict[str, Any]:
        """Return a stable JSON-serializable condition representation."""

        return {
            "condition_id": self.condition_id,
            "name": self.name,
            "prompt_mode": self.prompt_mode,
            "info_partition": self.info_partition,
            "no_communication": self.no_communication,
            "no_coarse_map": self.no_coarse_map,
            "full_shared_information": self.full_shared_information,
            "shared_constraints": self.shared_constraints,
        }


def _canonical_condition(name: str) -> ConditionSpec:
    if name == "main":
        return ConditionSpec("main", prompt_mode="minimal", info_partition="spatial")
    if name == "no_communication":
        return ConditionSpec("no_communication", prompt_mode="minimal", info_partition="spatial", no_communication=True)
    if name == "full_information":
        return ConditionSpec(
            "full_information",
            prompt_mode="minimal",
            info_partition="spatial",
            full_shared_information=True,
            shared_constraints=True,
        )
    if name == "cooperative_scaffold":
        return ConditionSpec("cooperative_scaffold", prompt_mode="cooperative", info_partition="spatial")
    raise KeyError(name)


def resolve_condition(
    name: str | ConditionSpec = "main",
    *,
    prompt_mode: PromptMode | str | None = None,
    info_partition: InfoPartition | str | None = None,
) -> ConditionSpec:
    """Resolve a canonical or legacy condition with optional CLI overrides."""

    if isinstance(name, ConditionSpec):
        spec = name
    else:
        condition_name = str(name).strip()
        if condition_name in POC_CONDITION_NAMES:
            spec = _canonical_condition(condition_name)
        elif condition_name in LEGACY_ABLATION_NAMES:
            # Legacy names retain their original environment toggles and use
            # the new default spatial partition for deterministic POC runs.
            legacy_kwargs = {
                "no_communication": condition_name == "no_communication",
                "no_coarse_map": condition_name == "no_coarse_map",
                "shared_constraints": condition_name == "shared_constraints",
                "full_shared_information": condition_name == "full_shared_information",
            }
            spec = ConditionSpec(condition_name, prompt_mode="minimal", info_partition="spatial", **legacy_kwargs)
        else:
            expected = ", ".join(all_condition_names())
            raise KeyError(f"Unknown condition {name!r}; expected one of {expected}")

    if prompt_mode is not None or info_partition is not None:
        updates: dict[str, Any] = {}
        if prompt_mode is not None:
            updates["prompt_mode"] = normalize_prompt_mode(prompt_mode)
        if info_partition is not None:
            updates["info_partition"] = info_partition
        # Import locally to keep the module's public dataclass dependency
        # surface small and retain the frozen value semantics.
        from dataclasses import replace

        spec = replace(spec, **updates)
    return spec


def condition_spec(name: str | ConditionSpec = "main", **overrides: Any) -> ConditionSpec:
    """Backward-friendly alias for :func:`resolve_condition`."""

    return resolve_condition(name, **overrides)


def all_condition_names() -> list[str]:
    """Return the unique canonical-plus-legacy condition name union."""

    return list(ALL_CONDITION_NAMES)


def poc_condition_names() -> list[str]:
    """Return exactly the four conditions used by ``--ablation-matrix``."""

    return list(POC_CONDITION_NAMES)


def ablation_kwargs(name: str) -> dict[str, Any]:
    """Return complete environment overrides for any named condition."""

    if name not in ABLATIONS:
        expected = ", ".join(all_condition_names())
        raise KeyError(f"Unknown condition {name!r}; expected one of {expected}")
    return dict(ABLATIONS[name])


def minimal_ablation_names() -> list[str]:
    """Return the original V0 ablation order for compatibility."""

    return list(LEGACY_ABLATION_NAMES)


# Historical callers import ``ABLATIONS`` directly.  Build it from the same
# resolved specs used by the runner so no helper can silently disagree about a
# condition's environment flags or information partition.
CONDITIONS: dict[str, ConditionSpec] = {name: resolve_condition(name) for name in ALL_CONDITION_NAMES}
ABLATIONS: dict[str, dict[str, Any]] = {
    name: spec.env_kwargs() for name, spec in CONDITIONS.items()
}
