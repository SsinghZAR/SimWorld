"""Deterministic, agent-readable evidence for successful venue inspections.

The benchmark deliberately keeps the source-of-truth trait values in the
evaluator record while exposing a small natural-language evidence list to the
agent.  This module is pure Python: it has no Unreal or model dependency and
therefore remains deterministic in offline tests and replay tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


# Keep this ordering stable.  It is the public evidence ordering and is also
# the canonical decision-fact vocabulary used by scoring/social metrics.
DECISION_FACT_TRAITS: tuple[str, ...] = (
    "open",
    "reachable",
    "accessible",
    "food_drink",
    "quiet",
    "uncrowded",
    "shelter",
    "near_transit",
    "capacity",
)


@dataclass(frozen=True)
class InspectionEvidence:
    """Traceable evidence generated from one canonical fact mapping.

    ``by_trait`` is an evaluator-side trace showing exactly which sentence was
    generated for each source trait.  ``sentences`` is the concise ordered list
    safe to put in an agent observation.  The dataclass is frozen so callers
    cannot accidentally replace either record; both values are copied by the
    builder before construction.
    """

    by_trait: dict[str, str]
    sentences: tuple[str, ...]

    @property
    def internal_mapping(self) -> dict[str, str]:
        """Backward/ergonomic name for the traceable trait-to-sentence map."""

        return dict(self.by_trait)

    @property
    def public_evidence(self) -> list[str]:
        """Return a mutable JSON-friendly copy for observation assembly."""

        return list(self.sentences)

    @property
    def trait_to_sentence(self) -> dict[str, str]:
        """Explicit name for the traceability mapping."""

        return dict(self.by_trait)

    @property
    def evidence(self) -> list[str]:
        """Short alias for the public sentence list."""

        return list(self.sentences)

    def __iter__(self):
        """Allow ``mapping, evidence = build_inspection_evidence(...)``."""

        yield dict(self.by_trait)
        yield list(self.sentences)


def _bool_sentence(trait: str, value: bool) -> str:
    """Render a stable human-readable sentence for one boolean trait."""

    positive = {
        "open": "The entrance appears open.",
        "reachable": "The approach appears unobstructed.",
        "accessible": "A step-free route or ramp is visible.",
        "food_drink": "A menu or food-and-drink service cue is visible.",
        "quiet": "The surrounding area appears calm.",
        "uncrowded": "The surrounding area appears uncrowded.",
        "shelter": "Covered or indoor shelter is visible.",
        "near_transit": "A nearby public-transit cue is visible.",
    }
    negative = {
        "open": "The entrance appears closed.",
        "reachable": "The approach appears blocked.",
        "accessible": "Only steps are visible; no step-free route is apparent.",
        "food_drink": "No menu or food-and-drink service cue is visible.",
        "quiet": "The surrounding area appears noisy.",
        "uncrowded": "The surrounding area appears crowded.",
        "shelter": "No covered or indoor shelter is visible.",
        "near_transit": "No nearby public-transit cue is visible.",
    }
    sentences = positive if value else negative
    # A guarded fallback keeps the function total if this module is extended
    # with a new boolean trait before a specialised sentence is added.
    return sentences.get(trait, f"{trait.replace('_', ' ').capitalize()}: {'yes' if value else 'no'}.")


def _sentence_for_trait(trait: str, value: Any) -> str:
    """Render one trait value without exposing implementation thresholds."""

    if trait == "capacity":
        try:
            capacity = int(value)
        except (TypeError, ValueError):
            return f"Capacity information: {value}."
        return f"Visible seating or space suggests room for {capacity} people."
    return _bool_sentence(trait, bool(value))


def build_inspection_evidence(
    facts: Mapping[str, Any],
    *,
    venue_id: str | None = None,
    traits: Iterable[str] = DECISION_FACT_TRAITS,
) -> InspectionEvidence:
    """Build deterministic public evidence from canonical inspection facts.

    Only keys in ``traits`` and the exact decision-fact vocabulary are emitted;
    unknown/internal fields are ignored.  ``venue_id`` is accepted for callers
    that want to label their own surrounding record, but is intentionally not
    interpolated into sentences so evidence remains compact and stable.
    """

    del venue_id  # The public sentence itself should not depend on an id alias.
    allowed = set(DECISION_FACT_TRAITS)
    by_trait: dict[str, str] = {}
    for trait in traits:
        if trait not in allowed or trait not in facts:
            continue
        by_trait[trait] = _sentence_for_trait(trait, facts[trait])
    return InspectionEvidence(by_trait=by_trait, sentences=tuple(by_trait.values()))


__all__ = ["DECISION_FACT_TRAITS", "InspectionEvidence", "build_inspection_evidence"]
