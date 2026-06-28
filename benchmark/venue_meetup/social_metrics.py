"""Process metrics for the social information-sharing behaviors we measure.

These score the *communication process*, not just the outcome (see notes.md
section 7). They are computed post-hoc from a logged trajectory plus the hidden
scenario, so they run offline with no UE.

Key behaviors operationalized:
- sharing_completeness: of the decision-relevant facts an agent learned first-hand,
  what fraction did it actually communicate?
- other_regarding_ratio: of the facts an agent shared, what fraction were relevant
  to a *teammate's* need vs. its own? (audience-aware / theory-of-mind sharing)
- redundancy: fraction of messages that conveyed nothing new.
- necessity (must_pool): could an agent have identified the group optimum alone?
- uptake: did the agent that depends on a report actually reach the optimum?

NOTE: message content is free text, so fact extraction is heuristic (venue alias +
trait-keyword co-mention). It is intentionally conservative; treat the sharing and
other-regarding numbers as approximate. The structural metrics (must_pool,
solvable_alone, optimum) are exact.
"""

from __future__ import annotations

from typing import Any

from benchmark.venue_meetup.scenario import Requirement, Scenario, Venue
from benchmark.venue_meetup.scoring import satisfies, score_venue

# Trait -> surface keywords used to detect that a message talks about that trait.
TRAIT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "accessible": ("accessible", "accessibility", "step-free", "step free", "stairs", "stair", "wheelchair", "ramp"),
    "food_drink": ("food", "drink", "coffee", "menu", "eat", "snack", "bar", "cafe", "café"),
    "quiet": ("quiet", "calm", "noisy", "noise", "loud"),
    "open": ("open", "closed", "shut"),
    "uncrowded": ("crowd", "crowded", "busy", "packed", "empty"),
    "shelter": ("shelter", "indoor", "covered", "roof", "sheltered"),
    "near_transit": ("transit", "station", "metro", "bus", "train", "subway"),
}


def _venue_aliases(venue: Venue) -> list[str]:
    """Lowercased surface forms that plausibly refer to a venue in chat."""

    aliases = {venue.venue_id.lower(), venue.venue_type.lower().replace("_", " ")}
    tokens = [t for t in venue.venue_id.lower().split("_") if t and t != "venue"]
    if tokens:
        aliases.add(" ".join(tokens))
        # Last token is usually the type-ish noun (cafe/market/hotel/hall).
        aliases.add(tokens[-1])
    return [a for a in aliases if len(a) >= 3]


def _mentioned_venues(text: str, venues: list[Venue]) -> set[str]:
    """Venue ids plausibly referenced by a message."""

    low = text.lower()
    hits = set()
    for venue in venues:
        if any(alias in low for alias in _venue_aliases(venue)):
            hits.add(venue.venue_id)
    return hits


def _mentioned_traits(text: str) -> set[str]:
    """Trait keys whose keywords appear in a message."""

    low = text.lower()
    return {trait for trait, words in TRAIT_KEYWORDS.items() if any(w in low for w in words)}


def _message_fact_pairs(text: str, venues: list[Venue]) -> set[tuple[str, str]]:
    """Heuristic (venue_id, trait) pairs a message conveys (co-mention)."""

    venue_hits = _mentioned_venues(text, venues)
    trait_hits = _mentioned_traits(text)
    return {(venue_id, trait) for venue_id in venue_hits for trait in trait_hits}


def _agent_relevant_traits(scenario: Scenario) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Return (all relevant traits, own-needs per agent, partner-needs per agent)."""

    own = {a.agent_id: set(a.private_requirement_keys) for a in scenario.agents}
    relevant: set[str] = set()
    for keys in own.values():
        relevant |= keys
    partner = {a.agent_id: (relevant - own[a.agent_id]) for a in scenario.agents}
    return relevant, own, partner


def _observed_facts(trajectory: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Reconstruct each agent's first-hand inspected facts from the action log."""

    observed: dict[str, dict[str, dict[str, Any]]] = {}
    for step in trajectory:
        actions = (step.get("info") or {}).get("actions") or {}
        for agent_id, result in actions.items():
            if not isinstance(result, dict):
                continue
            if str(result.get("result", "")).startswith("INSPECT_OK") and isinstance(result.get("facts"), dict):
                observed.setdefault(agent_id, {})[result["venue_id"]] = result["facts"]
    return observed


def _transcript(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the cumulative message transcript (sender, content, step)."""

    for step in reversed(trajectory):
        comms = (step.get("info") or {}).get("comms") or {}
        transcript = comms.get("transcript")
        if transcript:
            return transcript
    return []


def compute_social_metrics(scenario: Scenario, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the social process metrics for one episode."""

    venues = list(scenario.venues)
    relevant, own_needs, partner_needs = _agent_relevant_traits(scenario)
    observed = _observed_facts(trajectory)
    transcript = _transcript(trajectory)

    # The unique group-feasible optimum (exact, from hidden scoring).
    feasible = [v for v in venues if not score_venue(v, scenario).hard_failures]
    optimum_id = feasible[0].venue_id if len(feasible) == 1 else None

    # Per-agent shared pairs + redundancy, scanning messages in order.
    shared_pairs: dict[str, set[tuple[str, str]]] = {a.agent_id: set() for a in scenario.agents}
    seen_pairs: set[tuple[str, str]] = set()
    msg_count: dict[str, int] = {a.agent_id: 0 for a in scenario.agents}
    redundant_msgs: dict[str, int] = {a.agent_id: 0 for a in scenario.agents}
    optimum_cross_communicated = False

    for message in sorted(transcript, key=lambda m: m.get("step", 0)):
        sender = message.get("sender")
        if sender not in shared_pairs:
            continue
        msg_count[sender] += 1
        pairs = _message_fact_pairs(str(message.get("content", "")), venues)
        relevant_pairs = {(v, t) for (v, t) in pairs if t in relevant}
        shared_pairs[sender] |= relevant_pairs
        if relevant_pairs and relevant_pairs.issubset(seen_pairs):
            redundant_msgs[sender] += 1
        seen_pairs |= relevant_pairs
        # Did anyone communicate the optimum's partner-decisive fact?
        if optimum_id is not None:
            for (v, t) in relevant_pairs:
                if v == optimum_id and t in partner_needs.get(sender, set()):
                    optimum_cross_communicated = True

    per_agent: dict[str, Any] = {}
    for agent in scenario.agents:
        aid = agent.agent_id
        observed_relevant = {
            (vid, trait)
            for vid, facts in observed.get(aid, {}).items()
            for trait in relevant
            if trait in facts
        }
        shared_relevant = shared_pairs[aid]
        shared_in_observed = shared_relevant & observed_relevant
        completeness = (len(shared_in_observed) / len(observed_relevant)) if observed_relevant else None
        partner_shared = {(v, t) for (v, t) in shared_relevant if t in partner_needs[aid]}
        other_regarding = (len(partner_shared) / len(shared_relevant)) if shared_relevant else None
        # must_pool: can't solve alone if own zone has no feasible venue, or >=2
        # zone venues tie on the agent's own need (cannot disambiguate solo).
        zone_venues = [v for v in venues if v.zone_id == agent.zone_id] if agent.zone_id else venues
        zone_feasible = [v for v in zone_venues if not score_venue(v, scenario).hard_failures]
        own_ok = [
            v
            for v in zone_venues
            if all(satisfies(v, Requirement(key=k, weight=1.0), scenario) for k in own_needs[aid])
        ]
        must_pool = (len(zone_feasible) == 0) or (len(own_ok) >= 2)
        per_agent[aid] = {
            "zone_id": agent.zone_id,
            "own_needs": sorted(own_needs[aid]),
            "partner_needs": sorted(partner_needs[aid]),
            "venues_inspected": sorted(observed.get(aid, {}).keys()),
            "messages_sent": msg_count[aid],
            "redundant_messages": redundant_msgs[aid],
            "observed_relevant_facts": len(observed_relevant),
            "shared_relevant_facts": len(shared_relevant),
            "sharing_completeness": _round(completeness),
            "other_regarding_ratio": _round(other_regarding),
            "must_pool": must_pool,
        }

    # Uptake: did the agent(s) whose zone lacks the optimum end at the optimum?
    final_positions = {}
    if trajectory:
        final_positions = (trajectory[-1].get("info") or {}).get("positions_internal") or {}
    dependent_ids = [a.agent_id for a in scenario.agents if optimum_id and not any(
        v.zone_id == a.zone_id and v.venue_id == optimum_id for v in venues)]
    reached = _agents_in_optimum_region(scenario, optimum_id, final_positions)

    return {
        "optimum_venue_id": optimum_id,
        "optimum_cross_communicated": optimum_cross_communicated,
        "dependent_agents": dependent_ids,
        "dependent_reached_optimum": sorted(set(dependent_ids) & reached),
        "agents_at_optimum": sorted(reached),
        "per_agent": per_agent,
        "notes": "message fact-extraction is heuristic; sharing/other-regarding are approximate.",
    }


def _agents_in_optimum_region(scenario: Scenario, optimum_id: str | None, positions: dict[str, Any]) -> set[str]:
    if optimum_id is None:
        return set()
    venue = next((v for v in scenario.venues if v.venue_id == optimum_id), None)
    if venue is None:
        return set()
    reached = set()
    for agent_id, point in positions.items():
        try:
            if venue.region.contains((float(point[0]), float(point[1]))):
                reached.add(agent_id)
        except (TypeError, IndexError, ValueError):
            continue
    return reached


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
