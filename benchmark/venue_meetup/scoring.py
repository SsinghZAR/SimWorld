"""Hidden-metadata scoring for Venue Meetup episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.venue_meetup.scenario import Requirement, Scenario, Venue


@dataclass(frozen=True)
class VenueScore:
    """Score details for one venue."""

    venue_id: str
    score: float
    satisfied_weight: float
    total_weight: float
    satisfied: dict[str, bool]
    hard_failures: list[str]

    def compact(self) -> dict[str, Any]:
        """Return JSON-serializable score detail."""

        return {
            "venue_id": self.venue_id,
            "score": self.score,
            "satisfied_weight": self.satisfied_weight,
            "total_weight": self.total_weight,
            "satisfied": self.satisfied,
            "hard_failures": self.hard_failures,
        }


def venue_decision_facts(venue: Venue, soft_weights: dict[str, float]) -> dict[str, Any]:
    """Decision-relevant ground-truth traits for a venue, as booleans/ints.

    Shared by the env (structured inspect reveal) and the social metrics so both
    speak the same fact vocabulary. ``quiet``/``uncrowded`` are thresholded from
    their continuous scores using the scenario's soft weights.
    """

    props = venue.properties
    quiet_threshold = float(soft_weights.get("quiet_threshold", 0.65))
    crowding_threshold = float(soft_weights.get("crowding_threshold", 0.5))
    return {
        "open": bool(props.open),
        "reachable": bool(props.reachable),
        "accessible": bool(props.accessible),
        "food_drink": bool(props.food_drink),
        "quiet": bool(props.quiet_score >= quiet_threshold),
        "uncrowded": bool(props.crowding_score <= crowding_threshold),
        "shelter": bool(props.shelter),
        "near_transit": bool(props.near_transit),
        "capacity": int(props.capacity),
    }


def satisfies(venue: Venue, requirement: Requirement, scenario: Scenario, *, num_agents: int | None = None) -> bool:
    """Return whether a venue satisfies one requirement."""

    props = venue.properties
    if requirement.key == "open":
        return props.open
    if requirement.key == "reachable":
        return props.reachable
    if requirement.key == "accessible":
        return props.accessible
    if requirement.key == "shelter":
        return props.shelter
    if requirement.key == "food_drink":
        return props.food_drink
    if requirement.key == "quiet":
        return props.quiet_score >= float(scenario.soft_weights.get("quiet_threshold", 0.65))
    if requirement.key == "uncrowded":
        return props.crowding_score <= float(scenario.soft_weights.get("crowding_threshold", 0.5))
    if requirement.key == "capacity":
        return props.capacity >= (num_agents or len(scenario.agents))
    if requirement.key == "near_transit":
        return props.near_transit
    return False


def score_venue(venue: Venue, scenario: Scenario, *, num_agents: int | None = None) -> VenueScore:
    """Score one venue against scenario requirements."""

    satisfied: dict[str, bool] = {}
    hard_failures: list[str] = []
    satisfied_weight = 0.0
    total_weight = 0.0
    for requirement in scenario.requirements:
        total_weight += float(requirement.weight)
        is_satisfied = satisfies(venue, requirement, scenario, num_agents=num_agents)
        satisfied[requirement.key] = is_satisfied
        if is_satisfied:
            satisfied_weight += float(requirement.weight)
        elif requirement.hard:
            hard_failures.append(requirement.key)
    score = satisfied_weight / total_weight if total_weight else 0.0
    if hard_failures:
        # Hard failures should be visible in diagnostics and make the venue a
        # poor final choice, while preserving the weighted detail for analysis.
        score = min(score, 0.25)
    return VenueScore(
        venue_id=venue.venue_id,
        score=round(score, 4),
        satisfied_weight=round(satisfied_weight, 4),
        total_weight=round(total_weight, 4),
        satisfied=satisfied,
        hard_failures=hard_failures,
    )


def best_available_score(scenario: Scenario, *, num_agents: int | None = None) -> VenueScore:
    """Return the best venue in the generated scenario."""

    return max((score_venue(venue, scenario, num_agents=num_agents) for venue in scenario.venues), key=lambda score: score.score)


def final_venue_from_positions(scenario: Scenario, positions: dict[str, tuple[float, float]]) -> tuple[str | None, dict[str, list[str]]]:
    """Infer final venue from physical convergence regions."""

    by_venue: dict[str, list[str]] = {venue.venue_id: [] for venue in scenario.venues}
    for agent_id, point in positions.items():
        for venue in scenario.venues:
            if venue.region.contains(point):
                by_venue[venue.venue_id].append(agent_id)
                break
    final_venue_id = None
    if by_venue:
        final_venue_id, agents = max(by_venue.items(), key=lambda item: len(item[1]))
        if not agents:
            final_venue_id = None
    return final_venue_id, by_venue


def episode_score(
    scenario: Scenario,
    positions: dict[str, tuple[float, float]],
    *,
    inspected_venues: set[str] | None = None,
    message_count: int = 0,
    token_count: int = 0,
    timed_out: bool = False,
) -> dict[str, Any]:
    """Compute episode-level metrics."""

    inspected_venues = inspected_venues or set()
    final_venue_id, venue_agents = final_venue_from_positions(scenario, positions)
    best = best_available_score(scenario, num_agents=len(scenario.agents))
    final = score_venue(scenario.venue_by_id(final_venue_id), scenario, num_agents=len(scenario.agents)) if final_venue_id else None

    final_score = final.score if final else 0.0
    normalised = final_score / best.score if best.score else 0.0
    final_agent_count = len(venue_agents.get(final_venue_id, [])) if final_venue_id else 0
    convergence = final_agent_count / len(scenario.agents) if scenario.agents else 0.0
    return {
        "scenario_id": scenario.scenario_id,
        "map_template_id": scenario.map_template_id,
        "seed": scenario.seed,
        "num_agents": len(scenario.agents),
        "final_venue_id": final_venue_id,
        "final_venue_requirement_score": round(final_score, 4),
        "best_available_venue_id": best.venue_id,
        "best_available_venue_score": best.score,
        "normalised_best_available_score": round(normalised, 4),
        "arrival_score": round(convergence, 4),
        "episode_score": round(normalised * convergence, 4),
        "venue_agent_membership": venue_agents,
        "final_venue_detail": final.compact() if final else None,
        "best_available_detail": best.compact(),
        "venues_inspected": len(inspected_venues),
        "inspected_venue_ids": sorted(inspected_venues),
        "total_candidate_venues": len(scenario.venues),
        "message_count": message_count,
        "token_count": token_count,
        "timed_out": timed_out,
    }
