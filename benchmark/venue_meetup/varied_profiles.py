"""Seeded private needs with explicit feasibility/decoy checks, without new assets."""

from __future__ import annotations

import random
from dataclasses import replace

from benchmark.venue_meetup.scenario import Requirement, Scenario, VenueProperties
from benchmark.venue_meetup.scoring import satisfies, score_venue

NEED_TEXT = {
    "accessible": "I require step-free access; stairs without an alternative are not acceptable.",
    "food_drink": "I require food or drink service at the meeting venue.",
    "shelter": "I require a covered or indoor meeting area.",
    "quiet": "I require a quiet meeting area; a noisy venue is not acceptable.",
    "uncrowded": "I require an uncrowded meeting area.",
    "capacity": "I require enough seating/space for both of us together (at least 2 people).",
}


def _properties(values: dict[str, bool], rng: random.Random) -> VenueProperties:
    return VenueProperties(
        open=True, reachable=True,
        capacity=rng.randint(2, 8) if values["capacity"] else 1,
        accessible=values["accessible"], shelter=values["shelter"],
        food_drink=values["food_drink"],
        quiet_score=rng.uniform(.7, .9) if values["quiet"] else rng.uniform(.1, .5),
        crowding_score=rng.uniform(.1, .4) if values["uncrowded"] else rng.uniform(.6, .9),
        near_transit=rng.choice((False, True)),
    )


def varied_profile(scenario: Scenario) -> Scenario:
    """Overlay one/two disjoint hard needs each, using already-authored zones.

    A distinct requirement per agent prevents identical private goals. Irrelevant
    attributes are randomized, not all maximized at the winning venue. The
    construction preserves local ambiguity, not information-theoretic necessity:
    a policy can still guess, which must be measured with no-communication runs.
    """

    if len(scenario.agents) != 2:
        raise ValueError("Targeted hidden profiles require exactly two agents")
    rng = random.Random(f"targeted-profile-v1:{scenario.seed}")
    agents = list(scenario.agents)
    if len({agent.zone_id for agent in agents}) != 2 or any(agent.zone_id is None for agent in agents):
        raise ValueError("Generate a spatial hidden profile before applying varied requirements")
    keys = list(NEED_TEXT)
    rng.shuffle(keys)
    first_count, second_count = rng.randint(1, 2), rng.randint(1, 2)
    needs = (keys[:first_count], keys[first_count:first_count + second_count])
    agents = [replace(agent, private_requirement_keys=list(own),
                      private_constraint="Hard requirements: " + " ".join(NEED_TEXT[key] for key in own))
              for agent, own in zip(agents, needs)]
    optimum_agent = rng.choice(agents)
    dependent = next(agent for agent in agents if agent != optimum_agent)
    own, partner = optimum_agent.private_requirement_keys, dependent.private_requirement_keys
    local = [venue for venue in scenario.venues if venue.zone_id == optimum_agent.zone_id]
    remote = [venue for venue in scenario.venues if venue.zone_id == dependent.zone_id]
    if min(len(local), len(remote)) < 2:
        raise ValueError("Each zone needs at least two venues")
    rng.shuffle(local)
    rng.shuffle(remote)
    optimum, decoy = local[:2]
    required = set(own + partner)
    properties = {}
    for venue in scenario.venues:
        values = {key: rng.choice((False, True)) for key in NEED_TEXT}
        if venue == optimum:
            values.update({key: True for key in required})
        elif venue == decoy:
            values.update({key: True for key in required})
            values[rng.choice(partner)] = False
        elif venue in remote[:2]:
            values.update({key: True for key in required})
            values[rng.choice(own)] = False
        else:
            values[rng.choice(sorted(required))] = False
        properties[venue.venue_id] = _properties(values, rng)
    venues = []
    for venue in scenario.venues:
        props = properties[venue.venue_id]
        venues.append(replace(venue, properties=props, entrances=[
            replace(entrance, status="accessible" if props.accessible else "stairs_only")
            for entrance in venue.entrances]))
    requirements = [Requirement(key=key, weight=1.0, hard=True,
                                description=NEED_TEXT.get(key, f"Venue must be {key}."))
                    for key in ["open", "reachable", *sorted(required)]]
    built = replace(scenario, scenario_id=f"{scenario.map_template_id}_targeted_seed_{scenario.seed}_n2",
                    agents=agents, venues=venues, requirements=requirements,
                    soft_weights={"quiet_threshold": .65, "crowding_threshold": .5})
    validate_varied_profile(built)
    return built


def validate_varied_profile(scenario: Scenario) -> None:
    feasible = [venue for venue in scenario.venues if not score_venue(venue, scenario).hard_failures]
    if len(feasible) != 1:
        raise ValueError("Targeted profile must have exactly one feasible venue")
    optimum = feasible[0]
    observer = next(agent for agent in scenario.agents if agent.zone_id == optimum.zone_id)
    partner = next(agent for agent in scenario.agents if agent.agent_id != observer.agent_id)

    def own_ok(venue, agent):
        return all(satisfies(venue, Requirement(key=key, weight=1), scenario)
                   for key in agent.private_requirement_keys)

    if not any(venue != optimum and venue.zone_id == observer.zone_id and own_ok(venue, observer)
               and not own_ok(venue, partner) for venue in scenario.venues):
        raise ValueError("Missing locally attractive decoy")
    if sum(venue.zone_id == partner.zone_id and own_ok(venue, partner) and not own_ok(venue, observer)
           for venue in scenario.venues) < 2:
        raise ValueError("Missing partner-zone traps")
