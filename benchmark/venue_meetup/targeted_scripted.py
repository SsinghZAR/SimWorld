"""Information-constrained smoke baseline; consumes only ordinary observations."""

from __future__ import annotations

import json
import re
from collections import defaultdict

from benchmark.venue_meetup._core.action_space import VenueAgentTurn
from benchmark.venue_meetup.inspection_evidence import build_inspection_evidence
from benchmark.venue_meetup.interactions import INTERACTION_KINDS
from benchmark.venue_meetup.varied_profiles import NEED_TEXT


def facts_from_evidence(sentences):
    """Decode deterministic evidence for this scripted baseline, not for the VLM."""

    facts = {}
    for key in (*NEED_TEXT, "open", "reachable"):
        if key == "capacity":
            for sentence in sentences:
                match = re.search(r"room for (\d+) people", sentence)
                if match:
                    facts[key] = int(match[1])
            continue
        for value in (False, True):
            canonical = build_inspection_evidence({key: value}).public_evidence
            if canonical and canonical[0] in sentences:
                facts[key] = value
    return facts


class TargetedScriptedPolicy:
    def __init__(self):
        self.announced = set()
        self.proposed = set()
        self.targets = {}
        self.rejected = defaultdict(set)

    def _act(self, agent, observation):
        own = [key for key, text in NEED_TEXT.items() if text in observation["private_constraint"]]
        if agent not in self.announced:
            self.announced.add(agent)
            return VenueAgentTurn(choice=4, message=json.dumps({"needs": own}))
        required = set(own) | {"open", "reachable"}
        proposal = None
        for message in observation.get("group_chat", []):
            try:
                payload = json.loads(message["content"])
            except (ValueError, KeyError):
                continue
            required.update(payload.get("needs", []))
            if payload.get("meet_at"):
                proposal = payload["meet_at"]
        guidance = {target["id"]: target for target in observation.get("navigation", {}).get("targets", [])}
        if proposal:
            return VenueAgentTurn(choice=0 if guidance.get(proposal, {}).get("arrived") else 5,
                                  target_venue_id=proposal)
        knowledge = {venue: facts_from_evidence(evidence)
                     for venue, evidence in observation.get("known_venue_evidence", {}).items()}
        for venue, facts in knowledge.items():
            if any(key in facts and (facts[key] < 2 if key == "capacity" else not facts[key]) for key in required):
                self.rejected[agent].add(venue)
            elif required <= facts.keys():
                if agent not in self.proposed:
                    self.proposed.add(agent)
                    return VenueAgentTurn(choice=4, message=json.dumps({"meet_at": venue, "facts": facts}))
                return VenueAgentTurn(choice=0)
        target = self.targets.get(agent)
        if not target or target in self.rejected[agent]:
            candidates = [venue["venue_id"] for venue in observation["candidate_venues"]
                          if venue.get("can_inspect", True) and venue["venue_id"] not in self.rejected[agent]]
            if not candidates:
                return VenueAgentTurn(choice=0)
            target = min(candidates, key=lambda ident: guidance.get(ident, {}).get("duration_ticks") or 999)
            self.targets[agent] = target
        if not guidance.get(target, {}).get("arrived"):
            return VenueAgentTurn(choice=5, target_venue_id=target)
        facts = knowledge.get(target, {})
        for kind in INTERACTION_KINDS:
            if (required - facts.keys()) & set(kind.traits):
                point = next((point for point in observation.get("nearby_interactables", [])
                              if point["venue_id"] == target and point["kind"] == kind.key), None)
                if point and point["visible"]:
                    return VenueAgentTurn(choice=3, target_venue_id=target,
                                          target_interactable_id=point["interaction_id"])
                return VenueAgentTurn(choice=2, angle=30, clockwise=True)
        return VenueAgentTurn(choice=0)

    def act_all(self, observations, **kwargs):
        turns = {agent: self._act(agent, observation) for agent, observation in observations.items()}
        return turns, [{"agent_id": agent, "baseline": "targeted_information_constrained",
                        "parsed_turn": turn.compact()} for agent, turn in turns.items()]
