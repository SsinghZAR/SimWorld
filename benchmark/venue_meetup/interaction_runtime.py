"""UE adapter for physical, individually visible information panels."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import cv2
import numpy as np

from benchmark.venue_meetup.actions import count_mask_pixels
from benchmark.venue_meetup.building_catalog import asset_path
from benchmark.venue_meetup.inspection_evidence import build_inspection_evidence
from benchmark.venue_meetup.interactions import (
    INTERACTION_MIN_PIXELS,
    INTERACTION_RANGE_CM,
    InteractionPoint,
    interaction_points,
)


class InteractionRuntime:
    """Keep target geometry/masks and evidence provenance outside the env loop."""

    def __init__(self, env):
        self.env = env
        self.points = interaction_points(env.scenario)
        self.history: dict[str, list[dict[str, Any]]] = {agent: [] for agent in env.agent_ids}

    def spawn(self) -> None:
        uc = self.env.communicator.unrealcv
        for point in self.points.values():
            uc.spawn_bp_asset(asset_path("BP_Table_C"), point.actor_name)
            uc.set_location(point.position, point.actor_name)
            uc.set_orientation((90, point.yaw_deg, 0), point.actor_name)
            uc.set_scale(point.scale, point.actor_name)
            uc.set_color(point.actor_name, point.mask_color)
            uc.set_collision(point.actor_name, False)
            uc.set_movable(point.actor_name, False)

    def resolve(self, action) -> InteractionPoint | None:
        point = self.points.get(action.target_interactable_id)
        if point and action.target_venue_id not in (None, point.venue_id):
            return None
        return point

    def precheck(self, agent: str, action) -> str | None:
        point = self.resolve(action)
        if point is None:
            return "Unknown interaction target; select an interaction_id from nearby_interactables."
        venue = self.env.scenario.venue_by_id(point.venue_id)
        if not self.env._can_inspect_zone(agent, venue):
            return "This venue is outside your inspection area."
        position = self.env._actor_xy(self.env.get_agent_state(agent).actor_name)
        if not venue.region.contains(position) or math.dist(position, point.position[:2]) > INTERACTION_RANGE_CM:
            return "The interaction point is out of range."
        return None

    def inspect(self, agent: str, action) -> dict[str, Any]:
        error = self.precheck(agent, action)
        point = self.resolve(action)
        result = {"turn": action.compact(), "result": "INSPECT_FAILED",
                  "interaction_id": action.target_interactable_id,
                  "venue_id": point.venue_id if point else action.target_venue_id}
        if error:
            return {**result, "reason": error}
        mask = self.env._capture_frames("object_mask")[agent]
        pixels = count_mask_pixels(mask, point.mask_color)
        if pixels < INTERACTION_MIN_PIXELS:
            return {**result, "reason": "The selected information point is not visible from your current view.",
                    "mask_pixels_internal": pixels}
        venue = self.env.scenario.venue_by_id(point.venue_id)
        facts = {key: value for key, value in self.env._venue_facts(venue).items() if key in point.kind.traits}
        evidence = build_inspection_evidence(facts).public_evidence
        if point.kind.key == "hours":
            evidence.append(f"Posted closing time: {self.env.timing.shops_close_at}.")
        known = self.env.revealed_facts.setdefault(agent, {}).setdefault(point.venue_id, {})
        repeated = any(item["interaction_id"] == point.interaction_id for item in self.history[agent])
        known.update(deepcopy(facts))
        sentences = self.env.revealed_evidence.setdefault(agent, {}).setdefault(point.venue_id, [])
        for sentence in evidence:
            if sentence not in sentences:
                sentences.append(sentence)
        self.env.inspected_venues.add(point.venue_id)
        self.history[agent].append({"interaction_id": point.interaction_id, "venue_id": point.venue_id,
                                    "observed_tick": self.env.step_index, "evidence": list(evidence)})
        return {**result, "result": "INSPECT_OK", "facts": facts, "evidence": evidence,
                "mask_pixels_internal": pixels, "repeat_internal": repeated}

    def augment(self, observations: dict[str, dict[str, Any]]) -> None:
        """Label visible physical targets, without painting hidden answers into images."""

        nearby = {}
        for agent in observations:
            position = self.env._actor_xy(self.env.get_agent_state(agent).actor_name)
            nearby[agent] = [point for point in self.points.values()
                             if math.dist(position, point.position[:2]) <= INTERACTION_RANGE_CM]
        masks = self.env._capture_frames("object_mask") if any(nearby.values()) else {}
        for agent, observation in observations.items():
            records = []
            frame = observation["ego_view"].copy()
            for index, point in enumerate(nearby[agent]):
                visible = count_mask_pixels(masks[agent], point.mask_color) >= INTERACTION_MIN_PIXELS
                item = point.public()
                venue = self.env.scenario.venue_by_id(point.venue_id)
                item.update(visible=visible, permitted=self.env._can_inspect_zone(agent, venue))
                position = self.env._actor_xy(self.env.get_agent_state(agent).actor_name)
                bearing = math.degrees(math.atan2(point.position[1] - position[1], point.position[0] - position[0]))
                yaw = self.env.get_kinematic_state(agent)["yaw_deg"]
                item.update(distance_m=round(math.dist(position, point.position[:2]) / 100, 1),
                            relative_angle_deg=round((bearing - yaw + 180) % 360 - 180, 1))
                item["checked"] = any(record["interaction_id"] == point.interaction_id for record in self.history[agent])
                records.append(item)
                if visible:
                    matching = np.all(masks[agent][:, :, :3] == np.array(point.mask_color[::-1]), axis=2)
                    ys, xs = np.where(matching)
                    if len(xs):
                        cv2.rectangle(frame, (int(xs.min()), int(ys.min())),
                                      (int(xs.max()), int(ys.max())), (80, 230, 255), 2)
                    text = f"{point.kind.key} ({point.kind.ticks} ticks)"
                    cv2.putText(frame, text, (8, 22 + 23 * index), cv2.FONT_HERSHEY_SIMPLEX,
                                .48, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, text, (8, 22 + 23 * index), cv2.FONT_HERSHEY_SIMPLEX,
                                .48, (80, 230, 255), 1, cv2.LINE_AA)
            observation["ego_view"] = frame
            observation["nearby_interactables"] = records
            observation["inspection_history"] = deepcopy(self.history[agent])
