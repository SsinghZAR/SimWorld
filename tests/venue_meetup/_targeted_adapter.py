"""Deterministic engine double, not a substitute for physical UE validation."""

from types import SimpleNamespace

import numpy as np

from benchmark.venue_meetup.targeted_env import TargetedVenueEnv
from simworld.utils.vector import Vector


class OfflineTargetedEnv(TargetedVenueEnv):
    def __init__(self, scenario, **kwargs):
        super().__init__(SimpleNamespace(), scenario, info_partition="spatial", navigate_mode="walk", **kwargs)
        self.positions = {agent.agent_id: tuple(agent.position[:2]) for agent in scenario.agents}
        self.yaws = {agent.agent_id: agent.yaw_deg for agent in scenario.agents}

    def reset(self):
        return self._build_observations()

    def get_agent_state(self, agent):
        return SimpleNamespace(actor_name=agent, humanoid=SimpleNamespace(id=agent))

    def _actor_xy(self, actor):
        return self.positions[actor]

    def get_kinematic_state(self, agent):
        return {"position": Vector(*self.positions[agent]), "yaw_deg": self.yaws[agent]}

    def _capture_frames(self, viewmode=None):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        if viewmode == "object_mask":
            for index, point in enumerate(self.interactions.points.values()):
                x, y = (index % 32) * 20, (index // 32) * 20
                frame[y:y + 10, x:x + 10] = point.mask_color[::-1]
        return {agent: frame.copy() for agent in self.agent_ids}

    def _walk_segment(self, agent, point, *, last_venue):
        import math
        moved = math.dist(self.positions[agent], point)
        self.positions[agent] = tuple(point)
        self._record_movement_point(agent, point)
        return moved, True

    def _face_point(self, *args):
        pass

    def _settle_camera_after_turn(self):
        pass

    def execute_action(self, agent, action):
        return {"result": "WAIT" if action.choice == 0 else "TURN_OK", "turn": action.compact()}
