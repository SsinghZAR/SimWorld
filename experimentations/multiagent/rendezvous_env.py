"""UE-grounded N-agent rendezvous environment for debugging multi-agent policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from benchmark.profile_unrealcv_loop import spawn_positions

from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.config import Config
from simworld.local_planner.action_space import LowLevelAction, LowLevelActionSpace
from simworld.utils.vector import Vector

from .action_space import MultiAgentTurn, action_to_dict, sanitize_turn
from .comms import BroadcastRouter, CommsRouter, MessageBus, messages_from_turns


AGENT_BLUEPRINT = "/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C"


@dataclass
class AgentState:
    """Runtime mapping from benchmark agent id to SimWorld humanoid actor."""

    agent_id: str
    humanoid: Humanoid
    actor_name: str


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-180, 180]."""

    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def vector_to_dict(vector: Vector) -> dict[str, float]:
    """Serialize a Vector."""

    return {"x": float(vector.x), "y": float(vector.y)}


def direction_from_yaw(yaw: float) -> Vector:
    """Build a unit Vector from a yaw angle."""

    return Vector(math.cos(math.radians(yaw)), math.sin(math.radians(yaw))).normalize()


class RendezvousEnv:
    """Parallel multi-agent rendezvous environment backed by a live UE server."""

    def __init__(
        self,
        communicator: Communicator,
        *,
        num_agents: int,
        config: Config | None = None,
        radius: float = 200.0,
        max_steps: int = 12,
        meeting_mode: str = "fixed_known",
        meeting_point: tuple[float, float] = (1000.0, 0.0),
        observe_others: bool = True,
        reveal_target_to_all: bool | None = None,
        spawn_origin: tuple[float, float, float] = (0.0, 0.0, 600.0),
        spawn_spacing: float = 250.0,
        agent_blueprint: str = AGENT_BLUEPRINT,
        speed: float = 200.0,
        resolution: tuple[int, int] = (640, 360),
        viewmode: str = "lit",
        camera_mode: str = "direct",
        tick_interval: float = 0.05,
        tick_count: int = 1,
        clear_on_first_reset: bool = True,
        router: CommsRouter | None = None,
        inbox_history: int = 8,
    ):
        if num_agents < 2:
            raise ValueError("RendezvousEnv requires at least two agents.")
        if meeting_mode not in {"fixed_known", "anchor_known"}:
            raise ValueError("meeting_mode must be 'fixed_known' or 'anchor_known'.")

        self.communicator = communicator
        self.config = config or Config()
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{index}" for index in range(num_agents)]
        self.radius = radius
        self.max_steps = max_steps
        self.meeting_mode = meeting_mode
        self.meeting_point = Vector(*meeting_point)
        self.observe_others = observe_others
        self.reveal_target_to_all = meeting_mode == "fixed_known" if reveal_target_to_all is None else reveal_target_to_all
        self.spawn_origin = spawn_origin
        self.spawn_spacing = spawn_spacing
        self.agent_blueprint = agent_blueprint
        self.speed = speed
        self.resolution = resolution
        self.viewmode = viewmode
        self.camera_mode = camera_mode
        self.tick_interval = tick_interval
        self.tick_count = tick_count
        self.clear_on_first_reset = clear_on_first_reset

        self.agent_states: dict[str, AgentState] = {}
        self.spawned = False
        self.step_index = 0
        self.bus = MessageBus(self.agent_ids, router=router or BroadcastRouter(), max_history=inbox_history)
        self.last_actions: dict[str, dict[str, Any]] = {}

    def reset(self) -> dict[str, dict[str, Any]]:
        """Reset the episode and return per-agent observations with ego renders."""

        self.step_index = 0
        self.last_actions = {}
        self.bus.reset(self.agent_ids)
        self.communicator.unrealcv.set_mode("sync", self.tick_interval)

        if not self.spawned:
            if self.clear_on_first_reset:
                self.communicator.clear_env(keep_roads=True)
                Humanoid._id_counter = 0
                Humanoid._camera_id_counter = 1
            self._spawn_agents()
            self.spawned = True
        else:
            self._reposition_agents()

        for _ in range(max(1, self.tick_count)):
            self.communicator.unrealcv.tick()
        return self._build_observations()

    def step(
        self,
        turns: dict[str, MultiAgentTurn | dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, float], bool, dict[str, Any]]:
        """Apply a simultaneous multi-agent step."""

        parsed_turns = {
            agent_id: MultiAgentTurn.from_json(turn)
            for agent_id, turn in turns.items()
            if agent_id in self.agent_ids
        }
        for agent_id in self.agent_ids:
            parsed_turns.setdefault(agent_id, MultiAgentTurn())

        positions_before = self._positions()
        messages = messages_from_turns(parsed_turns, step=self.step_index)
        inboxes = self.bus.deliver(messages, positions=positions_before)

        executed_actions = {}
        action_tick_count = self.tick_count
        for agent_id, turn in parsed_turns.items():
            state = self.get_agent_state(agent_id)
            relative_angle = self.relative_angle_to_target(agent_id) if self.target_known_to(agent_id) else 0.0
            action = sanitize_turn(turn, relative_angle=relative_angle)
            executed_actions[agent_id] = self.execute_action(state.humanoid, action)
            action_tick_count = max(action_tick_count, executed_actions[agent_id]["sync_ticks"])

        for _ in range(max(1, action_tick_count)):
            self.communicator.unrealcv.tick()

        self.step_index += 1
        observations = self._build_observations(inboxes=inboxes)
        distances = self.distances_to_target()
        reward = -sum(distances.values()) / len(distances)
        rewards = {agent_id: reward for agent_id in self.agent_ids}
        done = all(distance <= self.radius for distance in distances.values()) or self.step_index >= self.max_steps
        info = {
            "step": self.step_index,
            "meeting_point": vector_to_dict(self.meeting_point),
            "distances": distances,
            "spread": self.spread(),
            "success": all(distance <= self.radius for distance in distances.values()),
            "actions": executed_actions,
            "comms": self.bus.snapshot(),
            "collisions": self.collision_snapshot(),
        }
        self.last_actions = executed_actions
        return observations, rewards, done, info

    def _spawn_agents(self) -> None:
        """Spawn all humanoid agents once."""

        self.agent_states = {}
        for agent_id, position in zip(self.agent_ids, spawn_positions(self.num_agents, self.spawn_origin, self.spawn_spacing)):
            humanoid = Humanoid(
                position=Vector(position[0], position[1]),
                direction=Vector(1, 0),
                communicator=self.communicator,
                config=self.config,
            )
            self.communicator.spawn_agent(
                humanoid,
                name=None,
                position=position,
                model_path=self.agent_blueprint,
                type="humanoid",
            )
            self.communicator.humanoid_set_speed(humanoid.id, self.speed)
            self.communicator.unrealcv.set_camera_resolution(humanoid.camera_id, self.resolution)
            actor_name = self.communicator.get_humanoid_name(humanoid.id)
            self.agent_states[agent_id] = AgentState(agent_id=agent_id, humanoid=humanoid, actor_name=actor_name)

    def _reposition_agents(self) -> None:
        """Return existing agents to their start grid."""

        for agent_id, position in zip(self.agent_ids, spawn_positions(self.num_agents, self.spawn_origin, self.spawn_spacing)):
            state = self.get_agent_state(agent_id)
            self.communicator.unrealcv.set_location(position, state.actor_name)
            self.communicator.unrealcv.set_orientation((0, 0, 0), state.actor_name)
            state.humanoid.position = Vector(position[0], position[1])
            state.humanoid.direction = 0
            self.communicator.humanoid_stop(state.humanoid.id)

    def get_agent_state(self, agent_id: str) -> AgentState:
        """Return runtime state for an agent id."""

        return self.agent_states[agent_id]

    def get_kinematic_state(self, agent_id: str) -> dict[str, Any]:
        """Read actor position and heading from UE."""

        state = self.get_agent_state(agent_id)
        location = self.communicator.unrealcv.get_location(state.actor_name)
        orientation = self.communicator.unrealcv.get_orientation(state.actor_name)
        position = Vector(float(location[0]), float(location[1]))
        yaw = float(orientation[1])
        direction = direction_from_yaw(yaw)
        state.humanoid.position = position
        state.humanoid.direction = yaw
        return {"position": position, "yaw_deg": yaw, "direction": direction}

    def _positions(self) -> dict[str, tuple[float, float]]:
        """Return current 2D positions by agent id."""

        positions = {}
        for agent_id in self.agent_ids:
            state = self.get_kinematic_state(agent_id)
            position = state["position"]
            positions[agent_id] = (position.x, position.y)
        return positions

    def _capture_frames(self) -> dict[str, Any]:
        """Capture compulsory ego renders for every agent."""

        camera_ids = [self.get_agent_state(agent_id).humanoid.camera_id for agent_id in self.agent_ids]
        frames = self.communicator.get_camera_observation_multicam(camera_ids, self.viewmode, mode=self.camera_mode)
        if not isinstance(frames, list):
            frames = [frames]
        if len(frames) != len(self.agent_ids):
            raise RuntimeError(f"Expected {len(self.agent_ids)} ego renders, got {len(frames)}")
        return dict(zip(self.agent_ids, frames))

    def _build_observations(self, inboxes: dict[str, list[Any]] | None = None) -> dict[str, dict[str, Any]]:
        """Build per-agent observations with private state, inbox, roster, and image."""

        frames = self._capture_frames()
        all_states = {agent_id: self.get_kinematic_state(agent_id) for agent_id in self.agent_ids}
        inboxes = inboxes or {agent_id: list(self.bus.inboxes[agent_id]) for agent_id in self.agent_ids}
        observations: dict[str, dict[str, Any]] = {}
        for agent_id, state in all_states.items():
            target_known = self.target_known_to(agent_id)
            distance = state["position"].distance(self.meeting_point) if target_known else None
            relative_angle = self.relative_angle_to_target(agent_id, state=state) if target_known else None
            others = {}
            if self.observe_others:
                others = {
                    other_id: {
                        "position": vector_to_dict(other_state["position"]),
                        "yaw_deg": other_state["yaw_deg"],
                    }
                    for other_id, other_state in all_states.items()
                    if other_id != agent_id
                }
            observations[agent_id] = {
                "agent_id": agent_id,
                "step": self.step_index,
                "max_steps": self.max_steps,
                "position": vector_to_dict(state["position"]),
                "yaw_deg": state["yaw_deg"],
                "direction": vector_to_dict(state["direction"]),
                "target_known": target_known,
                "target": vector_to_dict(self.meeting_point) if target_known else None,
                "distance_to_target": distance,
                "relative_angle_deg": relative_angle,
                "radius": self.radius,
                "roster": self.agent_ids,
                "others": others,
                "inbox": [message.compact() for message in inboxes.get(agent_id, [])],
                "last_action": self.last_actions.get(agent_id),
                "ego_view": frames[agent_id],
            }
        return observations

    def target_known_to(self, agent_id: str) -> bool:
        """Return whether this agent is allowed to observe the meeting point."""

        return self.reveal_target_to_all or agent_id == "agent_0"

    def relative_angle_to_target(self, agent_id: str, state: dict[str, Any] | None = None) -> float:
        """Return relative angle from agent heading to meeting point."""

        state = state or self.get_kinematic_state(agent_id)
        target_vector = self.meeting_point - state["position"]
        target_yaw = math.degrees(math.atan2(target_vector.y, target_vector.x))
        return normalize_angle(target_yaw - float(state["yaw_deg"]))

    def execute_action(self, humanoid: Humanoid, action: LowLevelActionSpace) -> dict[str, Any]:
        """Execute one low-level movement action."""

        actor_name = self.communicator.get_humanoid_name(humanoid.id)
        sync_ticks = self.tick_count
        if action.choice == LowLevelAction.STEP_FORWARD:
            duration = float(action.duration or 0.2)
            direction = int(action.direction or 0)
            self._kinematic_step(actor_name, humanoid, duration, direction)
            sync_ticks = self._duration_to_ticks(duration)
            result = f"STEP_FORWARD duration={duration:.2f} direction={direction}"
        elif action.choice == LowLevelAction.TURN_AROUND:
            angle = float(action.angle or 45)
            direction = "right" if action.clockwise else "left"
            self._kinematic_rotate(actor_name, humanoid, angle, direction)
            sync_ticks = self._duration_to_ticks(1.0)
            result = f"TURN_AROUND angle={angle:.1f} direction={direction}"
        else:
            self.communicator.humanoid_stop(humanoid.id)
            result = "DO_NOTHING"
        return {"action": action_to_dict(action), "result": result, "sync_ticks": sync_ticks}

    def _duration_to_ticks(self, duration: float) -> int:
        """Convert action duration seconds to sync ticks."""

        return max(1, math.ceil(duration / max(self.tick_interval, 1e-6)) + 1)

    def _kinematic_step(self, actor_name: str, humanoid: Humanoid, duration: float, direction: int) -> None:
        """Move the actor deterministically; packaged StepForward is unreliable in sync mode."""

        location = self.communicator.unrealcv.get_location(actor_name)
        orientation = self.communicator.unrealcv.get_orientation(actor_name)
        yaw = float(orientation[1])
        sign = -1 if direction == 1 else 1
        distance = sign * self.speed * duration
        yaw_rad = math.radians(yaw)
        new_location = (
            float(location[0]) + math.cos(yaw_rad) * distance,
            float(location[1]) + math.sin(yaw_rad) * distance,
            float(location[2]),
        )
        self.communicator.unrealcv.set_location(new_location, actor_name)
        humanoid.position = Vector(new_location[0], new_location[1])

    def _kinematic_rotate(self, actor_name: str, humanoid: Humanoid, angle: float, direction: str) -> None:
        """Rotate the actor deterministically; packaged TurnAround is unreliable in sync mode."""

        orientation = self.communicator.unrealcv.get_orientation(actor_name)
        yaw = float(orientation[1])
        yaw_delta = -angle if direction == "right" else angle
        new_yaw = normalize_angle(yaw + yaw_delta)
        self.communicator.unrealcv.set_orientation((float(orientation[0]), new_yaw, float(orientation[2])), actor_name)
        humanoid.direction = new_yaw

    def distances_to_target(self) -> dict[str, float]:
        """Return distance from each agent to the shared meeting point."""

        return {
            agent_id: self.get_kinematic_state(agent_id)["position"].distance(self.meeting_point)
            for agent_id in self.agent_ids
        }

    def spread(self) -> float:
        """Return max pairwise distance between agents."""

        positions = [self.get_kinematic_state(agent_id)["position"] for agent_id in self.agent_ids]
        max_distance = 0.0
        for left_index, left in enumerate(positions):
            for right in positions[left_index + 1:]:
                max_distance = max(max_distance, left.distance(right))
        return max_distance

    def collision_snapshot(self) -> dict[str, dict[str, int] | None]:
        """Best-effort collision counters by agent id."""

        snapshot: dict[str, dict[str, int] | None] = {}
        for agent_id in self.agent_ids:
            humanoid = self.get_agent_state(agent_id).humanoid
            try:
                human, obj, building, vehicle = self.communicator.get_collision_number(humanoid.id)
            except Exception:
                snapshot[agent_id] = None
                continue
            snapshot[agent_id] = {
                "human": human,
                "object": obj,
                "building": building,
                "vehicle": vehicle,
            }
        return snapshot

    def observation_summary(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Drop image arrays from observations for compact logs."""

        return {key: value for key, value in observation.items() if key != "ego_view"}

    def disconnect(self) -> None:
        """Disconnect the underlying UnrealCV client."""

        self.communicator.unrealcv.disconnect()
