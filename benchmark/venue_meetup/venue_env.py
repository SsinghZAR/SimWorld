"""UE-grounded Venue Meetup environment."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmark.venue_meetup._core.action_space import VenueAction, VenueAgentTurn, sanitize_turn
from benchmark.venue_meetup._core.comms import BroadcastRouter, CommsRouter, MessageBus, messages_from_turns
from benchmark.venue_meetup.building_catalog import asset_path
from benchmark.venue_meetup.navigation import Obstacle, building_obstacles, path_length, plan_path
from benchmark.venue_meetup.scenario import Region, Scenario, Venue
from benchmark.venue_meetup.scoring import episode_score, final_venue_from_positions, venue_decision_facts
from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.config import Config
from simworld.utils.vector import Vector

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


def direction_from_yaw(yaw: float) -> Vector:
    """Build a unit Vector from a yaw angle."""

    return Vector(math.cos(math.radians(yaw)), math.sin(math.radians(yaw))).normalize()


_COMPASS_POINTS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


def compass_label(angle_deg: float) -> str:
    """Map a world angle (0=east/+x, 90=north/+y, CCW) to an 8-point compass label."""

    index = int((angle_deg % 360 + 22.5) // 45) % 8
    return _COMPASS_POINTS[index]


def turn_to_face(heading_deg: float, bearing_deg: float, *, tolerance: float = 8.0) -> dict[str, Any]:
    """Describe the TURN_AROUND that aligns ``heading_deg`` onto ``bearing_deg``.

    Uses the env's movement convention: ``clockwise=False`` increases yaw (a
    counter-clockwise / left turn on the north-up coarse map), ``clockwise=True``
    decreases it. Following this then STEP_FORWARD is guaranteed to approach the
    target because forward motion is along ``(cos yaw, sin yaw)``.
    """

    delta = normalize_angle(bearing_deg - heading_deg)
    if abs(delta) <= tolerance:
        return {"instruction": "already facing it (within ~8 deg) - STEP_FORWARD to approach", "needs_turn": False}
    if delta > 0:
        return {
            "instruction": f"turn LEFT ~{round(delta)} deg, then STEP_FORWARD",
            "needs_turn": True,
            "action": {"choice": 2, "clockwise": False, "angle": round(delta)},
        }
    return {
        "instruction": f"turn RIGHT ~{round(-delta)} deg, then STEP_FORWARD",
        "needs_turn": True,
        "action": {"choice": 2, "clockwise": True, "angle": round(-delta)},
    }


def vector_to_dict(vector: Vector) -> dict[str, float]:
    """Serialize a Vector."""

    return {"x": float(vector.x), "y": float(vector.y)}


class VenueMeetupEnv:
    """Small-N venue-meetup environment backed by a live UE server."""

    def __init__(
        self,
        communicator: Communicator,
        scenario: Scenario,
        *,
        config: Config | None = None,
        resolution: tuple[int, int] = (640, 360),
        viewmode: str = "lit",
        frame_gamma: float = 0.5,
        camera_mode: str = "direct",
        tick_interval: float = 0.05,
        tick_count: int = 1,
        speed: float = 1000.0,
        step_duration: float = 0.45,
        min_step_duration: float = 0.2,
        max_step_duration: float = 0.8,
        block_ratio: float = 0.4,
        move_settle_sec: float = 0.15,
        spawn_settle_sec: float = 0.7,
        record_motion: bool = False,
        motion_fps: float = 10.0,
        agent_blueprint: str = AGENT_BLUEPRINT,
        inspect_range: float = 5000.0,
        inspect_min_mask_pixels: int = 50,
        navigate_max_tries: int = 4,
        navigate_mode: str = "teleport",
        walk_clearance: float = 700.0,
        walk_landmark_radius: float = 2000.0,
        walk_arrive_radius: float = 700.0,
        walk_waypoint_radius: float = 250.0,
        walk_max_bursts: int = 24,
        walk_block_ratio: float = 0.3,
        walk_max_stalls: int = 2,
        walk_max_replans: int = 5,
        walk_discovery_radius: float = 900.0,
        walk_discovery_ahead: float = 500.0,
        entrance_block_radius: float = 400.0,
        sun_rotation: tuple[float, float, float] = (-50.0, 180.0, 180.0),
        no_communication: bool = False,
        no_coarse_map: bool = False,
        full_shared_information: bool = False,
        shared_constraints: bool = False,
        info_partition: str = "none",
        router: CommsRouter | None = None,
    ):
        self.communicator = communicator
        self.scenario = scenario
        self.config = config or Config()
        self.resolution = resolution
        self.viewmode = viewmode
        self.frame_gamma = float(frame_gamma)
        # The packaged base maps render very dark via UnrealCV scene capture and
        # offer no API lever to raise exposure/light intensity, so we brighten the
        # lit ego frame in Python (gamma < 1) before it reaches the model + video.
        # Mask frames used for inspect scoring are captured separately and stay raw.
        if abs(self.frame_gamma - 1.0) > 1e-3:
            self._frame_lut = ((np.arange(256) / 255.0) ** self.frame_gamma * 255.0).clip(0, 255).astype(np.uint8)
        else:
            self._frame_lut = None
        self.camera_mode = camera_mode
        self.tick_interval = tick_interval
        self.tick_count = tick_count
        self.speed = speed
        self.step_duration = step_duration
        self.min_step_duration = min_step_duration
        self.max_step_duration = max_step_duration
        self.block_ratio = block_ratio
        self.move_settle_sec = move_settle_sec
        self.spawn_settle_sec = spawn_settle_sec
        # When recording, sample ego frames *during* the walk so the saved video
        # shows continuous locomotion instead of one settled snapshot per step.
        self.record_motion = record_motion
        self.motion_fps = motion_fps
        self._motion_frames: list[dict[str, Any]] = []
        self.agent_blueprint = agent_blueprint
        self.inspect_range = inspect_range
        self.inspect_min_mask_pixels = inspect_min_mask_pixels
        self.navigate_max_tries = max(1, int(navigate_max_tries))
        # NAVIGATE locomotion: "teleport" drops the agent at the meeting region
        # (abstracted movement, fast social-only runs); "walk" plans an
        # obstacle-aware route and physically walks it with real StepForward
        # locomotion so the agent traverses the world instead of teleporting.
        if navigate_mode not in ("teleport", "walk"):
            raise ValueError(f"navigate_mode must be 'teleport' or 'walk', got {navigate_mode!r}")
        self.navigate_mode = navigate_mode
        self.walk_clearance = walk_clearance
        self.walk_landmark_radius = walk_landmark_radius
        self.walk_arrive_radius = walk_arrive_radius
        self.walk_waypoint_radius = walk_waypoint_radius
        self.walk_max_bursts = max(1, int(walk_max_bursts))
        self.walk_block_ratio = walk_block_ratio
        self.walk_max_stalls = max(1, int(walk_max_stalls))
        self.walk_max_replans = max(0, int(walk_max_replans))
        self.walk_discovery_radius = walk_discovery_radius
        self.walk_discovery_ahead = walk_discovery_ahead
        # Static building keep-out discs (lazily built once; scene is fixed for
        # the env's lifetime) and the most recent planned route per agent (for
        # debug overlays).
        self._obstacles: list[Obstacle] | None = None
        self._last_path: dict[str, list[tuple[float, float]]] = {}
        self.entrance_block_radius = entrance_block_radius
        self.sun_rotation = sun_rotation
        self._sun_actor: str | None = None
        self.no_communication = no_communication
        self.no_coarse_map = no_coarse_map
        self.full_shared_information = full_shared_information
        self.shared_constraints = shared_constraints
        # Info-partition mode: "none" (any agent can inspect any venue) or
        # "spatial" (an agent can only inspect venues whose zone_id matches its
        # own). See notes.md sections 6-6a; "skill_check" is reserved for V2.
        if info_partition not in ("none", "spatial"):
            raise ValueError(f"info_partition must be 'none' or 'spatial', got {info_partition!r}")
        self.info_partition = info_partition
        self._agent_zone = {agent.agent_id: agent.zone_id for agent in scenario.agents}

        self.agent_ids = scenario.agent_ids()
        self.agent_states: dict[str, AgentState] = {}
        self.bus = MessageBus(self.agent_ids, router=router or BroadcastRouter())
        self.step_index = 0
        self.spawned = False
        self.last_actions: dict[str, dict[str, Any]] = {}
        self.last_inspections: dict[str, dict[str, Any]] = {}
        self.inspected_venues: set[str] = set()
        # Per-agent ground-truth facts revealed by that agent's own successful
        # inspections: agent_id -> {venue_id -> {trait -> value}}. This is the
        # embodied, egocentric knowledge each agent has gathered first-hand and is
        # the basis for the social process metrics (see notes.md section 7).
        self.revealed_facts: dict[str, dict[str, dict[str, Any]]] = {agent_id: {} for agent_id in self.agent_ids}

    def reset(self) -> dict[str, dict[str, Any]]:
        """Reset the episode and return initial observations."""

        self.step_index = 0
        self.last_actions = {}
        self.last_inspections = {}
        self.inspected_venues = set()
        self.revealed_facts = {agent_id: {} for agent_id in self.agent_ids}
        self.bus.reset(self.agent_ids)
        # Async (running) mode so the engine character movement actually walks and
        # is blocked by building collision. Sync/pause only advances on manual
        # ticks, under which the packaged StepForward does not progress.
        self.communicator.unrealcv.set_mode("async")
        self.communicator.clear_env(keep_roads=True)
        Humanoid._id_counter = 0
        Humanoid._camera_id_counter = 1
        self._setup_lighting()
        self._spawn_static_scene()
        self._spawn_agents()
        self.spawned = True
        self._settle()
        return self._build_observations()

    def _settle(self) -> None:
        """Let freshly spawned agents drop to the ground and come to rest."""

        for agent_id in self.agent_ids:
            try:
                self.communicator.humanoid_stop(self.get_agent_state(agent_id).humanoid.id)
            except Exception:  # noqa: BLE001 - settling is best-effort.
                pass
        time.sleep(self.spawn_settle_sec)
        self._tick()

    def _setup_lighting(self) -> None:
        """Force a deterministic, well-lit sun on the otherwise dim empty map.

        The packaged empty map ships with a low/odd sun angle and no usable
        weather-manager blueprint, so the only reliable lever is rotating the
        scene's existing directional light. Re-applying it every reset also
        undoes any orientation left behind by earlier episodes.
        """

        if self._sun_actor is None:
            try:
                objects = list(self.communicator.unrealcv.get_objects())
            except Exception:  # noqa: BLE001 - lighting is best-effort.
                objects = []
            self._sun_actor = next((name for name in objects if name.lower().startswith("directionallight")), "")
        if not self._sun_actor:
            return
        try:
            self.communicator.unrealcv.set_orientation(self.sun_rotation, self._sun_actor)
        except Exception:  # noqa: BLE001 - never fail an episode over lighting.
            pass

    def step(
        self,
        turns: dict[str, VenueAgentTurn | dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, float], bool, dict[str, Any]]:
        """Apply one simultaneous multi-agent step."""

        parsed_turns = {
            agent_id: VenueAgentTurn.from_json(turn)
            for agent_id, turn in turns.items()
            if agent_id in self.agent_ids
        }
        for agent_id in self.agent_ids:
            parsed_turns.setdefault(agent_id, VenueAgentTurn())

        positions_before = self._positions()
        inboxes = {agent_id: list(self.bus.inboxes[agent_id]) for agent_id in self.agent_ids}
        if not self.no_communication:
            messages = messages_from_turns(parsed_turns, step=self.step_index)
            inboxes = self.bus.deliver(messages, positions=positions_before)

        actions = {}
        for agent_id, turn in parsed_turns.items():
            relative_angle = self.relative_angle_to_venue(agent_id, turn.target_venue_id) if turn.target_venue_id else 0.0
            actions[agent_id] = sanitize_turn(turn, relative_angle=relative_angle)

        executed_actions: dict[str, Any] = {}
        move_ctx: dict[str, dict[str, Any]] = {}
        self._motion_frames = []
        # Phase 1: issue concurrent forward intents (engine character movement).
        for agent_id, action in actions.items():
            if action.choice == VenueAction.STEP_FORWARD.value:
                ctx = self._begin_step(agent_id, action)
                if ctx.get("blocked_entrance"):
                    executed_actions[agent_id] = {
                        "turn": action.compact(),
                        "result": "BLOCKED_BY_ENTRANCE",
                        "moved_cm": 0.0,
                        "location": ctx["start"],
                    }
                else:
                    move_ctx[agent_id] = ctx
        # Phase 2: let all movers walk simultaneously in real time, then halt them.
        if move_ctx:
            walk_time = max(ctx["duration"] for ctx in move_ctx.values()) + self.move_settle_sec
            if self.record_motion:
                self._sample_motion(walk_time)
            else:
                time.sleep(walk_time)
            for agent_id in move_ctx:
                self.communicator.humanoid_stop(self.get_agent_state(agent_id).humanoid.id)
            self._tick()
        # Phase 3: finalize movers (measure actual displacement) and run other actions.
        for agent_id, action in actions.items():
            if agent_id in move_ctx:
                executed_actions[agent_id] = self._finalize_step(agent_id, action, move_ctx[agent_id])
            elif agent_id not in executed_actions:
                executed_actions[agent_id] = self.execute_action(agent_id, action)

        self._tick()
        self.step_index += 1
        observations = self._build_observations(inboxes=inboxes)
        done = self._converged() or self.step_index >= self.scenario.max_steps
        final_positions = self._positions()
        scores = episode_score(
            self.scenario,
            final_positions,
            inspected_venues=self.inspected_venues,
            message_count=len(self.bus.transcript),
            timed_out=self.step_index >= self.scenario.max_steps and not self._converged(),
        )
        rewards = {agent_id: scores["episode_score"] for agent_id in self.agent_ids}
        info = {
            "step": self.step_index,
            "actions": executed_actions,
            "comms": self.bus.snapshot(),
            "inspections": dict(self.last_inspections),
            "positions_internal": final_positions,
            "scores": scores,
            "success": self._converged(),
        }
        self.last_actions = executed_actions
        return observations, rewards, done, info

    def _spawn_static_scene(self) -> None:
        """Spawn venues, landmarks, and visible dressing props."""

        for venue in self.scenario.venues:
            actor_name = self.venue_actor_name(venue.venue_id)
            self.communicator.spawn_object(actor_name, venue.asset_path, venue.position, (0.0, venue.yaw_deg, 0.0))
            self.communicator.unrealcv.set_color(actor_name, venue.mask_color_rgb)
            for prop in venue.props:
                prop_name = self.prop_actor_name(prop.prop_id)
                self.communicator.spawn_object(prop_name, asset_path(prop.asset_key), prop.position, (0.0, prop.yaw_deg, 0.0))
                self.communicator.unrealcv.set_scale(prop.scale, prop_name)
                if prop.color_rgb is not None:
                    self.communicator.unrealcv.set_color(prop_name, prop.color_rgb)

        for landmark in self.scenario.landmarks:
            actor_name = self.landmark_actor_name(landmark.landmark_id)
            self.communicator.spawn_object(actor_name, landmark.asset_path, landmark.position, (0.0, landmark.yaw_deg, 0.0))
            self.communicator.unrealcv.set_color(actor_name, landmark.mask_color_rgb)

    def _spawn_agents(self) -> None:
        """Spawn all humanoid agents."""

        self.agent_states = {}
        for agent in self.scenario.agents:
            direction = direction_from_yaw(agent.yaw_deg)
            humanoid = Humanoid(position=Vector(agent.position[0], agent.position[1]), direction=direction, communicator=self.communicator, config=self.config)
            self.communicator.spawn_agent(
                humanoid,
                name=None,
                position=agent.position,
                model_path=self.agent_blueprint,
                type="humanoid",
            )
            actor_name = self.communicator.get_humanoid_name(humanoid.id)
            self.communicator.unrealcv.set_orientation((0.0, agent.yaw_deg, 0.0), actor_name)
            self.communicator.humanoid_set_speed(humanoid.id, self.speed)
            self.communicator.unrealcv.set_camera_resolution(humanoid.camera_id, self.resolution)
            self.agent_states[agent.agent_id] = AgentState(agent_id=agent.agent_id, humanoid=humanoid, actor_name=actor_name)

    def _tick(self) -> None:
        for _ in range(max(1, self.tick_count)):
            self.communicator.unrealcv.tick()

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

    def _capture_frames(self, viewmode: str | None = None) -> dict[str, Any]:
        """Capture ego renders for every agent."""

        viewmode = viewmode or self.viewmode
        camera_ids = [self.get_agent_state(agent_id).humanoid.camera_id for agent_id in self.agent_ids]
        frames = self.communicator.get_camera_observation_multicam(camera_ids, viewmode, mode=self.camera_mode)
        if not isinstance(frames, list):
            frames = [frames]
        if len(frames) != len(self.agent_ids):
            raise RuntimeError(f"Expected {len(self.agent_ids)} ego renders, got {len(frames)}")
        return dict(zip(self.agent_ids, frames))

    def _enhance_frame(self, frame: Any) -> Any:
        """Brighten a lit ego frame (gamma) so dim UE captures stay readable."""

        if self._frame_lut is None or frame is None:
            return frame
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            return frame
        return self._frame_lut[frame]

    def _target_cue(self, ident: str, kind: str, type_: str, target_pos: Any, agent_pos: Vector, yaw: float, region: Region | None = None) -> dict[str, Any]:
        """Build a world-frame bearing/turn/distance cue toward one target."""

        dx = float(target_pos[0]) - agent_pos.x
        dy = float(target_pos[1]) - agent_pos.y
        bearing = math.degrees(math.atan2(dy, dx))
        turn = turn_to_face(yaw, bearing)
        arrived = bool(region is not None and region.contains((agent_pos.x, agent_pos.y)))
        cue = {
            "id": ident,
            "kind": kind,
            "type": type_,
            "direction": compass_label(bearing),
            "bearing_deg": round(bearing),
            "distance_m": round(math.hypot(dx, dy) / 100.0),
            "guidance": "You are here (at this venue). Stop advancing; INSPECT or WAIT." if arrived else turn["instruction"],
        }
        if arrived:
            cue["arrived"] = True
        elif turn.get("action"):
            cue["suggested_action"] = turn["action"]
        return cue

    def _heading_cue(self, agent_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Compute proprioceptive heading and (map-gated) navigation cues."""

        kin = self.get_kinematic_state(agent_id)
        position, yaw = kin["position"], kin["yaw_deg"]
        self_pose = {
            "facing": compass_label(yaw),
            "heading_deg": round(yaw),
            "note": (
                "Your camera is third-person (you see your own back). Use this compass and the coarse map "
                "(north=up/+y, east=right/+x) for left/right decisions, not the image."
            ),
        }
        if self.no_coarse_map:
            return self_pose, None
        targets = [
            self._target_cue(venue.venue_id, "venue", venue.venue_type, venue.position, position, yaw, region=venue.region)
            for venue in self.scenario.venues
        ]
        targets += [
            self._target_cue(landmark.landmark_id, "landmark", landmark.landmark_type, landmark.position, position, yaw)
            for landmark in self.scenario.landmarks
        ]
        if self.navigate_mode == "walk":
            hint = (
                "Use NAVIGATE (choice=5, target_venue_id) to walk to a venue: it plans a route around the buildings and "
                "physically walks you toward that venue's meeting region (this takes real travel time and can be blocked "
                "by a building - if 'arrived' is false, NAVIGATE again to keep going). You must be in a venue's region to "
                "INSPECT it. STEP_FORWARD/TURN_AROUND are optional fine movement; venues are solid buildings you cannot "
                "walk through. When a venue shows 'arrived: true' you are physically at it - INSPECT it or WAIT."
            )
        else:
            hint = (
                "Prefer NAVIGATE (choice=5, target_venue_id) to travel to a venue in one action: it places you in that "
                "venue's meeting region. You must be in a venue's region to INSPECT it. STEP_FORWARD/TURN_AROUND are "
                "optional fine movement; venues are solid buildings you cannot walk through. When a venue shows "
                "'arrived: true' you are physically at it - INSPECT it or WAIT."
            )
        navigation = {
            "frame": "world bearings (north=up/+y, east=right/+x); matches the coarse map",
            "hint": hint,
            "targets": targets,
        }
        return self_pose, navigation

    def _build_observations(self, inboxes: dict[str, list[Any]] | None = None) -> dict[str, dict[str, Any]]:
        """Build per-agent observations while hiding evaluator-only state."""

        frames = self._capture_frames("lit")
        frames = {agent_id: self._enhance_frame(frame) for agent_id, frame in frames.items()}
        inboxes = inboxes or {agent_id: list(self.bus.inboxes[agent_id]) for agent_id in self.agent_ids}
        shared_constraint = "; ".join(agent.private_constraint for agent in self.scenario.agents)
        observations: dict[str, dict[str, Any]] = {}
        for agent in self.scenario.agents:
            private_constraint = shared_constraint if self.shared_constraints else agent.private_constraint
            if self.full_shared_information:
                venue_summaries = [venue.compact() if hasattr(venue, "compact") else venue.__dict__ for venue in self.scenario.venues]
                known_venue_facts: dict[str, Any] = {venue.venue_id: self._venue_facts(venue) for venue in self.scenario.venues}
            else:
                venue_summaries = []
                for venue in self.scenario.venues:
                    summary = {
                        "venue_id": venue.venue_id,
                        "venue_type": venue.venue_type,
                        "slot_id": venue.slot_id,
                        "visual_summary": venue.visual_summary,
                    }
                    if self.info_partition == "spatial":
                        summary["zone_id"] = venue.zone_id
                        summary["can_inspect"] = self._can_inspect_zone(agent.agent_id, venue)
                    venue_summaries.append(summary)
                known_venue_facts = dict(self.revealed_facts.get(agent.agent_id, {}))

            self_pose, navigation = self._heading_cue(agent.agent_id)
            observations[agent.agent_id] = {
                "agent_id": agent.agent_id,
                "step": self.step_index,
                "max_steps": self.scenario.max_steps,
                "role": "visitor",
                "objective": "Find the best feasible venue for everyone and physically meet there.",
                "private_constraint": private_constraint,
                "zone_id": self._agent_zone.get(agent.agent_id),
                "info_partition": self.info_partition,
                "coarse_map_text": None if self.no_coarse_map else self.scenario.coarse_map_text,
                "coarse_map_path": None if self.no_coarse_map else self.scenario.coarse_map_path,
                "self_pose": self_pose,
                "candidate_venues": venue_summaries,
                "known_venue_facts": known_venue_facts,
                "landmarks": [
                    {
                        "landmark_id": landmark.landmark_id,
                        "type": landmark.landmark_type,
                        "slot_id": landmark.slot_id,
                        "visual_summary": landmark.visual_summary,
                    }
                    for landmark in self.scenario.landmarks
                ],
                "group_chat": [message.compact() for message in inboxes.get(agent.agent_id, [])],
                "roster": self.agent_ids,
                "last_action": self.last_actions.get(agent.agent_id),
                "last_inspect_result": self.last_inspections.get(agent.agent_id),
                "valid_actions": {
                    "0": "WAIT",
                    "1": "STEP_FORWARD",
                    "2": "TURN_AROUND",
                    "3": "INSPECT target_venue_id (must be standing in that venue's region - NAVIGATE there first)",
                    "4": "COMMUNICATE message",
                    "5": "NAVIGATE target_venue_id (walk to a venue's meeting point)",
                },
                "ego_view": frames[agent.agent_id],
            }
            if navigation is not None:
                observations[agent.agent_id]["navigation"] = navigation
        return observations

    def execute_action(self, agent_id: str, action: VenueAgentTurn) -> dict[str, Any]:
        """Execute one benchmark action."""

        state = self.get_agent_state(agent_id)
        if action.choice == VenueAction.STEP_FORWARD.value:
            return self._engine_step_blocking(agent_id, action)
        if action.choice == VenueAction.TURN_AROUND.value:
            result = self._kinematic_rotate(state.actor_name, state.humanoid, float(action.angle or 45), "right" if action.clockwise else "left")
        elif action.choice == VenueAction.INSPECT.value:
            result = self._inspect(agent_id, action)
        elif action.choice == VenueAction.NAVIGATE.value:
            result = self._navigate(agent_id, action)
        elif action.choice == VenueAction.COMMUNICATE.value:
            result = {"result": "COMMUNICATE", "message": action.message}
        else:
            self.communicator.humanoid_stop(state.humanoid.id)
            result = {"result": "WAIT"}
        return {"turn": action.compact(), **result}

    def _full_stop(self, state: AgentState) -> None:
        """Halt a pawn's locomotion as fully as the API allows before a teleport.

        Pedestrian pawns keep walking on their own; ``StopAgent`` alone does not
        reliably cancel an in-flight step, so a queued forward walk overrides the
        next ``set_location`` (the teleport is silently swallowed). Issuing
        ``StopAction`` as well clears the current movement.
        """

        self.communicator.humanoid_stop(state.humanoid.id)
        try:
            self.communicator.unrealcv.humanoid_stop_current_action(state.actor_name)
        except Exception:  # noqa: BLE001 - older builds may lack StopAction.
            pass

    def _navigate(self, agent_id: str, action: VenueAgentTurn) -> dict[str, Any]:
        """High-level move to a venue's plaza-side meeting point.

        Dispatches on ``navigate_mode``: ``teleport`` places the agent at the
        meeting region (abstracted movement; see notes.md section 2), while
        ``walk`` plans an obstacle-aware route and physically walks it with real
        engine locomotion so the agent traverses the world (no teleporting).
        """

        venue = self._resolve_inspect_target(agent_id, action)
        if venue is None or (action.target_venue_id and venue.venue_id != action.target_venue_id and not action.target_description):
            return {"result": "NAVIGATE_FAILED", "reason": "unknown target"}

        if self.navigate_mode == "walk":
            return self._walk_navigate(agent_id, venue)
        return self._teleport_navigate(agent_id, venue)

    def _teleport_navigate(self, agent_id: str, venue: Venue) -> dict[str, Any]:
        """Abstracted NAVIGATE: drop the agent at the venue's meeting region.

        The agent is set down at the validated open standoff (the same point used
        for convergence) and faced toward the venue facade.
        """

        state = self.get_agent_state(agent_id)
        location = self.communicator.unrealcv.get_location(state.actor_name)
        z = float(location[2])
        cx, cy = venue.region.center
        # Fan agents slightly around the meeting point so they do not stack.
        index = self.agent_ids.index(agent_id)
        offset = 300.0
        angle = math.radians(90.0 * index)
        # Round the fan offset: math.cos(radians(90)) is 6.1e-17, not 0, so for a
        # venue centred on x=0 the target would be ~1.8e-14 - a value Python
        # renders in scientific notation ("1.8e-14"), which UnrealCV's set_location
        # parser cannot read, silently dropping the teleport. Rounding collapses
        # the float dust to a clean coordinate.
        target_x = round(cx + math.cos(angle) * offset, 2)
        target_y = round(cy + math.sin(angle) * offset, 2)

        # Place the pawn by dropping it in from above with BOTH its AI controller
        # and its collision temporarily disabled, then restore them and let it
        # fall to the floor. Why each piece is needed (the engine runs async /
        # real-time, so the movement component is live every frame):
        #  * enable_controller(0): a live AI walk re-asserts the pawn's x,y every
        #    frame and silently overrides set_location (nondeterministic per agent
        #    and worsens as navigations accumulate). Disabling it stops the fight.
        #  * set_collision(False): with the controller off, a plain set_location
        #    is swept against intervening building collision and gets pinned at the
        #    source. Disabling the pawn's own collision lets the placement commit.
        #  * drop from above (not the source z): teleporting to the source's z
        #    embeds the pawn below the destination ground (per-venue terrain height
        #    differs, ~90-150 cm here); an embedded pawn can't move afterwards.
        # NOTE: do NOT issue StopAgent/StopAction here (the old _full_stop). Those
        # latch a "stop and hold position" task on the still-live controller that
        # then swallows the very next set_location - the exact bug that stranded
        # an agent at the previous venue. Zeroing speed is enough to keep it from
        # walking; the controller is re-enabled (restoring gravity) before the
        # settle so the pawn actually falls to the floor instead of hovering.
        uc = self.communicator.unrealcv
        actor = state.actor_name
        place_z = 400.0
        self.communicator.humanoid_set_speed(state.humanoid.id, 0)

        def _teleport(x: float, y: float) -> tuple[float, float]:
            uc.enable_controller(actor, 0)
            uc.set_collision(actor, False)
            uc.set_location([x, y, place_z], actor)
            self._tick()
            uc.set_collision(actor, True)
            uc.enable_controller(actor, 1)
            # Async real-time fall; poll until the pawn rests on the ground.
            prev_z: float | None = None
            for _ in range(15):
                time.sleep(0.06)
                self._tick()
                cur_z = float(uc.get_location(actor)[2])
                if prev_z is not None and abs(cur_z - prev_z) < 2.0:
                    break
                prev_z = cur_z
            loc = uc.get_location(actor)
            return float(loc[0]), float(loc[1])

        arrived = False
        for _ in range(self.navigate_max_tries):
            ax, ay = _teleport(target_x, target_y)
            if venue.region.contains((ax, ay)):
                arrived = True
                break
        final = uc.get_location(actor)
        # Face the facade for the post-arrival ego frame, then restore walk speed.
        self._face_point(actor, state.humanoid, (venue.position[0], venue.position[1]))
        self.communicator.humanoid_set_speed(state.humanoid.id, self.speed)
        state.humanoid.position = Vector(float(final[0]), float(final[1]))
        return {
            "result": "NAVIGATE_OK" if arrived else "NAVIGATE_FAILED",
            "venue_id": venue.venue_id,
            "arrived": arrived,
            "reason": None if arrived else "could not reach the meeting point (placement blocked)",
            "location": (round(float(final[0]), 1), round(float(final[1]), 1)),
        }

    def _meeting_target(self, agent_id: str, venue: Venue) -> tuple[float, float]:
        """Fan agents slightly around the meeting point so they do not stack."""

        cx, cy = venue.region.center
        index = self.agent_ids.index(agent_id)
        angle = math.radians(90.0 * index)
        return round(cx + math.cos(angle) * 300.0, 2), round(cy + math.sin(angle) * 300.0, 2)

    def _walk_navigate(self, agent_id: str, venue: Venue) -> dict[str, Any]:
        """Walk-mode NAVIGATE: plan an obstacle-aware route and physically walk it.

        Locomotion is real (engine ``StepForward`` bursts, blocked by building
        collision), so the agent traverses the plaza rather than teleporting. The
        route is planned in Python (no engine navmesh is exposed) around building
        keep-out discs, then walked waypoint by waypoint, halting at the venue's
        meeting region. In-motion ego frames are captured along the way when
        recording so the saved video shows the walk.

        Because the disc footprints only approximate the real (asymmetric) building
        collision, the walk is collision-reactive: if the pawn stalls against
        unmodeled geometry, that spot is registered as a discovered obstacle and
        the route is replanned around it. This adapts the abstract map to the true
        collision the agent actually feels, so it routes around buildings instead
        of grinding into them.
        """

        state = self.get_agent_state(agent_id)
        actor = state.actor_name
        uc = self.communicator.unrealcv
        if self._obstacles is None:
            self._obstacles = building_obstacles(
                self.scenario,
                clearance=self.walk_clearance,
                landmark_radius=self.walk_landmark_radius,
            )

        target = self._meeting_target(agent_id, venue)
        # Make sure the pawn walks at the configured speed (teleport mode may have
        # zeroed it on a prior step).
        self.communicator.humanoid_set_speed(state.humanoid.id, self.speed)

        discovered: list[Obstacle] = []
        moved_total = 0.0
        planned_len = 0.0
        last_plan: list[tuple[float, float]] = []
        arrived = False
        for _ in range(self.walk_max_replans + 1):
            location = uc.get_location(actor)
            start = (float(location[0]), float(location[1]))
            if venue.region.contains(start):
                arrived = True
                break
            waypoints = plan_path(start, target, [*self._obstacles, *discovered])
            last_plan = [start, *waypoints]
            planned_len = path_length(start, waypoints)
            seg_moved, reached, blocked_point = self._walk_route(agent_id, waypoints, venue)
            moved_total += seg_moved
            final_xy = self._actor_xy(actor)
            if venue.region.contains(final_xy):
                arrived = True
                break
            if blocked_point is None:
                break
            discovered.append(Obstacle(blocked_point[0], blocked_point[1], self.walk_discovery_radius))

        final_xy = self._actor_xy(actor)
        self._last_path[agent_id] = last_plan or [final_xy]
        arrived = arrived or venue.region.contains(final_xy)
        # Face the facade for the post-arrival ego frame.
        self._face_point(actor, state.humanoid, (venue.position[0], venue.position[1]))
        self._tick()
        state.humanoid.position = Vector(final_xy[0], final_xy[1])
        if arrived:
            result, reason = "NAVIGATE_OK", None
        elif discovered:
            result = "NAVIGATE_BLOCKED"
            reason = "buildings blocked the route; try STEP_FORWARD/TURN_AROUND or NAVIGATE again"
        else:
            result = "NAVIGATE_PARTIAL"
            reason = "walked toward the venue but did not reach the meeting region; NAVIGATE again to continue"
        return {
            "result": result,
            "venue_id": venue.venue_id,
            "arrived": arrived,
            "mode": "walk",
            "path_waypoints": len(last_plan) - 1 if last_plan else 0,
            "replans": len(discovered),
            "planned_distance_cm": round(planned_len, 1),
            "moved_cm": round(moved_total, 1),
            "reason": reason,
            "location": (round(final_xy[0], 1), round(final_xy[1], 1)),
        }

    def _actor_xy(self, actor_name: str) -> tuple[float, float]:
        """Return the actor's current 2D position."""

        location = self.communicator.unrealcv.get_location(actor_name)
        return float(location[0]), float(location[1])

    def _walk_route(self, agent_id: str, waypoints: list[tuple[float, float]], venue: Venue) -> tuple[float, bool, tuple[float, float] | None]:
        """Walk a full waypoint list; on a stall, return the discovered block point.

        Returns ``(distance_moved, completed, blocked_point)``. ``blocked_point``
        is a spot just ahead of where the pawn stalled (toward the waypoint it was
        chasing), i.e. roughly where the unmodeled collision is, so the caller can
        register it as an obstacle and replan.
        """

        state = self.get_agent_state(agent_id)
        actor = state.actor_name
        moved_total = 0.0
        for index, waypoint in enumerate(waypoints):
            is_last = index == len(waypoints) - 1
            seg_moved, reached = self._walk_segment(agent_id, waypoint, last_venue=venue if is_last else None)
            moved_total += seg_moved
            if not reached:
                position = self._actor_xy(actor)
                distance = math.hypot(waypoint[0] - position[0], waypoint[1] - position[1])
                if distance > 1.0:
                    ux, uy = (waypoint[0] - position[0]) / distance, (waypoint[1] - position[1]) / distance
                    blocked_point = (position[0] + ux * self.walk_discovery_ahead, position[1] + uy * self.walk_discovery_ahead)
                else:
                    blocked_point = position
                return moved_total, False, blocked_point
        return moved_total, True, None

    def _walk_segment(self, agent_id: str, waypoint: tuple[float, float], *, last_venue: Venue | None) -> tuple[float, bool]:
        """Walk toward one waypoint in engine StepForward bursts.

        Returns ``(distance_moved, reached)``. ``reached`` is False if the pawn
        stalls against collision for ``walk_max_stalls`` consecutive bursts or the
        per-segment burst budget is exhausted. The final segment also succeeds as
        soon as the agent enters the venue's meeting region.
        """

        state = self.get_agent_state(agent_id)
        actor = state.actor_name
        uc = self.communicator.unrealcv
        arrive_radius = self.walk_arrive_radius if last_venue is not None else self.walk_waypoint_radius
        seg_moved = 0.0
        stalls = 0
        for _ in range(self.walk_max_bursts):
            location = uc.get_location(actor)
            position = (float(location[0]), float(location[1]))
            if last_venue is not None and last_venue.region.contains(position):
                return seg_moved, True
            distance = math.hypot(waypoint[0] - position[0], waypoint[1] - position[1])
            if distance <= arrive_radius:
                return seg_moved, True
            self._face_point(actor, state.humanoid, waypoint)
            self._tick()
            duration = max(self.min_step_duration, min(self.max_step_duration, distance / max(1.0, self.speed)))
            with uc.lock:
                uc.client.request(f"vbp {actor} StepForward {duration} 0")
            self._walk_wait(duration + self.move_settle_sec)
            self.communicator.humanoid_stop(state.humanoid.id)
            self._tick()
            after = uc.get_location(actor)
            moved = math.hypot(float(after[0]) - position[0], float(after[1]) - position[1])
            seg_moved += moved
            intended = min(distance, self.speed * duration)
            if intended > 1.0 and moved < self.walk_block_ratio * intended:
                stalls += 1
                if stalls >= self.walk_max_stalls:
                    return seg_moved, False
            else:
                stalls = 0
        return seg_moved, False

    def _walk_wait(self, duration: float) -> None:
        """Wait out a walk burst, sampling in-motion ego frames when recording."""

        if self.record_motion:
            self._sample_motion(duration)
        else:
            time.sleep(max(0.0, duration))

    def _step_duration(self, action: VenueAgentTurn) -> float:
        """Clamp the requested step duration into the configured range."""

        requested = float(action.duration) if action.duration else self.step_duration
        return max(self.min_step_duration, min(self.max_step_duration, requested))

    def _begin_step(self, agent_id: str, action: VenueAgentTurn) -> dict[str, Any]:
        """Start an engine-driven forward step; honor logically blocked entrances."""

        state = self.get_agent_state(agent_id)
        location = self.communicator.unrealcv.get_location(state.actor_name)
        orientation = self.communicator.unrealcv.get_orientation(state.actor_name)
        start = (float(location[0]), float(location[1]))
        duration = self._step_duration(action)
        direction = int(action.direction or 0)
        yaw = float(orientation[1])
        sign = -1 if direction == 1 else 1
        reach = sign * self.speed * duration
        yaw_rad = math.radians(yaw)
        end = (start[0] + math.cos(yaw_rad) * reach, start[1] + math.sin(yaw_rad) * reach)
        if self._blocked_by_entrance(start, end):
            self.communicator.humanoid_stop(state.humanoid.id)
            return {"blocked_entrance": True, "start": start, "duration": duration, "direction": direction}
        # Issue the engine StepForward without the wrapper's blocking sleep so
        # multiple agents can walk concurrently during one shared wait.
        with self.communicator.unrealcv.lock:
            self.communicator.unrealcv.client.request(f"vbp {state.actor_name} StepForward {duration} {direction}")
        return {"blocked_entrance": False, "start": start, "duration": duration, "direction": direction}

    def _finalize_step(self, agent_id: str, action: VenueAgentTurn, ctx: dict[str, Any]) -> dict[str, Any]:
        """Measure actual displacement after an engine step and flag collisions."""

        state = self.get_agent_state(agent_id)
        location = self.communicator.unrealcv.get_location(state.actor_name)
        end = (float(location[0]), float(location[1]))
        moved = math.hypot(end[0] - ctx["start"][0], end[1] - ctx["start"][1])
        intended = self.speed * ctx["duration"]
        blocked = intended > 1.0 and moved < self.block_ratio * intended
        state.humanoid.position = Vector(end[0], end[1])
        result = {
            "turn": action.compact(),
            "result": "BLOCKED_BY_OBSTACLE" if blocked else f"STEP_FORWARD duration={ctx['duration']:.2f} direction={ctx['direction']}",
            "moved_cm": round(moved, 1),
            "intended_cm": round(intended, 1),
            "location": end,
        }
        building = self._collision_counts(agent_id).get("building")
        if building is not None:
            result["building_collisions"] = building
        return result

    def _engine_step_blocking(self, agent_id: str, action: VenueAgentTurn) -> dict[str, Any]:
        """Single-agent engine step (issue, wait, halt, measure)."""

        ctx = self._begin_step(agent_id, action)
        if ctx.get("blocked_entrance"):
            return {"turn": action.compact(), "result": "BLOCKED_BY_ENTRANCE", "moved_cm": 0.0, "location": ctx["start"]}
        time.sleep(ctx["duration"] + self.move_settle_sec)
        self.communicator.humanoid_stop(self.get_agent_state(agent_id).humanoid.id)
        self._tick()
        return self._finalize_step(agent_id, action, ctx)

    def _sample_motion(self, walk_time: float) -> None:
        """Grab enhanced ego frames while agents walk so video shows the gait.

        Runs for ``walk_time`` of wall-clock (matching the non-recording sleep so
        the engine character travels the same distance) and captures every agent's
        camera roughly ``motion_fps`` times, buffering them for the runner.
        """

        start = time.monotonic()
        interval = 1.0 / max(1.0, self.motion_fps)
        next_sample = start
        while True:
            now = time.monotonic()
            if now - start >= walk_time:
                break
            try:
                frames = self._capture_frames("lit")
                self._motion_frames.append({agent_id: self._enhance_frame(frame) for agent_id, frame in frames.items()})
            except Exception:  # noqa: BLE001 - a dropped recording frame must never abort a step.
                pass
            next_sample += interval
            sleep_for = min(next_sample, start + walk_time) - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def drain_motion_frames(self) -> list[dict[str, Any]]:
        """Return and clear the in-motion frames captured during the last step."""

        frames = self._motion_frames
        self._motion_frames = []
        return frames

    def _collision_counts(self, agent_id: str) -> dict[str, int]:
        """Best-effort cumulative collision counts (schema-tolerant)."""

        try:
            raw = self.communicator.unrealcv.get_collision_num(self.get_agent_state(agent_id).actor_name)
            data = json.loads(raw)
            return {key.replace("Collision", "").lower(): int(value) for key, value in data.items()}
        except Exception:  # noqa: BLE001 - collision counters are diagnostics only.
            return {}

    def _kinematic_rotate(self, actor_name: str, humanoid: Humanoid, angle: float, direction: str) -> dict[str, Any]:
        """Rotate deterministically."""

        orientation = self.communicator.unrealcv.get_orientation(actor_name)
        yaw = float(orientation[1])
        yaw_delta = -angle if direction == "right" else angle
        new_yaw = normalize_angle(yaw + yaw_delta)
        self.communicator.unrealcv.set_orientation((float(orientation[0]), new_yaw, float(orientation[2])), actor_name)
        humanoid.direction = new_yaw
        return {"result": f"TURN_AROUND angle={angle:.1f} direction={direction}", "yaw_deg": new_yaw}

    def _inspect(self, agent_id: str, action: VenueAgentTurn) -> dict[str, Any]:
        """Validate a visually grounded venue inspection."""

        venue = self._resolve_inspect_target(agent_id, action)
        if venue is None:
            result = {"result": "INSPECT_FAILED", "reason": "unknown target"}
            self.last_inspections[agent_id] = result
            return result

        if not self._can_inspect_zone(agent_id, venue):
            result = {
                "result": "INSPECT_FAILED",
                "venue_id": venue.venue_id,
                "reason": "outside your area",
                "agent_visible_result": "this venue is in your teammate's area; ask them to inspect it and report back",
            }
            self.last_inspections[agent_id] = result
            return result

        state = self.get_agent_state(agent_id)
        kin = self.get_kinematic_state(agent_id)
        agent_xy = (kin["position"].x, kin["position"].y)
        # Physical-presence gate: you must be standing in the venue's meeting
        # region to inspect it - the *same* region used for convergence. Building
        # meshes are large and `venue.position` is the mesh pivot (often deep
        # behind the facade), so the old "distance-to-pivot <= inspect_range"
        # gate was unsatisfiable for big buildings even while standing at them,
        # and the mask-pixel threshold was flaky. Reusing region.contains makes a
        # NAVIGATE landing inspect-valid by construction (see notes.md section 2).
        at_venue = venue.region.contains(agent_xy)
        distance_to_center = kin["position"].distance(Vector(venue.region.center[0], venue.region.center[1]))
        # Auto-face the facade and capture an object-mask frame. Visibility is now
        # a logged diagnostic, not a hard gate: the redesign defers image-grounded
        # traits ("P vs PV" in notes.md), and presence + auto-face already put the
        # building in view. mask_pixels stays in the log so a future visual mode
        # can re-promote it to a gate.
        self._face_point(state.actor_name, state.humanoid, (venue.position[0], venue.position[1]))
        self._tick()
        mask_frame = self.communicator.get_camera_observation(state.humanoid.camera_id, "object_mask", mode=self.camera_mode)
        mask_pixels = self._count_mask_pixels(mask_frame, venue.mask_color_rgb)
        valid = at_venue
        result = {
            "result": "INSPECT_OK" if valid else "INSPECT_FAILED",
            "venue_id": venue.venue_id,
            "target_description": action.target_description,
            "distance_to_center_internal": round(distance_to_center, 2),
            "region_radius_internal": round(float(venue.region.radius), 2),
            "mask_pixels_internal": int(mask_pixels),
            "agent_visible_result": (
                "focused camera frame returned"
                if valid
                else "you are not at this venue yet - NAVIGATE to it first, then INSPECT"
            ),
        }
        if valid:
            self.inspected_venues.add(venue.venue_id)
            # Structured reveal: hand the agent the decision-relevant traits as
            # text (the social/pure-perception decision in notes.md), and record
            # them as this agent's first-hand knowledge for the process metrics.
            facts = self._venue_facts(venue)
            result["facts"] = facts
            self.revealed_facts.setdefault(agent_id, {})[venue.venue_id] = facts
        self.last_inspections[agent_id] = result
        return result

    def _can_inspect_zone(self, agent_id: str, venue: Venue) -> bool:
        """Return whether the partition mode lets this agent inspect this venue."""

        if self.info_partition != "spatial":
            return True
        agent_zone = self._agent_zone.get(agent_id)
        # Unzoned venues (or unzoned agents) stay public even under partitioning.
        if venue.zone_id is None or agent_zone is None:
            return True
        return venue.zone_id == agent_zone

    def _venue_facts(self, venue: Venue) -> dict[str, Any]:
        """Decision-relevant ground-truth traits, surfaced on a successful inspect."""

        return venue_decision_facts(venue, self.scenario.soft_weights)

    def _resolve_inspect_target(self, agent_id: str, action: VenueAgentTurn) -> Venue | None:
        """Resolve inspect target by id or description."""

        if action.target_venue_id:
            for venue in self.scenario.venues:
                if venue.venue_id == action.target_venue_id:
                    return venue
        if action.target_description:
            query = action.target_description.lower()
            for venue in self.scenario.venues:
                if query in venue.venue_id.lower() or query in venue.visual_summary.lower() or query in venue.venue_type:
                    return venue
        kin = self.get_kinematic_state(agent_id)
        return min(self.scenario.venues, key=lambda venue: kin["position"].distance(Vector(venue.region.center[0], venue.region.center[1])))

    def _face_point(self, actor_name: str, humanoid: Humanoid, point: tuple[float, float]) -> None:
        """Set actor yaw toward a target point."""

        location = self.communicator.unrealcv.get_location(actor_name)
        dx = float(point[0]) - float(location[0])
        dy = float(point[1]) - float(location[1])
        yaw = math.degrees(math.atan2(dy, dx))
        orientation = self.communicator.unrealcv.get_orientation(actor_name)
        self.communicator.unrealcv.set_orientation((float(orientation[0]), yaw, float(orientation[2])), actor_name)
        humanoid.direction = yaw

    def _count_mask_pixels(self, frame: np.ndarray, color_rgb: tuple[int, int, int]) -> int:
        """Count approximate venue-color pixels in an object-mask frame."""

        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 3:
            return 0
        rgb = np.array(color_rgb, dtype=np.int16)
        bgr = np.array((color_rgb[2], color_rgb[1], color_rgb[0]), dtype=np.int16)
        pixels = frame[:, :, :3].astype(np.int16)
        rgb_hits = np.all(np.abs(pixels - rgb) <= 8, axis=2)
        bgr_hits = np.all(np.abs(pixels - bgr) <= 8, axis=2)
        return int(np.count_nonzero(rgb_hits | bgr_hits))

    def _blocked_by_entrance(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        """Logically block movement through blocked venue entrances."""

        for venue in self.scenario.venues:
            for entrance in venue.entrances:
                if entrance.status != "blocked":
                    continue
                entrance_xy = (entrance.position[0], entrance.position[1])
                if self._point_to_segment_distance(entrance_xy, start, end) <= self.entrance_block_radius:
                    return True
        return False

    def _point_to_segment_distance(self, point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
        """Distance from point to segment."""

        px, py = point
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
        cx = sx + t * dx
        cy = sy + t * dy
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    def _converged(self) -> bool:
        """Return whether all agents are at the same venue region."""

        final_venue_id, venue_agents = final_venue_from_positions(self.scenario, self._positions())
        return bool(final_venue_id and len(venue_agents.get(final_venue_id, [])) == len(self.agent_ids))

    def relative_angle_to_venue(self, agent_id: str, venue_id: str | None) -> float:
        """Return relative angle from agent heading to a venue."""

        if not venue_id:
            return 0.0
        venue = self.scenario.venue_by_id(venue_id)
        state = self.get_kinematic_state(agent_id)
        target_vector = Vector(venue.position[0], venue.position[1]) - state["position"]
        target_yaw = math.degrees(math.atan2(target_vector.y, target_vector.x))
        return normalize_angle(target_yaw - float(state["yaw_deg"]))

    def observation_summary(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Drop image arrays from observations for compact logs."""

        return {key: value for key, value in observation.items() if key != "ego_view"}

    def disconnect(self) -> None:
        """Disconnect the underlying UnrealCV client."""

        self.communicator.unrealcv.disconnect()

    @staticmethod
    def venue_actor_name(venue_id: str) -> str:
        """Return deterministic UE actor name for a venue.

        The ``GEN_BP_`` prefix lets ``clear_env(keep_roads=True)`` wipe every
        scene actor on the next reset; otherwise venues/landmarks/props would
        accumulate and collide across episodes and scenarios.
        """

        return f"GEN_BP_VENUE_{venue_id}"

    @staticmethod
    def landmark_actor_name(landmark_id: str) -> str:
        """Return deterministic UE actor name for a landmark."""

        return f"GEN_BP_LANDMARK_{landmark_id}"

    @staticmethod
    def prop_actor_name(prop_id: str) -> str:
        """Return deterministic UE actor name for a prop."""

        return f"GEN_BP_PROP_{prop_id}"
