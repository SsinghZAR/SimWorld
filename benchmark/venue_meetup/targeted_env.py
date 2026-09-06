"""Targeted, independently timed protocol over the existing physical UE adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from benchmark.venue_meetup._core.action_space import (
    VenueAction,
    VenueAgentTurn,
    sanitize_turn,
)
from benchmark.venue_meetup._core.comms import MessageBus, messages_from_turns
from benchmark.venue_meetup.interaction_runtime import InteractionRuntime
from benchmark.venue_meetup.interactions import action_durations
from benchmark.venue_meetup.observations import public_action_result
from benchmark.venue_meetup.scoring import episode_score
from benchmark.venue_meetup.timed_navigation import (
    advance_route,
    finish_route,
    plan_timed_route,
)
from benchmark.venue_meetup.timing import ActionScheduler, TimingConfig
from benchmark.venue_meetup.venue_env import VenueMeetupEnv


class TargetedVenueEnv(VenueMeetupEnv):
    """No incremental rewards; final scores are evaluator-only terminal records."""

    def __init__(self, *args, timing: TimingConfig | None = None, **kwargs):
        self.timing = timing or TimingConfig()
        super().__init__(*args, episode_clock=self.timing, **kwargs)
        # Preserve every message received during a long, non-interruptible trip.
        self.bus = MessageBus(self.agent_ids, router=self.bus.router,
                              max_history=self.timing.max_ticks * len(self.agent_ids))
        self.scheduler = ActionScheduler(self.agent_ids, self.timing)
        self.interactions = InteractionRuntime(self)
        self.terminated = False

    def reset(self):
        self.scheduler = ActionScheduler(self.agent_ids, self.timing)
        self.interactions = InteractionRuntime(self)
        self.terminated = False
        return super().reset()

    def _spawn_static_scene(self):
        super()._spawn_static_scene()
        self.interactions.spawn()

    def _inspect(self, agent_id, action):
        """Never inherit the legacy full-venue inspection fallback."""
        return self.interactions.inspect(agent_id, action)

    @property
    def ready_agent_ids(self):
        return self.scheduler.ready if not self.terminated else ()

    def _build_observations(self, inboxes=None):
        observations = super()._build_observations(inboxes)
        self.interactions.augment(observations)
        for agent, observation in observations.items():
            observation["protocol"] = "targeted_v1"
            observation["timing_config"] = asdict(self.timing)
            observation["action_durations_ticks"] = action_durations()
            observation["own_activity"] = self.scheduler.activity(agent)
            observation["valid_actions"] = {**observation["valid_actions"],
                                            "3": "INSPECT target_interactable_id (permitted, nearby and visible)"}
            navigation = observation.get("navigation")
            if navigation:
                navigation["hint"] = "NAVIGATE follows the walkable route and occupies its estimated ticks. INSPECT targets a nearby information point."
                for target in navigation["targets"]:
                    if target["kind"] != "venue":
                        continue
                    target["guidance"] = "At this venue." if target.get("arrived") else "NAVIGATE to travel here."
                    if agent in self.ready_agent_ids:
                        try:
                            route = plan_timed_route(self, agent, self.scenario.venue_by_id(target["id"]))
                            target["duration_ticks"] = len(route.chunks)
                        except ValueError:
                            target["duration_ticks"] = None
        return observations

    def _prepare(self, agent: str, raw) -> None:
        turn = VenueAgentTurn.from_json(raw)
        try:
            relative = self.relative_angle_to_venue(agent, turn.target_venue_id) if turn.target_venue_id else 0.0
        except KeyError:
            relative = 0.0
        action = sanitize_turn(turn, relative_angle=relative)
        duration, payload = 1, {}
        if action.choice == VenueAction.INSPECT.value:
            error = self.interactions.precheck(agent, action)
            if error:
                payload["failure"] = {"result": "INSPECT_FAILED", "reason": error}
            else:
                duration = self.interactions.resolve(action).kind.ticks
        elif action.choice == VenueAction.NAVIGATE.value:
            venue = None
            if action.target_venue_id:
                try:
                    venue = self.scenario.venue_by_id(action.target_venue_id)
                except KeyError:
                    pass
            elif action.target_description:
                venue = self._resolve_inspect_target(agent, action)
            if venue is None:
                payload["failure"] = {"result": "NAVIGATE_FAILED", "reason": "Unknown venue target."}
            else:
                try:
                    route = plan_timed_route(self, agent, venue)
                    duration, payload["route"] = len(route.chunks), route
                except ValueError as exc:
                    payload["failure"] = {"result": "NAVIGATE_FAILED", "reason": str(exc)}
        self.scheduler.start(agent, action, duration, payload)

    def _complete(self, agent, pending):
        action, payload = pending.turn, pending.payload
        if "failure" in payload:
            result = dict(payload["failure"])
        elif "route" in payload:
            result = finish_route(self, agent, payload["route"])
        elif action.choice == VenueAction.INSPECT.value:
            result = self._inspect(agent, action)
        elif action.choice == VenueAction.COMMUNICATE.value:
            result = {"result": "COMMUNICATE", "message": action.message}
        else:
            result = self.execute_action(agent, action)
        result.update(turn=action.compact(), started_tick=pending.started_tick,
                      completed_tick=self.step_index, duration_ticks=self.step_index - pending.started_tick)
        if action.choice == VenueAction.INSPECT.value:
            self.last_inspections_internal[agent] = deepcopy(result)
            self.last_inspections_public[agent] = public_action_result(result)
        return result

    def step(self, turns):
        if self.terminated:
            raise RuntimeError("Episode already completed")
        if set(turns) - set(self.ready_agent_ids):
            raise ValueError("Only available agents may submit actions")
        positions = self._positions()
        self._step_movement_paths = {agent: [point] for agent, point in positions.items()}
        self._motion_frames = []
        # Prepare every action before changing the shared scene.
        for agent in self.ready_agent_ids:
            self._prepare(agent, turns.get(agent, VenueAgentTurn()))
        for agent, pending in self.scheduler.pending.items():
            route = pending.payload.get("route")
            if route is not None:
                index = self.scheduler.tick - pending.started_tick
                if advance_route(self, agent, route, index):
                    pending.duration_ticks = index + 1
        completed = self.scheduler.advance()
        self.step_index = self.scheduler.tick
        actions = {agent: self._complete(agent, pending) for agent, pending in completed.items()}
        # Deliver together, after all completions. No model runs between them.
        messages = messages_from_turns({agent: pending.turn for agent, pending in completed.items()}, step=self.step_index)
        if not self.no_communication:
            self.bus.deliver(messages, positions=self._positions())
        for agent, result in actions.items():
            self.last_actions_internal[agent] = deepcopy(result)
            self.last_actions_public[agent] = public_action_result(result)
        final_positions = self._positions()
        for agent, position in final_positions.items():
            self._record_movement_point(agent, position)
        converged = self._converged() and not self.scheduler.pending
        deadline = self.timing.expired(self.step_index)
        self.terminated = converged or deadline
        info = {
            "step": self.step_index, "actions": actions,
            "activities_internal": {agent: self.scheduler.activity(agent) for agent in self.agent_ids},
            "comms": self.bus.snapshot(), "positions_internal": final_positions,
            "movement_paths_internal": deepcopy(self._step_movement_paths),
            "navigation_mode": self.navigate_mode,
            "closing_clock": self.timing.snapshot(self.step_index),
            "success": False,
        }
        if self.terminated:
            # In-flight agents cannot earn arrival credit by passing through a
            # meeting region on their way elsewhere at the deadline.
            settled = {agent: point for agent, point in final_positions.items()
                       if agent not in self.scheduler.pending}
            scores = episode_score(self.scenario, settled, inspected_venues=self.inspected_venues,
                                   message_count=len(self.bus.transcript), timed_out=deadline and not converged)
            detail = scores.get("final_venue_detail") or {}
            info["success"] = bool(converged and detail and not detail.get("hard_failures"))
            info["scores"] = scores
            info["termination_reason"] = "meetup" if converged else "closing_deadline"
        return self._build_observations(), {}, self.terminated, info
