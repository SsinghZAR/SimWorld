"""Plan once, consume physical route chunks on the deterministic episode clock."""

from __future__ import annotations

import math
from dataclasses import dataclass

from benchmark.venue_meetup.navigation import path_length, plan_layout_route, plan_path


def route_chunks(start, waypoints, chunk_cm):
    """Split a polyline without cutting corners; each chunk <= the distance budget."""

    if not math.isfinite(chunk_cm) or chunk_cm <= 0:
        raise ValueError("Chunk length must be positive and finite")
    position = tuple(start)
    chunks, current = [], []
    remaining = chunk_cm
    for endpoint in waypoints:
        endpoint = tuple(endpoint)
        distance = math.dist(position, endpoint)
        while distance > remaining + 1e-6:
            ratio = remaining / distance
            position = tuple(a + (b - a) * ratio for a, b in zip(position, endpoint))
            current.append(position)
            chunks.append(current)
            current, remaining = [], chunk_cm
            distance = math.dist(position, endpoint)
        if distance > 1e-6:
            current.append(endpoint)
            remaining -= distance
        position = endpoint
        if remaining <= 1e-6:
            chunks.append(current)
            current, remaining = [], chunk_cm
    if current:
        chunks.append(current)
    return chunks or [[]]


@dataclass
class TimedRoute:
    venue_id: str
    chunks: list[list[tuple[float, float]]]
    distance_cm: float
    end_node_id: str | None
    moved_cm: float = 0.0
    blocked: bool = False
    arrived_early: bool = False


def plan_timed_route(env, agent, venue) -> TimedRoute:
    start = env._actor_xy(env.get_agent_state(agent).actor_name)
    target = env._meeting_target(agent, venue)
    end_node = env._frontage_node_id_for_venue(venue)
    if venue.region.contains(start):
        return TimedRoute(venue.venue_id, [[]], 0.0, end_node)
    origin_node = env._agent_walk_nodes.get(agent)
    if env.scenario.layout is not None and origin_node:
        route = plan_layout_route(env.scenario.layout, origin_node, venue_slot_id=venue.slot_id)
        if route is None:
            raise ValueError("No walkable route to this venue")
        # Return along the current frontage's access path before taking graph
        # edges. Omitting this segment can cut a corner through a building.
        current = next((candidate for candidate in env.scenario.venues if candidate.region.contains(start)), None)
        prefix = []
        if current is not None:
            back = plan_layout_route(env.scenario.layout, origin_node, venue_slot_id=current.slot_id)
            if back:
                prefix = list(reversed(back.access_path))
                node = next(node for node in env.scenario.layout.walk_nodes if node.node_id == origin_node)
                prefix.append(tuple(node.position))
        points = [*prefix, *route.waypoints, *route.access_path, target]
    else:
        points = plan_path(start, target, env._obstacles or [])
        if points is None:
            raise ValueError("No walkable route to this venue")
    distance = path_length(start, points)
    chunks = route_chunks(start, points, env.timing.travel_metres_per_tick * 100)
    return TimedRoute(venue.venue_id, chunks, distance, end_node)


def advance_route(env, agent, route: TimedRoute, chunk_index: int) -> bool:
    """Return whether a physical blockage ended the action early."""

    venue = env.scenario.venue_by_id(route.venue_id)
    final = chunk_index == len(route.chunks) - 1
    if route.arrived_early:
        return False
    if env.navigate_mode == "teleport":
        # Abstracted social control: no physical motion until the travel budget
        # has elapsed. It still cannot beat the same route-based deadline.
        if final:
            env._teleport_navigate(agent, venue)
        return False
    points = route.chunks[chunk_index]
    for index, point in enumerate(points):
        moved, reached = env._walk_segment(agent, point,
                                            last_venue=venue if final and index == len(points) - 1 else None)
        route.moved_cm += moved
        if not reached:
            position = env._actor_xy(env.get_agent_state(agent).actor_name)
            if venue.region.contains(position):
                # Keep the advertised completion time even if the controller
                # meets the region tolerance before the exact fan waypoint.
                route.arrived_early = True
                return False
            route.blocked = True
            return True
    return False


def finish_route(env, agent, route: TimedRoute):
    venue = env.scenario.venue_by_id(route.venue_id)
    state = env.get_agent_state(agent)
    position = env._actor_xy(state.actor_name)
    # A waypoint can be obstructed by the waiting teammate after the traveller
    # has already entered the valid meeting region. Region arrival, not reaching
    # the exact fan point, is the task's physical completion condition.
    arrived = venue.region.contains(position)
    if arrived:
        env._agent_walk_nodes[agent] = route.end_node_id
        env._face_point(state.actor_name, state.humanoid, venue.position[:2])
        env._settle_camera_after_turn()
    elif route.blocked:
        # Do not route a subsequent attempt from an out-of-date graph origin.
        env._agent_walk_nodes[agent] = None
    return {"result": "NAVIGATE_OK" if arrived else "NAVIGATE_BLOCKED",
            "venue_id": route.venue_id, "arrived": arrived, "mode": env.navigate_mode,
            "planned_distance_cm": route.distance_cm, "moved_cm": route.moved_cm,
            "reason": None if arrived else "The route ended before reaching the venue."}
