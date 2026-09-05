"""Obstacle-aware and layout-graph path planning for walk-mode NAVIGATE.

In walk mode an agent must physically traverse the plaza to a venue instead of
teleporting (see notes.md: teleporting collapses the embodied task into pure
graph traversal). SimWorld plans navigation in Python and then drives the
humanoid's locomotion - there is no engine-side navmesh ``MoveTo`` exposed over
UnrealCV - so we plan a collision-aware polyline here and let the env walk it
with real ``StepForward`` locomotion.

When a scenario carries a :class:`~benchmark.venue_meetup.layout.DistrictLayout`
and the agent has a known walk-graph node, routes prefer that authored sidewalk /
crossing / bridge graph. Building keep-out discs remain the legacy free-space
fallback for plaza templates without usable layout graph data.

Building keep-out regions are derived purely from scenario geometry, so the
planner is template-agnostic and needs no UE bounding-box query (UnrealCV
exposes none). Venue and landmark proxies remain single inflated discs. Authored
district shells use their measured world-axis AABBs instead: each shell is a
deterministic overlapping chain of discs along its long axis, with radius equal
to the short half-extent plus caller clearance. This avoids the unnecessary
lateral over-width of one circumscribed disc. The route is found with an
8-connected grid A* and then simplified by line-of-sight string-pulling into a
few waypoints.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from benchmark.venue_meetup.building_catalog import building_bbox
from benchmark.venue_meetup.layout import (DistrictLayout, Frontage, WalkEdge,
                                           WalkRouteKind)

if TYPE_CHECKING:
    from benchmark.venue_meetup.district_geometry import DistrictShellFootprint

Point = tuple[float, float]
WalkPlannerKind = Literal["layout_graph", "obstacle_astar"]


def meeting_target(
    agent_index: int,
    center: Point,
    *,
    offset: float = 300.0,
    frontage_yaw_deg: float | None = None,
    agent_count: int | None = None,
) -> Point:
    """Return the deterministic per-agent fan point around a meeting center.

    When frontage orientation and group size are supplied, agents fan along the
    storefront tangent. This keeps every pawn the same safe distance from a
    rotated facade instead of placing one member inward into the building.
    Omitting those arguments preserves the legacy quarter-turn fan. Both
    NAVIGATE modes use the same rounded coordinates so teleport and walk agree.
    """

    if frontage_yaw_deg is None:
        angle_deg = 90.0 * agent_index
    else:
        count = max(1, int(agent_count or 1))
        angle_deg = float(frontage_yaw_deg) - 90.0 + 360.0 * agent_index / count
    angle = math.radians(angle_deg)
    return round(center[0] + math.cos(angle) * offset, 2), round(center[1] + math.sin(angle) * offset, 2)


@dataclass(frozen=True)
class LayoutRoute:
    """Deterministic walk-graph route from a start node to a venue frontage.

    ``node_ids`` is the graph path from the spawn/current node to the
    frontage's *approach* node (a public sidewalk/crossing/intersection node).
    ``waypoints`` are flattened edge polylines along that path (start
    exclusive, approach-node inclusive).  ``access_path`` goes from the
    approach node to the meeting-region centre.
    """

    node_ids: tuple[str, ...]
    edge_ids: tuple[tuple[str, str], ...]
    waypoints: tuple[Point, ...]
    graph_distance_cm: float
    access_path: tuple[Point, ...]
    access_distance_cm: float
    total_distance_cm: float
    route_kinds: tuple[WalkRouteKind, ...]
    frontage_id: str
    end_node_id: str
    used_bridge: bool


def select_walk_planner(*, layout: DistrictLayout | None, walk_node_id: str | None) -> WalkPlannerKind:
    """Choose graph-backed planning when layout + current walk node are usable."""

    if layout is not None and walk_node_id is not None and layout.walk_nodes:
        return "layout_graph"
    return "obstacle_astar"


def _resolve_frontage(
    layout: DistrictLayout,
    *,
    venue_slot_id: str | None,
    frontage_id: str | None,
) -> Frontage:
    """Resolve a frontage from either ``venue_slot_id`` or ``frontage_id``."""

    if (venue_slot_id is None) == (frontage_id is None):
        raise ValueError("Provide exactly one of venue_slot_id or frontage_id")

    if frontage_id is not None:
        return layout.frontage_by_id(frontage_id)

    assert venue_slot_id is not None
    matches = [frontage for frontage in layout.frontages if frontage.venue_slot_id == venue_slot_id]
    if not matches:
        raise ValueError(f"Unknown venue_slot_id: {venue_slot_id}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate venue_slot_id: {venue_slot_id}")
    return matches[0]


def _find_route_edge(
    layout: DistrictLayout,
    start_node_id: str,
    end_node_id: str,
) -> WalkEdge:
    """Return the best enabled undirected edge between two adjacent path nodes."""

    matches: list[WalkEdge] = []
    for edge in layout.walk_edges:
        if not edge.enabled:
            continue
        if {edge.start_node_id, edge.end_node_id} == {start_node_id, end_node_id}:
            matches.append(edge)
    if not matches:
        raise ValueError(f"Missing walk edge on path: {start_node_id} -> {end_node_id}")
    return sorted(matches, key=lambda e: e.route_kind)[0]


def _edge_route_kind(layout: DistrictLayout, start_node_id: str, end_node_id: str) -> WalkRouteKind:
    """Return the route kind for the enabled undirected edge between two nodes."""

    return _find_route_edge(layout, start_node_id, end_node_id).route_kind


def plan_layout_route(
    layout: DistrictLayout,
    start_node_id: str,
    *,
    venue_slot_id: str | None = None,
    frontage_id: str | None = None,
) -> LayoutRoute | None:
    """Plan a deterministic layout-graph route to a venue slot or frontage.

    The route ends at the frontage's ``approach_node_id`` (a public walk
    node).  The frontage ``access_path`` is recorded separately and never
    participates in graph search.  Returns ``None`` when the approach node is
    unreachable via enabled edges.

    Legacy compatibility: when ``approach_node_id`` is *None*, falls back to
    using the frontage id as a walk-node id (the old "frontage-as-node"
    scheme) so stored layouts keep loading.
    """

    frontage = _resolve_frontage(layout, venue_slot_id=venue_slot_id, frontage_id=frontage_id)

    approach_id = frontage.approach_node_id
    if approach_id is not None:
        try:
            layout.node_by_id(approach_id)
        except ValueError as exc:
            raise ValueError(
                f"Frontage {frontage.frontage_id!r} approach_node_id="
                f"{approach_id!r} is not a walk node"
            ) from exc
    else:
        try:
            layout.node_by_id(frontage.frontage_id)
            approach_id = frontage.frontage_id
        except ValueError as exc:
            raise ValueError(
                f"Frontage {frontage.frontage_id!r} has no approach_node_id "
                "and no matching walk node"
            ) from exc

    path = layout.shortest_path(start_node_id, approach_id)
    if path is None:
        return None

    nodes = {node.node_id: node for node in layout.walk_nodes}
    flat_waypoints: list[Point] = []
    edge_ids: list[tuple[str, str]] = []
    route_kinds: list[WalkRouteKind] = []

    for left, right in zip(path, path[1:]):
        edge = _find_route_edge(layout, left, right)
        edge_ids.append((left, right))
        route_kinds.append(edge.route_kind)
        wps = edge.waypoints if edge.start_node_id == left else tuple(reversed(edge.waypoints))
        flat_waypoints.extend(wps)
        flat_waypoints.append(nodes[right].position)

    graph_distance = layout.path_length_cm(start_node_id, approach_id)
    if graph_distance is None:
        return None

    access = tuple(frontage.access_path)
    approach_pos = nodes[approach_id].position
    access_dist = path_length(approach_pos, list(access))

    return LayoutRoute(
        node_ids=tuple(path),
        edge_ids=tuple(edge_ids),
        waypoints=tuple(flat_waypoints),
        graph_distance_cm=float(graph_distance),
        access_path=access,
        access_distance_cm=access_dist,
        total_distance_cm=float(graph_distance) + access_dist,
        route_kinds=tuple(route_kinds),
        frontage_id=frontage.frontage_id,
        end_node_id=approach_id,
        used_bridge=any(kind == "bridge" for kind in route_kinds),
    )


@dataclass(frozen=True)
class Obstacle:
    """A circular keep-out region in Unreal centimeters."""

    cx: float
    cy: float
    radius: float

    def contains(self, point: Point) -> bool:
        """Return whether ``point`` lies inside the disc."""

        return math.hypot(point[0] - self.cx, point[1] - self.cy) <= self.radius

    def with_radius(self, radius: float) -> "Obstacle":
        """Return a copy with a different radius."""

        return Obstacle(self.cx, self.cy, max(0.0, radius))


def _shell_obstacles_for_footprint(
    footprint: "DistrictShellFootprint",
    *,
    clearance: float,
) -> tuple[Obstacle, ...]:
    """Cover a shell AABB with deterministic discs along its long axis.

    A single circumscribed disc is needlessly wide for a frontage shell.  A
    row of discs with radius ``short_half_extent + clearance`` follows the
    long axis instead.  Endpoint centres span the complete long half-extent;
    interior spacing is tightened to keep adjacent discs overlapping across
    the rectangle's short-edge corners whenever caller clearance is positive.
    """

    hx, hy = (float(value) for value in footprint.half_extents)
    if not all(math.isfinite(value) and value > 0.0 for value in (hx, hy)):
        return ()
    caller_clearance = float(clearance)
    if not math.isfinite(caller_clearance):
        raise ValueError(f"clearance must be finite: {clearance!r}")
    long_axis = 0 if hx >= hy else 1
    long_half = max(hx, hy)
    short_half = min(hx, hy)
    radius = short_half + caller_clearance
    if radius <= 0.0:
        return ()

    # At the short edge (|short-axis| == short_half), each disc covers a
    # long-axis half-span of sqrt(radius² - short_half²).  Use that span to
    # choose overlapping centres.  With zero caller clearance the exact span is
    # zero, so retain a finite centreline-overlap fallback for compatibility;
    # callers that need continuous edge coverage should provide positive
    # clearance (the normal walk-mode configuration does).
    edge_reach_sq = radius * radius - short_half * short_half
    edge_reach = math.sqrt(max(0.0, edge_reach_sq))
    spacing_limit = 2.0 * edge_reach if edge_reach > 1e-6 else 2.0 * radius
    spacing_limit = max(spacing_limit, 1e-6)
    span = 2.0 * long_half
    segments = max(1, int(math.ceil(span / spacing_limit)))
    step = span / segments
    cx, cy = (float(value) for value in footprint.position)
    centres: list[Obstacle] = []
    for index in range(segments + 1):
        offset = -long_half + step * index
        center = (cx + offset, cy) if long_axis == 0 else (cx, cy + offset)
        centres.append(Obstacle(center[0], center[1], radius))
    return tuple(centres)


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    """Shortest distance from ``point`` to segment ``start``-``end``."""

    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    cx = sx + t * dx
    cy = sy + t * dy
    return math.hypot(px - cx, py - cy)


def segment_clear(start: Point, end: Point, obstacles: list[Obstacle]) -> bool:
    """Return whether the straight segment misses every obstacle disc."""

    for obstacle in obstacles:
        if _point_to_segment_distance((obstacle.cx, obstacle.cy), start, end) < obstacle.radius:
            return False
    return True


def building_obstacles(
    scenario,
    *,
    clearance: float = 500.0,
    landmark_radius: float = 2000.0,
    min_radius: float = 700.0,
) -> list[Obstacle]:
    """Build keep-out discs for every solid building in the scene.

    A venue building spans from its plaza-side facade (its meeting region center)
    back to its mesh pivot, so we model it as a disc centered on the *midpoint*
    of that span with radius = half the depth (+ clearance). This is far better
    than a pivot-centered, reach-sized disc: it (a) covers the building's real
    extent including the near/facade side, (b) leaves the plaza-side meeting
    point routable instead of walling it off, and (c) does not balloon laterally
    for deep buildings (a hotel's disc stays centered deep in its own block
    rather than reaching across the plaza). Half-depth is used as the lateral
    proxy since UnrealCV exposes no real footprint; the walker's collision-aware
    recovery absorbs the residual inaccuracy.

    The target venue is intentionally NOT special-cased: an agent that starts
    behind a building must still route around it to reach the facade, and the
    goal stays free because it sits just outside the midpoint disc.
    """

    obstacles: list[Obstacle] = []
    for venue in scenario.venues:
        facade = (float(venue.region.center[0]), float(venue.region.center[1]))
        pivot = (float(venue.position[0]), float(venue.position[1]))
        depth = math.hypot(pivot[0] - facade[0], pivot[1] - facade[1])
        center = ((pivot[0] + facade[0]) / 2.0, (pivot[1] + facade[1]) / 2.0)
        radius = max(depth / 2.0, min_radius) + clearance
        obstacles.append(Obstacle(center[0], center[1], radius))
    for landmark in scenario.landmarks:
        obstacles.append(
            Obstacle(
                float(landmark.position[0]),
                float(landmark.position[1]),
                landmark_radius + clearance,
            )
        )
    for building in getattr(scenario, "buildings", ()):
        if not getattr(building, "collision", True):
            continue
        raw_x, raw_y, _raw_z = building_bbox(building.asset_key)
        half_x = raw_x * float(building.scale[0]) / 2.0
        half_y = raw_y * float(building.scale[1]) / 2.0
        radians = math.radians(float(building.yaw_deg))
        cosine, sine = abs(math.cos(radians)), abs(math.sin(radians))
        world_half_x = cosine * half_x + sine * half_y
        world_half_y = sine * half_x + cosine * half_y
        obstacles.append(
            Obstacle(
                float(building.position[0]),
                float(building.position[1]),
                math.hypot(world_half_x, world_half_y) + clearance,
            )
        )
    # Layout-backed district shells are authored visual geometry with measured
    # conservative AABBs.  Tile each shell with a deterministic disc chain so
    # obstacle-A* fallback cannot walk through a solid shell when a graph route
    # is unavailable, without adding the footprint's legacy circumscribed
    # margin a second time.  Import locally to keep the planner free of a
    # module-level dressing dependency and to preserve central-square scenarios
    # (which have no layout shells).
    if getattr(scenario, "layout", None) is not None:
        from benchmark.venue_meetup.district_dressing import \
            plan_shell_footprints

        for footprint in plan_shell_footprints(scenario):
            obstacles.extend(_shell_obstacles_for_footprint(footprint, clearance=clearance))
    return obstacles


def _free_obstacles(point: Point, obstacles: list[Obstacle]) -> list[Obstacle]:
    """Shrink any obstacle that swallows ``point`` so the cell is routable.

    The start (and occasionally a fanned goal) can land just inside an inflated
    disc; rather than declaring the route impossible, locally pull that disc in
    to just clear the point so A* has somewhere legal to begin/end.
    """

    adjusted: list[Obstacle] = []
    for obstacle in obstacles:
        distance = math.hypot(point[0] - obstacle.cx, point[1] - obstacle.cy)
        if distance < obstacle.radius:
            adjusted.append(obstacle.with_radius(max(0.0, distance - 1.0)))
        else:
            adjusted.append(obstacle)
    return adjusted


def _auto_bounds(points: list[Point], obstacles: list[Obstacle], margin: float) -> tuple[float, float, float, float]:
    """Axis-aligned planning bounds covering all points and obstacle extents."""

    xs = [p[0] for p in points] + [o.cx + o.radius for o in obstacles] + [o.cx - o.radius for o in obstacles]
    ys = [p[1] for p in points] + [o.cy + o.radius for o in obstacles] + [o.cy - o.radius for o in obstacles]
    if not xs:
        xs = [0.0]
    if not ys:
        ys = [0.0]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def _string_pull(points: list[Point], obstacles: list[Obstacle]) -> list[Point]:
    """Greedily drop intermediate points whose removal keeps line-of-sight."""

    if len(points) <= 2:
        return list(points)
    result = [points[0]]
    anchor = 0
    for i in range(2, len(points)):
        if not segment_clear(points[anchor], points[i], obstacles):
            result.append(points[i - 1])
            anchor = i - 1
    result.append(points[-1])
    return result


def _astar_grid(
    start: Point,
    goal: Point,
    obstacles: list[Obstacle],
    *,
    cell: float,
    bounds: tuple[float, float, float, float],
) -> list[Point] | None:
    """8-connected grid A* between the cells nearest ``start`` and ``goal``."""

    min_x, min_y, max_x, max_y = bounds
    nx = max(1, int(math.ceil((max_x - min_x) / cell)))
    ny = max(1, int(math.ceil((max_y - min_y) / cell)))

    def to_world(ix: int, iy: int) -> Point:
        return (min_x + ix * cell, min_y + iy * cell)

    def to_cell(point: Point) -> tuple[int, int]:
        ix = int(round((point[0] - min_x) / cell))
        iy = int(round((point[1] - min_y) / cell))
        return max(0, min(nx, ix)), max(0, min(ny, iy))

    start_cell = to_cell(start)
    goal_cell = to_cell(goal)

    def blocked(cell_xy: tuple[int, int]) -> bool:
        if cell_xy in (start_cell, goal_cell):
            return False
        world = to_world(*cell_xy)
        return any(obstacle.contains(world) for obstacle in obstacles)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (0.0, counter, start_cell))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}

    def heuristic(cell_xy: tuple[int, int]) -> float:
        return math.hypot(cell_xy[0] - goal_cell[0], cell_xy[1] - goal_cell[1]) * cell

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal_cell:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            return [to_world(ix, iy) for ix, iy in cells]
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] <= nx and 0 <= neighbor[1] <= ny):
                continue
            if blocked(neighbor):
                continue
            if dx != 0 and dy != 0:
                # Forbid cutting through a blocked diagonal corner.
                if blocked((current[0] + dx, current[1])) and blocked((current[0], current[1] + dy)):
                    continue
            step_cost = cell * (math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0)
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic(neighbor), counter, neighbor))
    return None


def plan_path(
    start: Point,
    goal: Point,
    obstacles: list[Obstacle],
    *,
    cell: float = 400.0,
    bounds: tuple[float, float, float, float] | None = None,
) -> list[Point]:
    """Return walk waypoints from ``start`` (exclusive) to ``goal`` (inclusive).

    Falls back gracefully: a clear straight shot returns ``[goal]``; if the grid
    search fails (e.g. over-inflated discs seal the goal) the obstacles are
    relaxed and retried, and as a last resort a straight line is returned so the
    walker can still make progress (and surface a block).
    """

    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    safe_obstacles = _free_obstacles(goal, _free_obstacles(start, obstacles))

    if segment_clear(start, goal, safe_obstacles):
        return [goal]

    if bounds is None:
        bounds = _auto_bounds([start, goal], safe_obstacles, margin=2.0 * cell)

    for shrink in (1.0, 0.85, 0.7):
        attempt = [o.with_radius(o.radius * shrink) for o in safe_obstacles] if shrink != 1.0 else safe_obstacles
        cells = _astar_grid(start, goal, attempt, cell=cell, bounds=bounds)
        if cells is None:
            continue
        points = [start] + cells + [goal]
        pulled = _string_pull(points, attempt)
        waypoints = [wp for wp in pulled[1:]]
        if waypoints and waypoints[-1] != goal:
            waypoints.append(goal)
        return waypoints or [goal]

    return [goal]


def path_length(start: Point, waypoints: list[Point]) -> float:
    """Total polyline length from ``start`` through ``waypoints``."""

    total = 0.0
    prev = start
    for wp in waypoints:
        total += math.hypot(wp[0] - prev[0], wp[1] - prev[1])
        prev = wp
    return total
