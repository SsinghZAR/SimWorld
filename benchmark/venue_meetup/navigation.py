"""Obstacle-aware path planning for walk-mode NAVIGATE.

In walk mode an agent must physically traverse the plaza to a venue instead of
teleporting (see notes.md: teleporting collapses the embodied task into pure
graph traversal). SimWorld plans navigation in Python and then drives the
humanoid's locomotion - there is no engine-side navmesh ``MoveTo`` exposed over
UnrealCV - so we plan a collision-aware polyline here and let the env walk it
with real ``StepForward`` locomotion.

Building footprints are modeled as inflated keep-out discs derived purely from
scenario geometry (a building's pivot, and for a venue its plaza-side meeting
region), so the planner is template-agnostic and needs no UE bounding-box query
(UnrealCV exposes none). The route is found with an 8-connected grid A* and then
simplified by line-of-sight string-pulling into a few waypoints.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

Point = tuple[float, float]


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
        obstacles.append(Obstacle(float(landmark.position[0]), float(landmark.position[1]), landmark_radius + clearance))
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


def _astar_grid(start: Point, goal: Point, obstacles: list[Obstacle], *, cell: float, bounds: tuple[float, float, float, float]) -> list[Point] | None:
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
