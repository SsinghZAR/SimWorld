"""Independent geometry calculations used by authored-district tests.

This module intentionally imports no benchmark production code.  It accepts
duck-typed layout/scenario records and a bounding-box lookup supplied by the
tests, so a production predicate cannot make an assertion pass tautologically.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


Point = tuple[float, float]
Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class Edge:
    index: int
    start: Point
    end: Point
    length: float
    tangent: Point
    outward: Point


def distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def segment_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return distance(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return distance(point, (start[0] + t * dx, start[1] + t * dy))


def minimum_route_distance(point: Point, routes: Iterable[tuple[Point, ...]]) -> float:
    return min(
        (segment_distance(point, start, end) for route in routes for start, end in zip(route, route[1:])),
        default=math.inf,
    )


def polygon_area(polygon: Iterable[Point]) -> float:
    points = tuple(polygon)
    return 0.5 * sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(points, (*points[1:], points[0]))
    )


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    points = tuple((float(x), float(y)) for x, y in polygon)
    if len(points) < 3:
        return False
    inside = False
    for left, right in zip(points, (*points[1:], points[0])):
        if segment_distance(point, left, right) <= 1e-6:
            return True
        if (left[1] > point[1]) != (right[1] > point[1]):
            crossing_x = (right[0] - left[0]) * (point[1] - left[1]) / (right[1] - left[1]) + left[0]
            if point[0] < crossing_x:
                inside = not inside
    return inside


def edge_frames(block) -> tuple[Edge, ...]:
    polygon = tuple((float(x), float(y)) for x, y in block.footprint)
    ccw = polygon_area(polygon) >= 0.0
    edges = []
    for index, (start, end) in enumerate(zip(polygon, (*polygon[1:], polygon[0]))):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        tangent = (dx / length, dy / length)
        outward = (tangent[1], -tangent[0]) if ccw else (-tangent[1], tangent[0])
        edges.append(Edge(index, start, end, length, tangent, outward))
    return tuple(edges)


def edge_projection(point: Point, edge: Edge) -> tuple[float, float]:
    dx, dy = point[0] - edge.start[0], point[1] - edge.start[1]
    along = max(0.0, min(edge.length, dx * edge.tangent[0] + dy * edge.tangent[1]))
    return along, abs(dx * edge.outward[0] + dy * edge.outward[1])


def frontage_edge(block, frontage) -> Edge:
    target = float(frontage.yaw_deg) % 360.0
    return min(
        edge_frames(block),
        key=lambda edge: (
            abs((math.degrees(math.atan2(edge.outward[1], edge.outward[0])) - target + 180.0) % 360.0 - 180.0),
            edge_projection(frontage.position[:2], edge)[1],
            edge.index,
        ),
    )


def measured_half_extents(item, bbox_lookup: Callable[[str], tuple[float, float, float]]) -> Point:
    raw_x, raw_y, _raw_z = bbox_lookup(item.asset_key)
    hx, hy = abs(raw_x * float(item.scale[0])) / 2.0, abs(raw_y * float(item.scale[1])) / 2.0
    radians = math.radians(float(item.yaw_deg))
    cosine, sine = abs(math.cos(radians)), abs(math.sin(radians))
    return cosine * hx + sine * hy, sine * hx + cosine * hy


def item_bounds(item, bbox_lookup: Callable[[str], tuple[float, float, float]], margin: float = 0.0) -> Bounds:
    hx, hy = measured_half_extents(item, bbox_lookup)
    x, y = float(item.position[0]), float(item.position[1])
    return x - hx - margin, y - hy - margin, x + hx + margin, y + hy + margin


def box_inside_block(block, bounds: Bounds) -> bool:
    xmin, ymin, xmax, ymax = bounds
    return all(point_in_polygon((x, y), block.footprint) for x in (xmin, xmax) for y in (ymin, ymax))


def boxes_separate(first: Bounds, second: Bounds, margin: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    return ax1 + margin <= bx0 or bx1 + margin <= ax0 or ay1 + margin <= by0 or by1 + margin <= ay0


def segment_hits_expanded_box(start: Point, end: Point, bounds: Bounds, clearance: float) -> bool:
    xmin, ymin, xmax, ymax = bounds
    xmin, ymin, xmax, ymax = xmin - clearance, ymin - clearance, xmax + clearance, ymax + clearance
    dx, dy = end[0] - start[0], end[1] - start[1]
    lower_t, upper_t = 0.0, 1.0
    for origin, delta, lower, upper in ((start[0], dx, xmin, xmax), (start[1], dy, ymin, ymax)):
        if abs(delta) <= 1e-9:
            if origin < lower or origin > upper:
                return False
            continue
        first, last = (lower - origin) / delta, (upper - origin) / delta
        if first > last:
            first, last = last, first
        lower_t, upper_t = max(lower_t, first), min(upper_t, last)
        if lower_t > upper_t:
            return False
    return True


def box_clear_of_routes(bounds: Bounds, routes: Iterable[tuple[Point, ...]], clearance: float) -> bool:
    return not any(
        segment_hits_expanded_box(start, end, bounds, clearance)
        for polyline in routes
        for start, end in zip(polyline, polyline[1:])
    )


def enabled_routes(layout) -> tuple[tuple[Point, ...], ...]:
    return tuple(
        tuple(layout.edge_polyline(edge))
        for edge in layout.walk_edges
        if edge.enabled and len(layout.edge_polyline(edge)) >= 2
    )


def enabled_bridge_routes(layout) -> tuple[tuple[Point, ...], ...]:
    return tuple(
        tuple(layout.edge_polyline(edge))
        for edge in layout.walk_edges
        if edge.enabled and edge.route_kind == "bridge" and len(layout.edge_polyline(edge)) >= 2
    )


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    ordered = sorted((min(left, right), max(left, right)) for left, right in intervals)
    if not ordered:
        return ()
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for left, right in ordered[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return tuple((left, right) for left, right in merged)


def interval_complement(length: float, gaps: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    cursor, result = 0.0, []
    for left, right in merge_intervals(gaps):
        if left > cursor + 1e-6:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < length - 1e-6:
        result.append((cursor, length))
    return tuple(result)


def fixture_gaps(
    scenario,
    block,
    edge: Edge,
    gap_fixtures: Mapping[str, Mapping[str, object]],
    extra_gaps: Iterable[tuple[str, int, float, float]] = (),
) -> tuple[tuple[float, float], ...]:
    layout = scenario.layout
    assert layout is not None
    fixture = gap_fixtures[layout.layout_id]
    declared = fixture["gaps"][block.block_id][edge.index]  # type: ignore[index]
    extras = (
        (left, right)
        for block_id, edge_index, left, right in extra_gaps
        if block_id == block.block_id and edge_index == edge.index
    )
    return merge_intervals((*declared, *extras))  # type: ignore[arg-type]


def shell_edge_metrics(
    scenario,
    gap_fixtures: Mapping[str, Mapping[str, object]],
    shell_records_fn: Callable[[object], Iterable[object]],
    *,
    extra_gaps: Iterable[tuple[str, int, float, float]] = (),
) -> dict[str, object]:
    layout = scenario.layout
    assert layout is not None
    records = tuple(shell_records_fn(scenario))
    total_eligible = covered = max_gap = 0.0
    per_block: dict[str, tuple[int, float, float, float]] = {}
    for block in layout.blocks:
        shells = [record for record in records if record.footprint.block_id == block.block_id]
        block_eligible = block_covered = block_gap = 0.0
        for edge in edge_frames(block):
            for left, right in interval_complement(edge.length, fixture_gaps(scenario, block, edge, gap_fixtures, extra_gaps)):
                span_length = right - left
                total_eligible += span_length
                block_eligible += span_length
                spans = []
                for record in shells:
                    footprint = record.footprint
                    assert footprint is not None
                    if footprint.edge_index != edge.index:
                        continue
                    along, _ = edge_projection(footprint.position, edge)
                    spans.append((max(left, along - footprint.tangent_half_extent), min(right, along + footprint.tangent_half_extent)))
                merged: list[list[float]] = []
                for span_left, span_right in sorted(span for span in spans if span[1] > span[0]):
                    if merged and span_left <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], span_right)
                    else:
                        merged.append([span_left, span_right])
                cursor = left
                for span_left, span_right in merged:
                    block_gap = max(block_gap, span_left - cursor)
                    cursor = max(cursor, span_right)
                block_gap = max(block_gap, right - cursor)
                span_covered = sum(span_right - span_left for span_left, span_right in merged)
                covered += span_covered
                block_covered += span_covered
        max_gap = max(max_gap, block_gap)
        per_block[block.block_id] = (len(shells), block_eligible, block_covered, block_gap)
    return {
        "eligible_cm": total_eligible,
        "covered_cm": covered,
        "coverage": covered / total_eligible if total_eligible else 0.0,
        "max_gap_cm": max_gap,
        "per_block": per_block,
    }


def disc_chain(half_extents: Point, position: Point, clearance: float) -> tuple[tuple[float, float, float], ...]:
    """Return independent overlapping discs covering a shell AABB."""

    hx, hy = (float(value) for value in half_extents)
    radius = min(hx, hy) + float(clearance)
    long_half = max(hx, hy)
    edge_reach = math.sqrt(max(0.0, radius * radius - min(hx, hy) ** 2))
    spacing_limit = 2.0 * edge_reach if edge_reach > 1e-6 else 2.0 * radius
    segments = max(1, int(math.ceil((2.0 * long_half) / max(spacing_limit, 1e-6))))
    step = (2.0 * long_half) / segments
    cx, cy = (float(value) for value in position)
    if hx >= hy:
        return tuple((cx - long_half + step * index, cy, radius) for index in range(segments + 1))
    return tuple((cx, cy - long_half + step * index, radius) for index in range(segments + 1))


__all__ = [
    "Bounds", "Edge", "Point", "box_clear_of_routes", "box_inside_block", "boxes_separate",
    "disc_chain", "distance", "edge_frames", "edge_projection", "enabled_bridge_routes",
    "enabled_routes", "fixture_gaps", "frontage_edge", "interval_complement", "item_bounds",
    "measured_half_extents", "minimum_route_distance", "point_in_polygon", "segment_distance",
    "shell_edge_metrics",
]
