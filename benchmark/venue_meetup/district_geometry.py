"""Pure geometry and perimeter tiling for authored district dressing.

The helpers in this module are intentionally independent of UnrealCV and of
the actor-record adapter.  They operate on authored block/frontage/layout
geometry and return deterministic shell placements or clearance decisions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from benchmark.venue_meetup.building_catalog import building_bbox
from benchmark.venue_meetup.layout import Block, Frontage


_BUILDING_CLEARANCE_CM = 3_400.0
_WALK_NODE_CLEARANCE_CM = 1_500.0
_ROUTE_CLEARANCE_CM = 1_600.0
_BRIDGE_CLEARANCE_CM = 2_200.0
_ANCHOR_CLEARANCE_CM = 1_200.0
_SHELL_EDGE_END_GAP_CM = 2_500.0
_SHELL_FRONTAGE_GAP_CM = 1_000.0
_SHELL_FRONTAGE_BUFFER_CM = 500.0
_SHELL_EDGE_SETBACK_CM = 1_200.0
# Keep the authored frontage inset as the first choice, then move a shell
# further into its owning parcel when a measured route segment or a neighbouring
# corner envelope occupies the nominal setback.  The larger values remain well
# within the 12--20 m parcel depths used by the district templates.
_SHELL_SETBACK_OPTIONS_CM = (1_200.0, 1_800.0, 2_400.0, 3_000.0, 3_800.0, 4_800.0)
_SHELL_COLLISION_MARGIN_CM = 450.0
_SHELL_SEAM_GAP_CM = 300.0
_SHELL_ROUTE_CLEARANCE_CM = 400.0
_SHELL_BRIDGE_CLEARANCE_CM = 500.0
_SHELL_NODE_CLEARANCE_CM = 400.0
_SHELL_ASSET_MARGIN_CM = 200.0
# Shells remain readable, collision-enabled buildings even when a residual
# frontage slot is narrower than the preferred asset.  Never emit a
# microscopic actor to satisfy a nominal count.
_SHELL_MIN_SCALE = 0.18
_SHELL_MIN_FOOTPRINT_AREA_CM2 = 1e-6

_SHELL_TARGET_MEDIUM = 16
_SHELL_TARGET_LARGE = 24

_SHELL_SCALES = {
    # Scales are deliberately larger than the venue's 0.25 visual scale so
    # the measured shells read as a continuous urban frontage.  Every choice
    # is still fit against its asset-specific bbox and slot AABB below.
    "BP_Building_05_C": 0.42,
    "BP_Building_06_C": 0.39,
    "BP_Building_20_C": 0.32,
    "BP_Building_24_C": 0.39,
    "BP_Building_25_C": 0.35,
    "BP_Building_44_C": 0.35,
    "BP_Building_87_C": 0.25,
    "BP_Building_95_C": 0.25,
    "BP_Building_99_C": 0.20,
    "BP_Building_101_C": 0.29,
    "BP_Building_123_C": 0.18,
}

_SHELL_RHYTHM = (
    "BP_Building_123_C",
    "BP_Building_99_C",
    "BP_Building_87_C",
    "BP_Building_95_C",
    "BP_Building_101_C",
    "BP_Building_20_C",
    "BP_Building_44_C",
    "BP_Building_25_C",
    "BP_Building_24_C",
    "BP_Building_06_C",
    "BP_Building_05_C",
)


@dataclass(frozen=True, slots=True)
class DistrictShellFootprint:
    """Conservative world-axis footprint for one collision-enabled shell."""

    actor_name: str
    asset_key: str
    block_id: str
    edge_index: int
    position: tuple[float, float]
    yaw_deg: float
    scale: tuple[float, float, float]
    half_extents: tuple[float, float]
    tangent_half_extent: float
    normal_half_extent: float

    @property
    def radius(self) -> float:
        return math.hypot(*self.half_extents) + _SHELL_COLLISION_MARGIN_CM

    @property
    def area(self) -> float:
        """Return the conservative world-axis footprint area in cm²."""

        return 4.0 * self.half_extents[0] * self.half_extents[1]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        x, y = self.position
        hx, hy = self.half_extents
        return x - hx, y - hy, x + hx, y + hy

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Compatibility alias for callers that name the envelope ``bbox``."""

        return self.bounds

    @property
    def center(self) -> tuple[float, float]:
        """Alias used by obstacle and validation callers."""

        return self.position


@dataclass(frozen=True, slots=True)
class _EdgeFrame:
    index: int
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    tangent: tuple[float, float]
    outward: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _ShellPlacement:
    point: tuple[float, float]
    yaw_deg: float
    edge_index: int
    asset_key: str
    scale: tuple[float, float, float]
    footprint: DistrictShellFootprint


def _distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _point_to_segment_distance_sq(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return _distance_sq(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return _distance_sq(point, (start[0] + t * dx, start[1] + t * dy))


def _minimum_segment_distance_sq(point, polylines) -> float:
    return min(
        (_point_to_segment_distance_sq(point, start, end)
         for polyline in polylines for start, end in zip(polyline, polyline[1:])),
        default=math.inf,
    )


def _offset(point, center, distance, lateral=0.0):
    dx, dy = center[0] - point[0], center[1] - point[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return point
    ux, uy = dx / length, dy / length
    return point[0] + ux * distance - uy * lateral, point[1] + uy * distance + ux * lateral


def frontages_by_block(layout) -> dict[str, tuple[Frontage, ...]]:
    return {
        block.block_id: tuple(f for f in layout.frontages if f.block_id == block.block_id)
        for block in layout.blocks
    }


def route_polylines(layout):
    polylines = []
    for edge in layout.walk_edges:
        if not edge.enabled:
            continue
        try:
            polyline = layout.edge_polyline(edge)
        except (KeyError, ValueError):
            continue
        if len(polyline) >= 2:
            polylines.append(polyline)
    return tuple(polylines)


def bridge_gap_polylines(layout):
    polylines = []
    for edge in layout.walk_edges:
        if not edge.enabled or edge.route_kind != "bridge":
            continue
        try:
            polyline = layout.edge_polyline(edge)
        except (KeyError, ValueError):
            continue
        if len(polyline) >= 2:
            polylines.append(polyline)
    return tuple(polylines)


def protected_anchors(scenario):
    anchors = []
    for venue in scenario.venues:
        anchors.extend(((venue.region.center, float(venue.region.radius) + _ANCHOR_CLEARANCE_CM),
                        ((venue.position[0], venue.position[1]), _BUILDING_CLEARANCE_CM)))
        anchors.extend(((e.position[0], e.position[1]), _ANCHOR_CLEARANCE_CM) for e in venue.entrances)
    anchors.extend(((l.position[0], l.position[1]), _BUILDING_CLEARANCE_CM) for l in scenario.landmarks)
    return tuple(anchors)


def clear(point, anchors, nodes, routes, occupied=(), spacing=0.0, *, bridge_polylines=()):
    if any(_distance_sq(point, anchor) < clearance ** 2 for anchor, clearance in anchors):
        return False
    if any(_distance_sq(point, node) < _WALK_NODE_CLEARANCE_CM ** 2 for node in nodes):
        return False
    if _minimum_segment_distance_sq(point, routes) < _ROUTE_CLEARANCE_CM ** 2:
        return False
    if _minimum_segment_distance_sq(point, bridge_polylines) < _BRIDGE_CLEARANCE_CM ** 2:
        return False
    return not spacing or all(_distance_sq(point, other) >= spacing ** 2 for other in occupied)


def inside_block(block: Block, point) -> bool:
    polygon = block.footprint
    if len(polygon) < 3:
        return False
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        if _point_to_segment_distance_sq(point, (x1, y1), (x2, y2)) <= 1e-6:
            return True
        if (y1 > point[1]) != (y2 > point[1]):
            crossing_x = (x2 - x1) * (point[1] - y1) / (y2 - y1) + x1
            if point[0] < crossing_x:
                inside = not inside
    return inside


def _polygon_area(polygon) -> float:
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, (*polygon[1:], polygon[0])))


def _edge_frames(block: Block) -> tuple[_EdgeFrame, ...]:
    polygon = block.footprint
    ccw = _polygon_area(polygon) >= 0.0
    result = []
    for index, (start, end) in enumerate(zip(polygon, (*polygon[1:], polygon[0]))):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        tangent = (dx / length, dy / length)
        outward = (tangent[1], -tangent[0]) if ccw else (-tangent[1], tangent[0])
        result.append(_EdgeFrame(index, start, end, length, tangent, outward))
    return tuple(result)


def _project_to_edge(point, edge: _EdgeFrame):
    dx, dy = point[0] - edge.start[0], point[1] - edge.start[1]
    along = max(0.0, min(edge.length, dx * edge.tangent[0] + dy * edge.tangent[1]))
    return along, abs(dx * edge.outward[0] + dy * edge.outward[1])


def _angle_distance(first, second):
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _frontage_edge(block, frontage, edges):
    if not edges:
        raise ValueError(f"Block {block.block_id!r} has no usable perimeter edges")
    target = float(frontage.yaw_deg) % 360.0
    return min(edges, key=lambda e: (_angle_distance(math.degrees(math.atan2(e.outward[1], e.outward[0])), target),
                                     _project_to_edge(frontage.position[:2], e)[1], e.index))


def _oriented_half_extents(asset_key, scale, yaw_deg):
    raw_x, raw_y, _ = building_bbox(asset_key)
    hx, hy = abs(raw_x * float(scale[0])) / 2.0, abs(raw_y * float(scale[1])) / 2.0
    radians = math.radians(float(yaw_deg))
    cosine, sine = abs(math.cos(radians)), abs(math.sin(radians))
    return cosine * hx + sine * hy, sine * hx + cosine * hy


def _footprint_extents_on_edge(half_extents, edge):
    hx, hy = half_extents
    return (abs(edge.tangent[0]) * hx + abs(edge.tangent[1]) * hy,
            abs(edge.outward[0]) * hx + abs(edge.outward[1]) * hy)

def _bounds_inside_block(block, bounds):
    """Return whether all four corners of an AABB stay inside a block."""

    xmin, ymin, xmax, ymax = bounds
    return all(
        inside_block(block, (x, y))
        for x in (xmin, xmax)
        for y in (ymin, ymax)
    )

def _segment_hits_expanded_bounds(start, end, bounds, clearance):
    """Return whether a segment intersects an AABB expanded by ``clearance``."""

    xmin, ymin, xmax, ymax = bounds
    xmin -= clearance
    ymin -= clearance
    xmax += clearance
    ymax += clearance
    dx, dy = end[0] - start[0], end[1] - start[1]
    t_min, t_max = 0.0, 1.0
    for origin, delta, lower, upper in (
        (start[0], dx, xmin, xmax),
        (start[1], dy, ymin, ymax),
    ):
        if abs(delta) <= 1e-9:
            if origin < lower or origin > upper:
                return False
            continue
        first = (lower - origin) / delta
        last = (upper - origin) / delta
        if first > last:
            first, last = last, first
        t_min = max(t_min, first)
        t_max = min(t_max, last)
        if t_min > t_max:
            return False
    return True

def _shell_clear_of_routes(bounds, routes, clearance):
    """Return whether a shell AABB leaves a corridor around every route segment."""

    return not any(
        _segment_hits_expanded_bounds(start, end, bounds, clearance)
        for polyline in routes
        for start, end in zip(polyline, polyline[1:])
    )

def _shell_clear_of_nodes(bounds, nodes, clearance):
    """Return whether graph nodes stay outside the shell AABB corridor."""

    xmin, ymin, xmax, ymax = bounds
    xmin -= clearance
    ymin -= clearance
    xmax += clearance
    ymax += clearance
    return all(not (xmin <= x <= xmax and ymin <= y <= ymax) for x, y in nodes)

def _bounds_separated(first, second, margin=0.0):
    """Return whether two world-axis AABBs are separated by ``margin``."""

    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    return (
        ax1 + margin <= bx0
        or bx1 + margin <= ax0
        or ay1 + margin <= by0
        or by1 + margin <= ay0
    )

def shell_protected_bounds(scenario, margin=_SHELL_ASSET_MARGIN_CM):
    """Return measured venue/landmark AABBs buffered for shell placement."""

    result = []
    for item in (*scenario.venues, *scenario.landmarks):
        half = _oriented_half_extents(item.asset_key, item.scale, item.yaw_deg)
        x, y = float(item.position[0]), float(item.position[1])
        result.append((x - half[0] - margin, y - half[1] - margin,
                       x + half[0] + margin, y + half[1] + margin))
    return tuple(result)

def _merge_intervals(intervals):
    if not intervals:
        return ()
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged = [[*ordered[0]]]
    for left, right in ordered[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return tuple((a, b) for a, b in merged)


def _edge_gap_intervals(block, edge, frontages, *, venue_by_slot=None, bridge_polylines=()):
    gaps = [(0.0, min(edge.length, _SHELL_EDGE_END_GAP_CM)),
            (max(0.0, edge.length - _SHELL_EDGE_END_GAP_CM), edge.length)]
    edges = _edge_frames(block)
    for frontage in frontages:
        if _frontage_edge(block, frontage, edges).index != edge.index:
            continue
        along, _ = _project_to_edge(frontage.position[:2], edge)
        tangent_half = 0.0
        if venue_by_slot is not None and frontage.venue_slot_id:
            venue = venue_by_slot.get(frontage.venue_slot_id)
            if venue is not None:
                half = _oriented_half_extents(venue.asset_key, venue.scale, venue.yaw_deg)
                tangent_half, _ = _footprint_extents_on_edge(half, edge)
        gap_half = max(_SHELL_FRONTAGE_GAP_CM, min(2_500.0, tangent_half + _SHELL_FRONTAGE_BUFFER_CM))
        gaps.append((max(0.0, along - gap_half), min(edge.length, along + gap_half)))
    for polyline in bridge_polylines:
        for portal in (polyline[0], polyline[-1]):
            along, distance = _project_to_edge(portal, edge)
            if distance <= 9_000.0:
                gaps.append((max(0.0, along - _SHELL_FRONTAGE_GAP_CM), min(edge.length, along + _SHELL_FRONTAGE_GAP_CM)))
    return _merge_intervals(gaps)


def _complement_intervals(length, gaps):
    cursor, result = 0.0, []
    for left, right in gaps:
        if left > cursor + 1e-6:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < length - 1e-6:
        result.append((cursor, length))
    return tuple(result)


def _allocate_interval_counts(intervals, count):
    if count <= 0 or not intervals:
        return tuple(0 for _ in intervals)
    lengths = [max(0.0, b - a) for a, b in intervals]
    total = sum(lengths)
    if total <= 1e-6:
        return tuple(0 for _ in intervals)
    raw = [count * length / total for length in lengths]
    allocated = [int(math.floor(value)) for value in raw]
    for index, length in enumerate(lengths):
        if length > 1_000.0 and allocated[index] == 0 and sum(allocated) < count:
            allocated[index] = 1
    while sum(allocated) < count:
        index = max(range(len(intervals)), key=lambda i: (raw[i] - allocated[i], lengths[i], -i))
        allocated[index] += 1
    while sum(allocated) > count:
        candidates = [i for i, value in enumerate(allocated) if value > 0]
        if not candidates:
            break
        index = min(candidates, key=lambda i: (raw[i] - allocated[i], lengths[i], i))
        allocated[index] -= 1
    return tuple(allocated)


def shell_yaw(block: Block, position, *, frontages=()):
    if frontages:
        nearest = min(frontages, key=lambda f: _distance_sq(position, f.position[:2]))
        if _distance_sq(position, nearest.position[:2]) <= 8_000.0 ** 2:
            return float(nearest.yaw_deg)
    xs, ys = zip(*block.footprint)
    return min(((abs(position[0] - min(xs)), 180.0), (abs(position[0] - max(xs)), 0.0),
                (abs(position[1] - min(ys)), -90.0), (abs(position[1] - max(ys)), 90.0)),
               key=lambda item: item[0])[1]


def _shell_asset_order(block_index, edge_index, slot_index):
    start = (block_index * 5 + edge_index * 3 + slot_index * 2) % len(_SHELL_RHYTHM)
    return _SHELL_RHYTHM[start:] + _SHELL_RHYTHM[:start]


def _make_shell_placement(block, edge, point, *, asset_key, scale_factor=1.0):
    factor = float(scale_factor)
    if not math.isfinite(factor):
        raise ValueError(f"Shell scale factor must be finite: {scale_factor!r}")
    base_scale = max(_SHELL_MIN_SCALE, _SHELL_SCALES[asset_key] * factor)
    scale = (base_scale, base_scale, base_scale)
    yaw = math.degrees(math.atan2(edge.outward[1], edge.outward[0]))
    half = _oriented_half_extents(asset_key, scale, yaw)
    tangent_half, normal_half = _footprint_extents_on_edge(half, edge)
    footprint = DistrictShellFootprint("", asset_key, block.block_id, edge.index, point, yaw, scale, half,
                                       tangent_half, normal_half)
    return _ShellPlacement(point, yaw, edge.index, asset_key, scale, footprint)


def _edge_point(edge, along, normal_half, setback):
    """Return a parcel-side point at ``along`` with an explicit inward inset."""

    boundary = (
        edge.start[0] + edge.tangent[0] * along,
        edge.start[1] + edge.tangent[1] * along,
    )
    return (
        boundary[0] - edge.outward[0] * (normal_half + setback),
        boundary[1] - edge.outward[1] * (normal_half + setback),
    )


def _placement_clear(placement, *, block, anchors, nodes, routes, bridge_polylines,
                     occupied_placements, occupied_positions, protected_bounds=()):
    """Validate a solid shell against authored geometry using its AABB."""

    if any(
        not math.isfinite(float(component)) or float(component) < _SHELL_MIN_SCALE
        for component in placement.scale
    ):
        return False
    if (
        not math.isfinite(placement.footprint.area)
        or placement.footprint.area <= _SHELL_MIN_FOOTPRINT_AREA_CM2
    ):
        return False
    bounds = placement.footprint.bounds
    if not _bounds_inside_block(block, bounds):
        return False
    for anchor, clearance in anchors:
        x, y = anchor
        anchor_bounds = (x - clearance, y - clearance, x + clearance, y + clearance)
        if not _bounds_separated(bounds, anchor_bounds):
            return False
    if any(not _bounds_separated(bounds, protected) for protected in protected_bounds):
        return False
    if not _shell_clear_of_nodes(bounds, nodes, _SHELL_NODE_CLEARANCE_CM):
        return False
    if not _shell_clear_of_routes(bounds, routes, _SHELL_ROUTE_CLEARANCE_CM):
        return False
    if not _shell_clear_of_routes(bounds, bridge_polylines, _SHELL_BRIDGE_CLEARANCE_CM):
        return False
    for other in occupied_placements:
        if not _bounds_separated(bounds, other.footprint.bounds, _SHELL_SEAM_GAP_CM):
            return False
    # Solid shells are accepted on measured AABB separation alone.  The old
    # pivot-distance/radius spacing rule rejected valid frontage spans and is
    # intentionally not applied to measured shell envelopes.
    return True

def _edge_gap_envelope_clear(placement, edge, gaps):
    """Return whether a shell tangent envelope stays outside every edge gap."""

    along, _ = _project_to_edge(placement.point, edge)
    tangent_half = placement.footprint.tangent_half_extent
    return all(
        along + tangent_half <= left + 1e-6 or along - tangent_half >= right - 1e-6
        for left, right in gaps
    )

def _tile_block(block, *, frontages, venue_by_slot, venue_positions, walk_node_positions,
                protected_anchors, route_polylines, bridge_polylines, occupied_placements,
                protected_bounds=(),
                occupied_positions, target_count, block_index):
    edges = _edge_frames(block)
    if not edges or target_count <= 0:
        return ()
    gaps = {e.index: _edge_gap_intervals(block, e, frontages, venue_by_slot=venue_by_slot,
                                          bridge_polylines=bridge_polylines) for e in edges}
    intervals = {e.index: _complement_intervals(e.length, gaps[e.index]) for e in edges}
    lengths = {e.index: sum(b - a for a, b in intervals[e.index]) for e in edges}
    perimeter = sum(lengths.values())
    if perimeter <= 1_000.0:
        return ()
    raw = {e.index: target_count * lengths[e.index] / perimeter for e in edges}
    counts = {e.index: int(math.floor(raw[e.index])) for e in edges}
    for e in edges:
        if lengths[e.index] >= 4_000.0 and counts[e.index] == 0:
            counts[e.index] = 1
    while sum(counts.values()) < target_count:
        edge = max(edges, key=lambda e: (raw[e.index] - counts[e.index], lengths[e.index], -e.index))
        counts[edge.index] += 1
    while sum(counts.values()) > target_count:
        choices = [e for e in edges if counts[e.index] > 1]
        if not choices:
            break
        edge = min(choices, key=lambda e: (raw[e.index] - counts[e.index], lengths[e.index], e.index))
        counts[edge.index] -= 1

    placed = []
    for edge in edges:
        local_slot = 0
        for (left, right), interval_count in zip(intervals[edge.index], _allocate_interval_counts(intervals[edge.index], counts[edge.index])):
            if interval_count <= 0:
                continue
            run_length = right - left
            slot_length = run_length / interval_count
            for slot in range(interval_count):
                along = left + slot_length * (slot + 0.5)
                yaw = math.degrees(math.atan2(edge.outward[1], edge.outward[0]))
                # Try the deterministic rhythm in order, but validate each
                # asset's measured tangent envelope and full AABB before
                # committing.  A wide shell can fail at a corner while a
                # narrower known-safe shell fits the same eligible span.
                asset_options = []
                for asset in _shell_asset_order(block_index, edge.index, local_slot):
                    scale = _SHELL_SCALES[asset]
                    half = _oriented_half_extents(asset, (scale,) * 3, yaw)
                    tangent_half, _ = _footprint_extents_on_edge(half, edge)
                    if 2.0 * tangent_half + _SHELL_SEAM_GAP_CM <= slot_length + 1e-6:
                        asset_options.append((asset, 1.0))
                # Very short residual spans (for example beside a frontage
                # interval) still receive a real, bbox-measured shell when
                # possible; scale only as much as the span requires.
                if not asset_options:
                    for asset in _shell_asset_order(block_index, edge.index, local_slot):
                        base = _SHELL_SCALES[asset]
                        half = _oriented_half_extents(asset, (base,) * 3, yaw)
                        tangent_half, _ = _footprint_extents_on_edge(half, edge)
                        minimum_factor = _SHELL_MIN_SCALE / base
                        if minimum_factor > 1.0:
                            continue
                        factor = min(
                            1.0,
                            max(
                                minimum_factor,
                                (slot_length - _SHELL_SEAM_GAP_CM)
                                / max(2.0 * tangent_half, 1.0),
                            ),
                        )
                        asset_options.append((asset, factor))
                selected = None
                shifts = (0.0, -0.22, 0.22, -0.38, 0.38, -0.50, 0.50)
                for shift in shifts:
                    shifted = max(left + 1.0, min(right - 1.0, along + shift * slot_length))
                    for asset, factor in asset_options:
                        scale = _SHELL_SCALES[asset] * factor
                        half = _oriented_half_extents(asset, (scale,) * 3, yaw)
                        _, normal_half = _footprint_extents_on_edge(half, edge)
                        # Probe deeper parcel-side insets when the nominal
                        # setback intersects an actual graph route or a
                        # neighbouring shell AABB.
                        for setback in _SHELL_SETBACK_OPTIONS_CM:
                            option = _make_shell_placement(
                                block, edge, _edge_point(edge, shifted, normal_half, setback),
                                asset_key=asset, scale_factor=factor,
                            )
                            if _edge_gap_envelope_clear(option, edge, gaps[edge.index]) and _placement_clear(
                                option, block=block, anchors=protected_anchors, nodes=walk_node_positions,
                                routes=route_polylines, bridge_polylines=bridge_polylines,
                                occupied_placements=(*occupied_placements, *placed), occupied_positions=occupied_positions,
                                protected_bounds=protected_bounds,
                            ):
                                selected = option
                                break
                        if selected is not None:
                            break
                    if selected is not None:
                        break
                if selected is not None:
                    placed.append(selected)
                local_slot += 1
    return tuple(placed)


def _augment_shell_shortfall(block, placements, *, minimum_count, frontages, venue_by_slot,
                             walk_node_positions, protected_anchors, route_polylines,
                             protected_bounds=(),
                            bridge_polylines, occupied_placements, block_index):
    if len(placements) >= minimum_count:
        return tuple(placements)
    edges = _edge_frames(block)
    gaps = {e.index: _edge_gap_intervals(block, e, frontages, venue_by_slot=venue_by_slot,
                                          bridge_polylines=bridge_polylines) for e in edges}
    additions = []
    narrow_assets = sorted(_SHELL_RHYTHM, key=lambda asset: (_SHELL_SCALES[asset], asset))
    for edge in edges:
        for left, right in _complement_intervals(edge.length, gaps[edge.index]):
            cursor = left + 250.0
            while cursor < right - 250.0 and len((*placements, *additions)) < minimum_count:
                yaw = math.degrees(math.atan2(edge.outward[1], edge.outward[0]))
                selected = None
                for asset in narrow_assets:
                    minimum_factor = _SHELL_MIN_SCALE / _SHELL_SCALES[asset]
                    if minimum_factor > 1.0:
                        continue
                    factors = tuple(
                        factor
                        for factor in (
                            minimum_factor,
                            0.22,
                            0.30,
                            0.40,
                            0.55,
                            0.70,
                            1.0,
                        )
                        if minimum_factor - 1e-9 <= factor <= 1.0
                    )
                    for factor in factors:
                        half = _oriented_half_extents(asset, (_SHELL_SCALES[asset] * factor,) * 3, yaw)
                        _, normal_half = _footprint_extents_on_edge(half, edge)
                        boundary = (edge.start[0] + edge.tangent[0] * cursor, edge.start[1] + edge.tangent[1] * cursor)
                        candidate = _make_shell_placement(block, edge,
                            (boundary[0] - edge.outward[0] * (normal_half + _SHELL_EDGE_SETBACK_CM),
                             boundary[1] - edge.outward[1] * (normal_half + _SHELL_EDGE_SETBACK_CM)),
                            asset_key=asset, scale_factor=factor)
                        if _edge_gap_envelope_clear(candidate, edge, gaps[edge.index]) and _placement_clear(
                            candidate, block=block, anchors=protected_anchors,
                            nodes=walk_node_positions, routes=route_polylines,
                            bridge_polylines=bridge_polylines,
                            occupied_placements=(*occupied_placements, *placements, *additions), occupied_positions=(),
                            protected_bounds=protected_bounds):
                            selected = candidate
                            break
                    if selected is not None:
                        break
                if selected is not None:
                    additions.append(selected)
                cursor += 250.0
    return tuple((*placements, *additions))


def shell_positions(block, venue_positions, walk_node_positions, *, frontages=(), protected_anchors=(),
                    route_polylines=(), bridge_polylines=(), occupied_positions=()):
    protected = (*protected_anchors, *((venue, _BUILDING_CLEARANCE_CM) for venue in venue_positions))
    placements = _tile_block(block, frontages=frontages, venue_by_slot=None, venue_positions=venue_positions,
        walk_node_positions=walk_node_positions, protected_anchors=protected, route_polylines=route_polylines,
        bridge_polylines=bridge_polylines, occupied_placements=(), occupied_positions=occupied_positions,
        target_count=4, block_index=0)
    return tuple(p.point for p in placements)


__all__ = [
    "DistrictShellFootprint", "_ShellPlacement", "frontages_by_block", "route_polylines",
    "bridge_gap_polylines", "protected_anchors", "clear", "inside_block", "shell_yaw",
    "shell_positions", "_make_shell_placement", "_tile_block", "_augment_shell_shortfall",
    "_SHELL_SCALES", "_SHELL_RHYTHM", "_SHELL_EDGE_END_GAP_CM",
    "_SHELL_FRONTAGE_GAP_CM", "_SHELL_FRONTAGE_BUFFER_CM", "_SHELL_EDGE_SETBACK_CM",
    "_SHELL_COLLISION_MARGIN_CM", "_SHELL_SEAM_GAP_CM", "_SHELL_TARGET_MEDIUM",
    "_SHELL_TARGET_LARGE",
    "_SHELL_MIN_SCALE", "_SHELL_MIN_FOOTPRINT_AREA_CM2",
    "_BUILDING_CLEARANCE_CM", "_WALK_NODE_CLEARANCE_CM", "_ROUTE_CLEARANCE_CM",
    "_BRIDGE_CLEARANCE_CM", "_ANCHOR_CLEARANCE_CM", "_distance_sq", "_minimum_segment_distance_sq",
    "_offset", "_oriented_half_extents", "_footprint_extents_on_edge",
]
