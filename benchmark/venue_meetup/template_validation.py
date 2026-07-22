"""Pure-Python validation for authored district layouts."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

from benchmark.venue_meetup.layout import DistrictLayout


class LayoutValidationError(ValueError):
    """Raised when a district layout fails pure geometric/graph checks."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        message = "; ".join(self.errors) if self.errors else "layout validation failed"
        super().__init__(message)


def collect_layout_errors(
    layout: DistrictLayout,
    *,
    required_paths: Iterable[tuple[str, str]] | None = None,
) -> list[str]:
    """Return human-readable layout problems without raising.

    Checks:
    - duplicate entity ids
    - walk-edge endpoints that are missing from walk nodes
    - frontage/block cross-reference mismatches
    - non-finite or non-positive edge lengths
    - unreachable required (start, end) node pairs when provided
    """

    errors: list[str] = []

    street_ids = [street.street_id for street in layout.streets]
    intersection_ids = [item.intersection_id for item in layout.intersections]
    block_ids = [block.block_id for block in layout.blocks]
    frontage_ids = [frontage.frontage_id for frontage in layout.frontages]
    node_ids = [node.node_id for node in layout.walk_nodes]

    errors.extend(_duplicate_id_errors("street_id", street_ids))
    errors.extend(_duplicate_id_errors("intersection_id", intersection_ids))
    errors.extend(_duplicate_id_errors("block_id", block_ids))
    errors.extend(_duplicate_id_errors("frontage_id", frontage_ids))
    errors.extend(_duplicate_id_errors("walk node_id", node_ids))

    known_nodes = set(node_ids)
    node_position = {node.node_id: node.position for node in layout.walk_nodes}
    authored_walk_data = layout.schema_version >= 2
    for index, edge in enumerate(layout.walk_edges):
        if edge.start_node_id not in known_nodes:
            errors.append(
                f"walk_edges[{index}] start_node_id={edge.start_node_id!r} is not a walk node"
            )
        if edge.end_node_id not in known_nodes:
            errors.append(
                f"walk_edges[{index}] end_node_id={edge.end_node_id!r} is not a walk node"
            )
        if not math.isfinite(edge.length_cm) or edge.length_cm <= 0.0:
            errors.append(
                f"walk_edges[{index}] length_cm={edge.length_cm!r} must be finite and > 0"
            )
        if authored_walk_data and not edge.waypoints:
            errors.append(
                f"walk_edges[{index}] has no explicit waypoint polyline "
                "(schema_version >= 2 layouts must author physical routes)"
            )
        for wp_idx, wp in enumerate(edge.waypoints):
            if not (math.isfinite(wp[0]) and math.isfinite(wp[1])):
                errors.append(
                    f"walk_edges[{index}] waypoint[{wp_idx}] has non-finite coordinates"
                )
        if (
            authored_walk_data
            and
            edge.start_node_id in node_position
            and edge.end_node_id in node_position
            and math.isfinite(edge.length_cm)
            and edge.length_cm > 0.0
        ):
            polyline = (
                node_position[edge.start_node_id],
                *edge.waypoints,
                node_position[edge.end_node_id],
            )
            actual_length = sum(
                math.hypot(right[0] - left[0], right[1] - left[1])
                for left, right in zip(polyline, polyline[1:])
            )
            if not math.isclose(edge.length_cm, actual_length, rel_tol=1e-6, abs_tol=1e-3):
                errors.append(
                    f"walk_edges[{index}] length_cm={edge.length_cm:.3f} does not match "
                    f"its authored polyline length={actual_length:.3f}"
                )

    known_blocks = set(block_ids)
    known_frontages = set(frontage_ids)
    frontage_block = {frontage.frontage_id: frontage.block_id for frontage in layout.frontages}

    _PUBLIC_WALK_KINDS = {"spawn", "sidewalk", "intersection", "crossing", "bridge"}
    node_kind = {node.node_id: node.kind for node in layout.walk_nodes}
    enabled_degree = {node_id: 0 for node_id in known_nodes}
    for edge in layout.walk_edges:
        if edge.enabled and edge.start_node_id in enabled_degree and edge.end_node_id in enabled_degree:
            enabled_degree[edge.start_node_id] += 1
            enabled_degree[edge.end_node_id] += 1

    for frontage in layout.frontages:
        if frontage.frontage_id in known_nodes:
            errors.append(
                f"frontage {frontage.frontage_id!r} reuses a walk node_id "
                "(frontage ids must not appear in the walk graph)"
            )
        if frontage.approach_node_id is not None:
            if frontage.approach_node_id not in known_nodes:
                errors.append(
                    f"frontage {frontage.frontage_id!r} approach_node_id="
                    f"{frontage.approach_node_id!r} is not a walk node"
                )
            elif node_kind.get(frontage.approach_node_id) not in _PUBLIC_WALK_KINDS:
                errors.append(
                    f"frontage {frontage.frontage_id!r} approach_node_id="
                    f"{frontage.approach_node_id!r} has non-public kind="
                    f"{node_kind.get(frontage.approach_node_id)!r}"
                )
        elif authored_walk_data:
            errors.append(
                f"frontage {frontage.frontage_id!r} has no explicit approach_node_id "
                "in a schema_version >= 2 layout"
            )
        if authored_walk_data and not frontage.access_path:
            errors.append(
                f"frontage {frontage.frontage_id!r} has no explicit access_path "
                "in a schema_version >= 2 layout"
            )
        if authored_walk_data and frontage.approach_node_id in node_position:
            approach = node_position[frontage.approach_node_id]
            center = frontage.meeting_region.center
            if math.isclose(approach[0], center[0], abs_tol=1e-6) and math.isclose(
                approach[1], center[1], abs_tol=1e-6
            ):
                errors.append(
                    f"frontage {frontage.frontage_id!r} approach_node_id is its meeting-region center"
                )
            if enabled_degree[frontage.approach_node_id] != 1:
                errors.append(
                    f"frontage {frontage.frontage_id!r} approach_node_id="
                    f"{frontage.approach_node_id!r} must be a one-edge public access leaf"
                )
        if frontage.access_path:
            end = frontage.access_path[-1]
            center = frontage.meeting_region.center
            dist = math.hypot(end[0] - center[0], end[1] - center[1])
            if authored_walk_data and dist > 1e-6:
                errors.append(
                    f"frontage {frontage.frontage_id!r} access_path endpoint must equal "
                    "the meeting region center in a schema_version >= 2 layout"
                )
            elif dist > frontage.meeting_region.radius:
                errors.append(
                    f"frontage {frontage.frontage_id!r} access_path endpoint "
                    f"is {dist:.0f} cm from meeting region center "
                    f"(must be within radius={frontage.meeting_region.radius:.0f})"
                )

    for frontage in layout.frontages:
        if frontage.block_id not in known_blocks:
            errors.append(
                f"frontage {frontage.frontage_id!r} references missing block_id={frontage.block_id!r}"
            )

    for block in layout.blocks:
        for frontage_id in block.frontage_ids:
            if frontage_id not in known_frontages:
                errors.append(
                    f"block {block.block_id!r} references missing frontage_id={frontage_id!r}"
                )
                continue
            owner_block = frontage_block.get(frontage_id)
            if owner_block != block.block_id:
                errors.append(
                    f"block {block.block_id!r} lists frontage {frontage_id!r} owned by block {owner_block!r}"
                )

    if required_paths is not None:
        node_id_counts = Counter(node_ids)
        if all(count == 1 for count in node_id_counts.values()):
            reachable_cache: dict[str, frozenset[str]] = {}
            for start_node_id, end_node_id in required_paths:
                if start_node_id not in known_nodes:
                    errors.append(
                        f"required path start_node_id={start_node_id!r} is not a walk node"
                    )
                    continue
                if end_node_id not in known_nodes:
                    errors.append(
                        f"required path end_node_id={end_node_id!r} is not a walk node"
                    )
                    continue
                if start_node_id not in reachable_cache:
                    reachable_cache[start_node_id] = _reachable_from(layout, start_node_id)
                if end_node_id not in reachable_cache[start_node_id]:
                    errors.append(
                        f"required node {end_node_id!r} is unreachable from {start_node_id!r}"
                    )
        else:
            for start_node_id, end_node_id in required_paths:
                if start_node_id not in known_nodes:
                    errors.append(
                        f"required path start_node_id={start_node_id!r} is not a walk node"
                    )
                if end_node_id not in known_nodes:
                    errors.append(
                        f"required path end_node_id={end_node_id!r} is not a walk node"
                    )

    return errors


def validate_layout(
    layout: DistrictLayout,
    *,
    required_paths: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Validate ``layout`` and raise :class:`LayoutValidationError` on failure."""

    errors = collect_layout_errors(layout, required_paths=required_paths)
    if errors:
        raise LayoutValidationError(errors)


def _reachable_from(layout: DistrictLayout, start_node_id: str) -> frozenset[str]:
    """BFS over enabled edges without raising on malformed endpoints."""

    known_nodes = {node.node_id for node in layout.walk_nodes}
    adj: dict[str, list[str]] = {node_id: [] for node_id in known_nodes}
    for edge in layout.walk_edges:
        if not edge.enabled:
            continue
        if edge.start_node_id not in known_nodes or edge.end_node_id not in known_nodes:
            continue
        adj[edge.start_node_id].append(edge.end_node_id)
        adj[edge.end_node_id].append(edge.start_node_id)
    seen = {start_node_id}
    stack = [start_node_id]
    while stack:
        node_id = stack.pop()
        for neighbor_id in adj.get(node_id, []):
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                stack.append(neighbor_id)
    return frozenset(seen)


def _duplicate_id_errors(label: str, values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return [f"duplicate {label}: {item_id!r}" for item_id, count in sorted(counts.items()) if count > 1]
