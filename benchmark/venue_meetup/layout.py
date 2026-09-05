"""Typed city-district layout primitives and walk-graph helpers.

Pure Python (standard library only). Coordinates use Unreal centimetres.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
Polygon2D = tuple[Point2D, ...]

WalkNodeKind = Literal["spawn", "sidewalk", "intersection", "crossing", "frontage", "bridge"]
WalkRouteKind = Literal["sidewalk", "crossing", "bridge", "alley"]


def _as_point2d(value: Sequence[float]) -> Point2D:
    return (float(value[0]), float(value[1]))


def _as_point3d(value: Sequence[float]) -> Point3D:
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_polygon2d(value: Sequence[Sequence[float]]) -> Polygon2D:
    return tuple(_as_point2d(point) for point in value)


@dataclass(frozen=True)
class MeetingRegion:
    """Circular 2D meeting / trigger region in Unreal centimetres."""

    center: Point2D
    radius: float


@dataclass(frozen=True)
class StreetSegment:
    """One authored street corridor between two points."""

    street_id: str
    start: Point2D
    end: Point2D
    width_cm: float
    sidewalk_width_cm: float


@dataclass(frozen=True)
class Intersection:
    """Named street intersection, optionally tied to a landmark."""

    intersection_id: str
    position: Point2D
    landmark_id: str | None = None


@dataclass(frozen=True)
class Block:
    """City block footprint with attached frontage ids and visual character.

    visual_style and shell_target are renderer hints rather than navigation
    semantics. Keeping them on the authored block lets a template describe a
    coherent facade rhythm without teaching the generic district renderer
    about template ids or block-name conventions.
    """

    block_id: str
    footprint: Polygon2D
    frontage_ids: tuple[str, ...] = ()
    visual_style: str = "mixed"
    shell_target: int | None = None


@dataclass(frozen=True)
class Frontage:
    """Public storefront / meeting attachment on a block."""

    frontage_id: str
    block_id: str
    position: Point3D
    yaw_deg: float
    entrance_point: Point3D
    meeting_region: MeetingRegion
    venue_slot_id: str | None = None
    approach_node_id: str | None = None
    access_path: tuple[Point2D, ...] = ()


@dataclass(frozen=True)
class WalkNode:
    """Node in the district walk graph."""

    node_id: str
    position: Point2D
    kind: WalkNodeKind


@dataclass(frozen=True)
class WalkEdge:
    """Walkable connection between two walk nodes.

    Treated as undirected unless a future field says otherwise.
    Disabled edges are ignored by path and reachability helpers.
    """

    start_node_id: str
    end_node_id: str
    length_cm: float
    enabled: bool = True
    route_kind: WalkRouteKind = "sidewalk"
    waypoints: tuple[Point2D, ...] = ()


@dataclass(frozen=True)
class DistrictLayout:
    """Authored district geometry plus walk graph used by later templates."""

    layout_id: str
    streets: tuple[StreetSegment, ...] = ()
    intersections: tuple[Intersection, ...] = ()
    blocks: tuple[Block, ...] = ()
    frontages: tuple[Frontage, ...] = ()
    walk_nodes: tuple[WalkNode, ...] = ()
    walk_edges: tuple[WalkEdge, ...] = ()
    schema_version: int = 1

    def street_by_id(self, street_id: str) -> StreetSegment:
        """Return a street segment by id."""

        return self._unique_lookup(
            items=self.streets,
            attr="street_id",
            key=street_id,
            label="street_id",
        )

    def intersection_by_id(self, intersection_id: str) -> Intersection:
        """Return an intersection by id."""

        return self._unique_lookup(
            items=self.intersections,
            attr="intersection_id",
            key=intersection_id,
            label="intersection_id",
        )

    def block_by_id(self, block_id: str) -> Block:
        """Return a block by id."""

        return self._unique_lookup(
            items=self.blocks,
            attr="block_id",
            key=block_id,
            label="block_id",
        )

    def frontage_by_id(self, frontage_id: str) -> Frontage:
        """Return a frontage by id."""

        return self._unique_lookup(
            items=self.frontages,
            attr="frontage_id",
            key=frontage_id,
            label="frontage_id",
        )

    def node_by_id(self, node_id: str) -> WalkNode:
        """Return a walk node by id."""

        nodes = self._node_index()
        if node_id not in nodes:
            raise ValueError(f"Unknown walk node_id: {node_id}")
        return nodes[node_id]

    def edge_polyline(self, edge: WalkEdge) -> tuple[Point2D, ...]:
        """Full polyline for *edge*: start-node position, waypoints, end-node position."""

        nodes = self._node_index()
        start = nodes[edge.start_node_id].position
        end = nodes[edge.end_node_id].position
        return (start, *edge.waypoints, end)

    def _unique_lookup(self, *, items: Sequence[Any], attr: str, key: str, label: str) -> Any:
        found: Any | None = None
        for item in items:
            item_id = getattr(item, attr)
            if item_id != key:
                continue
            if found is not None:
                raise ValueError(f"Duplicate {label}: {key}")
            found = item
        if found is None:
            raise ValueError(f"Unknown {label}: {key}")
        return found

    def _node_index(self) -> dict[str, WalkNode]:
        index: dict[str, WalkNode] = {}
        for node in self.walk_nodes:
            if node.node_id in index:
                raise ValueError(f"Duplicate walk node_id: {node.node_id}")
            index[node.node_id] = node
        return index

    def _adjacency(self) -> dict[str, list[tuple[str, float]]]:
        """Return undirected adjacency for enabled edges.

        Neighbors are sorted by node id so path search is deterministic.
        Disabled edges are skipped entirely, even when endpoints or lengths are
        malformed. Enabled edges with unknown endpoints or non-finite /
        non-positive ``length_cm`` raise ``ValueError``.
        """

        nodes = self._node_index()
        adj: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in nodes}
        for edge in self.walk_edges:
            if not edge.enabled:
                continue
            if edge.start_node_id not in nodes:
                raise ValueError(f"Unknown walk edge endpoint: {edge.start_node_id}")
            if edge.end_node_id not in nodes:
                raise ValueError(f"Unknown walk edge endpoint: {edge.end_node_id}")
            weight = float(edge.length_cm)
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    f"Invalid walk edge length_cm: {edge.length_cm!r} "
                    f"({edge.start_node_id} -> {edge.end_node_id}); "
                    "must be finite and > 0"
                )
            adj[edge.start_node_id].append((edge.end_node_id, weight))
            adj[edge.end_node_id].append((edge.start_node_id, weight))
        for node_id, neighbors in adj.items():
            neighbors.sort(key=lambda item: (item[0], item[1]))
            adj[node_id] = neighbors
        return adj

    def shortest_path(self, start_node_id: str, end_node_id: str) -> list[str] | None:
        """Return the deterministic weighted shortest path as node ids.

        Returns ``None`` when the destination is unreachable. Unknown requested
        node ids, unknown enabled-edge endpoints, and non-finite or non-positive
        enabled edge lengths raise ``ValueError``. Equal-length ties prefer the
        lexicographically smaller frontier node and keep the first predecessor
        found.
        """

        nodes = self._node_index()
        if start_node_id not in nodes:
            raise ValueError(f"Unknown walk node_id: {start_node_id}")
        if end_node_id not in nodes:
            raise ValueError(f"Unknown walk node_id: {end_node_id}")

        adj = self._adjacency()
        if start_node_id == end_node_id:
            return [start_node_id]

        dist: dict[str, float] = {start_node_id: 0.0}
        prev: dict[str, str | None] = {start_node_id: None}
        heap: list[tuple[float, str]] = [(0.0, start_node_id)]

        while heap:
            distance, node_id = heapq.heappop(heap)
            if node_id == end_node_id:
                break
            if distance > dist.get(node_id, math.inf):
                continue
            for neighbor_id, weight in adj.get(node_id, []):
                candidate = distance + weight
                best = dist.get(neighbor_id, math.inf)
                if candidate < best:
                    dist[neighbor_id] = candidate
                    prev[neighbor_id] = node_id
                    heapq.heappush(heap, (candidate, neighbor_id))

        if end_node_id not in prev:
            return None

        path: list[str] = []
        cursor: str | None = end_node_id
        while cursor is not None:
            path.append(cursor)
            cursor = prev.get(cursor)
        path.reverse()
        return path

    def reachable_nodes(self, start_node_id: str) -> frozenset[str]:
        """Return all node ids reachable from ``start_node_id`` via enabled edges.

        Unknown ``start_node_id``, unknown enabled-edge endpoints, and
        non-finite or non-positive enabled edge lengths raise ``ValueError``.
        """

        nodes = self._node_index()
        if start_node_id not in nodes:
            raise ValueError(f"Unknown walk node_id: {start_node_id}")

        adj = self._adjacency()
        seen = {start_node_id}
        stack = [start_node_id]
        while stack:
            node_id = stack.pop()
            for neighbor_id, _weight in adj.get(node_id, []):
                if neighbor_id not in seen:
                    seen.add(neighbor_id)
                    stack.append(neighbor_id)
        return frozenset(seen)

    def is_reachable(self, start_node_id: str, end_node_id: str) -> bool:
        """Return whether ``end_node_id`` is reachable from ``start_node_id``."""

        nodes = self._node_index()
        if end_node_id not in nodes:
            raise ValueError(f"Unknown walk node_id: {end_node_id}")
        return end_node_id in self.reachable_nodes(start_node_id)

    def path_length_cm(self, start_node_id: str, end_node_id: str) -> float | None:
        """Return total enabled-edge length of the shortest path, or None.

        Raises the same ``ValueError`` conditions as ``shortest_path``.
        """

        path = self.shortest_path(start_node_id, end_node_id)
        if path is None:
            return None
        if len(path) <= 1:
            return 0.0

        adj = self._adjacency()
        total = 0.0
        for left, right in zip(path, path[1:]):
            weight = next((w for neighbor_id, w in adj[left] if neighbor_id == right), None)
            if weight is None:
                raise ValueError(f"Missing walk edge on path: {left} -> {right}")
            total += weight
        return total

    def path_length(self, start_node_id: str, end_node_id: str) -> float | None:
        """Alias for ``path_length_cm``."""

        return self.path_length_cm(start_node_id, end_node_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        return self.compact()

    def compact(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistrictLayout:
        """Rehydrate a layout from a JSON-compatible mapping."""

        return district_layout_from_dict(payload)


def district_layout_from_dict(payload: Mapping[str, Any]) -> DistrictLayout:
    """Rehydrate a :class:`DistrictLayout` from JSON-compatible data."""

    streets = tuple(
        StreetSegment(
            street_id=str(item["street_id"]),
            start=_as_point2d(item["start"]),
            end=_as_point2d(item["end"]),
            width_cm=float(item["width_cm"]),
            sidewalk_width_cm=float(item["sidewalk_width_cm"]),
        )
        for item in payload.get("streets", ())
    )
    intersections = tuple(
        Intersection(
            intersection_id=str(item["intersection_id"]),
            position=_as_point2d(item["position"]),
            landmark_id=item.get("landmark_id"),
        )
        for item in payload.get("intersections", ())
    )
    blocks = tuple(
        Block(
            block_id=str(item["block_id"]),
            footprint=_as_polygon2d(item["footprint"]),
            frontage_ids=tuple(str(fid) for fid in item.get("frontage_ids", ())),
            visual_style=str(item.get("visual_style", "mixed")),
            shell_target=(
                int(item["shell_target"])
                if item.get("shell_target") is not None
                else None
            ),
        )
        for item in payload.get("blocks", ())
    )
    frontages = tuple(
        Frontage(
            frontage_id=str(item["frontage_id"]),
            block_id=str(item["block_id"]),
            position=_as_point3d(item["position"]),
            yaw_deg=float(item["yaw_deg"]),
            entrance_point=_as_point3d(item["entrance_point"]),
            meeting_region=MeetingRegion(
                center=_as_point2d(item["meeting_region"]["center"]),
                radius=float(item["meeting_region"]["radius"]),
            ),
            venue_slot_id=item.get("venue_slot_id"),
            approach_node_id=item.get("approach_node_id"),
            access_path=tuple(_as_point2d(p) for p in item.get("access_path", ())),
        )
        for item in payload.get("frontages", ())
    )
    walk_nodes = tuple(
        WalkNode(
            node_id=str(item["node_id"]),
            position=_as_point2d(item["position"]),
            kind=item["kind"],
        )
        for item in payload.get("walk_nodes", ())
    )
    walk_edges = tuple(
        WalkEdge(
            start_node_id=str(item["start_node_id"]),
            end_node_id=str(item["end_node_id"]),
            length_cm=float(item["length_cm"]),
            enabled=bool(item.get("enabled", True)),
            route_kind=item.get("route_kind", "sidewalk"),
            waypoints=tuple(_as_point2d(p) for p in item.get("waypoints", ())),
        )
        for item in payload.get("walk_edges", ())
    )
    return DistrictLayout(
        layout_id=str(payload["layout_id"]),
        streets=streets,
        intersections=intersections,
        blocks=blocks,
        frontages=frontages,
        walk_nodes=walk_nodes,
        walk_edges=walk_edges,
        schema_version=int(payload.get("schema_version", 1)),
    )
