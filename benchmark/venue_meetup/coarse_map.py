"""Coarse schematic map generation for Venue Meetup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from benchmark.venue_meetup.layout import DistrictLayout
from benchmark.venue_meetup.scenario import Scenario

# Trait / constraint names that must never appear as keys in the public schematic JSON.
HIDDEN_SCHEMATIC_KEYS = frozenset(
    {
        "open",
        "accessible",
        "food_drink",
        "crowding",
        "crowding_score",
        "quiet",
        "quiet_score",
        "capacity",
        "requirements",
        "private_constraint",
        "private_requirement_keys",
        "properties",
        "entrances",
        "soft_weights",
    }
)


def _world_to_image(point: tuple[float, float], *, size: int, extent: float) -> tuple[int, int]:
    """Convert Unreal 2D coordinates to image coordinates."""

    x = int(size / 2 + (point[0] / extent) * (size / 2 - 48))
    y = int(size / 2 - (point[1] / extent) * (size / 2 - 48))
    return x, y


def _layout_extent_coords(layout: DistrictLayout) -> list[float]:
    """Collect absolute coordinates from authored layout geometry."""

    coords: list[float] = []
    for street in layout.streets:
        coords += [abs(street.start[0]), abs(street.start[1]), abs(street.end[0]), abs(street.end[1])]
        # Half-width padding so thick corridors stay inside the frame.
        half = 0.5 * float(street.width_cm)
        coords.append(half)
    for intersection in layout.intersections:
        coords += [abs(intersection.position[0]), abs(intersection.position[1])]
    for block in layout.blocks:
        for point in block.footprint:
            coords += [abs(point[0]), abs(point[1])]
    for frontage in layout.frontages:
        coords += [
            abs(frontage.position[0]),
            abs(frontage.position[1]),
            abs(frontage.entrance_point[0]),
            abs(frontage.entrance_point[1]),
            abs(frontage.meeting_region.center[0]),
            abs(frontage.meeting_region.center[1]),
            float(frontage.meeting_region.radius),
        ]
    for node in layout.walk_nodes:
        coords += [abs(node.position[0]), abs(node.position[1])]
    return coords


def _auto_extent(scenario: Scenario) -> float:
    """Return a half-extent that comfortably fits every placed element."""

    coords = [0.0]
    for venue in scenario.venues:
        coords += [abs(venue.position[0]), abs(venue.position[1])]
    for landmark in scenario.landmarks:
        coords += [abs(landmark.position[0]), abs(landmark.position[1])]
    for agent in scenario.agents:
        coords += [abs(agent.position[0]), abs(agent.position[1])]
    if scenario.layout is not None:
        coords += _layout_extent_coords(scenario.layout)
    return max(coords) * 1.15 or 1200.0


def _load_fonts() -> tuple[Any, Any]:
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    return font, small_font


def _cm_to_px(length_cm: float, *, size: int, extent: float) -> int:
    scale = (size / 2 - 48) / extent
    return max(1, int(round(abs(length_cm) * scale)))


def _is_bridge_street(street_id: str) -> bool:
    return "bridge" in street_id.lower()


def _draw_street_segment(
    draw: ImageDraw.ImageDraw,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    width_cm: float,
    size: int,
    extent: float,
    fill: str,
) -> None:
    p0 = _world_to_image(start, size=size, extent=extent)
    p1 = _world_to_image(end, size=size, extent=extent)
    width = _cm_to_px(width_cm, size=size, extent=extent)
    draw.line((p0[0], p0[1], p1[0], p1[1]), fill=fill, width=max(2, width))


def _render_central_square_fallback(
    scenario: Scenario,
    draw: ImageDraw.ImageDraw,
    *,
    size: int,
    extent: float,
    font: Any,
    small_font: Any,
) -> None:
    """Preserve the original central-square schematic drawing."""

    margin = 48
    center = size // 2
    draw.line((margin, center, size - margin, center), fill="gray", width=4)
    draw.line((center, margin, center, size - margin), fill="gray", width=4)
    draw.ellipse((center - 70, center - 70, center + 70, center + 70), outline="gray", width=3)
    draw.text((center - 54, center - 10), "central square", fill="black", font=small_font)

    for landmark in scenario.landmarks:
        x, y = _world_to_image((landmark.position[0], landmark.position[1]), size=size, extent=extent)
        draw.rectangle((x - 26, y - 26, x + 26, y + 26), outline="navy", width=3)
        label = landmark.landmark_type.replace("_", " ")
        draw.text((x - 54, y + 32), label, fill="navy", font=small_font)

    for venue in scenario.venues:
        x, y = _world_to_image((venue.position[0], venue.position[1]), size=size, extent=extent)
        draw.ellipse((x - 30, y - 30, x + 30, y + 30), outline="darkgreen", width=3)
        label = venue.venue_type.replace("_", " ")
        draw.text((x - 46, y + 34), label, fill="darkgreen", font=small_font)

    for agent in scenario.agents:
        x, y = _world_to_image((agent.position[0], agent.position[1]), size=size, extent=extent)
        draw.polygon([(x, y - 18), (x - 14, y + 12), (x + 14, y + 12)], outline="black", fill="lightgray")
        draw.text((x - 28, y + 18), agent.agent_id, fill="black", font=small_font)

    draw.text((20, 18), "Venue Meetup V0 coarse map", fill="black", font=font)
    draw.text(
        (20, size - 38),
        "Map hides: open/closed, accessibility, crowding, food/drink, blocked entrances.",
        fill="black",
        font=small_font,
    )


def _render_layout_district(
    scenario: Scenario,
    layout: DistrictLayout,
    draw: ImageDraw.ImageDraw,
    *,
    size: int,
    extent: float,
    font: Any,
    small_font: Any,
) -> None:
    """Render an authored district from ``scenario.layout`` (public cues only)."""

    # Blocks first (footprints behind streets).
    for block in layout.blocks:
        if len(block.footprint) < 3:
            continue
        points = [_world_to_image(point, size=size, extent=extent) for point in block.footprint]
        draw.polygon(points, outline="dimgray", fill="#e8e8e8")
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        draw.text((cx - 28, cy - 6), block.block_id.replace("_", " "), fill="dimgray", font=small_font)

    # Street corridors with authored widths; bridges drawn distinctly.
    for street in layout.streets:
        bridge = _is_bridge_street(street.street_id)
        _draw_street_segment(
            draw,
            start=street.start,
            end=street.end,
            width_cm=street.width_cm,
            size=size,
            extent=extent,
            fill="#c45c26" if bridge else "gray",
        )
        if bridge:
            # Outline so bridges remain visible even when thin.
            mid = (
                0.5 * (street.start[0] + street.end[0]),
                0.5 * (street.start[1] + street.end[1]),
            )
            mx, my = _world_to_image(mid, size=size, extent=extent)
            draw.text((mx - 24, my - 10), "bridge", fill="#8b3a0f", font=small_font)

    # Walk graph (enabled edges); bridge edges use the bridge accent.
    node_index = {node.node_id: node for node in layout.walk_nodes}
    for edge in layout.walk_edges:
        if not edge.enabled:
            continue
        start = node_index.get(edge.start_node_id)
        end = node_index.get(edge.end_node_id)
        if start is None or end is None:
            continue
        p0 = _world_to_image(start.position, size=size, extent=extent)
        p1 = _world_to_image(end.position, size=size, extent=extent)
        if edge.route_kind == "bridge":
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill="#c45c26", width=5)
        else:
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill="#b0b0b0", width=1)

    # Intersections (or walk nodes when intersections are absent).
    if layout.intersections:
        for intersection in layout.intersections:
            x, y = _world_to_image(intersection.position, size=size, extent=extent)
            draw.regular_polygon((x, y, 7), n_sides=4, rotation=45, fill="white", outline="black")
    else:
        for node in layout.walk_nodes:
            if node.kind not in {"intersection", "crossing", "bridge", "spawn"}:
                continue
            x, y = _world_to_image(node.position, size=size, extent=extent)
            if node.kind == "bridge":
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#c45c26", outline="#8b3a0f")
            elif node.kind == "spawn":
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="lightgray", outline="black")
            else:
                draw.regular_polygon((x, y, 5), n_sides=4, rotation=45, fill="white", outline="gray")

    # Named frontages (public storefront labels only).
    for frontage in layout.frontages:
        x, y = _world_to_image((frontage.position[0], frontage.position[1]), size=size, extent=extent)
        draw.rectangle((x - 10, y - 10, x + 10, y + 10), outline="#2e7d32", width=2)
        label = (frontage.venue_slot_id or frontage.frontage_id).replace("_", " ")
        draw.text((x - 40, y + 12), label, fill="#2e7d32", font=small_font)

    # Venue markers (type labels; never hidden traits).
    for venue in scenario.venues:
        x, y = _world_to_image((venue.position[0], venue.position[1]), size=size, extent=extent)
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), outline="darkgreen", width=3)
        label = venue.venue_type.replace("_", " ")
        draw.text((x - 36, y - 34), label, fill="darkgreen", font=small_font)

    for landmark in scenario.landmarks:
        x, y = _world_to_image((landmark.position[0], landmark.position[1]), size=size, extent=extent)
        draw.rectangle((x - 22, y - 22, x + 22, y + 22), outline="navy", width=3)
        label = landmark.landmark_type.replace("_", " ")
        draw.text((x - 48, y + 26), label, fill="navy", font=small_font)

    for agent in scenario.agents:
        x, y = _world_to_image((agent.position[0], agent.position[1]), size=size, extent=extent)
        draw.polygon([(x, y - 18), (x - 14, y + 12), (x + 14, y + 12)], outline="black", fill="lightgray")
        draw.text((x - 28, y + 18), agent.agent_id, fill="black", font=small_font)

    title = f"Venue Meetup coarse map ({layout.layout_id})"
    draw.text((20, 18), title, fill="black", font=font)
    draw.text(
        (20, size - 38),
        "Map hides: open/closed, accessibility, crowding, food/drink, blocked entrances.",
        fill="black",
        font=small_font,
    )


def public_coarse_map_compact(scenario: Scenario) -> dict[str, Any]:
    """Return a public-only schematic summary suitable for JSON logs.

    Never includes hidden venue traits, requirements, or private constraints.
    """

    payload: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "map_template_id": scenario.map_template_id,
        "venues": [
            {
                "venue_id": venue.venue_id,
                "slot_id": venue.slot_id,
                "venue_type": venue.venue_type,
                "position": list(venue.position),
            }
            for venue in scenario.venues
        ],
        "landmarks": [
            {
                "landmark_id": landmark.landmark_id,
                "landmark_type": landmark.landmark_type,
                "position": list(landmark.position),
            }
            for landmark in scenario.landmarks
        ],
        "agents": [
            {
                "agent_id": agent.agent_id,
                "spawn_slot": agent.spawn_slot,
                "position": list(agent.position),
                "walk_node_id": agent.walk_node_id,
            }
            for agent in scenario.agents
        ],
    }
    if scenario.layout is not None:
        layout = scenario.layout
        payload["layout"] = {
            "layout_id": layout.layout_id,
            "schema_version": layout.schema_version,
            "streets": [
                {
                    "street_id": street.street_id,
                    "start": list(street.start),
                    "end": list(street.end),
                    "width_cm": street.width_cm,
                    "sidewalk_width_cm": street.sidewalk_width_cm,
                    "is_bridge": _is_bridge_street(street.street_id),
                }
                for street in layout.streets
            ],
            "intersections": [
                {
                    "intersection_id": item.intersection_id,
                    "position": list(item.position),
                    "landmark_id": item.landmark_id,
                }
                for item in layout.intersections
            ],
            "blocks": [
                {
                    "block_id": block.block_id,
                    "footprint": [list(point) for point in block.footprint],
                    "frontage_ids": list(block.frontage_ids),
                }
                for block in layout.blocks
            ],
            "frontages": [
                {
                    "frontage_id": frontage.frontage_id,
                    "block_id": frontage.block_id,
                    "position": list(frontage.position),
                    "venue_slot_id": frontage.venue_slot_id,
                }
                for frontage in layout.frontages
            ],
            "walk_nodes": [
                {
                    "node_id": node.node_id,
                    "position": list(node.position),
                    "kind": node.kind,
                }
                for node in layout.walk_nodes
            ],
            "walk_edges": [
                {
                    "start_node_id": edge.start_node_id,
                    "end_node_id": edge.end_node_id,
                    "length_cm": edge.length_cm,
                    "enabled": edge.enabled,
                    "route_kind": edge.route_kind,
                }
                for edge in layout.walk_edges
            ],
        }
    return payload


def assert_public_schematic_payload(payload: dict[str, Any]) -> None:
    """Raise ``ValueError`` if a public schematic payload leaks hidden fields."""

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in HIDDEN_SCHEMATIC_KEYS:
                    raise ValueError(f"Hidden schematic field leaked at {path}.{key}")
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                walk(value, f"{path}[{index}]")

    walk(payload)


def render_coarse_map(scenario: Scenario, output_path: Path, *, size: int = 768, extent: float | None = None) -> Path:
    """Render a schematic PNG coarse map without hidden venue state."""

    extent = extent or _auto_extent(scenario)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    font, small_font = _load_fonts()

    if scenario.layout is not None:
        _render_layout_district(
            scenario,
            scenario.layout,
            draw,
            size=size,
            extent=extent,
            font=font,
            small_font=small_font,
        )
    else:
        _render_central_square_fallback(
            scenario,
            draw,
            size=size,
            extent=extent,
            font=font,
            small_font=small_font,
        )

    image.save(output_path)
    return output_path


def with_rendered_coarse_map(scenario: Scenario, output_dir: Path) -> Scenario:
    """Render a coarse map and return a scenario copy with the path attached."""

    from dataclasses import replace

    path = render_coarse_map(scenario, output_dir / f"{scenario.scenario_id}_coarse_map.png")
    return replace(scenario, coarse_map_path=str(path))
