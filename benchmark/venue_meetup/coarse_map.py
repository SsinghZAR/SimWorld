"""Coarse schematic map generation for Venue Meetup."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from benchmark.venue_meetup.scenario import Scenario


def _world_to_image(point: tuple[float, float], *, size: int, extent: float) -> tuple[int, int]:
    """Convert Unreal 2D coordinates to image coordinates."""

    x = int(size / 2 + (point[0] / extent) * (size / 2 - 48))
    y = int(size / 2 - (point[1] / extent) * (size / 2 - 48))
    return x, y


def _auto_extent(scenario: Scenario) -> float:
    """Return a half-extent that comfortably fits every placed element."""

    coords = [0.0]
    for venue in scenario.venues:
        coords += [abs(venue.position[0]), abs(venue.position[1])]
    for landmark in scenario.landmarks:
        coords += [abs(landmark.position[0]), abs(landmark.position[1])]
    for agent in scenario.agents:
        coords += [abs(agent.position[0]), abs(agent.position[1])]
    return max(coords) * 1.15 or 1200.0


def render_coarse_map(scenario: Scenario, output_path: Path, *, size: int = 768, extent: float | None = None) -> Path:
    """Render a schematic PNG coarse map without hidden venue state."""

    extent = extent or _auto_extent(scenario)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

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
    draw.text((20, size - 38), "Map hides: open/closed, accessibility, crowding, food/drink, blocked entrances.", fill="black", font=small_font)
    image.save(output_path)
    return output_path


def with_rendered_coarse_map(scenario: Scenario, output_dir: Path) -> Scenario:
    """Render a coarse map and return a scenario copy with the path attached."""

    from dataclasses import replace

    path = render_coarse_map(scenario, output_dir / f"{scenario.scenario_id}_coarse_map.png")
    return replace(scenario, coarse_map_path=str(path))
