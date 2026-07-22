"""Offline coarse-map rendering tests (no UE / network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from benchmark.venue_meetup.coarse_map import (
    HIDDEN_SCHEMATIC_KEYS,
    assert_public_schematic_payload,
    public_coarse_map_compact,
    render_coarse_map,
)
from benchmark.venue_meetup.templates.central_square import build_fixed_scenario as build_central
from benchmark.venue_meetup.templates.riverside_market import build_fixed_scenario as build_riverside
from benchmark.venue_meetup.templates.station_quarter import build_fixed_scenario as build_station


def _sample_hash(image: Image.Image, *, stride: int = 32) -> str:
    """Stable content fingerprint from a regular pixel grid sample."""

    width, height = image.size
    samples: list[tuple[int, int, int, int, int]] = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            pixel = image.getpixel((x, y))
            assert isinstance(pixel, tuple) and len(pixel) == 3
            samples.append((x, y, int(pixel[0]), int(pixel[1]), int(pixel[2])))
    digest = hashlib.sha256(repr(samples).encode("utf-8")).hexdigest()
    return digest


def _assert_valid_rgb_png(path: Path, *, size: int) -> Image.Image:
    assert path.is_file()
    assert path.stat().st_size > 0
    image = Image.open(path)
    image.load()
    assert image.format == "PNG"
    assert image.mode == "RGB"
    assert image.size == (size, size)
    return image


def _assert_public_json_free_of_hidden(scenario) -> None:
    public = scenario.compact(include_hidden=False)
    assert "requirements" not in public
    assert "soft_weights" not in public
    for venue in public["venues"]:
        assert "properties" not in venue
        assert "entrances" not in venue
        for key in HIDDEN_SCHEMATIC_KEYS:
            assert key not in venue
        # Nested property values must not appear under venues either.
        blob = json.dumps(venue)
        for token in (
            "food_drink",
            "crowding_score",
            "quiet_score",
            "capacity",
            "accessible",
        ):
            assert token not in blob

    schematic = public_coarse_map_compact(scenario)
    assert_public_schematic_payload(schematic)
    schematic_blob = json.dumps(schematic)
    for token in (
        '"properties"',
        '"requirements"',
        '"private_constraint"',
        '"private_requirement_keys"',
        '"food_drink"',
        '"crowding_score"',
        '"quiet_score"',
        '"capacity"',
        '"accessible"',
        '"open"',
    ):
        assert token not in schematic_blob


def test_central_square_fallback_png(tmp_path: Path) -> None:
    size = 512
    scenario = build_central(seed=7)
    assert scenario.layout is None
    path = tmp_path / "central_coarse_map.png"
    render_coarse_map(scenario, path, size=size)
    image = _assert_valid_rgb_png(path, size=size)
    # Fallback silhouette: axis cross through the image center.
    center = size // 2
    assert image.getpixel((center, 64)) != (255, 255, 255)
    assert image.getpixel((64, center)) != (255, 255, 255)
    _assert_public_json_free_of_hidden(scenario)


def test_station_quarter_layout_map_distinct_from_central(tmp_path: Path) -> None:
    size = 640
    station = build_station(seed=11)
    central = build_central(seed=7)
    assert station.layout is not None
    assert central.layout is None

    station_path = tmp_path / "station.png"
    central_path = tmp_path / "central.png"
    render_coarse_map(station, station_path, size=size)
    render_coarse_map(central, central_path, size=size)

    station_img = _assert_valid_rgb_png(station_path, size=size)
    central_img = _assert_valid_rgb_png(central_path, size=size)
    station_hash = _sample_hash(station_img)
    central_hash = _sample_hash(central_img)
    assert station_hash != central_hash

    _assert_public_json_free_of_hidden(station)


def test_riverside_layout_map_distinct_from_central_and_station(tmp_path: Path) -> None:
    size = 640
    riverside = build_riverside(seed=31)
    station = build_station(seed=11)
    central = build_central(seed=7)
    assert riverside.layout is not None

    riverside_path = tmp_path / "riverside.png"
    station_path = tmp_path / "station.png"
    central_path = tmp_path / "central.png"
    render_coarse_map(riverside, riverside_path, size=size)
    render_coarse_map(station, station_path, size=size)
    render_coarse_map(central, central_path, size=size)

    riverside_img = _assert_valid_rgb_png(riverside_path, size=size)
    station_img = _assert_valid_rgb_png(station_path, size=size)
    central_img = _assert_valid_rgb_png(central_path, size=size)

    riverside_hash = _sample_hash(riverside_img)
    station_hash = _sample_hash(station_img)
    central_hash = _sample_hash(central_img)
    assert riverside_hash != central_hash
    assert riverside_hash != station_hash

    # Bridge accent color should appear on the riverside schematic.
    bridge_accent = (0xC4, 0x5C, 0x26)
    found_bridge = False
    width, height = riverside_img.size
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            if riverside_img.getpixel((x, y)) == bridge_accent:
                found_bridge = True
                break
        if found_bridge:
            break
    assert found_bridge

    _assert_public_json_free_of_hidden(riverside)


def test_layout_extent_includes_geometry_not_only_actors() -> None:
    from benchmark.venue_meetup.coarse_map import _auto_extent

    riverside = build_riverside(seed=31)
    assert riverside.layout is not None
    with_layout = _auto_extent(riverside)
    # Drop layout so extent falls back to actors only.
    from dataclasses import replace

    actors_only = replace(riverside, layout=None)
    without_layout = _auto_extent(actors_only)
    assert with_layout >= without_layout
