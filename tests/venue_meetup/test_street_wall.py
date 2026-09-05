"""Pure contract tests for the isolated continuous street-wall primitive."""

from __future__ import annotations

import pytest

from benchmark.venue_meetup.busy_street import (
    BAR_ASSETS,
    BOOKSHOP_ASSETS,
    HOUSE_ASSETS,
    RESTAURANT_ASSETS,
    SKYSCRAPER_ASSETS,
    plan_busy_street,
    plan_busy_street_props,
)
from benchmark.venue_meetup.street_wall import plan_street_wall, street_wall_metrics


def test_wall_fills_segment_with_hard_maximum_seam() -> None:
    wall = plan_street_wall(
        (-12000.0, 3000.0),
        (12000.0, 3000.0),
        outward=(0.0, -1.0),
        gap_cm=50.0,
    )
    metrics = street_wall_metrics(wall, 24000.0)

    assert metrics.actor_count == 19
    assert metrics.coverage == pytest.approx(0.9625)
    assert metrics.maximum_gap_cm <= 50.0 + 1e-6
    assert metrics.leading_gap_cm == pytest.approx(0.0, abs=1e-6)
    assert metrics.trailing_gap_cm == pytest.approx(0.0, abs=1e-6)
    assert all(
        right.tangent_start_cm - left.tangent_end_cm == pytest.approx(50.0)
        for left, right in zip(wall, wall[1:])
    )


def test_zero_gap_wall_is_continuous_end_to_end() -> None:
    wall = plan_street_wall(
        (-10000.0, 0.0),
        (10000.0, 0.0),
        outward=(0.0, -1.0),
        gap_cm=0.0,
    )
    metrics = street_wall_metrics(wall, 20000.0)

    assert metrics.coverage == pytest.approx(1.0, abs=1e-9)
    assert metrics.maximum_gap_cm == pytest.approx(0.0, abs=1e-6)
    assert all(
        right.tangent_start_cm == pytest.approx(left.tangent_end_cm, abs=1e-6)
        for left, right in zip(wall, wall[1:])
    )


def test_visual_fill_ratio_overlaps_oversized_measured_bounds() -> None:
    wall = plan_street_wall(
        (-10000.0, 0.0),
        (10000.0, 0.0),
        outward=(0.0, -1.0),
        gap_cm=0.0,
        facade_fill_ratio=0.85,
    )
    metrics = street_wall_metrics(wall, 20000.0)

    assert metrics.coverage == pytest.approx(1.0)
    assert all(
        item.measured_tangent_width_cm > item.tangent_width_cm
        for item in wall
    )
    assert all(
        left.tangent_end_cm == pytest.approx(right.tangent_start_cm, abs=1e-6)
        for left, right in zip(wall, wall[1:])
    )


def test_wall_is_deterministic_and_respects_scale_bounds() -> None:
    kwargs = {
        "outward": (-1.0, 0.0),
        "gap_cm": 25.0,
        "min_scale": 0.20,
        "max_scale": 0.50,
        "height_factor": 1.6,
    }
    first = plan_street_wall((4000.0, -9000.0), (4000.0, 9000.0), **kwargs)
    second = plan_street_wall((4000.0, -9000.0), (4000.0, 9000.0), **kwargs)

    assert first == second
    assert first
    assert all(0.20 <= item.scale[0] <= 0.50 for item in first)
    assert all(item.scale[0] == item.scale[1] for item in first)
    assert all(item.scale[2] == pytest.approx(item.scale[0] * 1.6) for item in first)
    assert {item.yaw_deg for item in first} == {180.0}


def test_per_asset_height_factors_create_distinct_silhouettes() -> None:
    wall = plan_street_wall(
        (-8000.0, 0.0),
        (8000.0, 0.0),
        outward=(0.0, -1.0),
        asset_keys=("BP_Building_05_C", "BP_Building_06_C"),
        preferred_scales={
            "BP_Building_05_C": 0.42,
            "BP_Building_06_C": 0.39,
        },
        height_factors={
            "BP_Building_05_C": 1.1,
            "BP_Building_06_C": 1.8,
        },
    )

    assert all(
        item.scale[2]
        == pytest.approx(
            item.scale[0] * (1.1 if item.asset_key == "BP_Building_05_C" else 1.8)
        )
        for item in wall
    )


def test_busy_street_composes_restaurants_and_houses_without_seams() -> None:
    buildings = plan_busy_street(
        (-15000.0, 3000.0),
        (15000.0, 3000.0),
        outward=(0.0, 1.0),
    )
    placements = tuple(building.placement for building in buildings)
    metrics = street_wall_metrics(placements, 30000.0)

    assert metrics.coverage == pytest.approx(1.0)
    assert metrics.maximum_gap_cm == pytest.approx(0.0, abs=1e-6)
    assert {building.use for building in buildings} == {
        "restaurant",
        "house",
        "bookshop",
        "bar",
        "skyscraper_lobby",
    }
    assert all(
        building.placement.asset_key in RESTAURANT_ASSETS
        for building in buildings
        if building.use == "restaurant"
    )
    assert all(
        building.placement.asset_key in HOUSE_ASSETS
        for building in buildings
        if building.use == "house"
    )
    role_assets = {
        "bookshop": BOOKSHOP_ASSETS,
        "bar": BAR_ASSETS,
        "skyscraper_lobby": SKYSCRAPER_ASSETS,
    }
    assert all(
        building.placement.asset_key in role_assets[building.use]
        for building in buildings
        if building.use in role_assets
    )
    venue_buildings = [building for building in buildings if building.venue_id]
    assert len(venue_buildings) == 12
    assert len({building.venue_id for building in venue_buildings}) == 12
    assert len({building.mask_color_rgb for building in venue_buildings}) == 12
    restaurant_heights = [
        building.placement.scale[2]
        for building in buildings
        if building.use == "restaurant"
    ]
    house_heights = [
        building.placement.scale[2]
        for building in buildings
        if building.use == "house"
    ]
    assert max(house_heights) > min(restaurant_heights)


def test_busy_street_props_make_restaurants_readable_and_stay_outside_wall() -> None:
    start = (-15000.0, 3000.0)
    end = (15000.0, 3000.0)
    buildings = plan_busy_street(start, end, outward=(0.0, 1.0))
    props = plan_busy_street_props(
        buildings,
        start,
        end,
        outward=(0.0, 1.0),
    )

    restaurants = [item for item in buildings if item.use == "restaurant"]
    tables = [item for item in props if item.use == "restaurant_seating"]
    book_displays = [item for item in props if item.use == "book_display"]
    bar_tables = [item for item in props if item.use == "bar_seating"]
    furniture = [item for item in props if item.use == "street_furniture"]

    assert len(tables) == len(restaurants)
    assert len(book_displays) == len(
        [item for item in buildings if item.use == "bookshop"]
    )
    assert len(bar_tables) == len([item for item in buildings if item.use == "bar"])
    assert furniture
    assert {item.asset_key for item in tables} == {"BP_Table_C", "BP_Table2_C"}
    assert all(item.position[1] > 3000.0 for item in props)
    assert all(item.building_index < len(buildings) for item in props)


def test_explicit_wall_target_count_is_enforced() -> None:
    wall = plan_street_wall(
        (-11000.0, 0.0),
        (11000.0, 0.0),
        outward=(0.0, 1.0),
        target_count=20,
    )

    assert len(wall) == 20

    with pytest.raises(ValueError, match="target_count"):
        plan_street_wall(
            (-1000.0, 0.0),
            (1000.0, 0.0),
            outward=(0.0, 1.0),
            target_count=20,
        )


@pytest.mark.parametrize(
    ("start", "end", "outward", "message"),
    (
        ((0.0, 0.0), (0.0, 0.0), (0.0, -1.0), "non-zero"),
        ((0.0, 0.0), (1000.0, 0.0), (1.0, 0.0), "perpendicular"),
    ),
)
def test_wall_rejects_invalid_geometry(start, end, outward, message) -> None:
    with pytest.raises(ValueError, match=message):
        plan_street_wall(start, end, outward=outward)
