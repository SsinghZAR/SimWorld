"""Measured packing primitive for one continuous row of building facades.

This module deliberately knows nothing about venues, blocks, navigation, or
UnrealCV. It solves one smaller problem: fill a straight authored frontage
with measured building envelopes while enforcing a hard maximum seam gap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from benchmark.venue_meetup.building_catalog import building_bbox

Point2D = tuple[float, float]

DEFAULT_FACADE_ASSETS = (
    "BP_Building_05_C",
    "BP_Building_06_C",
    "BP_Building_24_C",
    "BP_Building_25_C",
)

DEFAULT_PREFERRED_SCALES = {
    "BP_Building_05_C": 0.42,
    "BP_Building_06_C": 0.39,
    "BP_Building_24_C": 0.39,
    "BP_Building_25_C": 0.35,
}


@dataclass(frozen=True, slots=True)
class StreetWallPlacement:
    """One building packed into a straight street-wall run."""

    index: int
    asset_key: str
    position: Point2D
    yaw_deg: float
    scale: tuple[float, float, float]
    tangent_start_cm: float
    tangent_end_cm: float
    tangent_width_cm: float
    measured_tangent_width_cm: float
    normal_half_extent_cm: float


@dataclass(frozen=True, slots=True)
class StreetWallMetrics:
    """Continuity diagnostics for a packed wall."""

    length_cm: float
    actor_count: int
    covered_cm: float
    coverage: float
    maximum_gap_cm: float
    leading_gap_cm: float
    trailing_gap_cm: float


def _finite_point(name: str, point: Point2D) -> Point2D:
    result = (float(point[0]), float(point[1]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite coordinates: {point!r}")
    return result


def _normalize(name: str, vector: Point2D) -> Point2D:
    x, y = _finite_point(name, vector)
    length = math.hypot(x, y)
    if length <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return x / length, y / length


def _oriented_half_extents(
    asset_key: str, scale: float, yaw_deg: float
) -> Point2D:
    raw_x, raw_y, _raw_z = building_bbox(asset_key)
    hx = raw_x * scale / 2.0
    hy = raw_y * scale / 2.0
    radians = math.radians(yaw_deg)
    cosine, sine = abs(math.cos(radians)), abs(math.sin(radians))
    return cosine * hx + sine * hy, sine * hx + cosine * hy


def _projected_half_extent(axis: Point2D, half_extents: Point2D) -> float:
    return abs(axis[0]) * half_extents[0] + abs(axis[1]) * half_extents[1]


def _unit_tangent_width(asset_key: str, yaw_deg: float, tangent: Point2D) -> float:
    half = _oriented_half_extents(asset_key, 1.0, yaw_deg)
    return 2.0 * _projected_half_extent(tangent, half)


def _asset_sequence(asset_keys: Sequence[str], count: int) -> tuple[str, ...]:
    return tuple(asset_keys[index % len(asset_keys)] for index in range(count))


def _choose_count(
    length_cm: float,
    gap_cm: float,
    asset_keys: Sequence[str],
    yaw_deg: float,
    tangent: Point2D,
    preferred_scales: Mapping[str, float],
    min_scale: float,
    max_scale: float,
    facade_fill_ratio: float,
    target_count: int | None,
) -> tuple[str, ...]:
    unit_widths = {
        asset: _unit_tangent_width(asset, yaw_deg, tangent) * facade_fill_ratio
        for asset in asset_keys
    }
    if target_count is not None:
        sequence = _asset_sequence(asset_keys, target_count)
        seam_total = gap_cm * max(0, target_count - 1)
        width_budget = length_cm - seam_total
        minimum = sum(unit_widths[asset] * min_scale for asset in sequence)
        maximum = sum(unit_widths[asset] * max_scale for asset in sequence)
        if width_budget <= 0.0 or not (
            minimum - 1e-6 <= width_budget <= maximum + 1e-6
        ):
            raise ValueError(
                f"Wall length cannot fit target_count={target_count} within "
                "the requested scale and gap limits"
            )
        return sequence
    minimum_module = min(unit_widths.values()) * min_scale + gap_cm
    max_count = max(1, int(math.ceil((length_cm + gap_cm) / minimum_module)) + 2)
    feasible: list[tuple[float, int, tuple[str, ...]]] = []
    for count in range(1, max_count + 1):
        sequence = _asset_sequence(asset_keys, count)
        seam_total = gap_cm * max(0, count - 1)
        width_budget = length_cm - seam_total
        if width_budget <= 0.0:
            continue
        minimum = sum(unit_widths[asset] * min_scale for asset in sequence)
        maximum = sum(unit_widths[asset] * max_scale for asset in sequence)
        if minimum - 1e-6 <= width_budget <= maximum + 1e-6:
            preferred = sum(
                unit_widths[asset] * preferred_scales[asset]
                for asset in sequence
            )
            feasible.append((abs(preferred - width_budget), count, sequence))
    if not feasible:
        raise ValueError(
            "Wall length cannot be packed within the requested scale and gap limits"
        )
    return min(feasible, key=lambda item: (item[0], item[1]))[2]


def _fit_scales(
    sequence: Sequence[str],
    width_budget: float,
    unit_widths: Mapping[str, float],
    preferred_scales: Mapping[str, float],
    min_scale: float,
    max_scale: float,
) -> tuple[float, ...]:
    def scales(multiplier: float) -> tuple[float, ...]:
        return tuple(
            min(max_scale, max(min_scale, preferred_scales[asset] * multiplier))
            for asset in sequence
        )

    def width(multiplier: float) -> float:
        return sum(
            unit_widths[asset] * scale
            for asset, scale in zip(sequence, scales(multiplier))
        )

    low, high = 0.0, 1.0
    while width(high) < width_budget:
        high *= 2.0
        if high > 1e6:
            raise ValueError("Unable to fit wall scales")
    for _ in range(80):
        middle = (low + high) / 2.0
        if width(middle) < width_budget:
            low = middle
        else:
            high = middle
    return scales((low + high) / 2.0)


def plan_street_wall(
    start: Point2D,
    end: Point2D,
    *,
    outward: Point2D,
    asset_keys: Sequence[str] = DEFAULT_FACADE_ASSETS,
    preferred_scales: Mapping[str, float] = DEFAULT_PREFERRED_SCALES,
    gap_cm: float = 50.0,
    setback_cm: float = 200.0,
    min_scale: float = 0.18,
    max_scale: float = 0.55,
    height_factor: float = 1.5,
    facade_fill_ratio: float = 1.0,
    height_factors: Mapping[str, float] | None = None,
    target_count: int | None = None,
) -> tuple[StreetWallPlacement, ...]:
    """Pack measured buildings across one straight frontage.

    The requested gap is used between every adjacent conservative envelope.
    Scale is solved across the whole run so leading and trailing gaps are
    effectively zero instead of accumulating an arbitrary remainder.
    """

    start = _finite_point("start", start)
    end = _finite_point("end", end)
    tangent = _normalize("wall direction", (end[0] - start[0], end[1] - start[1]))
    outward = _normalize("outward", outward)
    if abs(tangent[0] * outward[0] + tangent[1] * outward[1]) > 1e-6:
        raise ValueError("outward must be perpendicular to the wall segment")
    length_cm = math.dist(start, end)
    values = {
        "gap_cm": gap_cm,
        "setback_cm": setback_cm,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "height_factor": height_factor,
        "facade_fill_ratio": facade_fill_ratio,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError(f"Wall parameters must be finite: {values!r}")
    if gap_cm < 0.0 or setback_cm < 0.0:
        raise ValueError("gap_cm and setback_cm must be non-negative")
    if min_scale <= 0.0 or max_scale < min_scale or height_factor <= 0.0:
        raise ValueError("Scale limits and height_factor must be positive and ordered")
    if not 0.0 < facade_fill_ratio <= 1.0:
        raise ValueError("facade_fill_ratio must be in (0, 1]")
    if target_count is not None and (
        isinstance(target_count, bool) or target_count <= 0
    ):
        raise ValueError("target_count must be a positive integer or None")
    assets = tuple(str(asset) for asset in asset_keys)
    if not assets:
        raise ValueError("asset_keys must not be empty")
    missing = [asset for asset in assets if asset not in preferred_scales]
    if missing:
        raise ValueError(f"Missing preferred scales for assets: {missing}")
    for asset in assets:
        building_bbox(asset)
    per_asset_heights = {
        asset: float(height_factors.get(asset, height_factor))
        if height_factors is not None
        else float(height_factor)
        for asset in assets
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in per_asset_heights.values()
    ):
        raise ValueError("height_factors values must be finite and positive")

    yaw_deg = math.degrees(math.atan2(outward[1], outward[0]))
    sequence = _choose_count(
        length_cm,
        gap_cm,
        assets,
        yaw_deg,
        tangent,
        preferred_scales,
        min_scale,
        max_scale,
        facade_fill_ratio,
        target_count,
    )
    unit_widths = {
        asset: _unit_tangent_width(asset, yaw_deg, tangent) * facade_fill_ratio
        for asset in assets
    }
    width_budget = length_cm - gap_cm * max(0, len(sequence) - 1)
    fitted_scales = _fit_scales(
        sequence,
        width_budget,
        unit_widths,
        preferred_scales,
        min_scale,
        max_scale,
    )

    widths = tuple(
        unit_widths[asset] * scale
        for asset, scale in zip(sequence, fitted_scales)
    )
    packed_length = sum(widths) + gap_cm * max(0, len(widths) - 1)
    leading = max(0.0, (length_cm - packed_length) / 2.0)
    cursor = leading
    placements: list[StreetWallPlacement] = []
    for index, (asset, scale_xy, width_cm) in enumerate(
        zip(sequence, fitted_scales, widths)
    ):
        tangent_start = cursor
        tangent_end = tangent_start + width_cm
        along = (tangent_start + tangent_end) / 2.0
        boundary = (
            start[0] + tangent[0] * along,
            start[1] + tangent[1] * along,
        )
        half = _oriented_half_extents(asset, scale_xy, yaw_deg)
        normal_half = _projected_half_extent(outward, half)
        position = (
            boundary[0] - outward[0] * (normal_half + setback_cm),
            boundary[1] - outward[1] * (normal_half + setback_cm),
        )
        placements.append(
            StreetWallPlacement(
                index=index,
                asset_key=asset,
                position=position,
                yaw_deg=yaw_deg,
                scale=(
                    scale_xy,
                    scale_xy,
                    scale_xy * per_asset_heights[asset],
                ),
                tangent_start_cm=tangent_start,
                tangent_end_cm=tangent_end,
                tangent_width_cm=width_cm,
                measured_tangent_width_cm=width_cm / facade_fill_ratio,
                normal_half_extent_cm=normal_half,
            )
        )
        cursor = tangent_end + gap_cm
    return tuple(placements)


def street_wall_metrics(
    placements: Sequence[StreetWallPlacement], length_cm: float
) -> StreetWallMetrics:
    """Return conservative coverage and gap diagnostics for a wall."""

    length = float(length_cm)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length_cm must be finite and positive")
    ordered = tuple(sorted(placements, key=lambda item: item.tangent_start_cm))
    if not ordered:
        return StreetWallMetrics(length, 0, 0.0, 0.0, length, length, length)
    leading = max(0.0, ordered[0].tangent_start_cm)
    trailing = max(0.0, length - ordered[-1].tangent_end_cm)
    seams = tuple(
        max(0.0, right.tangent_start_cm - left.tangent_end_cm)
        for left, right in zip(ordered, ordered[1:])
    )
    covered = sum(item.tangent_width_cm for item in ordered)
    return StreetWallMetrics(
        length_cm=length,
        actor_count=len(ordered),
        covered_cm=covered,
        coverage=covered / length,
        maximum_gap_cm=max((leading, trailing, *seams)),
        leading_gap_cm=leading,
        trailing_gap_cm=trailing,
    )


__all__ = [
    "DEFAULT_FACADE_ASSETS",
    "DEFAULT_PREFERRED_SCALES",
    "StreetWallMetrics",
    "StreetWallPlacement",
    "plan_street_wall",
    "street_wall_metrics",
]
