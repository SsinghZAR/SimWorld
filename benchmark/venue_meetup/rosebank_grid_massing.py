"""Bounded actor massing for scalable Rosebank-inspired grid layouts."""

from __future__ import annotations

from benchmark.venue_meetup.building_catalog import (asset_path,
                                                     building_description)
from benchmark.venue_meetup.rosebank_grid import (AlleyAxis, RosebankGridPlan,
                                                  RosebankVenueSite,
                                                  frontage_tangent)
from benchmark.venue_meetup.scenario import StaticBuilding

_CELL_OFFSET_CM = 1_800.0
_MASSING_CELLS = (
    (-_CELL_OFFSET_CM, _CELL_OFFSET_CM),
    (_CELL_OFFSET_CM, -_CELL_OFFSET_CM),
    (_CELL_OFFSET_CM, _CELL_OFFSET_CM),
    (-_CELL_OFFSET_CM, -_CELL_OFFSET_CM),
    (0.0, _CELL_OFFSET_CM),
    (0.0, -_CELL_OFFSET_CM),
    (_CELL_OFFSET_CM, 0.0),
    (-_CELL_OFFSET_CM, 0.0),
    (0.0, 0.0),
)
_SIDE_CELL = {
    "north": (0.0, _CELL_OFFSET_CM),
    "east": (_CELL_OFFSET_CM, 0.0),
    "south": (0.0, -_CELL_OFFSET_CM),
    "west": (-_CELL_OFFSET_CM, 0.0),
}
_TARGETS = {
    "residential": 4,
    "mixed_use": 6,
    "transit_core": 7,
    "civic": 5,
    "garden": 0,
}
_PALETTES = {
    "residential": (
        "BP_Building_05_C",
        "BP_Building_24_C",
        "BP_Building_06_C",
        "BP_Building_05_C",
    ),
    "mixed_use": (
        "BP_Building_05_C",
        "BP_Building_24_C",
        "BP_Building_25_C",
        "BP_Building_44_C",
        "BP_Building_06_C",
        "BP_Building_95_C",
    ),
    "transit_core": (
        "BP_Building_44_C",
        "BP_Building_25_C",
        "BP_Building_20_C",
        "BP_Building_101_C",
        "BP_Building_06_C",
        "BP_Building_24_C",
        "BP_Building_95_C",
    ),
    "civic": (
        "BP_Building_95_C",
        "BP_Building_44_C",
        "BP_Building_24_C",
        "BP_Building_20_C",
        "BP_Building_06_C",
    ),
}
_ASSET_SCALES = {
    "BP_Building_05_C": (0.54, 0.54, 0.62),
    "BP_Building_06_C": (0.50, 0.50, 0.62),
    "BP_Building_20_C": (0.38, 0.38, 0.62),
    "BP_Building_24_C": (0.42, 0.42, 0.58),
    "BP_Building_25_C": (0.38, 0.38, 0.48),
    "BP_Building_44_C": (0.42, 0.42, 0.58),
    "BP_Building_95_C": (0.17, 0.17, 0.28),
    "BP_Building_101_C": (0.25, 0.25, 0.55),
}
_ZONE_HEIGHT_FACTOR = {
    "residential": 0.82,
    "mixed_use": 1.0,
    "transit_core": 1.32,
    "civic": 1.08,
}
_TREE_ASSETS = ("BP_Tree1_C", "BP_Tree2_C")
_TREE_SCALE = (1.38, 1.38, 1.38)


def _available_cells(
    *,
    alley_axes: frozenset[AlleyAxis],
    reserved_cell: tuple[float, float] | None,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        cell
        for cell in _MASSING_CELLS
        if cell != reserved_cell
        and not ("horizontal" in alley_axes and cell[1] == 0.0)
        and not ("vertical" in alley_axes and cell[0] == 0.0)
    )


def _venue_cell(site: RosebankVenueSite | None) -> tuple[float, float] | None:
    """Return the parcel cell reserved for an interactive frontage."""

    if site is None:
        return None
    side_cell = _SIDE_CELL[site.side]
    tangent = frontage_tangent(site.side)
    target = (
        side_cell[0] + tangent[0] * site.frontage_offset_cm,
        side_cell[1] + tangent[1] * site.frontage_offset_cm,
    )
    return min(
        _MASSING_CELLS,
        key=lambda cell: (cell[0] - target[0]) ** 2 + (cell[1] - target[1]) ** 2,
    )


def _tree(
    block_id: str,
    index: int,
    position: tuple[float, float],
) -> StaticBuilding:
    asset_key = _TREE_ASSETS[index % len(_TREE_ASSETS)]
    return StaticBuilding(
        building_id=f"rosebank_tree_{block_id.lower()}_{index:02d}",
        asset_key=asset_key,
        asset_path=asset_path(asset_key),
        position=(*position, 0.0),
        yaw_deg=float((index * 73) % 360),
        scale=_TREE_SCALE,
        collision=False,
        visual_summary="Leafy streetscape tree used as a district navigation cue.",
    )


def plan_rosebank_massing(
    plan: RosebankGridPlan,
) -> tuple[StaticBuilding, ...]:
    """Return dense but bounded building cells plus non-blocking green cues."""

    venue_site_by_block = {
        site.block_id: site for site in plan.venue_sites
    }
    records: list[StaticBuilding] = []
    for block_index, block in enumerate(plan.blocks):
        if block.zone == "garden":
            for tree_index, local in enumerate(
                ((-1_500.0, -1_500.0), (1_500.0, -1_500.0), (0.0, 0.0),
                 (-1_500.0, 1_500.0), (1_500.0, 1_500.0))
            ):
                records.append(
                    _tree(
                        block.block_id,
                        tree_index,
                        (block.center[0] + local[0], block.center[1] + local[1]),
                    )
                )
            continue

        if block.landmark_role is not None:
            for tree_index, local in enumerate(
                ((-2_250.0, 2_250.0), (2_250.0, -2_250.0))
            ):
                records.append(
                    _tree(
                        block.block_id,
                        tree_index,
                        (block.center[0] + local[0], block.center[1] + local[1]),
                    )
                )
            continue

        palette = _PALETTES[block.zone]
        target = _TARGETS[block.zone]
        cells = _available_cells(
            alley_axes=block.alley_axes,
            reserved_cell=_venue_cell(venue_site_by_block.get(block.block_id)),
        )
        for local_index, local in enumerate(cells[:target]):
            asset_key = palette[(block_index + local_index) % len(palette)]
            base_scale = _ASSET_SCALES[asset_key]
            scale = (
                base_scale[0],
                base_scale[1],
                base_scale[2] * _ZONE_HEIGHT_FACTOR[block.zone],
            )
            position = (
                block.center[0] + local[0],
                block.center[1] + local[1],
            )
            records.append(
                StaticBuilding(
                    building_id=(
                        f"rosebank_{block.block_id.lower()}_{local_index:02d}"
                    ),
                    asset_key=asset_key,
                    asset_path=asset_path(asset_key),
                    position=(*position, 0.0),
                    yaw_deg=float(
                        90 if (block.row + block.column + local_index) % 2 else 0
                    ),
                    scale=scale,
                    collision=True,
                    visual_summary=building_description(asset_key),
                )
            )

        if (
            block.zone == "residential"
            and not block.alley_axes
        ) or (
            block.zone == "mixed_use"
            and not block.alley_axes
            and (block.row + block.column) % 4 == 0
        ):
            records.append(_tree(block.block_id, 0, block.center))
    return tuple(records)


__all__ = ["plan_rosebank_massing"]
