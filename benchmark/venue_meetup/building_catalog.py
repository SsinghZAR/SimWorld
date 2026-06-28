"""Base SimWorld asset choices for Venue Meetup scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
UE_ASSETS_PATH = REPO_ROOT / "data" / "ue_assets.json"
DESCRIPTION_MAP_PATH = REPO_ROOT / "data" / "description_map.json"


@dataclass(frozen=True)
class BuildingChoice:
    """One known base building useful for venue meetup."""

    asset_key: str
    role_tags: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class PropChoice:
    """One known base prop useful for dressing hidden venue state."""

    asset_key: str
    role_tags: tuple[str, ...]
    rationale: str


BUILDING_CHOICES: tuple[BuildingChoice, ...] = (
    BuildingChoice("BP_Building_05_C", ("cafe", "restaurant", "shop"), "Low shop building with red awnings."),
    BuildingChoice("BP_Building_06_C", ("cafe", "restaurant"), "Storefronts and green awning, good cafe frontage."),
    BuildingChoice("BP_Building_24_C", ("shop", "convenience"), "Convenience-store building with red awnings."),
    BuildingChoice("BP_Building_25_C", ("shop", "convenience"), "Mini-market signage and blue-tiled lower facade."),
    BuildingChoice("BP_Building_44_C", ("hotel_lobby", "landmark"), "High residential/hotel with colorful signs."),
    BuildingChoice("BP_Building_95_C", ("hotel_lobby", "landmark"), "Brown brick hotel building with arched base."),
    BuildingChoice("BP_Building_20_C", ("clock_tower", "landmark"), "Large light-gray office building with clock tower."),
    BuildingChoice("BP_Building_87_C", ("hospital", "landmark", "accessible"), "Hospital with central sign and ramps."),
    BuildingChoice("BP_Building_99_C", ("museum", "landmark"), "Low museum with grand staircase."),
    BuildingChoice("BP_Building_101_C", ("clock_tower", "landmark"), "Office building with central clocktower."),
    BuildingChoice("BP_Building_123_C", ("public_square", "venue_hall", "landmark"), "Distinctive venue hall with glass domed roof."),
)

PROP_CHOICES: tuple[PropChoice, ...] = (
    PropChoice("RoadBlocker_C", ("blocked", "closed"), "Road blocker used to mark blocked entrances."),
    PropChoice("RoadCone_C", ("blocked", "closed"), "Cone used to mark blocked/unsafe access."),
    PropChoice("BP_Table_C", ("food_drink", "seating"), "Outdoor table indicating seating or venue service."),
    PropChoice("BP_Table2_C", ("food_drink", "seating"), "Alternate table for venue dressing."),
    PropChoice("BP_Can_C", ("food_drink",), "Can prop suggesting drinks are available."),
    PropChoice("BP_Soda1_C", ("food_drink",), "Soda prop suggesting drinks are available."),
    PropChoice("BP_Trash_bin_a_C", ("crowded", "street_furniture"), "Street furniture / clutter cue."),
    PropChoice("BP_Hydrant_C", ("street_landmark",), "Small street landmark / localization cue."),
)

MASK_COLORS: dict[str, tuple[int, int, int]] = {
    "venue_0": (255, 0, 0),
    "venue_1": (0, 255, 0),
    "venue_2": (0, 0, 255),
    "venue_3": (255, 255, 0),
    "venue_4": (255, 0, 255),
    "venue_5": (0, 255, 255),
    "landmark_0": (180, 80, 255),
    "landmark_1": (255, 128, 0),
    "landmark_2": (128, 255, 128),
}


@lru_cache(maxsize=1)
def ue_assets() -> dict[str, dict[str, str]]:
    """Load the base UE asset map."""

    return json.loads(UE_ASSETS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def description_map() -> dict[str, str]:
    """Load human-readable building descriptions."""

    return json.loads(DESCRIPTION_MAP_PATH.read_text(encoding="utf-8"))


def asset_path(asset_key: str) -> str:
    """Return the Unreal blueprint path for an asset key."""

    assets = ue_assets()
    if asset_key not in assets:
        raise KeyError(f"Asset not found in {UE_ASSETS_PATH}: {asset_key}")
    return assets[asset_key]["asset_path"]


def building_description(asset_key: str) -> str:
    """Return the human-readable description for a building asset."""

    return description_map().get(asset_key, "")


def buildings_for_tag(tag: str) -> list[BuildingChoice]:
    """Return known building choices for a venue or landmark tag."""

    return [choice for choice in BUILDING_CHOICES if tag in choice.role_tags]


def props_for_tag(tag: str) -> list[PropChoice]:
    """Return known prop choices for a hidden-state or visual-cue tag."""

    return [choice for choice in PROP_CHOICES if tag in choice.role_tags]


def assert_catalog_assets_exist(extra_asset_keys: Iterable[str] = ()) -> None:
    """Fail early if a selected base asset is missing from the local asset map."""

    keys = [choice.asset_key for choice in BUILDING_CHOICES]
    keys.extend(choice.asset_key for choice in PROP_CHOICES)
    keys.extend(extra_asset_keys)
    for key in sorted(set(keys)):
        asset_path(key)
