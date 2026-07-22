"""Scenario metadata for the Venue Meetup benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from benchmark.venue_meetup.layout import DistrictLayout

VenueType = Literal[
    "cafe",
    "pub",
    "restaurant",
    "hotel_lobby",
    "shop",
    "station_entrance",
    "public_square",
]
LandmarkType = Literal[
    "clock_tower",
    "hotel",
    "hospital",
    "museum",
    "venue_hall",
    "commercial_tower",
    "street_landmark",
]
EntranceStatus = Literal["open", "blocked", "stairs_only", "accessible"]
RequirementKey = Literal[
    "open",
    "reachable",
    "accessible",
    "shelter",
    "food_drink",
    "quiet",
    "uncrowded",
    "capacity",
    "near_transit",
]


@dataclass(frozen=True)
class Region:
    """Circular 2D scoring or trigger region in Unreal centimeters."""

    center: tuple[float, float]
    radius: float

    def contains(self, point: tuple[float, float]) -> bool:
        """Return whether a 2D point is inside this region."""

        dx = float(point[0]) - float(self.center[0])
        dy = float(point[1]) - float(self.center[1])
        return (dx * dx + dy * dy) ** 0.5 <= self.radius


@dataclass(frozen=True)
class Entrance:
    """Hidden entrance metadata plus the visible marker location."""

    entrance_id: str
    status: EntranceStatus
    position: tuple[float, float, float]
    yaw_deg: float
    visible_cues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VenueProperties:
    """Ground-truth venue attributes hidden from agents."""

    open: bool
    reachable: bool
    capacity: int
    accessible: bool
    shelter: bool
    food_drink: bool
    quiet_score: float
    crowding_score: float
    near_transit: bool = False


@dataclass(frozen=True)
class PropSpec:
    """One visible dressing prop spawned for a venue or landmark."""

    prop_id: str
    asset_key: str
    position: tuple[float, float, float]
    yaw_deg: float = 0.0
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    semantic: str = ""
    color_rgb: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class Venue:
    """A candidate meeting venue with hidden scoring state."""

    venue_id: str
    slot_id: str
    venue_type: VenueType
    asset_key: str
    asset_path: str
    position: tuple[float, float, float]
    yaw_deg: float
    region: Region
    mask_color_rgb: tuple[int, int, int]
    properties: VenueProperties
    entrances: list[Entrance]
    props: list[PropSpec] = field(default_factory=list)
    visual_summary: str = ""
    # Explicit visual/collision scale.  Authored city blocks use this to keep
    # source assets within their documented street clearances.
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Partition zone for the ``spatial`` info-partition mode. An agent may only
    # successfully INSPECT a venue when ``zone_id is None`` (public) or it matches
    # the agent's ``zone_id``. ``None`` => no partition (legacy behavior).
    zone_id: str | None = None


@dataclass(frozen=True)
class Landmark:
    """A visible localization cue."""

    landmark_id: str
    slot_id: str
    landmark_type: LandmarkType
    asset_key: str
    asset_path: str
    position: tuple[float, float, float]
    yaw_deg: float
    mask_color_rgb: tuple[int, int, int]
    visual_summary: str = ""
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class AgentSpec:
    """Agent spawn and private objective information."""

    agent_id: str
    spawn_slot: str
    position: tuple[float, float, float]
    yaw_deg: float
    private_constraint: str
    private_requirement_keys: list[RequirementKey]
    # Which partition zone this agent can INSPECT under the ``spatial`` mode.
    # ``None`` => the agent can inspect any zone (legacy behavior).
    zone_id: str | None = None
    # Optional DistrictLayout walk-graph node used as the agent's route origin.
    # This is a graph ``WalkNode.node_id`` (e.g. ``spawn_clock_tower``), not the
    # human-facing ``spawn_slot`` name. ``None`` preserves legacy scenarios that
    # only record spawn slots/positions.
    walk_node_id: str | None = None


@dataclass(frozen=True)
class Requirement:
    """One global or private venue requirement."""

    key: RequirementKey
    weight: float
    hard: bool = False
    description: str = ""


@dataclass(frozen=True)
class Scenario:
    """Fully materialized hidden scenario metadata."""

    scenario_id: str
    map_template_id: str
    seed: int
    venues: list[Venue]
    landmarks: list[Landmark]
    agents: list[AgentSpec]
    requirements: list[Requirement]
    soft_weights: dict[str, float]
    coarse_map_text: str
    coarse_map_path: str | None = None
    max_steps: int = 24
    layout: DistrictLayout | None = None

    def venue_by_id(self, venue_id: str) -> Venue:
        """Return a venue by id."""

        for venue in self.venues:
            if venue.venue_id == venue_id:
                return venue
        raise KeyError(f"Unknown venue_id: {venue_id}")

    def agent_ids(self) -> list[str]:
        """Return benchmark agent ids in scenario order."""

        return [agent.agent_id for agent in self.agents]

    def compact(self, *, include_hidden: bool = True) -> dict[str, Any]:
        """Return JSON-serializable metadata.

        When ``include_hidden`` is false, hidden venue properties are removed for
        prompt/log views intended for agents.
        """

        payload = asdict(self)
        # Keep legacy public JSON shape when no layout is attached.
        if payload.get("layout") is None:
            payload.pop("layout", None)
        if not include_hidden:
            for venue in payload["venues"]:
                venue.pop("properties", None)
                venue.pop("entrances", None)
                venue.pop("mask_color_rgb", None)
            payload.pop("requirements", None)
            payload.pop("soft_weights", None)
        return payload

    def save_json(self, path: Path) -> None:
        """Write scenario metadata to disk."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.compact(), indent=2) + "\n", encoding="utf-8")


def scenario_from_dict(payload: dict[str, Any]) -> Scenario:
    """Rehydrate a Scenario from JSON-compatible dictionaries."""

    from benchmark.venue_meetup.layout import DistrictLayout

    venues = []
    for venue in payload["venues"]:
        venue = dict(venue)
        venue["region"] = Region(**venue["region"])
        venue["properties"] = VenueProperties(**venue["properties"])
        venue["entrances"] = [Entrance(**entrance) for entrance in venue.get("entrances", [])]
        venue["props"] = [PropSpec(**prop) for prop in venue.get("props", [])]
        venues.append(Venue(**venue))

    landmarks = [Landmark(**landmark) for landmark in payload.get("landmarks", [])]
    agents = [AgentSpec(**agent) for agent in payload.get("agents", [])]
    requirements = [Requirement(**requirement) for requirement in payload.get("requirements", [])]
    layout_payload = payload.get("layout")
    layout = DistrictLayout.from_dict(layout_payload) if layout_payload is not None else None
    return Scenario(
        scenario_id=payload["scenario_id"],
        map_template_id=payload["map_template_id"],
        seed=int(payload["seed"]),
        venues=venues,
        landmarks=landmarks,
        agents=agents,
        requirements=requirements,
        soft_weights=dict(payload.get("soft_weights", {})),
        coarse_map_text=payload.get("coarse_map_text", ""),
        coarse_map_path=payload.get("coarse_map_path"),
        max_steps=int(payload.get("max_steps", 24)),
        layout=layout,
    )


def load_scenario(path: Path) -> Scenario:
    """Load a scenario from a JSON file."""

    return scenario_from_dict(json.loads(path.read_text(encoding="utf-8")))
