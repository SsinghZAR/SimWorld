"""Authored medium station-quarter district for Venue Meetup.

``station_quarter_medium_v1`` is a deterministic 8-venue, 4-block commercial
district (not a central-square ring). Venue positions and meeting regions are
taken from named :class:`~benchmark.venue_meetup.layout.Frontage` entries on one
:class:`~benchmark.venue_meetup.layout.DistrictLayout`. Coordinates are Unreal
centimetres (~400 m east-west footprint).
"""

from __future__ import annotations

import math

from benchmark.venue_meetup.building_catalog import MASK_COLORS, asset_path, building_description
from benchmark.venue_meetup.layout import (
    Block,
    DistrictLayout,
    Frontage,
    Intersection,
    MeetingRegion,
    StreetSegment,
    WalkEdge,
    WalkNode,
)
from benchmark.venue_meetup.scenario import (
    AgentSpec,
    Entrance,
    Landmark,
    PropSpec,
    Region,
    Requirement,
    Scenario,
    Venue,
    VenueProperties,
)

MAP_TEMPLATE_ID = "station_quarter_medium_v1"
LAYOUT_ID = "station_quarter_medium_v1"

# District extents (~400 m E-W, ~320 m N-S between Market Street and the alley).
WEST_X = -20000.0
EAST_X = 20000.0
MID_X = 0.0
MARKET_Y = 16000.0
CROSS_Y = 0.0
ALLEY_Y = -16000.0

STREET_WIDTH = 1400.0
ALLEY_WIDTH = 700.0
SIDEWALK_WIDTH = 300.0
MEET_RADIUS = 900.0
AGENT_Z = 150.0
CITY_BUILDING_SCALE = 0.25
BLOCK_CHAMFER = 1500.0

# Sidewalk centerlines offset from street centre toward the adjacent block.
_MARKET_SW_Y = MARKET_Y - STREET_WIDTH / 2.0 - SIDEWALK_WIDTH / 2.0
_CROSS_N_SW_Y = CROSS_Y + STREET_WIDTH / 2.0 + SIDEWALK_WIDTH / 2.0
_CROSS_S_SW_Y = CROSS_Y - STREET_WIDTH / 2.0 - SIDEWALK_WIDTH / 2.0
_ALLEY_SW_Y = ALLEY_Y + ALLEY_WIDTH / 2.0 + SIDEWALK_WIDTH / 2.0
_WEST_SW_X = WEST_X + STREET_WIDTH / 2.0 + SIDEWALK_WIDTH / 2.0
_EAST_SW_X = EAST_X - STREET_WIDTH / 2.0 - SIDEWALK_WIDTH / 2.0

# Catalog mask colours; last two venues reuse distinct extras past venue_5.
_VENUE_MASKS: tuple[tuple[int, int, int], ...] = (
    MASK_COLORS["venue_0"],
    MASK_COLORS["venue_1"],
    MASK_COLORS["venue_2"],
    MASK_COLORS["venue_3"],
    MASK_COLORS["venue_4"],
    MASK_COLORS["venue_5"],
    (255, 160, 64),
    (64, 160, 255),
)
_LANDMARK_MASKS: tuple[tuple[int, int, int], ...] = (
    MASK_COLORS["landmark_0"],
    MASK_COLORS["landmark_1"],
    MASK_COLORS["landmark_2"],
    (220, 180, 40),
    (40, 180, 220),
)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_length(points: tuple[tuple[float, float], ...]) -> float:
    return sum(_dist(left, right) for left, right in zip(points, points[1:]))


def _chamfered_rect(
    left: float,
    bottom: float,
    right: float,
    top: float,
    *,
    chamfer: float = BLOCK_CHAMFER,
) -> tuple[tuple[float, float], ...]:
    """Return an Eixample-like block with eight clipped street corners."""

    if right - left <= 2.0 * chamfer or top - bottom <= 2.0 * chamfer:
        raise ValueError("Chamfer must leave a positive block edge")
    return (
        (left + chamfer, bottom),
        (right - chamfer, bottom),
        (right, bottom + chamfer),
        (right, top - chamfer),
        (right - chamfer, top),
        (left + chamfer, top),
        (left, top - chamfer),
        (left, bottom + chamfer),
    )


def _edge(
    nodes: dict[str, WalkNode],
    start_id: str,
    end_id: str,
    *,
    route_kind: str = "sidewalk",
) -> WalkEdge:
    start = nodes[start_id].position
    end = nodes[end_id].position
    # Every authored edge records the path actually driven by walk navigation.
    # The route lattice is axis-aligned or safely outside the block interiors;
    # a midpoint keeps that physical segment explicit for diagnostics and UE
    # preflight rather than silently treating it as an abstract graph edge.
    waypoints = (((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0),)
    return WalkEdge(
        start_node_id=start_id,
        end_node_id=end_id,
        length_cm=_polyline_length((start, *waypoints, end)),
        route_kind=route_kind,  # type: ignore[arg-type]
        waypoints=waypoints,
    )


def build_district_layout() -> DistrictLayout:
    """Return the fixed station-quarter district layout."""

    # Two N-S through streets (west/east avenues), one E-W cross street, one alley.
    # Market Street is the northern through corridor closing the station forecourt.
    streets = (
        StreetSegment(
            street_id="market_street",
            start=(WEST_X, MARKET_Y),
            end=(EAST_X, MARKET_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="west_avenue",
            start=(WEST_X, ALLEY_Y),
            end=(WEST_X, MARKET_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="east_avenue",
            start=(EAST_X, ALLEY_Y),
            end=(EAST_X, MARKET_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="cross_street",
            start=(WEST_X, CROSS_Y),
            end=(EAST_X, CROSS_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="service_alley",
            start=(WEST_X, ALLEY_Y),
            end=(EAST_X, ALLEY_Y),
            width_cm=ALLEY_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
    )

    intersections = (
        Intersection("ix_market_west", (WEST_X, MARKET_Y)),
        Intersection("ix_market_east", (EAST_X, MARKET_Y)),
        Intersection("ix_clock_tower", (WEST_X, CROSS_Y), landmark_id="landmark_clock_tower"),
        Intersection("ix_cross_mid", (MID_X, CROSS_Y)),
        Intersection(
            "ix_station_forecourt",
            (EAST_X, CROSS_Y),
            landmark_id="landmark_station_forecourt",
        ),
        Intersection("ix_hotel_corner", (WEST_X, ALLEY_Y), landmark_id="landmark_hotel_corner"),
        Intersection("ix_bus_stop", (EAST_X, ALLEY_Y), landmark_id="landmark_bus_stop"),
        Intersection("ix_market_mid", (MID_X, MARKET_Y), landmark_id="landmark_market_canopy"),
    )

    # Building pivots inside blocks; meeting regions on public sidewalks.
    frontages = (
        Frontage(
            frontage_id="front_nw_market_cafe",
            block_id="block_nw",
            position=(-11000.0, 9000.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(-11000.0, 11800.0, 0.0),
            meeting_region=MeetingRegion(center=(-11000.0, _MARKET_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="nw_market_cafe",
            approach_node_id="approach_nw_cafe",
            access_path=((-11000.0, _MARKET_SW_Y),),
        ),
        Frontage(
            frontage_id="front_nw_cross_bistro",
            block_id="block_nw",
            position=(-9000.0, 4500.0, 0.0),
            yaw_deg=-90.0,
            entrance_point=(-9000.0, 2200.0, 0.0),
            meeting_region=MeetingRegion(center=(-9000.0, _CROSS_N_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="nw_cross_bistro",
            approach_node_id="approach_nw_bistro",
            access_path=((-9000.0, _CROSS_N_SW_Y),),
        ),
        Frontage(
            frontage_id="front_ne_market_shop",
            block_id="block_ne",
            position=(11000.0, 9000.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(11000.0, 11800.0, 0.0),
            meeting_region=MeetingRegion(center=(11000.0, _MARKET_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="ne_market_shop",
            approach_node_id="approach_ne_shop",
            access_path=((11000.0, _MARKET_SW_Y),),
        ),
        Frontage(
            frontage_id="front_ne_cross_deli",
            block_id="block_ne",
            position=(9000.0, 4500.0, 0.0),
            yaw_deg=-90.0,
            entrance_point=(9000.0, 2200.0, 0.0),
            meeting_region=MeetingRegion(center=(9000.0, _CROSS_N_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="ne_cross_deli",
            approach_node_id="approach_ne_deli",
            access_path=((9000.0, _CROSS_N_SW_Y),),
        ),
        Frontage(
            frontage_id="front_sw_cross_pub",
            block_id="block_sw",
            position=(-9000.0, -4500.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(-9000.0, -2200.0, 0.0),
            meeting_region=MeetingRegion(center=(-9000.0, _CROSS_S_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="sw_cross_pub",
            approach_node_id="approach_sw_pub",
            access_path=((-9000.0, _CROSS_S_SW_Y),),
        ),
        Frontage(
            frontage_id="front_sw_alley_lobby",
            block_id="block_sw",
            position=(-11000.0, -9000.0, 0.0),
            yaw_deg=-90.0,
            entrance_point=(-11000.0, -11800.0, 0.0),
            meeting_region=MeetingRegion(center=(-11000.0, _ALLEY_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="sw_alley_lobby",
            approach_node_id="approach_sw_lobby",
            access_path=((-11000.0, _ALLEY_SW_Y),),
        ),
        Frontage(
            frontage_id="front_se_cross_hall",
            block_id="block_se",
            position=(9000.0, -4500.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(9000.0, -2200.0, 0.0),
            meeting_region=MeetingRegion(center=(9000.0, _CROSS_S_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="se_cross_hall",
            approach_node_id="approach_se_hall",
            access_path=((9000.0, _CROSS_S_SW_Y),),
        ),
        Frontage(
            frontage_id="front_se_alley_market",
            block_id="block_se",
            position=(11000.0, -9000.0, 0.0),
            yaw_deg=-90.0,
            entrance_point=(11000.0, -11800.0, 0.0),
            meeting_region=MeetingRegion(center=(11000.0, _ALLEY_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="se_alley_market",
            approach_node_id="approach_se_market",
            access_path=((11000.0, _ALLEY_SW_Y),),
        ),
    )

    blocks = (
        Block(
            block_id="block_nw",
            footprint=_chamfered_rect(-19000.0, 1200.0, -1200.0, 14800.0),
            frontage_ids=("front_nw_market_cafe", "front_nw_cross_bistro"),
            visual_style="station_west",
            shell_target=22,
        ),
        Block(
            block_id="block_ne",
            footprint=_chamfered_rect(1200.0, 1200.0, 19000.0, 14800.0),
            frontage_ids=("front_ne_market_shop", "front_ne_cross_deli"),
            visual_style="station_east",
            shell_target=22,
        ),
        Block(
            block_id="block_sw",
            footprint=_chamfered_rect(-19000.0, -14800.0, -1200.0, -1200.0),
            frontage_ids=("front_sw_cross_pub", "front_sw_alley_lobby"),
            visual_style="station_west",
            shell_target=22,
        ),
        Block(
            block_id="block_se",
            footprint=_chamfered_rect(1200.0, -14800.0, 19000.0, -1200.0),
            frontage_ids=("front_se_cross_hall", "front_se_alley_market"),
            visual_style="station_east",
            shell_target=22,
        ),
    )

    # Public route spines sit street-facing of the meeting regions.  A venue
    # approach is a distinct one-edge leaf; only its target access path enters
    # the meeting region, so routes to other venues can never cut through it.
    walk_nodes = (
        # Interior decision-point spawns face along Cross Street. Four built
        # block faces now surround each initial camera instead of an empty-map
        # boundary occupying half the field of view.
        WalkNode("spawn_clock_tower", (-14000.0, _CROSS_N_SW_Y), "spawn"),
        WalkNode("spawn_station_forecourt", (14000.0, _CROSS_N_SW_Y), "spawn"),
        # Market Street south sidewalk (west -> east)
        WalkNode("swk_market_west", (_WEST_SW_X, _MARKET_SW_Y), "intersection"),
        WalkNode("swk_nw_cafe", (-11000.0, 15750.0), "sidewalk"),
        WalkNode("swk_market_mid", (MID_X, _MARKET_SW_Y), "crossing"),
        WalkNode("swk_ne_shop", (11000.0, 15750.0), "sidewalk"),
        WalkNode("swk_market_east", (_EAST_SW_X, _MARKET_SW_Y), "intersection"),
        # Cross Street north sidewalk
        WalkNode("swk_cross_n_west", (_WEST_SW_X, _CROSS_N_SW_Y), "intersection"),
        WalkNode("swk_nw_bistro", (-9000.0, 250.0), "sidewalk"),
        WalkNode("swk_cross_n_mid", (MID_X, _CROSS_N_SW_Y), "crossing"),
        WalkNode("swk_ne_deli", (9000.0, 250.0), "sidewalk"),
        WalkNode("swk_cross_n_east", (_EAST_SW_X, _CROSS_N_SW_Y), "intersection"),
        # Cross Street south sidewalk
        WalkNode("swk_cross_s_west", (_WEST_SW_X, _CROSS_S_SW_Y), "intersection"),
        WalkNode("swk_sw_pub", (-9000.0, -250.0), "sidewalk"),
        WalkNode("swk_cross_s_mid", (MID_X, _CROSS_S_SW_Y), "crossing"),
        WalkNode("swk_se_hall", (9000.0, -250.0), "sidewalk"),
        WalkNode("swk_cross_s_east", (_EAST_SW_X, _CROSS_S_SW_Y), "intersection"),
        # Service alley north sidewalk
        WalkNode("swk_alley_west", (_WEST_SW_X, _ALLEY_SW_Y), "intersection"),
        WalkNode("swk_sw_lobby", (-11000.0, -16100.0), "sidewalk"),
        WalkNode("swk_alley_mid", (MID_X, _ALLEY_SW_Y), "crossing"),
        WalkNode("swk_se_market", (11000.0, -16100.0), "sidewalk"),
        # Target-only public approach leaves; none is a through-route vertex.
        WalkNode("approach_nw_cafe", (-11000.0, 15450.0), "sidewalk"),
        WalkNode("approach_nw_bistro", (-9000.0, 550.0), "sidewalk"),
        WalkNode("approach_ne_shop", (11000.0, 15450.0), "sidewalk"),
        WalkNode("approach_ne_deli", (9000.0, 550.0), "sidewalk"),
        WalkNode("approach_sw_pub", (-9000.0, -550.0), "sidewalk"),
        WalkNode("approach_sw_lobby", (-11000.0, -15800.0), "sidewalk"),
        WalkNode("approach_se_hall", (9000.0, -550.0), "sidewalk"),
        WalkNode("approach_se_market", (11000.0, -15800.0), "sidewalk"),
        WalkNode("swk_alley_east", (_EAST_SW_X, _ALLEY_SW_Y), "intersection"),
    )
    nodes = {node.node_id: node for node in walk_nodes}

    edge_specs: tuple[tuple[str, str, str], ...] = (
        # Spawns onto the interior Cross Street axis.
        ("spawn_clock_tower", "swk_nw_bistro", "sidewalk"),
        ("spawn_station_forecourt", "swk_ne_deli", "sidewalk"),
        # Market Street sidewalk
        ("swk_market_west", "swk_market_mid", "sidewalk"),
        ("swk_market_mid", "swk_market_east", "sidewalk"),
        # West avenue sidewalks
        ("swk_market_west", "swk_cross_n_west", "sidewalk"),
        ("swk_cross_n_west", "swk_cross_s_west", "crossing"),
        ("swk_cross_s_west", "swk_alley_west", "sidewalk"),
        # East avenue sidewalks
        ("swk_market_east", "swk_cross_n_east", "sidewalk"),
        ("swk_cross_n_east", "swk_cross_s_east", "crossing"),
        ("swk_cross_s_east", "swk_alley_east", "sidewalk"),
        # Cross Street north sidewalk
        ("swk_cross_n_west", "swk_cross_n_mid", "sidewalk"),
        ("swk_cross_n_mid", "swk_cross_n_east", "sidewalk"),
        # Cross Street south sidewalk
        ("swk_cross_s_west", "swk_cross_s_mid", "sidewalk"),
        ("swk_cross_s_mid", "swk_cross_s_east", "sidewalk"),
        # Crossings across Cross Street
        ("swk_cross_n_mid", "swk_cross_s_mid", "crossing"),
        ("swk_nw_bistro", "swk_sw_pub", "crossing"),
        ("swk_ne_deli", "swk_se_hall", "crossing"),
        # Service alley
        ("swk_alley_west", "swk_alley_mid", "alley"),
        ("swk_alley_mid", "swk_alley_east", "alley"),
        # One public edge into each target-specific approach leaf.
        ("swk_market_west", "swk_nw_cafe", "sidewalk"),
        ("swk_nw_cafe", "approach_nw_cafe", "sidewalk"),
        ("swk_cross_n_west", "swk_nw_bistro", "sidewalk"),
        ("swk_nw_bistro", "approach_nw_bistro", "sidewalk"),
        ("swk_market_east", "swk_ne_shop", "sidewalk"),
        ("swk_ne_shop", "approach_ne_shop", "sidewalk"),
        ("swk_cross_n_east", "swk_ne_deli", "sidewalk"),
        ("swk_ne_deli", "approach_ne_deli", "sidewalk"),
        ("swk_cross_s_west", "swk_sw_pub", "sidewalk"),
        ("swk_sw_pub", "approach_sw_pub", "sidewalk"),
        ("swk_alley_west", "swk_sw_lobby", "alley"),
        ("swk_sw_lobby", "approach_sw_lobby", "alley"),
        ("swk_cross_s_east", "swk_se_hall", "sidewalk"),
        ("swk_se_hall", "approach_se_hall", "sidewalk"),
        ("swk_alley_east", "swk_se_market", "alley"),
        ("swk_se_market", "approach_se_market", "alley"),
    )
    walk_edges = tuple(_edge(nodes, start, end, route_kind=kind) for start, end, kind in edge_specs)

    return DistrictLayout(
        layout_id=LAYOUT_ID,
        streets=streets,
        intersections=intersections,
        blocks=blocks,
        frontages=frontages,
        walk_nodes=walk_nodes,
        walk_edges=walk_edges,
        schema_version=3,
    )


def _region_from_frontage(frontage: Frontage) -> Region:
    meeting = frontage.meeting_region
    return Region(center=meeting.center, radius=meeting.radius)


def _venue_prop(
    venue_id: str,
    index: int,
    asset_key: str,
    position: tuple[float, float, float],
    semantic: str,
) -> PropSpec:
    return PropSpec(
        prop_id=f"{venue_id}_prop_{index}",
        asset_key=asset_key,
        position=position,
        semantic=semantic,
    )


def build_fixed_scenario(seed: int = 11) -> Scenario:
    """Return the deterministic station-quarter medium scenario."""

    layout = build_district_layout()
    frontage_by_slot = {
        frontage.venue_slot_id: frontage
        for frontage in layout.frontages
        if frontage.venue_slot_id is not None
    }

    venue_defs: list[dict[str, object]] = [
        {
            "venue_id": "venue_nw_market_cafe",
            "slot_id": "nw_market_cafe",
            "venue_type": "cafe",
            "asset_key": "BP_Building_05_C",
            "zone_id": "zone_west",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=6,
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=0.72,
                crowding_score=0.25,
                near_transit=False,
            ),
            "entrance_status": "accessible",
            "cues": ["red awning", "outdoor table", "drink prop"],
            "props": [
                ("BP_Table_C", (-10800.0, 12000.0, 0.0), "outdoor seating"),
                ("BP_Soda1_C", (-11200.0, 12100.0, 0.0), "food/drink cue"),
            ],
        },
        {
            "venue_id": "venue_nw_cross_bistro",
            "slot_id": "nw_cross_bistro",
            "venue_type": "restaurant",
            "asset_key": "BP_Building_06_C",
            "zone_id": "zone_west",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=8,
                accessible=True,
                shelter=True,
                food_drink=False,
                quiet_score=0.55,
                crowding_score=0.4,
                near_transit=False,
            ),
            "entrance_status": "accessible",
            "cues": ["green awning", "cross-street storefront"],
            "props": [],
        },
        {
            "venue_id": "venue_ne_market_shop",
            "slot_id": "ne_market_shop",
            "venue_type": "shop",
            "asset_key": "BP_Building_25_C",
            "zone_id": "zone_east",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=3,
                accessible=False,
                shelter=True,
                food_drink=True,
                quiet_score=0.45,
                crowding_score=0.72,
                near_transit=True,
            ),
            "entrance_status": "stairs_only",
            "cues": ["busy storefront", "narrow/raised entrance"],
            "props": [
                ("BP_Can_C", (11200.0, 12000.0, 0.0), "food/drink cue"),
                ("BP_Trash_bin_a_C", (10800.0, 11900.0, 0.0), "busy street cue"),
            ],
        },
        {
            "venue_id": "venue_ne_cross_deli",
            "slot_id": "ne_cross_deli",
            "venue_type": "shop",
            "asset_key": "BP_Building_24_C",
            "zone_id": "zone_east",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=5,
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=0.5,
                crowding_score=0.55,
                near_transit=True,
            ),
            "entrance_status": "accessible",
            "cues": ["convenience awning", "cross-street deli"],
            "props": [],
        },
        {
            "venue_id": "venue_sw_cross_pub",
            "slot_id": "sw_cross_pub",
            "venue_type": "pub",
            "asset_key": "BP_Building_44_C",
            "zone_id": "zone_west",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=10,
                accessible=False,
                shelter=True,
                food_drink=True,
                quiet_score=0.35,
                crowding_score=0.75,
                near_transit=False,
            ),
            "entrance_status": "stairs_only",
            "cues": ["colorful signs", "noisy pub front"],
            "props": [("BP_Trash_bin_a_C", (-8800.0, -2000.0, 0.0), "busy street cue")],
        },
        {
            "venue_id": "venue_sw_alley_lobby",
            "slot_id": "sw_alley_lobby",
            "venue_type": "hotel_lobby",
            "asset_key": "BP_Building_95_C",
            "zone_id": "zone_west",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=8,
                accessible=True,
                shelter=True,
                food_drink=False,
                quiet_score=0.82,
                crowding_score=0.2,
                near_transit=False,
            ),
            "entrance_status": "accessible",
            "cues": ["open arched lobby", "quiet facade"],
            "props": [],
        },
        {
            "venue_id": "venue_se_cross_hall",
            "slot_id": "se_cross_hall",
            "venue_type": "public_square",
            "asset_key": "BP_Building_99_C",
            "zone_id": "zone_east",
            "properties": VenueProperties(
                open=False,
                reachable=True,
                capacity=12,
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=0.6,
                crowding_score=0.3,
                near_transit=False,
            ),
            "entrance_status": "blocked",
            "cues": ["road blockers across entrance", "closed-looking frontage"],
            "props": [
                ("RoadBlocker_C", (8800.0, -2000.0, 0.0), "blocked/closed entrance"),
                ("RoadCone_C", (9200.0, -2000.0, 0.0), "blocked/closed entrance"),
            ],
        },
        {
            "venue_id": "venue_se_alley_market",
            "slot_id": "se_alley_market",
            "venue_type": "shop",
            "asset_key": "BP_Building_123_C",
            "zone_id": "zone_east",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=9,
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=0.65,
                crowding_score=0.45,
                near_transit=True,
            ),
            "entrance_status": "accessible",
            "cues": ["glass-domed hall", "alley market entrance"],
            "props": [],
        },
    ]

    venues: list[Venue] = []
    for index, spec in enumerate(venue_defs):
        slot_id = str(spec["slot_id"])
        frontage = frontage_by_slot[slot_id]
        asset_key = str(spec["asset_key"])
        venue_id = str(spec["venue_id"])
        props = [
            _venue_prop(venue_id, prop_index, asset, position, semantic)
            for prop_index, (asset, position, semantic) in enumerate(spec["props"])  # type: ignore[arg-type]
        ]
        venues.append(
            Venue(
                venue_id=venue_id,
                slot_id=slot_id,
                venue_type=spec["venue_type"],  # type: ignore[arg-type]
                asset_key=asset_key,
                asset_path=asset_path(asset_key),
                position=frontage.position,
                yaw_deg=frontage.yaw_deg,
                region=_region_from_frontage(frontage),
                mask_color_rgb=_VENUE_MASKS[index],
                properties=spec["properties"],  # type: ignore[arg-type]
                entrances=[
                    Entrance(
                        entrance_id=f"{slot_id}_front",
                        status=spec["entrance_status"],  # type: ignore[arg-type]
                        position=frontage.entrance_point,
                        yaw_deg=frontage.yaw_deg,
                        visible_cues=list(spec["cues"]),  # type: ignore[arg-type]
                    )
                ],
                props=props,
                visual_summary=building_description(asset_key),
                zone_id=str(spec["zone_id"]),
                scale=(CITY_BUILDING_SCALE, CITY_BUILDING_SCALE, CITY_BUILDING_SCALE),
            )
        )

    landmarks = [
        Landmark(
            landmark_id="landmark_clock_tower",
            slot_id="clock_tower",
            landmark_type="clock_tower",
            asset_key="BP_Building_20_C",
            asset_path=asset_path("BP_Building_20_C"),
            position=(-24000.0, CROSS_Y, 0.0),
            yaw_deg=0.0,
            mask_color_rgb=_LANDMARK_MASKS[0],
            visual_summary=building_description("BP_Building_20_C"),
            scale=(0.30, 0.30, 0.80),
        ),
        Landmark(
            landmark_id="landmark_station_forecourt",
            slot_id="station_forecourt",
            landmark_type="commercial_tower",
            asset_key="BP_Building_101_C",
            asset_path=asset_path("BP_Building_101_C"),
            position=(24000.0, CROSS_Y, 0.0),
            yaw_deg=180.0,
            mask_color_rgb=_LANDMARK_MASKS[1],
            visual_summary=building_description("BP_Building_101_C"),
            scale=(0.31, 0.31, 0.82),
        ),
        Landmark(
            landmark_id="landmark_hotel_corner",
            slot_id="hotel_corner",
            landmark_type="hotel",
            asset_key="BP_Building_95_C",
            asset_path=asset_path("BP_Building_95_C"),
            position=(WEST_X - 2500.0, ALLEY_Y - 5000.0, 0.0),
            yaw_deg=-135.0,
            mask_color_rgb=_LANDMARK_MASKS[2],
            visual_summary=building_description("BP_Building_95_C"),
            scale=(0.27, 0.27, 0.42),
        ),
        Landmark(
            landmark_id="landmark_bus_stop",
            slot_id="bus_stop",
            landmark_type="street_landmark",
            asset_key="BP_Building_44_C",
            asset_path=asset_path("BP_Building_44_C"),
            position=(EAST_X + 2500.0, ALLEY_Y - 5000.0, 0.0),
            yaw_deg=135.0,
            mask_color_rgb=_LANDMARK_MASKS[3],
            visual_summary=building_description("BP_Building_44_C"),
            scale=(0.27, 0.27, 0.44),
        ),
        Landmark(
            landmark_id="landmark_market_canopy",
            slot_id="market_canopy",
            landmark_type="venue_hall",
            asset_key="BP_Building_123_C",
            asset_path=asset_path("BP_Building_123_C"),
            position=(MID_X, MARKET_Y + 5000.0, 0.0),
            yaw_deg=0.0,
            mask_color_rgb=_LANDMARK_MASKS[4],
            visual_summary=building_description("BP_Building_123_C"),
            scale=(0.28, 0.28, 0.50),
        ),
    ]

    spawn_clock = layout.node_by_id("spawn_clock_tower").position
    spawn_station = layout.node_by_id("spawn_station_forecourt").position
    agents = [
        AgentSpec(
            agent_id="agent_0",
            spawn_slot="clock_tower_spawn",
            position=(spawn_clock[0], spawn_clock[1], AGENT_Z),
            yaw_deg=0.0,
            private_constraint="I need step-free access and cannot use stairs.",
            private_requirement_keys=["accessible"],
            zone_id="zone_west",
            walk_node_id="spawn_clock_tower",
        ),
        AgentSpec(
            agent_id="agent_1",
            spawn_slot="station_forecourt_spawn",
            position=(spawn_station[0], spawn_station[1], AGENT_Z),
            yaw_deg=180.0,
            private_constraint="I strongly prefer food or drink and a quiet place.",
            private_requirement_keys=["food_drink", "quiet"],
            zone_id="zone_east",
            walk_node_id="spawn_station_forecourt",
        ),
    ]

    coarse_map_text = (
        "Coarse map: a compact Cerdà-inspired station quarter. Four chamfered blocks form a legible 2x2 grid "
        "around Cross Street and the central north-south passage; Market Street closes the north edge and the "
        "service alley closes the south edge. Eight storefront venues occupy the continuous block fronts. "
        "Cross Street is the main orientation axis: the clock tower terminates its west view and the station "
        "tower terminates its east view. The glass market hall is due north; the hotel and bus-stop tower mark "
        "the southwest and southeast corners. Both agents start inside Cross Street's built frontage, facing "
        "toward the district centre rather than the empty-map boundary. "
        "Venue status, accessibility, crowding, food/drink, and entrance conditions are not on this map; inspect visually."
    )

    return Scenario(
        scenario_id=f"station_quarter_seed_{seed}",
        map_template_id=MAP_TEMPLATE_ID,
        seed=seed,
        venues=venues,
        landmarks=landmarks,
        agents=agents,
        requirements=[
            Requirement(key="open", weight=2.0, hard=True, description="Venue must be open."),
            Requirement(key="reachable", weight=2.0, hard=True, description="Venue must be physically reachable."),
            Requirement(key="accessible", weight=2.0, hard=True, description="At least one visitor needs step-free access."),
            Requirement(key="food_drink", weight=1.25, description="Food or drink available is valuable."),
            Requirement(key="quiet", weight=1.0, description="Quiet venues are preferred."),
            Requirement(key="shelter", weight=0.75, description="Indoor/sheltered venue preferred."),
        ],
        soft_weights={"quiet_threshold": 0.65, "crowding_threshold": 0.5},
        coarse_map_text=coarse_map_text,
        max_steps=64,
        layout=layout,
    )
