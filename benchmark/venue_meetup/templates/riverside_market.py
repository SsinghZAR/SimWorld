"""Authored large riverside-market district for Venue Meetup.

``riverside_market_large_v1`` is a deterministic 12-venue, 6-block civic/market
district split by a vertical canal/rail barrier. Venue positions and meeting
regions come from named :class:`~benchmark.venue_meetup.layout.Frontage`
entries on one :class:`~benchmark.venue_meetup.layout.DistrictLayout`.
Coordinates are Unreal centimetres (~800 m east-west footprint).
"""

from __future__ import annotations

import math
from dataclasses import replace

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

MAP_TEMPLATE_ID = "riverside_market_large_v1"
LAYOUT_ID = "riverside_market_large_v1"

# District extents (~800 m E-W, ~700 m N-S).
WEST_X = -40000.0
EAST_X = 40000.0
MID_X = 0.0
NORTH_Y = 35000.0
SOUTH_Y = -35000.0
MARKET_Y = 8000.0
CROSS_Y = -8000.0

# Vertical canal/rail barrier corridor (not directly traversable).
CANAL_HALF_W = 2500.0
WEST_PROMENADE_X = -CANAL_HALF_W - 1500.0  # -4000
EAST_PROMENADE_X = CANAL_HALF_W + 1500.0  # 4000
# Internal radial merchant lanes divide each 280 m superblock into two
# walkable, Amsterdam-like blocks with a narrow facade rhythm.
WEST_MERCHANT_LANE_X = -22000.0
EAST_MERCHANT_LANE_X = 22000.0
MERCHANT_LANE_WIDTH = 1000.0

PRIMARY_BRIDGE_Y = 5000.0
SECONDARY_BRIDGE_Y = -25000.0

STREET_WIDTH = 1400.0
ALLEY_WIDTH = 800.0
SIDEWALK_WIDTH = 300.0
MEET_RADIUS = 900.0
AGENT_Z = 150.0
CITY_BUILDING_SCALE = 0.25

_WEST_AV_SW_X = WEST_X + STREET_WIDTH / 2.0 + SIDEWALK_WIDTH / 2.0
_EAST_AV_SW_X = EAST_X - STREET_WIDTH / 2.0 - SIDEWALK_WIDTH / 2.0
_WEST_PROM_SW_X = WEST_PROMENADE_X - STREET_WIDTH / 2.0 - SIDEWALK_WIDTH / 2.0
_EAST_PROM_SW_X = EAST_PROMENADE_X + STREET_WIDTH / 2.0 + SIDEWALK_WIDTH / 2.0
_NORTH_SW_Y = NORTH_Y - STREET_WIDTH / 2.0 - SIDEWALK_WIDTH / 2.0
_MARKET_N_SW_Y = MARKET_Y + STREET_WIDTH / 2.0 + SIDEWALK_WIDTH / 2.0
_MARKET_S_SW_Y = MARKET_Y - STREET_WIDTH / 2.0 - SIDEWALK_WIDTH / 2.0
_CROSS_N_SW_Y = CROSS_Y + STREET_WIDTH / 2.0 + SIDEWALK_WIDTH / 2.0
_CROSS_S_SW_Y = CROSS_Y - STREET_WIDTH / 2.0 - SIDEWALK_WIDTH / 2.0
_SOUTH_SW_Y = SOUTH_Y + STREET_WIDTH / 2.0 + SIDEWALK_WIDTH / 2.0

_VENUE_MASKS: tuple[tuple[int, int, int], ...] = (
    MASK_COLORS["venue_0"],
    MASK_COLORS["venue_1"],
    MASK_COLORS["venue_2"],
    MASK_COLORS["venue_3"],
    MASK_COLORS["venue_4"],
    MASK_COLORS["venue_5"],
    (255, 160, 64),
    (64, 160, 255),
    (200, 80, 120),
    (80, 200, 120),
    (180, 180, 60),
    (120, 80, 200),
)
_LANDMARK_MASKS: tuple[tuple[int, int, int], ...] = (
    MASK_COLORS["landmark_0"],
    MASK_COLORS["landmark_1"],
    MASK_COLORS["landmark_2"],
    (220, 180, 40),
    (40, 180, 220),
    (200, 100, 60),
    (100, 200, 180),
    (160, 80, 200),
)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_length(points: tuple[tuple[float, float], ...]) -> float:
    return sum(_dist(left, right) for left, right in zip(points, points[1:]))


def _edge(
    nodes: dict[str, WalkNode],
    start_id: str,
    end_id: str,
    *,
    route_kind: str = "sidewalk",
) -> WalkEdge:
    start = nodes[start_id].position
    end = nodes[end_id].position
    # Persist the physical route segment instead of treating graph adjacency as
    # an implicit straight-line instruction.  Template routes are authored on
    # the public street/promenade lattice; the midpoint makes each segment
    # explicit in diagnostics and live walk preflight.
    waypoints = (((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0),)
    return WalkEdge(
        start_node_id=start_id,
        end_node_id=end_id,
        length_cm=_polyline_length((start, *waypoints, end)),
        route_kind=route_kind,  # type: ignore[arg-type]
        waypoints=waypoints,
    )


def build_district_layout() -> DistrictLayout:
    """Return the fixed riverside-market district layout."""

    streets = (
        StreetSegment(
            street_id="west_avenue",
            start=(WEST_X, SOUTH_Y),
            end=(WEST_X, NORTH_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="east_avenue",
            start=(EAST_X, SOUTH_Y),
            end=(EAST_X, NORTH_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="west_promenade",
            start=(WEST_PROMENADE_X, SOUTH_Y),
            end=(WEST_PROMENADE_X, NORTH_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="east_promenade",
            start=(EAST_PROMENADE_X, SOUTH_Y),
            end=(EAST_PROMENADE_X, NORTH_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="west_merchant_lane",
            start=(WEST_MERCHANT_LANE_X, SOUTH_Y),
            end=(WEST_MERCHANT_LANE_X, NORTH_Y),
            width_cm=MERCHANT_LANE_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="east_merchant_lane",
            start=(EAST_MERCHANT_LANE_X, SOUTH_Y),
            end=(EAST_MERCHANT_LANE_X, NORTH_Y),
            width_cm=MERCHANT_LANE_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="north_street_west",
            start=(WEST_X, NORTH_Y),
            end=(WEST_PROMENADE_X, NORTH_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="north_street_east",
            start=(EAST_PROMENADE_X, NORTH_Y),
            end=(EAST_X, NORTH_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="market_street_west",
            start=(WEST_X, MARKET_Y),
            end=(WEST_PROMENADE_X, MARKET_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="market_street_east",
            start=(EAST_PROMENADE_X, MARKET_Y),
            end=(EAST_X, MARKET_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="cross_street_west",
            start=(WEST_X, CROSS_Y),
            end=(WEST_PROMENADE_X, CROSS_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="cross_street_east",
            start=(EAST_PROMENADE_X, CROSS_Y),
            end=(EAST_X, CROSS_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="south_street_west",
            start=(WEST_X, SOUTH_Y),
            end=(WEST_PROMENADE_X, SOUTH_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="south_street_east",
            start=(EAST_PROMENADE_X, SOUTH_Y),
            end=(EAST_X, SOUTH_Y),
            width_cm=STREET_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="primary_bridge",
            start=(WEST_PROMENADE_X, PRIMARY_BRIDGE_Y),
            end=(EAST_PROMENADE_X, PRIMARY_BRIDGE_Y),
            width_cm=ALLEY_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
        StreetSegment(
            street_id="secondary_bridge",
            start=(WEST_PROMENADE_X, SECONDARY_BRIDGE_Y),
            end=(EAST_PROMENADE_X, SECONDARY_BRIDGE_Y),
            width_cm=ALLEY_WIDTH,
            sidewalk_width_cm=SIDEWALK_WIDTH,
        ),
    )

    intersections = (
        Intersection(
            "ix_civic_plaza",
            (WEST_MERCHANT_LANE_X, NORTH_Y),
            landmark_id="landmark_civic_tower",
        ),
        Intersection("ix_market_square", (WEST_PROMENADE_X, MARKET_Y), landmark_id="landmark_market_square"),
        Intersection("ix_main_bridge_west", (WEST_PROMENADE_X, PRIMARY_BRIDGE_Y), landmark_id="landmark_main_bridge"),
        Intersection("ix_main_bridge_east", (EAST_PROMENADE_X, PRIMARY_BRIDGE_Y), landmark_id="landmark_main_bridge"),
        Intersection(
            "ix_transit_entrance",
            (EAST_MERCHANT_LANE_X, NORTH_Y),
            landmark_id="landmark_transit_entrance",
        ),
        Intersection("ix_waterside_tower", (EAST_PROMENADE_X, MARKET_Y), landmark_id="landmark_waterside_tower"),
        Intersection(
            "ix_hospital",
            (WEST_MERCHANT_LANE_X, SOUTH_Y),
            landmark_id="landmark_hospital",
        ),
        Intersection(
            "ix_bus_stop",
            (EAST_MERCHANT_LANE_X, SOUTH_Y),
            landmark_id="landmark_bus_stop",
        ),
        Intersection(
            "ix_fountain",
            (WEST_MERCHANT_LANE_X, CROSS_Y),
            landmark_id="landmark_fountain",
        ),
        Intersection("ix_w_lane_market", (WEST_MERCHANT_LANE_X, MARKET_Y)),
        Intersection("ix_e_lane_market", (EAST_MERCHANT_LANE_X, MARKET_Y)),
        Intersection("ix_secondary_bridge_west", (WEST_PROMENADE_X, SECONDARY_BRIDGE_Y)),
        Intersection("ix_secondary_bridge_east", (EAST_PROMENADE_X, SECONDARY_BRIDGE_Y)),
    )

    frontages = (
        # West / north-zone bank (6 venues)
        Frontage(
            frontage_id="front_nw_civic_cafe",
            block_id="block_nw_civic",
            position=(-28000.0, 22000.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(-28000.0, 25000.0, 0.0),
            meeting_region=MeetingRegion(center=(-28000.0, _NORTH_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="nw_civic_cafe",
            approach_node_id="approach_nw_cafe",
            access_path=((-28000.0, _NORTH_SW_Y),),
        ),
        Frontage(
            frontage_id="front_nw_civic_shop",
            block_id="block_nw_canal",
            position=(-16000.0, 18000.0, 0.0),
            yaw_deg=-90.0,
            entrance_point=(-16000.0, 14000.0, 0.0),
            meeting_region=MeetingRegion(center=(-16000.0, _MARKET_N_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="nw_civic_shop",
            approach_node_id="approach_nw_shop",
            access_path=((-16000.0, _MARKET_N_SW_Y),),
        ),
        Frontage(
            frontage_id="front_w_market_bistro",
            block_id="block_w_market",
            position=(-26000.0, 2000.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(-26000.0, 5000.0, 0.0),
            meeting_region=MeetingRegion(center=(-26000.0, _MARKET_S_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="w_market_bistro",
            approach_node_id="approach_w_bistro",
            access_path=((-26000.0, _MARKET_S_SW_Y),),
        ),
        Frontage(
            frontage_id="front_w_market_stall",
            block_id="block_w_quay",
            position=(-12000.0, 2000.0, 0.0),
            yaw_deg=0.0,
            entrance_point=(-8000.0, 2000.0, 0.0),
            meeting_region=MeetingRegion(center=(_WEST_PROM_SW_X, 2000.0), radius=MEET_RADIUS),
            venue_slot_id="w_market_stall",
            approach_node_id="approach_w_stall",
            access_path=((_WEST_PROM_SW_X, 2000.0),),
        ),
        Frontage(
            frontage_id="front_sw_resid_pub",
            block_id="block_sw_residential",
            position=(-26000.0, -18000.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(-26000.0, -14000.0, 0.0),
            meeting_region=MeetingRegion(center=(-26000.0, _CROSS_S_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="sw_resid_pub",
            approach_node_id="approach_sw_pub",
            access_path=((-26000.0, _CROSS_S_SW_Y),),
        ),
        Frontage(
            frontage_id="front_sw_resid_lobby",
            block_id="block_sw_canal",
            position=(-16000.0, -26000.0, 0.0),
            yaw_deg=-90.0,
            entrance_point=(-16000.0, -30000.0, 0.0),
            meeting_region=MeetingRegion(center=(-16000.0, _SOUTH_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="sw_resid_lobby",
            approach_node_id="approach_sw_lobby",
            access_path=((-16000.0, _SOUTH_SW_Y),),
        ),
        # East / south-zone bank (6 venues)
        Frontage(
            frontage_id="front_ne_transit_shop",
            block_id="block_ne_transit",
            position=(28000.0, 22000.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(28000.0, 25000.0, 0.0),
            meeting_region=MeetingRegion(center=(28000.0, _NORTH_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="ne_transit_shop",
            approach_node_id="approach_ne_shop",
            access_path=((28000.0, _NORTH_SW_Y),),
        ),
        Frontage(
            frontage_id="front_ne_transit_deli",
            block_id="block_ne_canal",
            position=(16000.0, 18000.0, 0.0),
            yaw_deg=-90.0,
            entrance_point=(16000.0, 14000.0, 0.0),
            meeting_region=MeetingRegion(center=(16000.0, _MARKET_N_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="ne_transit_deli",
            approach_node_id="approach_ne_deli",
            access_path=((16000.0, _MARKET_N_SW_Y),),
        ),
        Frontage(
            frontage_id="front_e_water_restaurant",
            block_id="block_e_waterfront",
            position=(26000.0, 2000.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(26000.0, 5000.0, 0.0),
            meeting_region=MeetingRegion(center=(26000.0, _MARKET_S_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="e_water_restaurant",
            approach_node_id="approach_e_restaurant",
            access_path=((26000.0, _MARKET_S_SW_Y),),
        ),
        Frontage(
            frontage_id="front_e_water_hall",
            block_id="block_e_quay",
            position=(12000.0, 2000.0, 0.0),
            yaw_deg=180.0,
            entrance_point=(8000.0, 2000.0, 0.0),
            meeting_region=MeetingRegion(center=(_EAST_PROM_SW_X, 2000.0), radius=MEET_RADIUS),
            venue_slot_id="e_water_hall",
            approach_node_id="approach_e_hall",
            access_path=((_EAST_PROM_SW_X, 2000.0),),
        ),
        Frontage(
            frontage_id="front_se_hotel_cafe",
            block_id="block_se_hotel",
            position=(26000.0, -18000.0, 0.0),
            yaw_deg=90.0,
            entrance_point=(26000.0, -14000.0, 0.0),
            meeting_region=MeetingRegion(center=(26000.0, _CROSS_S_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="se_hotel_cafe",
            approach_node_id="approach_se_cafe",
            access_path=((26000.0, _CROSS_S_SW_Y),),
        ),
        Frontage(
            frontage_id="front_se_hotel_shop",
            block_id="block_se_canal",
            position=(16000.0, -26000.0, 0.0),
            yaw_deg=-90.0,
            entrance_point=(16000.0, -30000.0, 0.0),
            meeting_region=MeetingRegion(center=(16000.0, _SOUTH_SW_Y), radius=MEET_RADIUS),
            venue_slot_id="se_hotel_shop",
            approach_node_id="approach_se_shop",
            access_path=((16000.0, _SOUTH_SW_Y),),
        ),
    )

    blocks = (
        Block(
            block_id="block_nw_civic",
            footprint=(
                (-39000.0, 9000.0),
                (-23000.0, 9000.0),
                (-23000.0, 34000.0),
                (-39000.0, 34000.0),
            ),
            frontage_ids=("front_nw_civic_cafe",),
            visual_style="civic_masonry",
            shell_target=28,
        ),
        Block(
            block_id="block_nw_canal",
            footprint=(
                (-21000.0, 9000.0),
                (-5000.0, 9000.0),
                (-5000.0, 34000.0),
                (-21000.0, 34000.0),
            ),
            frontage_ids=("front_nw_civic_shop",),
            visual_style="canal_merchant",
            shell_target=28,
        ),
        Block(
            block_id="block_w_market",
            footprint=(
                (-39000.0, -7000.0),
                (-23000.0, -7000.0),
                (-23000.0, 7000.0),
                (-39000.0, 7000.0),
            ),
            frontage_ids=("front_w_market_bistro",),
            visual_style="civic_masonry",
            shell_target=24,
        ),
        Block(
            block_id="block_w_quay",
            footprint=(
                (-21000.0, -7000.0),
                (-5000.0, -7000.0),
                (-5000.0, 7000.0),
                (-21000.0, 7000.0),
            ),
            frontage_ids=("front_w_market_stall",),
            visual_style="canal_merchant",
            shell_target=24,
        ),
        Block(
            block_id="block_sw_residential",
            footprint=(
                (-39000.0, -34000.0),
                (-23000.0, -34000.0),
                (-23000.0, -9000.0),
                (-39000.0, -9000.0),
            ),
            frontage_ids=("front_sw_resid_pub",),
            visual_style="civic_masonry",
            shell_target=28,
        ),
        Block(
            block_id="block_sw_canal",
            footprint=(
                (-21000.0, -34000.0),
                (-5000.0, -34000.0),
                (-5000.0, -9000.0),
                (-21000.0, -9000.0),
            ),
            frontage_ids=("front_sw_resid_lobby",),
            visual_style="canal_merchant",
            shell_target=28,
        ),
        Block(
            block_id="block_ne_canal",
            footprint=(
                (5000.0, 9000.0),
                (21000.0, 9000.0),
                (21000.0, 34000.0),
                (5000.0, 34000.0),
            ),
            frontage_ids=("front_ne_transit_deli",),
            visual_style="canal_merchant",
            shell_target=28,
        ),
        Block(
            block_id="block_ne_transit",
            footprint=(
                (23000.0, 9000.0),
                (39000.0, 9000.0),
                (39000.0, 34000.0),
                (23000.0, 34000.0),
            ),
            frontage_ids=("front_ne_transit_shop",),
            visual_style="transit_mixed",
            shell_target=28,
        ),
        Block(
            block_id="block_e_quay",
            footprint=(
                (5000.0, -7000.0),
                (21000.0, -7000.0),
                (21000.0, 7000.0),
                (5000.0, 7000.0),
            ),
            frontage_ids=("front_e_water_hall",),
            visual_style="canal_merchant",
            shell_target=24,
        ),
        Block(
            block_id="block_e_waterfront",
            footprint=(
                (23000.0, -7000.0),
                (39000.0, -7000.0),
                (39000.0, 7000.0),
                (23000.0, 7000.0),
            ),
            frontage_ids=("front_e_water_restaurant",),
            visual_style="transit_mixed",
            shell_target=24,
        ),
        Block(
            block_id="block_se_canal",
            footprint=(
                (5000.0, -34000.0),
                (21000.0, -34000.0),
                (21000.0, -9000.0),
                (5000.0, -9000.0),
            ),
            frontage_ids=("front_se_hotel_shop",),
            visual_style="canal_merchant",
            shell_target=28,
        ),
        Block(
            block_id="block_se_hotel",
            footprint=(
                (23000.0, -34000.0),
                (39000.0, -34000.0),
                (39000.0, -9000.0),
                (23000.0, -9000.0),
            ),
            frontage_ids=("front_se_hotel_cafe",),
            visual_style="transit_mixed",
            shell_target=28,
        ),
    )

    walk_nodes = (
        # Interior market-axis spawns. Each sits at a four-block decision
        # point with facades on both sides and faces toward the canal centre.
        WalkNode(
            "spawn_civic_plaza",
            (WEST_MERCHANT_LANE_X, _MARKET_N_SW_Y),
            "spawn",
        ),
        WalkNode(
            "spawn_transit_forecourt",
            (EAST_MERCHANT_LANE_X, _MARKET_N_SW_Y),
            "spawn",
        ),
        # West bank sidewalk lattice
        WalkNode("swk_w_north_west", (_WEST_AV_SW_X, _NORTH_SW_Y), "intersection"),
        WalkNode("swk_nw_cafe", (-28000.0, 34750.0), "sidewalk"),
        WalkNode("swk_w_north_prom", (_WEST_PROM_SW_X, _NORTH_SW_Y), "intersection"),
        WalkNode("swk_w_market_n_west", (_WEST_AV_SW_X, _MARKET_N_SW_Y), "intersection"),
        WalkNode("swk_nw_shop", (-16000.0, 8150.0), "sidewalk"),
        WalkNode("swk_w_market_n_prom", (_WEST_PROM_SW_X, _MARKET_N_SW_Y), "intersection"),
        WalkNode("swk_w_market_s_west", (_WEST_AV_SW_X, _MARKET_S_SW_Y), "intersection"),
        WalkNode("swk_w_bistro", (-26000.0, 7850.0), "sidewalk"),
        WalkNode("swk_w_market_s_prom", (_WEST_PROM_SW_X, _MARKET_S_SW_Y), "intersection"),
        WalkNode("swk_w_stall", (-4250.0, 2000.0), "sidewalk"),
        WalkNode("swk_w_cross_n_west", (_WEST_AV_SW_X, _CROSS_N_SW_Y), "intersection"),
        WalkNode("swk_w_cross_n_prom", (_WEST_PROM_SW_X, _CROSS_N_SW_Y), "intersection"),
        WalkNode("swk_w_cross_s_west", (_WEST_AV_SW_X, _CROSS_S_SW_Y), "intersection"),
        WalkNode("swk_sw_pub", (-26000.0, -8150.0), "sidewalk"),
        WalkNode("swk_w_cross_s_prom", (_WEST_PROM_SW_X, _CROSS_S_SW_Y), "intersection"),
        WalkNode("swk_w_south_west", (_WEST_AV_SW_X, _SOUTH_SW_Y), "intersection"),
        WalkNode("swk_sw_lobby", (-16000.0, -34750.0), "sidewalk"),
        WalkNode("swk_w_south_prom", (_WEST_PROM_SW_X, _SOUTH_SW_Y), "intersection"),
        # West merchant-lane spine (the market-north node is the spawn).
        WalkNode("swk_w_lane_north", (WEST_MERCHANT_LANE_X, _NORTH_SW_Y), "intersection"),
        WalkNode(
            "swk_w_lane_market_s",
            (WEST_MERCHANT_LANE_X, _MARKET_S_SW_Y),
            "intersection",
        ),
        WalkNode(
            "swk_w_lane_cross_n",
            (WEST_MERCHANT_LANE_X, _CROSS_N_SW_Y),
            "intersection",
        ),
        WalkNode(
            "swk_w_lane_cross_s",
            (WEST_MERCHANT_LANE_X, _CROSS_S_SW_Y),
            "intersection",
        ),
        WalkNode("swk_w_lane_south", (WEST_MERCHANT_LANE_X, _SOUTH_SW_Y), "intersection"),
        # East bank sidewalk lattice
        WalkNode("swk_e_north_east", (_EAST_AV_SW_X, _NORTH_SW_Y), "intersection"),
        WalkNode("swk_ne_shop", (28000.0, 34750.0), "sidewalk"),
        WalkNode("swk_e_north_prom", (_EAST_PROM_SW_X, _NORTH_SW_Y), "intersection"),
        WalkNode("swk_e_market_n_east", (_EAST_AV_SW_X, _MARKET_N_SW_Y), "intersection"),
        WalkNode("swk_ne_deli", (16000.0, 8150.0), "sidewalk"),
        WalkNode("swk_e_market_n_prom", (_EAST_PROM_SW_X, _MARKET_N_SW_Y), "intersection"),
        WalkNode("swk_e_market_s_east", (_EAST_AV_SW_X, _MARKET_S_SW_Y), "intersection"),
        WalkNode("swk_e_restaurant", (26000.0, 7850.0), "sidewalk"),
        WalkNode("swk_e_market_s_prom", (_EAST_PROM_SW_X, _MARKET_S_SW_Y), "intersection"),
        WalkNode("swk_e_hall", (4250.0, 2000.0), "sidewalk"),
        WalkNode("swk_e_cross_n_east", (_EAST_AV_SW_X, _CROSS_N_SW_Y), "intersection"),
        WalkNode("swk_e_cross_n_prom", (_EAST_PROM_SW_X, _CROSS_N_SW_Y), "intersection"),
        WalkNode("swk_e_cross_s_east", (_EAST_AV_SW_X, _CROSS_S_SW_Y), "intersection"),
        WalkNode("swk_se_cafe", (26000.0, -8150.0), "sidewalk"),
        WalkNode("swk_e_cross_s_prom", (_EAST_PROM_SW_X, _CROSS_S_SW_Y), "intersection"),
        WalkNode("swk_e_south_east", (_EAST_AV_SW_X, _SOUTH_SW_Y), "intersection"),
        WalkNode("swk_se_shop", (16000.0, -34750.0), "sidewalk"),
        WalkNode("swk_e_south_prom", (_EAST_PROM_SW_X, _SOUTH_SW_Y), "intersection"),
        # East merchant-lane spine (the market-north node is the spawn).
        WalkNode("swk_e_lane_north", (EAST_MERCHANT_LANE_X, _NORTH_SW_Y), "intersection"),
        WalkNode(
            "swk_e_lane_market_s",
            (EAST_MERCHANT_LANE_X, _MARKET_S_SW_Y),
            "intersection",
        ),
        WalkNode(
            "swk_e_lane_cross_n",
            (EAST_MERCHANT_LANE_X, _CROSS_N_SW_Y),
            "intersection",
        ),
        WalkNode(
            "swk_e_lane_cross_s",
            (EAST_MERCHANT_LANE_X, _CROSS_S_SW_Y),
            "intersection",
        ),
        WalkNode("swk_e_lane_south", (EAST_MERCHANT_LANE_X, _SOUTH_SW_Y), "intersection"),
        # Bridge endpoints (only legal barrier crossings)
        WalkNode("bridge_primary_west", (WEST_PROMENADE_X, PRIMARY_BRIDGE_Y), "bridge"),
        WalkNode("bridge_primary_east", (EAST_PROMENADE_X, PRIMARY_BRIDGE_Y), "bridge"),
        WalkNode("bridge_secondary_west", (WEST_PROMENADE_X, SECONDARY_BRIDGE_Y), "bridge"),
        WalkNode("bridge_secondary_east", (EAST_PROMENADE_X, SECONDARY_BRIDGE_Y), "bridge"),
        # Target-only exterior leaves.  The final access segment into a venue
        # is intentionally not a usable through-route edge.
        WalkNode("approach_nw_cafe", (-28000.0, 34450.0), "sidewalk"),
        WalkNode("approach_nw_shop", (-16000.0, 8450.0), "sidewalk"),
        WalkNode("approach_w_bistro", (-26000.0, 7550.0), "sidewalk"),
        WalkNode("approach_w_stall", (-4550.0, 2000.0), "sidewalk"),
        WalkNode("approach_sw_pub", (-26000.0, -8450.0), "sidewalk"),
        WalkNode("approach_sw_lobby", (-16000.0, -34450.0), "sidewalk"),
        WalkNode("approach_ne_shop", (28000.0, 34450.0), "sidewalk"),
        WalkNode("approach_ne_deli", (16000.0, 8450.0), "sidewalk"),
        WalkNode("approach_e_restaurant", (26000.0, 7550.0), "sidewalk"),
        WalkNode("approach_e_hall", (4550.0, 2000.0), "sidewalk"),
        WalkNode("approach_se_cafe", (26000.0, -8450.0), "sidewalk"),
        WalkNode("approach_se_shop", (16000.0, -34450.0), "sidewalk"),
    )
    nodes = {node.node_id: node for node in walk_nodes}

    edge_specs: list[tuple[str, str, str]] = [
        # West horizontal streets cross the outer avenue, internal merchant
        # lane, and canal promenade in that order.
        ("swk_w_north_west", "swk_w_lane_north", "sidewalk"),
        ("swk_w_lane_north", "swk_w_north_prom", "sidewalk"),
        ("swk_w_market_n_west", "spawn_civic_plaza", "sidewalk"),
        ("spawn_civic_plaza", "swk_w_market_n_prom", "sidewalk"),
        ("swk_w_market_s_west", "swk_w_lane_market_s", "sidewalk"),
        ("swk_w_lane_market_s", "swk_w_market_s_prom", "sidewalk"),
        ("swk_w_cross_n_west", "swk_w_lane_cross_n", "sidewalk"),
        ("swk_w_lane_cross_n", "swk_w_cross_n_prom", "sidewalk"),
        ("swk_w_cross_s_west", "swk_w_lane_cross_s", "sidewalk"),
        ("swk_w_lane_cross_s", "swk_w_cross_s_prom", "sidewalk"),
        ("swk_w_south_west", "swk_w_lane_south", "sidewalk"),
        ("swk_w_lane_south", "swk_w_south_prom", "sidewalk"),
        # West avenue N-S
        ("swk_w_north_west", "swk_w_market_n_west", "sidewalk"),
        ("swk_w_market_n_west", "swk_w_market_s_west", "crossing"),
        ("swk_w_market_s_west", "swk_w_cross_n_west", "sidewalk"),
        ("swk_w_cross_n_west", "swk_w_cross_s_west", "crossing"),
        ("swk_w_cross_s_west", "swk_w_south_west", "sidewalk"),
        # West merchant lane N-S
        ("swk_w_lane_north", "spawn_civic_plaza", "sidewalk"),
        ("spawn_civic_plaza", "swk_w_lane_market_s", "crossing"),
        ("swk_w_lane_market_s", "swk_w_lane_cross_n", "sidewalk"),
        ("swk_w_lane_cross_n", "swk_w_lane_cross_s", "crossing"),
        ("swk_w_lane_cross_s", "swk_w_lane_south", "sidewalk"),
        # West promenade N-S
        ("swk_w_north_prom", "swk_w_market_n_prom", "sidewalk"),
        ("swk_w_market_n_prom", "swk_w_market_s_prom", "crossing"),
        ("swk_w_market_s_prom", "swk_w_stall", "sidewalk"),
        ("swk_w_market_s_prom", "bridge_primary_west", "sidewalk"),
        ("bridge_primary_west", "swk_w_cross_n_prom", "sidewalk"),
        ("swk_w_cross_n_prom", "swk_w_cross_s_prom", "crossing"),
        ("swk_w_cross_s_prom", "bridge_secondary_west", "sidewalk"),
        ("bridge_secondary_west", "swk_w_south_prom", "sidewalk"),
        # East horizontal streets mirror the west-bank structure.
        ("swk_e_north_east", "swk_e_lane_north", "sidewalk"),
        ("swk_e_lane_north", "swk_e_north_prom", "sidewalk"),
        ("swk_e_market_n_east", "spawn_transit_forecourt", "sidewalk"),
        ("spawn_transit_forecourt", "swk_e_market_n_prom", "sidewalk"),
        ("swk_e_market_s_east", "swk_e_lane_market_s", "sidewalk"),
        ("swk_e_lane_market_s", "swk_e_market_s_prom", "sidewalk"),
        ("swk_e_cross_n_east", "swk_e_lane_cross_n", "sidewalk"),
        ("swk_e_lane_cross_n", "swk_e_cross_n_prom", "sidewalk"),
        ("swk_e_cross_s_east", "swk_e_lane_cross_s", "sidewalk"),
        ("swk_e_lane_cross_s", "swk_e_cross_s_prom", "sidewalk"),
        ("swk_e_south_east", "swk_e_lane_south", "sidewalk"),
        ("swk_e_lane_south", "swk_e_south_prom", "sidewalk"),
        # East avenue N-S
        ("swk_e_north_east", "swk_e_market_n_east", "sidewalk"),
        ("swk_e_market_n_east", "swk_e_market_s_east", "crossing"),
        ("swk_e_market_s_east", "swk_e_cross_n_east", "sidewalk"),
        ("swk_e_cross_n_east", "swk_e_cross_s_east", "crossing"),
        ("swk_e_cross_s_east", "swk_e_south_east", "sidewalk"),
        # East merchant lane N-S
        ("swk_e_lane_north", "spawn_transit_forecourt", "sidewalk"),
        ("spawn_transit_forecourt", "swk_e_lane_market_s", "crossing"),
        ("swk_e_lane_market_s", "swk_e_lane_cross_n", "sidewalk"),
        ("swk_e_lane_cross_n", "swk_e_lane_cross_s", "crossing"),
        ("swk_e_lane_cross_s", "swk_e_lane_south", "sidewalk"),
        # East promenade N-S
        ("swk_e_north_prom", "swk_e_market_n_prom", "sidewalk"),
        ("swk_e_market_n_prom", "swk_e_market_s_prom", "crossing"),
        ("swk_e_market_s_prom", "swk_e_hall", "sidewalk"),
        ("swk_e_market_s_prom", "bridge_primary_east", "sidewalk"),
        ("bridge_primary_east", "swk_e_cross_n_prom", "sidewalk"),
        ("swk_e_cross_n_prom", "swk_e_cross_s_prom", "crossing"),
        ("swk_e_cross_s_prom", "bridge_secondary_east", "sidewalk"),
        ("bridge_secondary_east", "swk_e_south_prom", "sidewalk"),
        # One public edge into each venue approach leaf.
        ("swk_w_north_west", "swk_nw_cafe", "sidewalk"),
        ("swk_nw_cafe", "approach_nw_cafe", "sidewalk"),
        ("spawn_civic_plaza", "swk_nw_shop", "sidewalk"),
        ("swk_nw_shop", "approach_nw_shop", "sidewalk"),
        ("swk_w_market_s_west", "swk_w_bistro", "sidewalk"),
        ("swk_w_bistro", "approach_w_bistro", "sidewalk"),
        ("swk_w_market_s_prom", "swk_w_stall", "sidewalk"),
        ("swk_w_stall", "approach_w_stall", "sidewalk"),
        ("swk_w_cross_s_west", "swk_sw_pub", "sidewalk"),
        ("swk_sw_pub", "approach_sw_pub", "sidewalk"),
        ("swk_w_lane_south", "swk_sw_lobby", "sidewalk"),
        ("swk_sw_lobby", "approach_sw_lobby", "sidewalk"),
        ("swk_e_north_east", "swk_ne_shop", "sidewalk"),
        ("swk_ne_shop", "approach_ne_shop", "sidewalk"),
        ("spawn_transit_forecourt", "swk_ne_deli", "sidewalk"),
        ("swk_ne_deli", "approach_ne_deli", "sidewalk"),
        ("swk_e_market_s_east", "swk_e_restaurant", "sidewalk"),
        ("swk_e_restaurant", "approach_e_restaurant", "sidewalk"),
        ("swk_e_market_s_prom", "swk_e_hall", "sidewalk"),
        ("swk_e_hall", "approach_e_hall", "sidewalk"),
        ("swk_e_cross_s_east", "swk_se_cafe", "sidewalk"),
        ("swk_se_cafe", "approach_se_cafe", "sidewalk"),
        ("swk_e_lane_south", "swk_se_shop", "sidewalk"),
        ("swk_se_shop", "approach_se_shop", "sidewalk"),
    ]
    walk_edges = [_edge(nodes, start, end, route_kind=kind) for start, end, kind in edge_specs]

    # Exactly two barrier crossings: short primary bridge and longer secondary detour.
    primary = WalkEdge(
        start_node_id="bridge_primary_west",
        end_node_id="bridge_primary_east",
        length_cm=_dist(nodes["bridge_primary_west"].position, nodes["bridge_primary_east"].position),
        route_kind="bridge",
        waypoints=((0.0, PRIMARY_BRIDGE_Y),),
    )
    secondary_span = _dist(nodes["bridge_secondary_west"].position, nodes["bridge_secondary_east"].position)
    # Secondary crossing is a noticeably longer detour (winding rail span).
    secondary_waypoints = (
        (-1000.0, SECONDARY_BRIDGE_Y - 5000.0),
        (1000.0, SECONDARY_BRIDGE_Y - 5000.0),
    )
    secondary = WalkEdge(
        start_node_id="bridge_secondary_west",
        end_node_id="bridge_secondary_east",
        length_cm=_polyline_length(
            (nodes["bridge_secondary_west"].position, *secondary_waypoints, nodes["bridge_secondary_east"].position)
        ),
        route_kind="bridge",
        waypoints=secondary_waypoints,
    )
    walk_edges.extend([primary, secondary])

    return DistrictLayout(
        layout_id=LAYOUT_ID,
        streets=streets,
        intersections=intersections,
        blocks=blocks,
        frontages=frontages,
        walk_nodes=walk_nodes,
        walk_edges=tuple(walk_edges),
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


def _barrier_props(
    y_positions: tuple[float, ...], *, west: bool
) -> list[tuple[str, tuple[float, float, float], str]]:
    """Place RoadBlocker_C props along the canal face of a waterside venue."""

    x = -CANAL_HALF_W if west else CANAL_HALF_W
    return [("RoadBlocker_C", (x, y, 0.0), "canal/rail barrier") for y in y_positions]


def build_fixed_scenario(seed: int = 31) -> Scenario:
    """Return the deterministic riverside-market large scenario."""

    layout = build_district_layout()
    frontage_by_slot = {
        frontage.venue_slot_id: frontage
        for frontage in layout.frontages
        if frontage.venue_slot_id is not None
    }

    venue_defs: list[dict[str, object]] = [
        {
            "venue_id": "venue_nw_civic_cafe",
            "slot_id": "nw_civic_cafe",
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
                quiet_score=0.7,
                crowding_score=0.3,
                near_transit=False,
            ),
            "entrance_status": "accessible",
            "cues": ["red awning", "civic plaza seating"],
            "props": [
                ("BP_Table_C", (-27800.0, 25200.0, 0.0), "outdoor seating"),
                ("BP_Soda1_C", (-28200.0, 25300.0, 0.0), "food/drink cue"),
            ],
        },
        {
            "venue_id": "venue_nw_civic_shop",
            "slot_id": "nw_civic_shop",
            "venue_type": "shop",
            "asset_key": "BP_Building_24_C",
            "zone_id": "zone_west",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=4,
                accessible=True,
                shelter=True,
                food_drink=False,
                quiet_score=0.55,
                crowding_score=0.45,
                near_transit=False,
            ),
            "entrance_status": "accessible",
            "cues": ["civic storefront", "market-street awning"],
            "props": [],
        },
        {
            "venue_id": "venue_w_market_bistro",
            "slot_id": "w_market_bistro",
            "venue_type": "restaurant",
            "asset_key": "BP_Building_06_C",
            "zone_id": "zone_west",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=8,
                accessible=False,
                shelter=True,
                food_drink=True,
                quiet_score=0.5,
                crowding_score=0.55,
                near_transit=False,
            ),
            "entrance_status": "stairs_only",
            "cues": ["green awning", "market square bistro"],
            "props": [],
        },
        {
            "venue_id": "venue_w_market_stall",
            "slot_id": "w_market_stall",
            "venue_type": "shop",
            "asset_key": "BP_Building_25_C",
            "zone_id": "zone_west",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=5,
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=0.4,
                crowding_score=0.7,
                near_transit=False,
            ),
            "entrance_status": "accessible",
            "cues": ["busy market stall", "canal-side counter"],
            "props": [
                ("BP_Can_C", (-7500.0, 2200.0, 0.0), "food/drink cue"),
                # Leave PRIMARY_BRIDGE_Y open: it is the legal northern bank crossing.
                *_barrier_props((0.0, 2500.0), west=True),
            ],
        },
        {
            "venue_id": "venue_sw_resid_pub",
            "slot_id": "sw_resid_pub",
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
                quiet_score=0.3,
                crowding_score=0.8,
                near_transit=False,
            ),
            "entrance_status": "stairs_only",
            "cues": ["colorful signs", "noisy pub front"],
            "props": [("BP_Trash_bin_a_C", (-25800.0, -13800.0, 0.0), "busy street cue")],
        },
        {
            "venue_id": "venue_sw_resid_lobby",
            "slot_id": "sw_resid_lobby",
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
                quiet_score=0.85,
                crowding_score=0.2,
                near_transit=False,
            ),
            "entrance_status": "accessible",
            "cues": ["arched lobby", "quiet residential front"],
            "props": [
                # Leave SECONDARY_BRIDGE_Y open: it is the longer legal detour.
                *_barrier_props((-22000.0, -28000.0), west=True),
            ],
        },
        {
            "venue_id": "venue_ne_transit_shop",
            "slot_id": "ne_transit_shop",
            "venue_type": "shop",
            "asset_key": "BP_Building_25_C",
            "zone_id": "zone_east",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=4,
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=0.45,
                crowding_score=0.65,
                near_transit=True,
            ),
            "entrance_status": "accessible",
            "cues": ["transit plaza shop", "busy forecourt"],
            "props": [("BP_Hydrant_C", (28200.0, 25200.0, 0.0), "street landmark cue")],
        },
        {
            "venue_id": "venue_ne_transit_deli",
            "slot_id": "ne_transit_deli",
            "venue_type": "shop",
            "asset_key": "BP_Building_24_C",
            "zone_id": "zone_east",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=5,
                accessible=False,
                shelter=True,
                food_drink=True,
                quiet_score=0.5,
                crowding_score=0.5,
                near_transit=True,
            ),
            "entrance_status": "stairs_only",
            "cues": ["convenience awning", "raised deli step"],
            "props": [],
        },
        {
            "venue_id": "venue_e_water_restaurant",
            "slot_id": "e_water_restaurant",
            "venue_type": "restaurant",
            "asset_key": "BP_Building_06_C",
            "zone_id": "zone_east",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=9,
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=0.6,
                crowding_score=0.4,
                near_transit=False,
            ),
            "entrance_status": "accessible",
            "cues": ["waterfront dining", "canal view"],
            "props": [("BP_Table2_C", (26200.0, 5200.0, 0.0), "outdoor seating")],
        },
        {
            "venue_id": "venue_e_water_hall",
            "slot_id": "e_water_hall",
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
                quiet_score=0.55,
                crowding_score=0.35,
                near_transit=False,
            ),
            "entrance_status": "blocked",
            "cues": ["road blockers across entrance", "closed hall front"],
            "props": [
                ("RoadBlocker_C", (7800.0, 1800.0, 0.0), "blocked/closed entrance"),
                ("RoadCone_C", (8200.0, 1800.0, 0.0), "blocked/closed entrance"),
                *_barrier_props((0.0, 2500.0), west=False),
            ],
        },
        {
            "venue_id": "venue_se_hotel_cafe",
            "slot_id": "se_hotel_cafe",
            "venue_type": "cafe",
            "asset_key": "BP_Building_05_C",
            "zone_id": "zone_east",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=6,
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=0.65,
                crowding_score=0.35,
                near_transit=False,
            ),
            "entrance_status": "accessible",
            "cues": ["hotel-street cafe", "quiet tables"],
            "props": [],
        },
        {
            "venue_id": "venue_se_hotel_shop",
            "slot_id": "se_hotel_shop",
            "venue_type": "shop",
            "asset_key": "BP_Building_123_C",
            "zone_id": "zone_east",
            "properties": VenueProperties(
                open=True,
                reachable=True,
                capacity=7,
                accessible=True,
                shelter=True,
                food_drink=False,
                quiet_score=0.75,
                crowding_score=0.25,
                near_transit=True,
            ),
            "entrance_status": "accessible",
            "cues": ["glass-domed shop", "south bank entrance"],
            "props": [
                *_barrier_props((-22000.0, -28000.0), west=False),
            ],
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
            landmark_id="landmark_main_bridge",
            slot_id="main_bridge",
            landmark_type="street_landmark",
            asset_key="BP_Building_99_C",
            asset_path=asset_path("BP_Building_99_C"),
            position=(MID_X, PRIMARY_BRIDGE_Y + 5500.0, 0.0),
            yaw_deg=0.0,
            mask_color_rgb=_LANDMARK_MASKS[0],
            visual_summary=building_description("BP_Building_99_C"),
            scale=(0.24, 0.24, 0.36),
        ),
        Landmark(
            landmark_id="landmark_market_square",
            slot_id="market_square",
            landmark_type="venue_hall",
            asset_key="BP_Building_123_C",
            asset_path=asset_path("BP_Building_123_C"),
            position=(-15000.0, 21000.0, 0.0),
            yaw_deg=0.0,
            mask_color_rgb=_LANDMARK_MASKS[1],
            visual_summary=building_description("BP_Building_123_C"),
            scale=(0.22, 0.22, 0.40),
        ),
        Landmark(
            landmark_id="landmark_transit_entrance",
            slot_id="transit_entrance",
            landmark_type="commercial_tower",
            asset_key="BP_Building_20_C",
            asset_path=asset_path("BP_Building_20_C"),
            position=(EAST_MERCHANT_LANE_X, NORTH_Y + 4000.0, 0.0),
            yaw_deg=180.0,
            mask_color_rgb=_LANDMARK_MASKS[2],
            visual_summary=building_description("BP_Building_20_C"),
            scale=(0.30, 0.30, 0.54),
        ),
        Landmark(
            landmark_id="landmark_civic_tower",
            slot_id="civic_tower",
            landmark_type="clock_tower",
            asset_key="BP_Building_101_C",
            asset_path=asset_path("BP_Building_101_C"),
            position=(WEST_MERCHANT_LANE_X, NORTH_Y + 4000.0, 0.0),
            yaw_deg=0.0,
            mask_color_rgb=_LANDMARK_MASKS[3],
            visual_summary=building_description("BP_Building_101_C"),
            scale=(0.31, 0.31, 0.56),
        ),
        Landmark(
            landmark_id="landmark_waterside_tower",
            slot_id="waterside_tower",
            landmark_type="commercial_tower",
            asset_key="BP_Building_44_C",
            asset_path=asset_path("BP_Building_44_C"),
            position=(EAST_PROMENADE_X + 4000.0, MARKET_Y + 3500.0, 0.0),
            yaw_deg=-90.0,
            mask_color_rgb=_LANDMARK_MASKS[4],
            visual_summary=building_description("BP_Building_44_C"),
            scale=(0.30, 0.30, 0.52),
        ),
        Landmark(
            landmark_id="landmark_fountain",
            slot_id="fountain",
            landmark_type="street_landmark",
            asset_key="BP_Building_99_C",
            asset_path=asset_path("BP_Building_99_C"),
            position=(-33000.0, 2000.0, 0.0),
            yaw_deg=0.0,
            mask_color_rgb=_LANDMARK_MASKS[5],
            visual_summary=building_description("BP_Building_99_C"),
            scale=(0.18, 0.18, 0.24),
        ),
        Landmark(
            landmark_id="landmark_hospital",
            slot_id="hospital",
            landmark_type="hospital",
            asset_key="BP_Building_87_C",
            asset_path=asset_path("BP_Building_87_C"),
            position=(WEST_MERCHANT_LANE_X, SOUTH_Y - 5000.0, 0.0),
            yaw_deg=0.0,
            mask_color_rgb=_LANDMARK_MASKS[6],
            visual_summary=building_description("BP_Building_87_C"),
            scale=(0.30, 0.30, 0.42),
        ),
        Landmark(
            landmark_id="landmark_bus_stop",
            slot_id="bus_stop",
            landmark_type="street_landmark",
            asset_key="BP_Building_95_C",
            asset_path=asset_path("BP_Building_95_C"),
            position=(EAST_MERCHANT_LANE_X, SOUTH_Y - 5000.0, 0.0),
            yaw_deg=180.0,
            mask_color_rgb=_LANDMARK_MASKS[7],
            visual_summary=building_description("BP_Building_95_C"),
            scale=(0.26, 0.26, 0.44),
        ),
    ]

    spawn_civic = layout.node_by_id("spawn_civic_plaza").position
    spawn_transit = layout.node_by_id("spawn_transit_forecourt").position
    agents = [
        AgentSpec(
            agent_id="agent_0",
            spawn_slot="civic_plaza_spawn",
            position=(spawn_civic[0], spawn_civic[1], AGENT_Z),
            yaw_deg=0.0,
            private_constraint="I need step-free access and cannot use stairs.",
            private_requirement_keys=["accessible"],
            zone_id="zone_west",
            walk_node_id="spawn_civic_plaza",
        ),
        AgentSpec(
            agent_id="agent_1",
            spawn_slot="transit_forecourt_spawn",
            position=(spawn_transit[0], spawn_transit[1], AGENT_Z),
            yaw_deg=180.0,
            private_constraint="I strongly prefer food or drink and a quiet place.",
            private_requirement_keys=["food_drink", "quiet"],
            zone_id="zone_east",
            walk_node_id="spawn_transit_forecourt",
        ),
    ]

    coarse_map_text = (
        "Coarse map: an Amsterdam-inspired canal market district. The central canal runs north-south "
        "between two promenade streets. On each bank, an outer avenue, an internal merchant lane, and "
        "the canal promenade divide six narrow blocks, producing twelve blocks in a clear 2x6 rhythm. "
        "Only the market bridge and the longer south bridge cross the canal. The west merchant lane is "
        "terminated by the civic clock tower to the north and hospital to the south; the east lane is "
        "terminated by the transit tower and hotel marker. The glass market hall sits northwest of the "
        "main bridge, while the waterside tower marks its east approach. Both agents start at interior "
        "merchant-lane/Market Street decisions with built frontages in every horizontal direction. "
        "Venue status, accessibility, crowding, food/drink, and entrance conditions are not on this map; inspect visually."
    )

    return Scenario(
        scenario_id=f"riverside_market_seed_{seed}",
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
        max_steps=128,
        layout=layout,
    )


def layout_with_bridges_disabled(layout: DistrictLayout | None = None) -> DistrictLayout:
    """Return a copy of the layout with both bridge edges disabled."""

    base = layout if layout is not None else build_district_layout()
    edges = tuple(
        replace(edge, enabled=False) if edge.route_kind == "bridge" else edge
        for edge in base.walk_edges
    )
    return replace(base, walk_edges=edges)
