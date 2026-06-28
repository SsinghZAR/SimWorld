"""Central-square fixed scenario for the first Venue Meetup smoke test.

Scale note: SimWorld's ``BP_Building`` assets are full city blocks with large,
asymmetric collision volumes (empirically ~23 m of collision reach on the
entrance side and ~45 m on the bulk side, measured from the actor pivot). Agents
now use real engine locomotion and are physically blocked by that collision, so
the layout is sized for it: each venue sits at its own ring distance chosen from
its measured centre-facing reach, so an agent walking out from the plaza always
halts ~``MEET_STANDOFF`` from the centre at the building facade. Arrival and
convergence regions are centred on that plaza-side meeting point (not the
building pivot). All coordinates are Unreal centimetres; metadata is hidden.
"""

from __future__ import annotations

from benchmark.venue_meetup.building_catalog import MASK_COLORS, asset_path, building_description
from benchmark.venue_meetup.scenario import AgentSpec, Entrance, Landmark, PropSpec, Region, Requirement, Scenario, Venue, VenueProperties

MAP_TEMPLATE_ID = "central_square_v0"

# These base buildings have large, asymmetric collision volumes whose reach toward
# the plaza centre depends on both the asset and its yaw (measured per venue with
# engine locomotion). To keep the task uniform, each venue sits at its own ring
# distance = MEET_STANDOFF + (its measured centre-facing reach), so an agent
# walking out from the plaza always halts ~MEET_STANDOFF from the centre at the
# building facade. Arrival/convergence regions are centred on that plaza-side
# meeting point (not the building pivot) and are the same size for every venue.
MEET_STANDOFF = 2200.0
MEET_RADIUS = 1200.0
# Centre-facing collision reach per venue (cm), padded slightly above measurement.
REACH_W, REACH_E, REACH_S, REACH_N = 1500.0, 3400.0, 4500.0, 5200.0
RING_W = MEET_STANDOFF + REACH_W
RING_E = MEET_STANDOFF + REACH_E
RING_S = MEET_STANDOFF + REACH_S
RING_N = MEET_STANDOFF + REACH_N
# Landmarks are background reference buildings only; keep them far out on the
# diagonals so their collision never intrudes on the plaza or approach corridors.
LANDMARK_AXIS = 5200.0

# Agents start out by the two diagonal landmarks (clock tower NW, hospital SE)
# rather than in the plaza, so they begin ~98 m apart, must localize off their
# landmark, and have to cross the square to converge. Both spots were validated
# live (agent settles in the open and has a navigable path to the plaza/venues):
#  - agent_0 sits ~38 m S of the clock tower, just WEST of the west-cafe venue.
#    The cafe (which opens east toward the plaza) walls off the direct NW->plaza
#    line, so agent_0 must walk south down the open corridor west of the cafe,
#    round it through the SW, then approach venues from the plaza side. It faces
#    due south (yaw -90) down that open corridor; bearings in the obs let it
#    orient. (Validated: scripted south-around route reaches the cafe front.)
#  - agent_1 sits ~30 m NW of the hospital on the open SE diagonal with a clear
#    straight shot into the plaza, then a cross-square walk to the west cafe.
SPAWN_CLOCK_TOWER = (-5600.0, 1400.0, 150.0)
SPAWN_HOSPITAL = (3079.0, -3079.0, 150.0)


def _region_toward(ux: float, uy: float) -> Region:
    """Arrival/convergence region centred on the plaza-side meeting point."""

    return Region(center=(ux * MEET_STANDOFF, uy * MEET_STANDOFF), radius=MEET_RADIUS)


def _venue_prop(venue_id: str, index: int, asset_key: str, position: tuple[float, float, float], semantic: str) -> PropSpec:
    return PropSpec(prop_id=f"{venue_id}_prop_{index}", asset_key=asset_key, position=position, semantic=semantic)


def build_fixed_scenario(seed: int = 7) -> Scenario:
    """Return the deterministic smoke-test scenario."""

    venues = [
        Venue(
            venue_id="venue_red_awning_cafe",
            slot_id="west_venue",
            venue_type="cafe",
            asset_key="BP_Building_05_C",
            asset_path=asset_path("BP_Building_05_C"),
            position=(-RING_W, 0.0, 0.0),
            yaw_deg=180.0,  # storefront faces +x toward the plaza
            region=_region_toward(-1.0, 0.0),
            mask_color_rgb=MASK_COLORS["venue_0"],
            properties=VenueProperties(
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
            entrances=[
                Entrance(
                    entrance_id="red_awning_front",
                    status="accessible",
                    position=(-MEET_STANDOFF, -60.0, 0.0),
                    yaw_deg=180.0,
                    visible_cues=["clear doorway", "outdoor table", "drink prop"],
                )
            ],
            props=[
                _venue_prop("venue_red_awning_cafe", 0, "BP_Table_C", (-MEET_STANDOFF + 200.0, -900.0, 0.0), "outdoor seating"),
                _venue_prop("venue_red_awning_cafe", 1, "BP_Soda1_C", (-MEET_STANDOFF + 250.0, -650.0, 0.0), "food/drink cue"),
            ],
            visual_summary=building_description("BP_Building_05_C"),
        ),
        Venue(
            venue_id="venue_blue_market",
            slot_id="east_venue",
            venue_type="shop",
            asset_key="BP_Building_25_C",
            asset_path=asset_path("BP_Building_25_C"),
            position=(RING_E, 0.0, 0.0),
            yaw_deg=0.0,  # storefront faces -x toward the plaza
            region=_region_toward(1.0, 0.0),
            mask_color_rgb=MASK_COLORS["venue_1"],
            properties=VenueProperties(
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
            entrances=[
                Entrance(
                    entrance_id="blue_market_front",
                    status="stairs_only",
                    position=(MEET_STANDOFF, 60.0, 0.0),
                    yaw_deg=0.0,
                    visible_cues=["busy storefront", "narrow/raised entrance"],
                )
            ],
            props=[
                _venue_prop("venue_blue_market", 0, "BP_Can_C", (MEET_STANDOFF - 200.0, 900.0, 0.0), "food/drink cue"),
                _venue_prop("venue_blue_market", 1, "BP_Trash_bin_a_C", (MEET_STANDOFF - 250.0, 650.0, 0.0), "busy street cue"),
            ],
            visual_summary=building_description("BP_Building_25_C"),
        ),
        Venue(
            venue_id="venue_brown_hotel",
            slot_id="south_venue",
            venue_type="hotel_lobby",
            asset_key="BP_Building_95_C",
            asset_path=asset_path("BP_Building_95_C"),
            position=(0.0, -RING_S, 0.0),
            yaw_deg=-90.0,  # storefront faces +y toward the plaza
            region=_region_toward(0.0, -1.0),
            mask_color_rgb=MASK_COLORS["venue_2"],
            properties=VenueProperties(
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
            entrances=[
                Entrance(
                    entrance_id="hotel_lobby_front",
                    status="accessible",
                    position=(-60.0, -MEET_STANDOFF, 0.0),
                    yaw_deg=-90.0,
                    visible_cues=["open arched lobby", "quiet facade"],
                )
            ],
            props=[],
            visual_summary=building_description("BP_Building_95_C"),
        ),
        Venue(
            venue_id="venue_closed_hall",
            slot_id="north_venue",
            venue_type="public_square",
            asset_key="BP_Building_99_C",
            asset_path=asset_path("BP_Building_99_C"),
            position=(0.0, RING_N, 0.0),
            yaw_deg=90.0,  # storefront faces -y toward the plaza
            region=_region_toward(0.0, 1.0),
            mask_color_rgb=MASK_COLORS["venue_3"],
            properties=VenueProperties(
                open=False,
                reachable=True,
                capacity=12,
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=0.6,
                crowding_score=0.3,
                near_transit=True,
            ),
            entrances=[
                Entrance(
                    entrance_id="hall_blocked_front",
                    status="blocked",
                    position=(0.0, MEET_STANDOFF, 0.0),
                    yaw_deg=90.0,
                    visible_cues=["road blockers across entrance", "closed-looking frontage"],
                )
            ],
            props=[
                _venue_prop("venue_closed_hall", 0, "RoadBlocker_C", (-300.0, MEET_STANDOFF - 200.0, 0.0), "blocked/closed entrance"),
                _venue_prop("venue_closed_hall", 1, "RoadCone_C", (300.0, MEET_STANDOFF - 200.0, 0.0), "blocked/closed entrance"),
            ],
            visual_summary=building_description("BP_Building_99_C"),
        ),
    ]

    landmarks = [
        Landmark(
            landmark_id="landmark_clock_tower",
            slot_id="northwest_landmark",
            landmark_type="clock_tower",
            asset_key="BP_Building_20_C",
            asset_path=asset_path("BP_Building_20_C"),
            position=(-LANDMARK_AXIS, LANDMARK_AXIS, 0.0),
            yaw_deg=-45.0,
            mask_color_rgb=MASK_COLORS["landmark_0"],
            visual_summary=building_description("BP_Building_20_C"),
        ),
        Landmark(
            landmark_id="landmark_hospital_ramp",
            slot_id="southeast_landmark",
            landmark_type="hospital",
            asset_key="BP_Building_87_C",
            asset_path=asset_path("BP_Building_87_C"),
            position=(LANDMARK_AXIS, -LANDMARK_AXIS, 0.0),
            yaw_deg=135.0,
            mask_color_rgb=MASK_COLORS["landmark_1"],
            visual_summary=building_description("BP_Building_87_C"),
        ),
    ]

    agents = [
        AgentSpec(
            agent_id="agent_0",
            spawn_slot="clock_tower_spawn",
            position=SPAWN_CLOCK_TOWER,  # by the NW clock-tower landmark, west of the cafe
            yaw_deg=-90.0,  # face south down the open corridor west of the cafe
            private_constraint="I need step-free access and cannot use stairs.",
            private_requirement_keys=["accessible"],
        ),
        AgentSpec(
            agent_id="agent_1",
            spawn_slot="hospital_spawn",
            position=SPAWN_HOSPITAL,  # beside the SE hospital landmark
            yaw_deg=135.0,  # face NW toward the plaza
            private_constraint="I strongly prefer food or drink and a quiet place.",
            private_requirement_keys=["food_drink", "quiet"],
        ),
    ]

    coarse_map_text = (
        "Coarse map: four candidate venues surround a central square, roughly 20 m of walking out in each cardinal direction "
        "(each is a solid building you stop in front of). "
        "West: red-awning storefront. East: blue mini-market. South: brown hotel lobby. North: columned hall with a grand staircase. "
        "Major landmarks: a clock-tower building to the northwest and a hospital-ramp building to the southeast. "
        "Venue status, accessibility, crowding, food/drink, and entrance conditions are not on this map; inspect visually."
    )

    return Scenario(
        scenario_id=f"central_square_seed_{seed}",
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
        max_steps=32,
    )
