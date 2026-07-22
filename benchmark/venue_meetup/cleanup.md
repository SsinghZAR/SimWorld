# Venue Meetup: Cleanup and City-Scale Expansion Plan

## Purpose

This document is a step-by-step implementation handoff for taking Venue Meetup
from a working V1 proof of concept to a maintainable benchmark with two genuine,
city-scale layouts:

- `station_quarter_medium_v1`: a medium, authored station district.
- `riverside_market_large_v1`: a large, authored multi-block neighbourhood.

The goal is **not** to scatter more building actors over a larger coordinate
range. Each layout must represent an intentional urban district: streets,
intersections, blocks, frontages, sidewalks, landmarks, entrance points, and
route constraints all come from one fixed layout specification.

The benchmark's primary dependent variables remain social:

- Can agents pool private observations to identify the unique group-feasible
  venue?
- Do they communicate information that is useful to their teammates, including
  facts outside their own private needs?

Walking and visual embodiment should remain measurable, but must not silently
become the main cause of failure in the social benchmark. For this reason every
new template must support both:

1. **Social/reference mode**: high-level `NAVIGATE` with abstracted travel.
2. **Embodied/stress mode**: `NAVIGATE --walk` using real route traversal and
   separate navigation diagnostics.

## Current State and Gaps

### What is already working

- Scenario data model, asset spawning, coarse-map rendering, group chat,
  structured inspection facts, scoring, logs, and social-process metrics.
- A 2-agent hidden-profile generator that guarantees one group-feasible optimum
  for a 4-venue, 2-zone case.
- `info_partition=spatial`, private constraints, and the V0 ablation registry.
- Abstracted `NAVIGATE`, plus an optional physical walking implementation.
- A recorded successful live VLM episode for the 2-agent hidden-profile case.

### Important limitations to fix

1. **There is only one physical layout.**
   `station_street.py` and `canal_bridge.py` currently call the central-square
   builder and only replace descriptive text/template IDs. They are not distinct
   UE geometry, spawn, collision, or navigation layouts. Do not count them as
   separate map templates in future experiments.

2. **`venue_env.py` is doing too much.**
   It owns scene construction, agent lifecycle, camera capture, observation
   assembly, action dispatch, inspection, teleport navigation, physical walking,
   lighting, recording, and scoring integration. Its size makes changes risky.

3. **Current physical navigation is too geometric for a city district.**
   It plans direct routes around circular building keep-out regions. That is
   adequate for the plaza proof of concept, but it cannot express sidewalks,
   crossings, bridges, alleyways, one-way detours, or a canal/rail barrier.

4. **Hidden profiles are fixed to 2 agents and 4 venues.**
   The current generator correctly rejects other shapes. This is safe, but
   prevents medium/large information structures and 3-agent experiments.

5. **Free-text social metrics are approximate.**
   Metrics infer facts by matching venue aliases and trait keywords. They are
   useful diagnostics, but cannot reliably distinguish a true fact report from
   an unsupported, negated, or ambiguous claim.

6. **There is no benchmark-specific public documentation or automated test
   suite.**
   Existing validation is primarily dry runs, generator assertions, and manual
   live UE runs.

## Non-Goals

Do not add these in this cleanup unless explicitly requested:

- Full procedural city generation.
- New UE maps or custom UE assets.
- Deception, private messaging, resident/local agents, or mixed motives.
- Replacing the structured V1 inspection reveal with VLM visual trait
  recognition.
- Treating physical navigation success as a replacement for social metrics.

The new districts are deterministic Python-authored scene templates that spawn
into the existing SimWorld base/empty map, using the existing asset catalog.

## Target Architecture

### 1. Configuration and template data

Introduce typed dataclasses instead of passing a long set of loosely related
arguments through the CLI and `VenueMeetupEnv`.

Suggested new files:

```text
benchmark/venue_meetup/
  config.py                # VenueMeetupConfig, EvaluationConfig
  layout.py                # city blocks, street graph, parcels, utilities
  template_validation.py   # pure validation + optional UE preflight
  templates/
    shared.py              # common scenario/layout construction helpers
    station_quarter.py     # medium authored layout
    riverside_market.py    # large authored layout
```

Suggested config split:

```python
@dataclass(frozen=True)
class VenueMeetupConfig:
    resolution: tuple[int, int] = (640, 360)
    viewmode: str = "lit"
    navigate_mode: Literal["teleport", "walk"] = "teleport"
    info_partition: Literal["none", "spatial"] = "none"
    no_communication: bool = False
    no_coarse_map: bool = False
    full_shared_information: bool = False
    shared_constraints: bool = False
    # camera, recording, and movement tuning follow

@dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...]
    agent_counts: tuple[int, ...]
    policy: str
    model: str
    ablations: tuple[str, ...]
    output_dir: Path
```

The CLI should construct these objects once and persist their serialized form in
every run manifest. Keep existing flags initially for compatibility; they simply
populate the dataclasses.

### 2. Layout model: authored city structure

Add an explicit layout model. It must be sufficient to spawn the scene, render
the coarse map, validate routes, and describe walkable navigation.

```python
@dataclass(frozen=True)
class StreetSegment:
    street_id: str
    start: Point2D
    end: Point2D
    width_cm: float
    sidewalk_width_cm: float

@dataclass(frozen=True)
class Intersection:
    intersection_id: str
    position: Point2D
    landmark_id: str | None = None

@dataclass(frozen=True)
class Block:
    block_id: str
    footprint: Polygon2D
    frontage_ids: list[str]

@dataclass(frozen=True)
class Frontage:
    frontage_id: str
    block_id: str
    position: Point3D
    yaw_deg: float
    entrance_point: Point3D
    meeting_region: Region
    venue_slot_id: str | None

@dataclass(frozen=True)
class WalkNode:
    node_id: str
    position: Point2D
    kind: Literal["spawn", "intersection", "crossing", "frontage", "bridge"]

@dataclass(frozen=True)
class WalkEdge:
    start_node_id: str
    end_node_id: str
    length_cm: float
    enabled: bool = True
    route_kind: Literal["sidewalk", "crossing", "bridge", "alley"] = "sidewalk"
```

Use a graph for planned walking routes. A graph is preferable to the current
straight-line-plus-circle-avoidance approach because a city map needs policy
meaningful constraints: a bridge can be the only canal crossing, an alley can be
closed, and a venue can be reached from its correct frontage rather than by
crossing through a building.

Keep geometric collision checks as a second-level safety check, not as the
source of route topology.

### 3. Environment responsibility split

Refactor incrementally; do not rewrite the live environment in one change.

| Component | Responsibility | Extract from current code |
|---|---|---|
| `scene_builder.py` | Spawn/clear static actors, props, agents, lighting | scene setup/reset helpers |
| `observations.py` | Per-agent public/private observation construction | `_build_observations`, target cues |
| `actions.py` | Dispatch/validate WAIT, STEP, TURN, INSPECT, COMMUNICATE, NAVIGATE | action sections of `VenueMeetupEnv` |
| `walk_graph.py` or `navigation.py` | Graph route planning and route diagnostics | current walking planner |
| `venue_env.py` | Episode state, action ordering, termination, and orchestration only | retain a thin public environment API |

Preserve the public `VenueMeetupEnv.reset()`, `step()`, and `disconnect()` API.
This lets existing runner code and stored artifact conventions survive the
refactor.

### 4. Structured fact-report side channel

Continue to allow natural-language broadcast messages. Add an optional,
machine-readable field to the communication action:

```json
{
  "choice": 4,
  "message": "The east market has food and is not step-free.",
  "shared_facts": [
    {"venue_id": "venue_east_market", "trait": "food_drink", "value": true},
    {"venue_id": "venue_east_market", "trait": "accessible", "value": false}
  ]
}
```

Rules:

- The environment records claims but does not reveal whether they are true to
  the recipient.
- Scoring compares claims to the sender's `revealed_facts`, not global hidden
  truth. This avoids credit for a lucky or hallucinated claim.
- Preserve text-based heuristic metrics for older trajectories, but mark them
  as legacy/approximate.
- Report: valid first-hand fact share, unsupported claim count, fact novelty,
  partner-relevant share, and cross-zone optimum evidence.

Update the model prompt to make `shared_facts` encouraged but optional. Do not
make this a required action-schema field until provider structured-output
reliability has been checked.

## Map Specifications

All dimensions below are target *district extents*, not building-pivot spacing.
Every actor position must be confirmed against the actual collision geometry of
the selected UE asset. Unreal units are centimetres.

### A. `station_quarter_medium_v1`

#### Intent

A station-adjacent commercial district. It should feel like two real urban
blocks separated by a through street, with a station forecourt at one end and a
service alley as a secondary route.

#### Scale and composition

- Approximate district footprint: **350–500 m across**.
- 8 venue frontages across 4 blocks; do not place them in a circle.
- 4–5 strong landmarks: station entrance/forecourt, clock tower, bus stop,
  hotel corner, market canopy.
- 2 primary streets, 1 cross street, 1 service alley, and marked crossings.
- 2 agents initially; four inspectable venues per spatial zone.
- Default budget: **64 synchronous policy turns**.
- Recommended physical route length from a spawn to the opposite-zone venue:
  150–300 m.

#### Proposed spatial structure

```text
                 north

  clock tower ─── Market Street ─── station forecourt
       │        [NW block] [NE block]        │
       │          venues      venues         │
       │───────────cross street──────────────│
       │        [SW block] [SE block]        │
  hotel corner ─── service alley ─── bus stop

                 south
```

- The forecourt and a major intersection are public orientation anchors.
- Each zone should be one side of the district, defined by inspect permissions,
  not a simple Euclidean half-plane.
- The optimum can occur in either zone across seeds.
- At least two venues in each zone must be plausible under local/private
  information, so agents cannot inspect one storefront then immediately stop.

#### Acceptance criteria

- There are four distinct blocks, not copies of the central-square ring.
- UE walking follows sidewalks/crossings and never tries to cut through a block.
- Every venue has a validated meeting region on its accessible public frontage.
- Each spawn can reach every meeting region through the graph and live UE walk.
- Coarse-map roads/intersections agree with the scene graph.

### B. `riverside_market_large_v1`

#### Intent

A larger civic/market neighbourhood divided by a canal or rail-like barrier.
A bridge is the obvious fast crossing; a longer secondary crossing supports
route choice and a genuine navigation stress test.

#### Scale and composition

- Approximate district footprint: **700–900 m across**.
- 12–14 venue frontages across at least 6 blocks.
- 6–8 landmarks: bridge, market square, transit entrance, civic building,
  waterside tower, statue/fountain, hospital/accessible landmark, bus stop.
- 3 street corridors plus a market square and two legal crossings of the barrier.
- Begin with 2 agents and 6 venues per spatial zone. Design the topology so it
  can later support 3 agents with three information zones.
- Default budget: **128 synchronous policy turns**.
- Recommended cross-district physical route length: 350–650 m.

#### Proposed spatial structure

```text
  north bank / civic side

  civic block ── market square ── transit block
       │              │                 │
  north street ───── main bridge ─── waterfront street
       │              │                 │
  ======= canal or rail barrier (not directly traversable) =======
       │              │                 │
  south street ─── secondary bridge ── market lane
       │              │                 │
  residential block ─ food market ─ hotel / station block

  south bank / market side
```

- Do not fake the barrier only in the map text. Spawn a continuous visual and
  collision barrier, leaving validated gaps at bridges/crossings.
- The graph must include no edge across the barrier except bridges/crossings.
- At least one late-stage route decision should be observable in walk-mode
  diagnostics (bridge chosen, detour length, replan reason); it must not alter
  social scoring in reference mode.

#### Acceptance criteria

- It has a genuinely different street graph and scene silhouette from the
  medium template.
- The primary and secondary crossings are both real in UE walking mode.
- A direct cross-barrier path is impossible in the route graph and blocked in
  the engine.
- Each venue is attached to a named block/frontage and has an exterior meeting
  point that does not overlap building collision.
- The coarse map makes the barrier and crossings visible without revealing venue
  traits or the optimum.

## Generalized Hidden-Profile Generator

Do not extend the old 4-venue generator by adding arbitrary random properties.
Use a deliberate, testable information design.

### Required invariants for every generated social episode

1. Exactly one venue satisfies all group hard constraints.
2. No individual agent's inspectable facts plus its own private constraint can
   identify that venue uniquely.
3. At least one partner-only decisive fact must be observed by a different
   agent than the agent who needs it.
4. Each zone has at least one locally attractive decoy.
5. Every candidate in the intended inspection set is reachable from its zone's
   spawn in the layout graph.
6. The optimum is reachable by every agent.
7. All properties used in structural assertions are hidden from observations
   before a successful inspection or a communication report.

### Phased generalization

Implement in this order:

1. **Two agents, 8 venues** for the medium map.
   Use four venues per zone: one optimum/decoy pair in the optimum agent's zone
   and two partner-zone traps, plus distractors that require inspection to rule
   out. Keep two cross-agent hard needs initially (`accessible`, `food_drink`).

2. **Two agents, 12 venues** for the large map.
   Add multiple decoy classes, e.g. open-but-no-food, accessible-but-noisy,
   food-but-inaccessible, and blocked/closed. Ensure the number of extra venues
   increases evidence gathering rather than merely adding obvious bad options.

3. **Three agents** only after the first two tiers pass.
   Add three zones and a third distinct private hard need. Require a unique
   optimum whose decisive facts are distributed across agents. Do not claim a
   3-agent result until no single pair can solve it either.

Expose these values through a `HiddenProfileSpec` rather than hard-coded
`num_agents == 2` / `len(venues) == 4` checks.

## Step-by-Step Implementation Plan

### Phase 0 — Preserve the baseline

1. Record the current commit, CLI command, and successful `hp_vlm_64_s7`
   artifacts in the new benchmark README.
2. Do not change current `central_square_v0` behaviour during the first
   refactor phases.
3. Add a baseline dry-run check that asserts the scenario serializes and
   existing V0 scores remain unchanged.

**Exit gate:** the original hidden-profile dry run and scripted smoke command
still produce the same scenario shape and score.

### Phase 1 — Add tests before changing behaviour

Create `tests/venue_meetup/` (or follow the repository's adopted test location)
with pure-Python tests for:

- `score_venue`, `episode_score`, and convergence attribution.
- 4-venue hidden-profile invariants across at least 100 deterministic seeds.
- spatial partition inspect permissions.
- message routing/ablations.
- coarse map rendering with no hidden properties serialized publicly.
- social metric handling for structured claims and legacy free text.
- graph reachability and no-cross-barrier path assertions.

Use a fake/stub communicator for environment-level action tests. Keep live UE
checks as explicit smoke/preflight commands rather than making unit tests depend
on a local Unreal server.

**Exit gate:** `pytest` succeeds without UE or an API key.

### Phase 2 — Extract configuration and layout primitives

1. Add `config.py` and construct configs in `run_venue_eval.py`.
2. Add `layout.py` dataclasses, JSON serialization, graph shortest-path helper,
   and drawing primitives.
3. Extend `Scenario` with a versioned optional `layout` field. Keep loading old
   scenario JSON valid by defaulting this field to `None`.
4. Update `coarse_map.py` to render streets, blocks, crossings, bridge/barrier,
   frontages, landmarks, and spawn areas when layout data is available.
5. Add `template_validation.py` for all pure geometric/graph checks.

**Exit gate:** central square can be represented in the new model without
changing its visual scene or public observation schema.

### Phase 3 — Thin the environment safely

1. Extract scene setup/spawning into `scene_builder.py` without changing actor
   names or UnrealCV calls.
2. Extract observation assembly into `observations.py` and snapshot-test the
   public keys visible to each agent under every ablation.
3. Route `NAVIGATE --walk` through the layout graph where one is present; retain
   the current planner as a legacy fallback for `layout is None`.
4. Extract action execution/inspection into an action handler module or small
   action classes.
5. Keep `VenueMeetupEnv` as the only public orchestration class.

**Exit gate:** central-square live smoke, `--walk` smoke, and previous artifact
schema all continue to work.

### Phase 4 — Implement `station_quarter_medium_v1`

1. Author the fixed block/street/sidewalk graph in
   `templates/station_quarter.py`.
2. Add eight named frontage slots and fixed visual assets from
   `building_catalog.py`.
3. Use the same layout data for actor placement, meeting regions, graph nodes,
   and coarse map rendering. Do not duplicate coordinates in separate modules.
4. Define zone membership as explicit frontage IDs; do not infer it from nearest
   spawn location.
5. Add `station_quarter_medium_v1` to `TEMPLATE_BUILDERS`.
6. Run the pure template validator, then the UE preflight described below.
7. Add the two-agent, 8-venue hidden-profile specification and tests.

**Exit gate:** the medium map passes a 10-seed scripted social run in teleport
mode and a 3-seed physical-navigation smoke in walk mode.

### Phase 5 — Implement `riverside_market_large_v1`

1. Author the barrier, block network, market square, bridges, streets, and
   frontages in `templates/riverside_market.py`.
2. Create barrier collision/visual actors and explicit graph crossing edges.
3. Add 12–14 venue slots with placements that maintain usable sidewalk clearance
   around the actual asset collision volumes.
4. Add the large hidden-profile specification and tests.
5. Add a stress-run preset with a 128-turn limit, motion recording disabled by
   default, and detailed route diagnostics enabled.
6. Run walking preflight from every spawn to every venue, including both bridge
   routes and at least one forced detour.

**Exit gate:** the large map passes route preflight, 10-seed teleport social
smokes, and repeated walk-mode no-stuck tests.

### Phase 6 — Generalize information structures and metrics

1. Replace the hard-coded hidden-profile implementation with `HiddenProfileSpec`.
2. Add structured `shared_facts` parsing to `VenueAgentTurn`, comms logs, model
   prompts, and social metrics.
3. Preserve free-text broadcast semantics; structured claims supplement, not
   replace, conversational language.
4. Add exact metrics based on first-hand revealed facts and structured claims.
5. Keep the old heuristic metrics in output as `legacy_*` fields for comparison.
6. Define uptake correctly: first report a non-causal arrival proxy; only label
   it causal after implementing a deterministic counterfactual replay that
   removes one message while holding all policy responses fixed or using a
   suitable replay policy.

**Exit gate:** synthetic selfish, cooperative, redundant, unsupported, and
contradictory transcripts are separated by tests.

### Phase 7 — Documentation and reproducibility

Add these documents:

```text
benchmark/venue_meetup/README.md
docs/source/benchmarks/venue_meetup.rst   # add to Sphinx toctree
benchmark/venue_meetup/cleanup.md         # this implementation plan
```

The README must cover:

- Research question and what is/is not measured.
- Observation contract and what remains hidden.
- Actions, turn semantics, and the difference between teleport and walk modes.
- Scenario/layout schema and map authoring process.
- Hidden-profile invariants and ablations.
- Scoring and process metric definitions, including which are exact/heuristic.
- Setup, UE-server launch, dry-run, scripted smoke, VLM run, and walk-mode
  commands.
- Artifact directory schema and how to render a trajectory.
- Known limitations: third-person camera, structured inspection facts, current
  provider assumptions, and V2 `skill_check` status.

Add a `run_manifest.json` to every evaluation root with:

- benchmark/schema version;
- git commit (when available);
- CLI/config serialization;
- template IDs, scenario IDs, seeds, and ablations;
- model/provider configuration with no secrets;
- timestamps and package versions.

**Exit gate:** a new contributor can reproduce a dry run and UE scripted smoke
using only the README.

### Phase 8 — Evaluation matrix

Run and report the following in order. Do not interpret results until all
preflight failures are fixed or clearly categorized.

1. **Template smoke suite**
   - central, medium, large;
   - scripted policy;
   - teleport and walk;
   - 3 seeds each.

2. **Reference social evaluation**
   - central (4 venues), medium (8), large (12+);
   - teleport navigation;
   - hidden profile + spatial partition;
   - 10 seeds per template;
   - 2 agents initially.

3. **Ablation suite**
   - main;
   - no communication;
   - shared constraints;
   - full shared information;
   - no coarse map;
   - same seeds/scenarios across conditions.

4. **Embodied stress evaluation**
   - same selected cases in `--walk` mode;
   - report social score and navigation diagnostics separately;
   - do not pool these results with reference social-mode scores.

5. **Three-agent evaluation**
   - only after Phase 6's 3-agent generator invariants and unit tests are done.

## UE Preflight Checklist

Run a per-template preflight utility before VLM calls. It should emit a JSON
report and fail loudly on required checks.

- UE backend reachable and expected map loaded.
- Every configured asset path exists/spawns.
- Every venue actor receives a distinct object-mask colour.
- Every spawn settles in open space.
- Every meeting region is on a valid public frontage and outside the building's
  collision envelope.
- Every graph edge intended for walking can be traversed in UE.
- No graph edge crosses a blocked building or barrier.
- Every spawn can reach every venue meeting region.
- Each bridge/crossing behaves as represented in the graph.
- Camera frames are nonblank after spawn, navigation, and inspect.
- Spatial partition denies cross-zone inspection and permits own-zone inspection.
- Convergence detection agrees with the actual final actor positions.

Save this report beside the scenario artifacts and record its status in the run
manifest.

## Compatibility and Migration Rules

- Keep `central_square_v0` runnable until the medium and large templates have
  passed their validation gates.
- Do not overwrite existing run artifacts or change their meaning silently.
- Version scenario JSON and output schemas when adding `layout`, structured
  claims, or manifest fields.
- Older JSON should load with defaults wherever possible.
- Use clear deprecation warnings for `station_street_v0` and
  `canal_bridge_v0`; either replace them with real implementations or remove
  them from the advertised evaluation matrix.
- Do not remove free-text messages or legacy metrics until downstream analysis
  has migrated.

## Definition of Done

This cleanup is complete when all of the following are true:

1. `station_quarter_medium_v1` and `riverside_market_large_v1` are visibly and
   topologically distinct, authored city districts with fixed blocks and street
   graphs.
2. Each layout is validated both geometrically and in a live UE preflight.
3. A generalized hidden-profile generator supports the venue counts used by both
   maps and asserts its information-sharing invariants.
4. `VenueMeetupEnv` has a focused orchestration role and configuration/template
   data are typed and documented.
5. Walk-mode uses graph-constrained, city-like routes for layout-backed maps.
6. Exact structured-claim social metrics coexist with legacy text heuristics.
7. Unit tests, smoke commands, map-authoring documentation, and reproducible
   run manifests exist.
8. The benchmark reports separate social-reference and embodied-stress results.

