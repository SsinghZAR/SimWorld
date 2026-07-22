# Walk Graph Rewrite: Public Routes and Venue Access

## Goal

Replace the current "frontage as walk-node" topology with two linked layers:

```text
spawn -> sidewalk -> crossing -> bridge -> sidewalk
                                      |
                               venue access path
                                      |
                              meeting region
```

The route graph contains only public, physically traversable space. A venue
frontage describes how its meeting region attaches to that graph; it is never a
through-route vertex. This prevents graph paths from cutting through building
collision, which the live preflight found on both authored districts.

## Non-negotiable invariants

1. `WalkNode`s represent public walking locations only: spawn, sidewalk,
   intersection, crossing, or bridge deck.
2. A `Frontage` references exactly one public `approach_node_id` and owns a
   short `access_path` from that node to its `meeting_region`.
3. A `WalkEdge` owns an authored polyline. The planner follows its points, not
   a direct line between distant node positions.
4. `frontage_id` must not be a `WalkNode.node_id`; therefore it cannot appear
   in a graph route.
5. The target venue's access path is appended only after graph planning has
   reached its approach node.
6. Legacy layout JSON still loads: missing access fields use a compatibility
   fallback only for old stored scenarios. New templates must use the new
   fields explicitly.
7. Existing `VenueMeetupEnv.reset()`, `step()`, `NAVIGATE`, scenario compact
   payloads, and central-square obstacle-A* fallback remain usable.

## Implementation stages

### 1. Schema and serialization

Update `layout.py`:

- Add `"sidewalk"` to `WalkNode.kind`.
- Add `WalkEdge.waypoints: tuple[Point2D, ...] = ()` and a helper returning
  its full polyline (`start node`, intermediate waypoints, `end node`).
- Add `Frontage.approach_node_id: str | None` and
  `Frontage.access_path: tuple[Point2D, ...] = ()`. New authored frontages
  must end their access path at their meeting-region centre.
- Preserve JSON `compact()` / `from_dict()` compatibility for missing fields.

### 2. Route planning

Update `navigation.py` so `plan_layout_route()`:

- resolves a venue to its frontage;
- finds a graph route from the agent node to `frontage.approach_node_id`;
- flattens edge polylines in route order, then appends the frontage access path;
- returns diagnostics distinguishing graph node path, edge ids, graph distance,
  access distance, total planned distance, and bridge usage;
- never relies on a frontage id being a walk node.

Keep the legacy obstacle-A* path for scenarios without a layout.

### 3. Template re-authoring

Rewrite both authored templates:

- `station_quarter_medium_v1`: exterior sidewalk nodes around all four blocks,
  crossings at public intersections, and one leaf access path per venue.
- `riverside_market_large_v1`: bank-side sidewalks, bridge-deck nodes, legal
  bridge edges only, and leaf access paths off the sidewalks.

Use enough intermediate polyline points to follow streets around block
footprints. Do not use building pivots, facade positions, or meeting-region
centres as public through-route nodes.

### 4. Validation

Extend `template_validation.py` with data-level checks:

- all new frontages have a real approach node of a public walk kind;
- no walk node reuses a frontage id;
- access paths end at their meeting region;
- every edge polyline is finite and non-zero;
- every spawn reaches every frontage approach node;
- graph paths do not require any other venue's access path.

Use authored block footprints for static geometry screening. Treat this as an
early warning only; live UE collision traversal remains authoritative.

### 5. Environment integration

Update layout-backed walk navigation in `venue_env.py` to consume the planned
polyline exactly. Preserve route diagnostics and record the access segment
separately. A collision on an authored edge remains a failure; do not hide bad
map topology with an automatic free-space detour.

### 6. Tests

Update existing tests rather than adding duplicate micro-tests. Keep coverage
for:

- schema round-trip and legacy-field fallback;
- a graph route ending at an approach node then access path;
- frontage ids excluded from graph node paths;
- bridge routing and disconnected-bank behavior;
- every spawn-to-venue route in both authored templates;
- central-square legacy fallback.

Tests must assert behavior and topology, not arbitrary exact node counts or
private implementation cache fields.

### 7. Acceptance gate

After offline tests pass, run live UE preflight on both maps:

- reset/spawn/nonblank camera/own-zone and cross-zone inspection/convergence;
- traverse every authored graph edge start-to-end;
- traverse every spawn-to-venue route;
- save JSON and screenshots for any failed edge.

Do not claim the rewrite is complete until the live preflight passes.
