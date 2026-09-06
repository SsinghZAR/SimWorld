# Better embodied Venue Meetup: exploration experiment

Status: design and implementation plan only; the changes below are not yet
implemented. This branch initially adds only this document.

Baseline: `16c46a3` (`Add targeted timed Venue Meetup protocol`), preserved on
`exp/venue-meetup-poc-contract`. Experiment branch:
`codex/better-embodied-exploration`.

## 1. Purpose and evidence

Make the city relevant to discovering venues, choosing routes, finding entrances,
and communicating spatial knowledge. Keep reliable motor control, but stop the
controller from making the agent's exploration decisions.

The current targeted protocol is a useful social-coordination baseline, not yet
a strong test of unfamiliar-city navigation:

- All candidate venue identities, descriptions, directions, distances, and
  route-duration estimates are supplied from the initial observation.
- `NAVIGATE(venue_id)` plans a complete route and automatically walks it.
- Inspection requires physical proximity and source visibility; agents cannot
  inspect the whole city from their starting position.
- Consequently, most decisions concern which known venue to query next. A graph
  with hidden node attributes and travel costs approximates the core task.

In the final seed-7 MiniMax 5x5 run, the agents inspected five of eight candidates
across 25 blocks. Agent 1 checked E4 Pub and E3 Hotel Lobby before going to the
already-listed D2 Skyscraper Lobby. D2's decisive meeting-area evidence arrived
at 17:45:30; its confirmation message arrived at 17:46. Agent 0 had already
started toward D2 at 17:44:30, without confirmed suitability. They met at 17:48,
with 12 minutes remaining. Both agents used destination-level navigation and
neither used manual stepping or turning. This is one diagnostic run, not a
difficulty calibration or proof that images are unnecessary.

Local evidence is retained under the ignored directory
`runs/venue_meetup/targeted_v1_minimax_walk_seed7_final/` (chat, model observations,
actions, and movement replay). It is not part of this document's Git commit.

## 2. Preserve the existing benchmark

Introduce an explicitly opt-in exploration mode. Do not silently change the
current targeted or legacy action/observation contracts, defaults, archived
scenario IDs, or result interpretation. CLI names below are design concepts,
not runnable options that already exist.

The original rationale in [notes.md](benchmark/venue_meetup/notes.md) deliberately
automates locomotion to isolate social behavior. This experiment adds spatial
decision-making as a separate condition, not as a retroactive correction to
that research design. Follow [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) for Unreal
runtime constraints and [the targeted plan](benchmark/venue_meetup/targeted_plan.md)
for the existing timing and evidence contract.

Preserve these rules:

- Two agents with private requirements and a feasible physical meetup objective.
- Time is the only action cost. No intermediate rewards, point deductions,
  efficiency labels, suggested strategies, or correctness scores reach agents.
- The visible clock is independent of API latency and rendering speed.
- One action per agent at a time; busy agents cannot also communicate or inspect.
- Messages and inspection evidence arrive at action completion. Completion at
  closing is valid; unfinished actions do not complete after closing.
- Keep 512-character communication and the current source-specific inspection
  durations initially. Record any later timing calibration explicitly.
- Keep final social outcomes separate from evaluator-only navigation diagnostics.
- Generated screenshots, videos, traces, and logs stay under ignored `runs/`.
  Never put credentials in the plan, scripts, commands, artifacts, or commits.

## 3. Local discovery instead of a global venue directory

### Agent experience

An agent starts with its ego image, compass, clock, requirements, and own memory.
It is not given the complete venue roster, candidate count, exact target
coordinates, global bearings, or shortest-route estimates. A separately labelled
map-assisted condition may supply only an explicitly public main-road schematic,
without venue pins, private discoveries, or unobserved alley connections.

Discovery has distinct stages:

1. **Notice a feature:** a tower silhouette or storefront is visible.
2. **Identify a venue:** a nearby sign is sufficiently visible/readable to name it.
3. **Find an entrance or source:** observe the actual doorway or information panel.
4. **Inspect evidence:** approach and complete the appropriate targeted action.

A visible tower does not reveal its lobby entrance or suitability. A building
mask alone is not sufficient to reveal a venue name when its sign is around a
corner. Remember observations after they leave view; distinguish current
visibility from last-seen information.

### Implementation

- Add a per-agent discovery store, separate from evaluator ground truth. Record
  observed features, identified venues, entrances, traversed connections, and
  first/last observation ticks. Store only justified knowledge, not optimal routes.
- Reuse Unreal object-mask capture for visibility. Define separate feature/sign/
  source targets, view-size thresholds, distance limits, and orientation checks.
  Validate the thresholds at the actual model image resolution; pixel count alone
  does not establish human-like sign readability.
- Build public observations from that store and the current sensor result, never
  by serializing the full scenario and only removing one directory field.
- Audit `candidate_venues`, `navigation.targets`, landmarks, nearby sources,
  map text/images/paths, errors, prompts, and action results. The current nearby
  source list includes invisible points with `visible=false`; previously unseen
  hidden sources must not be listed in the exploration mode.
- Keep internal world coordinates and complete graph connectivity evaluator-only.
  Public location cues should be local/relative. If exact GPS or a map is later
  provided, treat it as an explicit assistance condition.
- Use public handles that do not encode hidden block coordinates or suitability.
  A visible street address may be read normally; an internal ID such as a graph
  node name must not disclose it before observation.
- Distinguish first-hand memory from partner reports. A report can be remembered
  but must not become verified sensor evidence or an automatically correct map pin.
- Reject guessed hidden IDs without revealing whether the corresponding hidden
  object exists. Apply equivalent checks to description-based target resolution.

Initial labels may act as a controlled, visibility-gated perception aid. Clearly
label that condition; it does not establish that the model can visually recognise
unlabelled venues or read signs without assistance.

## 4. Local route choices with reliable walking

### Agent experience

Replace unrestricted destination navigation in exploration mode with local
choices: follow a visible street segment, enter a visible alley, cross at a
visible crossing, approach a visible doorway, or turn to look.

At a junction, the observation might show a street ahead, an alley on the left,
and a clock tower to the right. It must not state that the alley is the shortest
route to an undiscovered venue. The agent selects the next movement, then gets
a new view. Hidden turns and detours are not selected on its behalf.

### Implementation

- Reuse `DistrictLayout` streets, walk nodes, edge polylines, and frontage paths
  internally. They remain the geometry/controller representation, not a globally
  exposed menu of destinations.
- Expose only locally justified movement affordances with opaque handles. An
  affordance identifies a visible corridor or crossing, not its hidden endpoint,
  downstream branches, venue contents, or full-graph path cost.
- Add bounded local movement to the action schema and executor. Working semantic
  names are `FOLLOW_SEGMENT`, `APPROACH`, and `LOOK`; finalize serialized IDs when
  implementing without renumbering legacy actions.
- The controller may steer within the selected corridor and avoid small local
  obstacles. It may not pick another street, cross an unchosen junction, or run a
  global shortest-path fallback when blocked.
- Stop at the next junction, an occluding corner requiring a new choice, a
  blockage, the selected local target, or the configured distance cap. Do not
  automatically rotate toward a venue's inspection panels on arrival.
- First timing proposal: one local movement action occupies one 30-second tick
  and travels at most 40 m, stopping earlier at a decision boundary. Turning or
  a failed attempt also consumes its configured tick; no free scanning or
  hidden-target probing. Report this rounding explicitly and calibrate the
  deadline after testing; many short segments otherwise inflate travel costs.
- Return operational results such as stopped-at-junction, reached-local-target,
  or blocked. Never return an undiscovered alternative route or strategy hint.
- In the initial prototype, revisiting a known venue still requires local route
  choices. Later route-following assistance may use a route explicitly supplied
  by the agent, but must not fill gaps with hidden graph knowledge.

Short automatic steps preserve stable motor control without asking the model to
micromanage Unreal steering. Even this mode has a graph representation internally;
the intended difference is partial, geometry-dependent knowledge and agent-owned
route decisions, not the absence of graphs.

## 5. Entrances and evidence that require spatial search

Separate seeing a building, finding its entrance, reaching an information source,
and entering the final meetup region. Today a frontage approach and a bundle of
nearby panels largely collapse these stages.

Example: a restaurant's menu is beside its main-street door; its step-free route
is on the side; its meeting area is in a courtyard reached through an alley.
Seeing the facade should not reveal all three sources.

Implementation:

- Extend frontage/source authoring to support independently placed entrances,
  signs, local access paths, and courtyard/side-passage meeting regions. The data
  model already has entrance points and access paths; avoid a separate conflicting
  coordinate system.
- Gate each inspection by its own reachable interaction area, range, and current
  visibility rather than requiring every source to occupy the final meeting zone.
- Keep partial evidence merging and source provenance. A menu does not answer
  access or crowding questions. No global or whole-building inspection fallback.
- Start with exterior recesses, side passages, and small courtyards; fully modelled
  interiors and staff dialogue are not prerequisites for this prototype.
- Ensure walls that define navigation actually collide and occlude. Do not assume
  decorative shells provide collision: the existing renderer supports non-colliding
  decoration. Build graph barriers, collision geometry, and rendered walls from a
  shared authored layout and verify them in Unreal.
- If a trait is represented physically, its geometry must agree with the evidence
  (for example, a labelled step-free route cannot lead through unavoidable stairs).
  Until suitable assets exist, label controlled information-panel evidence honestly
  rather than claiming every hidden trait is physically simulated.
- Use supported packaged assets first. Do not introduce the known-incompatible road
  blueprints or require an Unreal package rebuild for the initial experiment.

## 6. Spatial communication without an automatic knowledge channel

Keep the existing free-text communication action. Do not add a second, free map
sharing channel or silently interpret a correct venue name as a navigation command.

Example message:

> I found a suitable lobby. From the clock tower, take the alley beside the green
> bookshop. Its entrance is in the courtyard.

The recipient must recognise the landmarks, choose local movements, and find the
entrance. It can ask which side of the bookshop, but the clarification consumes
time. A message saying an alley exists does not create a visible movement handle
or reveal its ground-truth coordinates.

- Retain separate per-agent discoveries, inboxes, and first-hand evidence.
- Preserve the provenance of communicated claims; do not automatically validate
  messages against hidden truth for the recipient.
- Keep optional `shared_facts` evaluator-only, never recipient-visible.
- Do not expose partner pose, destination, active route, or discoveries unless
  observed under an explicit visibility rule or communicated normally.
- Choose landmark names and visual differences that support useful relative
  descriptions. Avoid a perfect shared address directory as an unintended shortcut.
- Log what the recipient actually knew before and after a message. Arrival after
  a report is not, by itself, evidence that the report caused the route choice.

### Inspection zones and the hidden-profile design

For the first controlled implementation, retain the existing inspection-zone rule
so discovery and navigation can be compared without simultaneously changing every
task mechanic. State that restriction plainly in the prompt; it is not physical
inaccessibility and must not be rendered as an invisible wall.

A later, separately configured experiment can allow either agent to inspect any
physically reachable source. Opposite starts, occlusion, distributed information,
and the deadline then create spatial information asymmetry naturally. This weakens
the old hard-partition assumptions: revalidate feasibility and test communication
benefits empirically. Do not claim that communication is mathematically necessary
just because one jointly feasible venue exists.

## 7. Layout changes: meaningful choices before larger maps

First exercise local discovery and movement on an existing small district. Then
author a compact experimental layout using the same layout primitives, instead of
immediately expanding the grid or adding decorative building counts.

Include a few testable spatial situations:

- A street wall that hides a venue until an agent rounds its corner.
- A tower visible from afar whose entrance cannot be inferred from its silhouette.
- A bookshop landmark beside an alley leading to a courtyard venue.
- A short, locally discoverable connection and a longer main-street alternative.
- A plausible dead end or visibly blocked passage requiring a return decision.
- Multiple candidates on different frontages, so scanning one road does not expose
  every option and one facade inspection does not resolve the whole block.

Render and walk every new connection. Dead ends and barriers must exist in the
image and collision world, not only as disabled graph edges. Avoid arbitrary
unannounced dynamic blockages in the first version. Keep all candidate entrances
reachable and retain a feasible meetup; use seed/layout checks and an informed
reference controller to detect impossible cases before model evaluations.

Not every building needs a task interaction. A building is functionally useful
when it creates an occluding street wall, route choice, landmark, or search area.
Measure that role rather than equating total block count with benchmark difficulty.

## 8. Suggested code boundaries

New module names are proposed, not existing interfaces:

| Responsibility | Reuse or extension |
| --- | --- |
| Per-agent observed knowledge and provenance | New `discovery.py`; separate from canonical scenario facts |
| Mask/geometry visibility and local affordances | New `local_perception.py`; reuse camera and mask capture |
| Bounded movement executor | New `local_navigation.py`; reuse layout polylines and walking primitives |
| Exploration observation/action orchestration | Small opt-in adapter; preserve `targeted_env.py` behavior |
| Entrances, signs, and source placement | Extend `layout.py`, scene authoring, and `interactions.py` |
| Local inspection gates | Extend `interaction_runtime.py` behind the explicit mode |
| Model contract and dispatch | `prompt.py`, `_core/action_space.py`, `_core/policy.py`, `run_venue_eval.py` |
| Evaluator-only discovery and route replay | Extend trajectory recording and `trajectory_minimap.py` |

Share timing, message delivery, evidence serialization, and scoring instead of
copying the targeted environment. Avoid moving all responsibilities into the legacy
environment monolith. Separate pure geometry/knowledge functions from Unreal calls
so the privacy boundary can be tested without a running engine.

## 9. Implementation sequence and acceptance gates

### Phase A: discovery and observation boundary

Implement the private discovery store, visibility rules, and mode-specific public
serializer. Remove global hints only in exploration mode. Add a prompt explaining
local knowledge, remembered observations, and unverified partner reports.

Gate: a venue behind a wall has no public identity, source handle, bearing, or
route estimate. Revealing its tower, then its sign, then its entrance yields only
the appropriate incremental knowledge. Leaving view retains memory, not live
hidden-state updates. Guessed IDs and description aliases do not bypass discovery.

### Phase B: local movement and timing

Implement local affordance selection and a bounded executor using existing walking
primitives. Disable destination-level navigation and global fallback routes in
exploration mode. Preserve independent action scheduling and no runtime scores.

Gate: an agent cannot cross a junction or traverse an unseen branch without a new
choice. Blocking the chosen passage produces a local failure, not an automatic
detour. Movement stays within the selected corridor and timing matches its logged
contract. Existing targeted/legacy tests continue to pass.

### Phase C: compact spatial-search layout

Add the side entrance, courtyard, occluded sign, landmark, and alternative-route
situations from section 7. Move sources independently from final meetup regions.

Gate: live screenshots from both sides of each relevant corner verify visibility;
walking checks verify collision, entrances, and connections. No through-wall
inspection and no automatic arrival alignment that solves entrance/source search.

### Phase D: communication, evaluation, and calibration

Run an observation-constrained scripted smoke before MiniMax. Preserve model
inputs, action traces, chat delivery times, discovery events, screenshots, and
top-down replays. The complete evaluator minimap must not enter agent observations.

Gate: the replay explains when a venue became visible, when it was identified,
which route choices were made, which sources were checked, and what each agent
knew before convergence. Run multiple held-out seeds and layouts, not only seed 7.
Check infrastructure errors separately from genuine task failures.

All phases above remain pending at the document-only commit.

## 10. Comparisons and definition of success

Use distinct controls; they answer different questions:

1. **Existing social reference:** known candidates and global automatic routes.
2. **Local exploration with images:** restricted knowledge and agent-chosen paths.
3. **Matched local text-only control:** identical locally available structured
   information, actions, clock, and memory, but no images. Do not compensate by
   supplying hidden graph topology or rich image descriptions unavailable to the
   image condition. Record what the structured perception aid already reveals.
4. **Full-map assistance:** same exploration task with explicitly disclosed map
   knowledge, to estimate the cost of route uncertainty separately.
5. **No-communication condition:** same layout/needs/seeds and no message delivery,
   to examine the benefit of sharing discoveries and requirements.

A separate graph simulation is useful only if its information availability,
action costs, and inspection rules are documented. A full-information graph is
an upper-bound control, not an information-matched replacement for the embodied
condition.

Preserve final venue-quality/arrival scoring. Compute additional diagnostics
after the episode, never as runtime feedback: distinct discoveries, street/frontage
coverage, entrance-search time, repeated traversal, movement distance, blocked
attempts, decision counts, communication redundancy, and remaining time. Coverage
is explanatory, not a new reward or an instruction to visit every block.

The experiment succeeds operationally when agents must make geometry-dependent
exploration decisions and their observations contain no global shortcuts. It
succeeds as a benchmark only after comparisons show what those decisions measure.
If the matched text-only control performs equally well, report a spatial
information-gathering task with visual redundancy, not demonstrated visual
reasoning. Poorer scores caused by broken movement or impossible deadlines are
not evidence of better benchmark design.

## 11. Deferred scope

Do not bundle full interiors, staff conversations, new dietary/price attributes,
traffic simulation, custom Unreal asset production, or expansion to larger cities
into the first prototype. Keep these as separate follow-ups after local discovery,
route choice, spatial inspection, and the evaluation controls work reliably.
