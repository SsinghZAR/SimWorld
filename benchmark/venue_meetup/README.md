# Venue Meetup

Venue Meetup is a two-agent, UE-grounded benchmark for social information
sharing. Each hidden-profile episode has one group-feasible venue. Agents must
inspect what they can reach, communicate useful information, and converge on
that venue. Movement is controlled so the reference result measures social
reasoning; `--walk` is a separate physical-navigation diagnostic.

## Research question and measurement

The main question is whether an agent integrates a partner's report and shares
facts that matter to the partner's private need (other-regarding communication),
rather than merely selecting a venue that suits itself. The primary outcome is
convergence at the unique group-feasible venue. `social_metrics.json` reports
the process: exact structured-claim checks, first-hand sharing completeness,
partner relevance, redundancy, and hidden-profile necessity. Free-text metrics
remain legacy heuristics; they are not treated as exact fact reports.

The benchmark does not make low-level locomotion the social dependent variable.
Default `NAVIGATE` uses teleport placement for a fast, controlled social
reference. `--walk` uses the authored sidewalk/crossing graph (or the legacy
obstacle planner for central-square scenarios) and reports route diagnostics
separately from social scores.

## Four information layers

| Layer | What an agent can see | What remains evaluator-only |
| --- | --- | --- |
| Public navigation | Candidate identity summaries (`venue_id`, type, slot, visual summary), coarse map, landmarks, roster, pose, and action feedback | Venue properties, regions, asset paths, mask colours, and internal diagnostics |
| First-hand evidence | A successful `INSPECT` adds deterministic, ordered, readable sentences to `known_venue_evidence` and the latest inspect result | The canonical trait dictionary used to generate those sentences |
| Hidden truth and constraints | In `main`, only the acting agent's own private constraint is shown | Canonical decision facts, the partner's constraint, soft weights, and the optimum; `full_information` intentionally exposes the facts and all group constraints as an upper bound |
| Communication | Only `COMMUNICATE` (`choice=4`) delivers optional text to other agents | `shared_facts` claims are evaluator annotations, never a recipient-visible side channel |

The default information partition is `spatial`: an agent may inspect only
venues in its assigned zone. `none` is an optional upper-bound partition
override. The selected condition supplies the default; `--info-partition` can
override it explicitly.

### Evidence, hidden truth, and the inspection gate

Evidence is not hidden truth. Evidence is generated from the canonical facts in
a stable vocabulary (`open`, `reachable`, `accessible`, `food_drink`, `quiet`,
`uncrowded`, `shelter`, `near_transit`, and `capacity`), but only the evidence
sentences reach a normal agent. A successful typed inspection requires, in order:

1. a permitted target under the information partition;
2. proximity: the agent is inside the venue meeting region and within the
   configured inspect range (5,000 cm by default);
3. sufficient pixels for the target in the current object-mask camera view
   (`inspect_min_mask_pixels`, 50 by default in the live environment).

The current orientation is checked before any refocus. Refocusing occurs only
after the mask threshold succeeds. Failed checks return readable failure text
and no facts. Successful evidence is deterministic and readable; canonical
facts and mask/proximity diagnostics stay in evaluator records.

Communication uses the same boundary. Text attached to `WAIT`, `INSPECT`,
`NAVIGATE`, or movement actions is never delivered. On `COMMUNICATE`, the
recipient serialization contains sender, recipients, text, step, and delivery
metadata, but no claims. The evaluator transcript retains the exact
`shared_facts` claims and checks them against the sender's first-hand canonical
inspection records.

## Actions and navigation

Agents act synchronously once per turn:

| Choice | Action |
| ---: | --- |
| 0 | `WAIT` |
| 1 | `STEP_FORWARD` (optional fine movement) |
| 2 | `TURN_AROUND` |
| 3 | `INSPECT` a permitted, nearby, currently visible venue |
| 4 | `COMMUNICATE` optional text and/or evaluator-only claims |
| 5 | `NAVIGATE` to a venue's meeting point |

`NAVIGATE` defaults to teleport mode. Add `--walk` to physically traverse the
layout graph, with route, bridge, stall, replan, and distance diagnostics in the
action log. A live walk smoke is not a social-evaluation score.

## Templates

| Template | Scale and structure | Default turns |
| --- | --- | ---: |
| `central_square_v0` | Four-venue plaza smoke layout | 32 |
| `station_quarter_medium_v1` | Cerdà-inspired station grid, 8 venues across 4 chamfered blocks | 64 |
| `riverside_market_large_v1` | Canal market, 12 venues across 12 narrow blocks and 2 bridges | 128 |
| `busy_street_playtest_v0` | Compact 24-facade city block with 4 entries, a courtyard loop, and 12 uniquely inspectable venues | 96 |
| `connected_blocks_playtest_v0` | Three dense 24-facade blocks joined by 2 portal-aligned pedestrian alleys | 192 |
| `rosebank_grid_3x3_v0` | Compact scale tier: 9 blocks, 4 venues, 3 landmarks, streets and alleys | 96 |
| `rosebank_grid_5x5_v0` | Intermediate scale tier: 25 blocks, 8 venues, 6 landmarks and garden anchors | 160 |
| `rosebank_grid_7x7_v0` | Large scale tier: 49 blocks, 12 venues, 6 landmarks and residential edges | 256 |
| `rosebank_grid_9x9_v0` | Rosebank-inspired 9x9 mixed-use district with 81 blocks, hierarchical streets, 37 alley axes, 36 venues, and 6 landmarks | 384 |

`busy_street_playtest_v0` is the fast visual/interaction sandbox. Six measured
facades line each side of one dense block, with centred north, east, south, and
west portals connecting the perimeter sidewalk to an internal courtyard loop.
Its six restaurants each have a unique identity, entrance, meeting region, mask
colour, and public visual summary. It also adds the `bookshop`, `bar`, and
`skyscraper_lobby` venue types; the remaining residences are solid scene
buildings rather than venue candidates.

`connected_blocks_playtest_v0` composes three copies of that proven block
primitive into West Market, Central Arcade, and East Tower blocks. The facing
east/west portals share explicit alley edges, giving both agents a continuous
collision-aware route through two connectors and three courtyard/perimeter
graphs. Venue ids and object-mask colors remain unique across all 36 candidates.

The Rosebank grid family is generated by one parameterized planner. The 3x3,
5x5, and 7x7 tiers pair increasing city extent with 4, 8, and 12 balanced
hidden-profile venues, respectively; each retains the Oxford/Tyrwhitt street
hierarchy, distinct landmark anchors, graph-backed sidewalks, and optional
mid-block shortcuts. These three tiers are the canonical scale comparison.

`rosebank_grid_9x9_v0` remains the maximum-size visual/navigation stress map: an
approximately 684 m square district with 81 blocks and 36 uniquely inspectable
venues. Its four garden blocks, 37 alley axes, and six landmark buildings remain
unchanged. The shared visual road planner scales carriageway slabs, raised
sidewalks, hierarchy-aware center markings, and Oxford/Tyrwhitt zebra crossings
to every grid size without changing pedestrian collision. This family is a
Rosebank-inspired benchmark abstraction, not a cadastral copy of Johannesburg.

The geometry and landmark rationale, including the municipal and heritage
references behind each layout, is documented in
[`city_design_references.md`](city_design_references.md).

The aliases `station_street_v0` and `canal_bridge_v0` are legacy names for
central-square geometry, not independent city maps. Hidden-profile generation
currently supports exactly two agents; each episode still has one optimum and
partner-only decisive facts.

## Canonical POC conditions

`--ablation` selects one condition and defaults to `main`. `--ablation-matrix`
runs the following four conditions in this order. `--prompt-mode` and
`--info-partition` are optional overrides; otherwise each condition's values in
the table are used.

The `minimal` prompt is the shared task/action contract without a strategy
scaffold. `cooperative` appends one explicit addendum asking the agent to
disclose needs, report teammate-useful evidence, pool observations, and
coordinate before convergence; it does not change environment flags.

| Condition | Prompt mode | Information/environment flags | Measurement role |
| --- | --- | --- | --- |
| `main` | `minimal` | `spatial`; communication enabled; private constraints | Social reference: hidden, partitioned evidence and ordinary task instructions |
| `no_communication` | `minimal` | `spatial`; `no_communication=true` | Egocentric floor: the real env step path suppresses message delivery |
| `full_information` | `minimal` | `spatial`; `full_shared_information=true`, `shared_constraints=true` | Upper bound: all canonical decision facts and all group constraints are visible |
| `cooperative_scaffold` | `cooperative` | Same environment kwargs as `main` | Prompt-only intervention: adds an explicit cooperative strategy scaffold |

### Legacy names

These names remain accepted for archived commands and artifacts, but are not the
canonical matrix: `main`, `no_communication`, `no_coarse_map`,
`shared_constraints`, and `full_shared_information`. The legacy full-information
name is distinct from the canonical `full_information` condition name; use the
canonical names for new runs.

## Setup and commands

Read the repository [PROJECT_HANDOFF.md](../../PROJECT_HANDOFF.md) for the
supported Unreal package, map, socket, and smoke-test details. For a live run,
launch the packaged backend on the empty map (do not substitute `demo_1` or
`demo_2`: their built-in geometry does not match the authored Venue Meetup
coordinates or walk graph):

```powershell
$simWorldUe = Start-Process `
  -FilePath 'D:\side_projects\simworld_ue\Windows\SimWorld.exe' `
  -ArgumentList '/Game/Maps/empty.umap' `
  -WorkingDirectory 'D:\side_projects\simworld_ue\Windows' `
  -PassThru
Test-NetConnection 127.0.0.1 -Port 9000
```

Use the repository virtual environment from PowerShell. A dry run requires no
Unreal Engine, model, key, or network:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
  --dry-run --hidden-profile --info-partition spatial `
  --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 `
  --ablation main --run-name dry_main
```

Run the complete canonical matrix offline:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
  --dry-run --hidden-profile --info-partition spatial `
  --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 `
  --ablation-matrix --run-name dry_matrix
```

The exact single-condition dry-run commands are:

```powershell
# main
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval --dry-run --hidden-profile --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 --ablation main
# no communication
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval --dry-run --hidden-profile --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 --ablation no_communication
# full information
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval --dry-run --hidden-profile --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 --ablation full_information
# cooperative scaffold
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval --dry-run --hidden-profile --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 --ablation cooperative_scaffold
```

With the UE package running at `127.0.0.1:9000`, run a scripted social smoke
(teleport/reference mode):

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
  --hidden-profile --info-partition spatial --policy scripted `
  --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 `
  --ablation main --output-dir runs\venue_meetup\social_reference
```

Run a one-step physical traversal smoke independently:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
  --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 `
  --policy nav_smoke --walk --max-steps 1 --speed 5000 --resolution 640x360 `
  --output-dir runs\venue_meetup\live_smoke
```

Generate the canonical 3x3/5x5/7x7 hidden-profile scale matrix offline with:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
  --dry-run --small-eval --hidden-profile --num-agents 2 `
  --seeds 7 --ablation-matrix --run-name scaled_dry_matrix
```

Capture overview, street, alley, object-mask, coarse-map, and overhead evidence
for each scale with:

```powershell
foreach ($size in 3, 5, 7) {
  .\.venv\Scripts\python.exe -m benchmark.venue_meetup.preview_rosebank_grid `
    --grid-size $size
}
```

The 2026-09-05 live scale preflight passed an opposite-gateway walk on every
tier with zero replans and inspected all 24 candidates successfully. Generated
evidence is recorded in `PROJECT_HANDOFF.md`.

For the retained 9x9 stress grid, use the packaged humanoid's stable 1,000 cm/s
speed. The two agents start at opposite ends of Tyrwhitt and walk to the same
west-side venue, so the east agent crosses almost the full district:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
  --template-id rosebank_grid_9x9_v0 --seeds 17 --num-agents 2 `
  --policy nav_smoke --walk --max-steps 1 --speed 1000 `
  --resolution 640x360 --output-dir runs\venue_meetup `
  --run-name rosebank_grid_9x9_walk
```

Regenerate its evidence with:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.preview_rosebank_grid `
  --grid-size 9
```

Each preview writes a fitted, centered 1,200 x 1,200 true-overhead frame named
`rosebank_grid_<N>x<N>_district_top_down.png`.

For a later VLM run, configure the provider credential in the environment (do
not put it in a command or commit it) and select the MiniMax-compatible policy:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
  --hidden-profile --info-partition spatial --policy minimax `
  --provider minimax --model MiniMax-M3 `
  --template-id rosebank_grid_3x3_v0 --seeds 7 --num-agents 2 `
  --ablation main --output-dir runs\venue_meetup\minimax_reference
```

This is the focused two-agent POC setting: nine city blocks and four candidate
venues. Use `--template-id riverside_market_large_v1` to exercise the larger
legacy layout.
`--small-eval` selects the 3x3, 5x5, and 7x7 scale family. It supports
`--hidden-profile` when `--num-agents 2`; the two-agent limit remains explicit.

## Artifacts and reproducibility

Every run root contains a manifest and aggregate summary:

```text
runs/venue_meetup/<run_name>/
  run_manifest.json
  summary.json
```

A dry run writes deterministic scenario artifacts for each case (but no policy
trajectory or model output):

```text
  <template_id>/<scenario_id>/<condition>/
    scenario_hidden.json
    scenario_public.json
    metadata.json
    <scenario_id>_coarse_map.png
```

A live policy run adds:

```text
  <template_id>/<scenario_id>/<condition>/
    trajectory.json
    trajectory_minimap.png      # final paths over the public coarse map
    trajectory_minimap.mp4      # paths revealed turn by turn
    social_metrics.json
    model_responses.jsonl
    agent_*.mp4                 # only with --save-video
```

The run manifest records the resolved `condition`/`condition_id`, `prompt_mode`,
`info_partition`, and `navigation_mode`, along with template/scenario/seed and
sanitized CLI arguments. Per-case metadata records the condition and prompt
information; its sanitized `args.walk` flag identifies walk versus teleport
mode (there is no separate case-level navigation key). Secrets are removed.
`scenario_public.json` omits hidden properties; `scenario_hidden.json` is for
evaluator use. Movement coordinates are recorded only under evaluator-facing
`info.movement_paths_internal`; they are not added to either agent's observation.
Physical movement is a solid line and abstracted `NAVIGATE` teleport movement
is dashed. The numbered dots identify the turn on which movement completed.
The minimap PNG and MP4 are generated automatically after every live episode.
Regenerate them together with the separate collision-diagnostic top-down view
(or render just the authored map) with:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.render_trajectory `
  --run-dir <absolute-case-directory>
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.render_trajectory `
  --run-dir <absolute-case-directory> --map-only
```

## Known limitations

- The ego camera is third-person, so the agent sees its own back; use the
  world-frame compass/coarse map for orientation.
- Hidden-profile generation is deliberately limited to two agents.
- Live object-mask calibration remains package-specific. The compact block,
  connected blocks, and Rosebank 9x9 playtests have reproducible live smoke
  coverage; the 3x3/5x5/7x7 tiers require their own live preflight after layout
  changes, and new assets or package revisions still require recalibration.
- `skill_check`/DnD perception is not implemented.
- Teleport/reference social scores and `--walk` navigation diagnostics must be
  reported separately. The current physical renderer uses authored shells and
  graph routes. Rosebank road dressing reuses measured, stable packaged
  blueprints because the package's native road blueprint is incompatible; it
  is not a replacement for a custom-authored UE street kit.
- Legacy free-text social metrics are heuristic. Exact sharing metrics require
  structured claims attached to `COMMUNICATE` and first-hand inspection records.
