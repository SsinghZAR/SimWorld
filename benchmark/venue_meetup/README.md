# Venue Meetup

UE-grounded multi-agent benchmark for **social information sharing**: agents must pool private observations to meet at a unique group-feasible venue.

Locomotion is a controlled variable, not the primary DV. Success depends on communication and other-regarding message content under a hidden-profile information structure.

## Research question

Does an agent integrate partner reports (group information), and does it share facts that matter to the partner's private needs (other-regarding message passing)? Outcome scoring rewards meeting at the group-feasible venue; process metrics diagnose *how* agents shared information.

## Information partition and hidden properties

- **Public to agents:** coarse map, candidate venues (ids/locations/types), own private hard requirement (in `main`), ego camera, inbox, known inspect facts.
- **Hidden:** venue ground-truth traits until a successful `INSPECT`; partner constraints (unless ablation); optimum id; soft weights used for scoring.
- **`--info-partition spatial`:** each agent may inspect only venues in its own zone; cross-zone facts must come from communication.
- **`--info-partition none`:** inspect returns traits without zone gating (easier upper bound).
- **`skill_check` (V2):** not implemented; do not expect DnD-style attribute rolls.

## Actions and turns

Agents act in **synchronous turns**. Each turn is one structured action:

| Action | Role |
|--------|------|
| `NAVIGATE` | High-level travel to a venue meeting region |
| `INSPECT` | Reveal structured traits when in range / allowed by partition |
| `COMMUNICATE` | Group or directed messages; optional free-text `message` and/or structured `shared_facts` claims `[{venue_id, trait, value}, ...]` for traits the sender personally inspected |
| `STEP_FORWARD` / `TURN_AROUND` | Optional fine movement |
| `WAIT` | No-op |

## Navigation modes

- **Teleport (default):** `NAVIGATE` places the agent at the venue meeting point. Use for social/reference evaluation.
- **`--walk`:** physical traversal. Prefer **graph-backed layout routes** (sidewalks, crossings, bridges) when the scenario carries a district layout; otherwise fall back to the **legacy obstacle-aware free-space planner** (plaza templates). Report navigation diagnostics separately from social scores.

## Map templates

Three real layouts (use these IDs):

| Template ID | Scale | Venues / structure | Default turn budget |
|-------------|-------|--------------------|---------------------|
| `central_square_v0` | Plaza smoke layout | 4 venues on a ring | 32 |
| `station_quarter_medium_v1` | ~350–500 m district | 8 venues, 4 blocks | 64 |
| `riverside_market_large_v1` | ~700–900 m district | 12 venues, 6+ blocks, canal/rail barrier + bridges | 128 |

**Legacy aliases (do not use as independent maps):** `station_street_v0` and `canal_bridge_v0` only relabel **central-square geometry**. They are not distinct UE layouts.

## Layout authoring model

Medium/large templates are deterministic Python-authored `DistrictLayout` specs: streets, blocks, frontages, landmarks, spawns, and a walk graph. Scenes spawn into the existing SimWorld base map via the asset catalog—no custom UE map assets required for these templates.

## Hidden-profile invariants

With `--hidden-profile` (current generator):

- Exactly one group-feasible optimum.
- No single agent can identify it from own-zone inspects + own constraint alone.
- Partner-only decisive facts exist (other-regarding sharing is measurable).

**Current limitation:** hidden-profile generation supports **2 agents** only. Do not claim 3-agent hidden-profile support.

## Scoring vs process metrics

- **Outcome (`episode_score`):** venue quality x arrival / convergence. Primary success signal for meetups.
- **Process metrics (`social_metrics.json`):**
  - **Structured `shared_facts` (exact):** claims attached to messages are evaluated exactly against the sender's first-hand inspection records and scenario decision facts. Reported categories include first-hand supported, unsupported, contradictory, duplicate/redundant, and partner-relevant claims (plus exact sharing completeness when applicable).
  - **Legacy free-text (heuristic):** venue/trait co-mention extraction for sharing completeness, other-regarding ratio, redundancy, necessity (`must_pool`), and related diagnostics. These remain approximate and are retained for comparison with the structured path.

## Ablations

`main`, `no_communication`, `no_coarse_map`, `shared_constraints`, `full_shared_information` (see `ablations.py`). Sweep with `--ablation-matrix`.

## Setup

1. Install the SimWorld Python package / project deps in your environment.
2. Launch the Unreal Engine SimWorld backend with UnrealCV reachable (default `127.0.0.1:9000`) for live runs.
3. For VLM policies, configure the provider credentials in the environment expected by the MiniMax/OpenAI-compatible client (defaults: `--provider minimax`, `--model MiniMax-M3`). Do not put secrets on the CLI.

## Commands

From the repository root:

```bash
# Dry-run (no UE): scenario artifacts + fake scores
python -m benchmark.venue_meetup.run_venue_eval \
  --dry-run --hidden-profile --info-partition spatial --seeds 7

# Live scripted smoke (UE required)
python -m benchmark.venue_meetup.run_venue_eval \
  --hidden-profile --info-partition spatial --policy scripted --seeds 7

# Live VLM
python -m benchmark.venue_meetup.run_venue_eval \
  --hidden-profile --info-partition spatial --policy minimax --seeds 7

# Walk mode (graph-backed when layout present)
python -m benchmark.venue_meetup.run_venue_eval \
  --template-id station_quarter_medium_v1 --walk --policy scripted --seeds 7
```

Useful flags: `--template-id`, `--seeds`, `--num-agents`, `--ablation` / `--ablation-matrix`, `--output-dir`, `--run-name`, `--save-video`, `--cinematic`.

## Artifact layout

```text
runs/venue_meetup/<run_name>/
  run_manifest.json          # written before cases execute
  summary.json
  <template_id>/<scenario_id>/<ablation>/
    scenario_hidden.json
    scenario_public.json
    coarse_map / metadata / trajectory / model_responses / social_metrics / videos...
```

`run_manifest.json` records schema version, timestamp, git commit (or `null`), sanitized CLI args (no secrets), discoverable runtime/package versions, template/scenario ids, seeds, agent counts, ablations, and navigation mode.

## Trajectory renderer

```bash
python -m benchmark.venue_meetup.render_trajectory --run-dir <case_dir>
# map only:
python -m benchmark.venue_meetup.render_trajectory --run-dir <case_dir> --map-only
```

## Known limitations

- Ego camera is **third-person** (agent sees its own back).
- UE preflight / live validation has **not** been claimed for the new medium/large maps in this documentation; treat UE smoke as an operator checklist, not a guaranteed gate.
- Default provider/model assumptions favor MiniMax-compatible APIs.
- V2 `skill_check` partition mode is absent.
- Hidden profile is 2-agent only.
- Structured `shared_facts` metrics are exact claim checks; legacy free-text social metrics remain heuristic.
