# SimWorld / Venue Meetup — Project Handoff

This is the operational entry point for future agents working in
`D:\side_projects\SimWorld`. It makes the local Unreal backend and the Venue
Meetup benchmark repeatable without rediscovering the machine setup.

## What this repository contains

- `simworld/` is the Python client for the SimWorld UnrealCV environment.
- `benchmark/venue_meetup/` is a two-agent hidden-profile social benchmark:
  agents privately inspect different venue information, exchange relevant
  facts, and meet at the unique group-feasible venue.
- `station_quarter_medium_v1` and `riverside_market_large_v1` are deterministic
  Python-authored districts. Their routes, blocks, venues, and landmarks are
  described in the benchmark; `district_scene.py` renders building shells into
  the runtime scene.
- `busy_street_playtest_v0` is the dense single-block visual primitive;
  `connected_blocks_playtest_v0` joins three varied copies through two
  collision-aware pedestrian alleys and exposes 36 unique venue identities.
- `rosebank_grid_9x9_v0` is the Rosebank-inspired city playtest: 81 mixed-zone
  blocks, Oxford/Tyrwhitt primary axes, 37 alley axes, 36 venues, 6 landmarks,
  and a graph that spans approximately 684 m.
- `runs/` holds generated evaluation artifacts and is intentionally ignored by
  Git.

Read `benchmark/venue_meetup/notes.md` before changing task semantics and
`benchmark/venue_meetup/cleanup.md` before changing the district renderer or
layouts.

## Local Unreal setup

The Unreal backend is **outside the Git repository**:

```text
Executable:        D:\side_projects\simworld_ue\Windows\SimWorld.exe
Working directory: D:\side_projects\simworld_ue\Windows
UnrealCV endpoint: 127.0.0.1:9000
```

The Venue Meetup medium and large districts are runtime overlays authored for
the otherwise empty packaged map. Launch them with:

```text
/Game/Maps/empty.umap
```

Do not substitute `demo_1` or `demo_2` for a Venue Meetup live run just because
they look more like a finished city: their built-in geometry does not match the
authored venue coordinates or walk graph. Those maps are valid for the generic
profiling workflow in `benchmark/README.md`, not for the district templates.

## Start the backend

From PowerShell, start the package and retain the returned process object if
you plan to stop it afterwards:

```powershell
$simWorldUeRoot = 'D:\side_projects\simworld_ue\Windows'
$simWorldUe = Start-Process `
  -FilePath "$simWorldUeRoot\SimWorld.exe" `
  -ArgumentList '/Game/Maps/empty.umap' `
  -WorkingDirectory $simWorldUeRoot `
  -PassThru
```

Wait until the package is ready, then check the UnrealCV socket:

```powershell
Test-NetConnection 127.0.0.1 -Port 9000
```

`TcpTestSucceeded : True` means a live benchmark can connect. Start only one
backend per port. If this command fails, wait for UE to finish loading; if it
continues to fail, inspect the package's launch log before changing Python.

When the run is complete, stop **only the process started above**:

```powershell
Stop-Process -Id $simWorldUe.Id
```

If the PowerShell session that owns `$simWorldUe` has closed, first identify the
correct process with `Get-Process -Name SimWorld`, then stop the intended PID.

## Python environment

Use the repository virtual environment rather than assuming a globally active
Python installation:

```powershell
.\.venv\Scripts\python.exe --version
```

All following commands are run from `D:\side_projects\SimWorld`.

## Fast validation order

1. Validate deterministic scenario generation without Unreal:

   ```powershell
   .\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
     --dry-run --hidden-profile --info-partition spatial `
     --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2
   ```

2. Run the focused test suite. Use an in-repository temporary directory; the
   default shared pytest temp/cache location has previously produced Windows
   permission errors:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest tests\venue_meetup -q `
     --basetemp D:\side_projects\SimWorld\runs\pytest_tmp
   ```

3. With Unreal running, perform a short physical traversal smoke. This checks
   the live renderer, actor lifecycle, graph-backed walk route, and UnrealCV
   connection; it is not a social-evaluation result:

   ```powershell
   .\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
     --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 `
     --policy nav_smoke --walk --max-steps 1 --speed 5000 `
     --resolution 640x360 --output-dir runs\venue_meetup\live_smoke
   ```

4. Run a real social episode separately. Teleport navigation is deliberate for
   the social reference condition; do not describe its movement as a physical
   navigation validation:

   ```powershell
   .\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
     --template-id station_quarter_medium_v1 --seeds 7 --num-agents 2 `
     --hidden-profile --info-partition spatial --policy scripted `
     --output-dir runs\venue_meetup\social_reference
   ```

5. Repeat either live command with `--template-id riverside_market_large_v1`
   to exercise the larger city-block layout. Begin with `nav_smoke`; it has a
   128-turn default budget for full episodes and is slower than the medium map.

For the compact connected-block playtest, regenerate the overview, alley, route,
mask, and coarse-map evidence with:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.preview_connected_blocks
```

Then validate an end-to-end physical route and one venue identity per block:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
  --template-id connected_blocks_playtest_v0 --seeds 17 --num-agents 2 `
  --policy nav_smoke --walk --max-steps 1 --speed 1000 `
  --resolution 640x360 --output-dir runs\venue_meetup `
  --run-name connected_blocks_alley_walk

.\.venv\Scripts\python.exe -m benchmark.venue_meetup.smoke_busy_street `
  --template-id connected_blocks_playtest_v0 `
  --venue-id venue_green_awning_bistro `
  --venue-id venue_green_awning_bistro_2 `
  --venue-id venue_green_awning_bistro_3 `
  --output runs\venue_meetup\connected_blocks_three_venue_smoke.json
```

For the Rosebank-inspired 9x9 district, regenerate all visual evidence, then
run the opposite-gateway walk and three-class interaction smoke with:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.preview_rosebank_grid

.\.venv\Scripts\python.exe -m benchmark.venue_meetup.run_venue_eval `
  --template-id rosebank_grid_9x9_v0 --seeds 17 --num-agents 2 `
  --policy nav_smoke --walk --max-steps 1 --speed 1000 `
  --resolution 640x360 --output-dir runs\venue_meetup `
  --run-name rosebank_grid_9x9_walk

.\.venv\Scripts\python.exe -m benchmark.venue_meetup.smoke_busy_street `
  --template-id rosebank_grid_9x9_v0 `
  --venue-id venue_red_awning_bistro `
  --venue-id venue_g5_bookshop `
  --venue-id venue_f2_skyscraper_lobby `
  --output runs\venue_meetup\rosebank_grid_interaction_smoke.json
```

The 2026-09-05 live walk passed with `NAVIGATE_OK` for both agents and zero
replans: 286.7 m planned from the west gateway and 590.7 m from the east. The
three interaction targets all returned `INSPECT_OK`. Generated screenshots and
the 1,400 px coarse map live under
`runs/city_landmark_redesign/rosebank_grid_9x9/`.
The road-equipped preview additionally writes the centered 1,200 x 1,200
`rosebank_grid_district_top_down.png` frame.
The subsequent road-equipped walk passed the same two routes with
`NAVIGATE_OK`, zero replans, and an episode score of 1.0.

For a VLM run, use `--policy minimax` and configure provider credentials in the
environment. Never pass secrets on the command line or commit them.

## Inspect results and recordings

Each evaluation writes a self-contained case beneath:

```text
runs/venue_meetup/<run-name>/<template>/<scenario>/<ablation>/
```

Important files are `run_manifest.json`, `summary.json`, `metadata.json`,
`trajectory.json`, `social_metrics.json`, and (when requested) per-agent MP4
files. Add `--save-video` to an evaluation command to record cameras. Use
`--cinematic` only when intentionally collecting a slower, denser walking clip.

Render a top-down trajectory after a run with:

```powershell
.\.venv\Scripts\python.exe -m benchmark.venue_meetup.render_trajectory `
  --run-dir <absolute case directory>
```

Add `--map-only` to inspect just the authored layout.

## Runtime constraints worth preserving

- Hidden-profile generation supports exactly **two agents**. Do not claim
  three-agent hidden-profile support.
- `NAVIGATE` teleports by default. `--walk` is the physical, graph-backed mode;
  report navigation and social outcomes separately.
- The visible districts currently use packaged building shells inside authored
  block footprints. The base map remains the empty packaged ground plane.
- Do not use catalogue road assets (`BP_Road_C` / `BP_Road1_C`) in this packaged
  empty-map renderer. They are catalogued but unavailable/incompatible in the
  installed package and previously caused a UE crash when spawned.
- `rosebank_roads.py` is the supported fallback: it plans 152 non-colliding
  visual actors from three measured, known-stable blueprints. They provide 20
  carriageways, 40 sidewalks, 12 center markings, and 80 Oxford/Tyrwhitt zebra
  bars. Do not replace them with `BP_Road1` without a separately re-authored UE
  package and live crash preflight.
- `DistrictSceneRenderer` intentionally uses raw UnrealCV spawning with
  collision disabled. `Communicator.spawn_object` enables collision by default
  and is appropriate for venues, interactive props, and the Rosebank planner's
  explicitly clearance-checked massing. Generic decorative district shells
  remain non-colliding.
- Keep the Rosebank physical smoke at `--speed 1000`. At 5,000 cm/s the packaged
  controller can overshoot short graph waypoints and trigger a false stall even
  on a clear street.
- Large scenes use bounded UnrealCV request batches once the static actor count
  reaches 96. Preserve the sequential compatibility path for small templates
  and test adapters.
- The `GEN_BP_` actor prefix is required so scene reset removes all generated
  actors. Preserve that naming convention.
- The packaged humanoid blueprint may register multiple FusionCam sensors in
  reverse spawn order. `SceneBuilder` matches cameras to pawns by live spatial
  proximity after spawning; do not restore an index-only camera assumption.

## Current implementation status

The current playtest stack includes the four-entry dense block, three-block
alley district, and Rosebank-inspired 9x9 district, with unique interactive
venues, graph-backed walking, alley-clear frontage placement, a visible road
hierarchy, and repeatable live visual evidence. Verify the current branch and
working tree before making further edits:

```powershell
git status --short
git log -1 --oneline
```

The next substantive visual upgrade is a custom-authored UE street kit with
materials, curbs, traffic signals, and vehicle lanes. It needs a re-authoring
pass for coordinate alignment and live physics rather than the incompatible
packaged `BP_Road1` blueprint.
