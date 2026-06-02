# SimWorld Benchmark Learning And Implementation Roadmap

This guide turns the benchmark ideas from the paper notes into a concrete, iterative plan for learning this repository and extending it toward an embodied multi-agent benchmark.

The north star is not just to "make SimWorld run." The goal is to learn the stack deeply enough to answer, with evidence, whether SimWorld can support tasks where agents must communicate, resolve referential ambiguity, build common ground, and act in a shared 3D world.

## Current Repo Snapshot

This checkout is a Python client and tooling layer around a separate Unreal Engine server. The Python side contains:

- `simworld/communicator`: high-level and low-level UnrealCV+ commands.
- `simworld/agent`: lightweight agent state classes such as humanoids, pedestrians, vehicles, and scooters.
- `simworld/local_planner`: high-level language plan parsing plus rule-based or VLM-guided navigation execution.
- `simworld/llm`: OpenAI-compatible LLM wrappers.
- `simworld/citygen`: procedural roads, buildings, elements, routes, and JSON export.
- `simworld/assets_rp`: natural-language asset retrieval and placement over existing assets.
- `simworld/traffic`: rule-based vehicles, pedestrians, traffic signals, and intersection logic.
- `simworld/map`: waypoint graph construction and shortest-path navigation.
- `examples`: notebooks showing the intended usage patterns.

Important current-state observations:

- A polished reusable Gym environment class is not packaged yet. The Gym-like environment appears mostly in `examples/gym_interface_demo.ipynb` and README snippets.
- `BaseLLM` already supports `provider="local"` for OpenAI-compatible local endpoints, but `A2ALLM.generate_instructions()` only routes `openai` and `openrouter`.
- `assets_rp` currently performs natural-language parsing, existing-asset retrieval, and placement JSON generation. This checkout does not contain a Hunyuan3D implementation by name.
- Natural-language scene editing is partly present as asset retrieval and placement, but it is construction-time oriented. `AssetsRetrieverPlacer.generate_assets_manually()` writes JSON that is then rendered by world generation.
- Synchronous and asynchronous UE modes are exposed at the UnrealCV layer through `UnrealCV.set_mode()` and `UnrealCV.tick()`.
- The traffic docs say the traffic system currently runs in asynchronous mode.
- There are no repository tests in this checkout, so every new feature should add its own small smoke test or dry-run validation where possible.

## Learning Order

Work through the repo from the simulator boundary inward. This gives every later agent feature a physical grounding.

1. UnrealCV boundary.
   Read `simworld/communicator/unrealcv.py` and `simworld/communicator/communicator.py`.
   Learn what commands actually cross the Python-to-UE bridge.

2. Agent state and embodiment.
   Read `simworld/agent/base_agent.py`, `simworld/agent/humanoid.py`, `simworld/agent/vehicle.py`, `simworld/agent/pedestrian.py`, and `simworld/agent/scooter.py`.
   Learn which state lives only in Python and which state must be refreshed from UE.

3. Observations and sensors.
   Read `docs/source/components/ue_detail.rst`, `examples/camera.ipynb`, and camera methods in `unrealcv.py`.
   Learn RGB, depth, object mask, camera pose, FOV, and resolution.

4. Map and navigation.
   Read `simworld/map/map.py` and `simworld/local_planner/local_planner.py`.
   Learn how roads become sidewalk/crosswalk graph nodes, how A* paths are computed, and how paths become humanoid movement.

5. LLM and local planner.
   Read `simworld/llm/base_llm.py`, `simworld/llm/a2a_llm.py`, `simworld/local_planner/action_space.py`, and `simworld/local_planner/prompt/prompt.py`.
   Learn where model calls happen, where action schemas are defined, and where the planner boundary should be drawn.

6. City generation and scene construction.
   Read `simworld/citygen/function_call/city_function_call.py`, `simworld/citygen/city/city_generator.py`, and `docs/source/components/citygen.md`.
   Learn how roads, buildings, elements, and routes become JSON.

7. Asset retrieval and placement.
   Read `simworld/assets_rp/AssetsRP.py`, `simworld/assets_rp/utils/input_parser.py`, and `simworld/assets_rp/utils/assets_rp_utils.py`.
   Learn how natural language becomes asset placement JSON.

8. Traffic and background dynamics.
   Read `simworld/traffic/controller/traffic_controller.py`, `simworld/traffic/manager/vehicle_manager.py`, `simworld/traffic/manager/pedestrian_manager.py`, and `docs/source/components/traffic_system.rst`.
   Learn what scales with vehicles, pedestrians, intersections, and polling rate.

## Iteration 0: Make A Reproducible Baseline

Goal: get one known environment running and record what works on the local machine before changing behavior.

Steps:

1. Create a local config from `config/example.yaml`.
2. Set absolute paths for roads, bounding boxes, asset images, vehicle types, and output directories.
3. Start the UE server with one base map.
4. Run `examples/camera.ipynb` to confirm image observations.
5. Run `examples/ue_command.ipynb` to confirm basic spawn, move, rotate, pickup, and social animation commands.
6. Run `examples/gym_interface_demo.ipynb` to confirm the minimal agent loop.
7. Record the map path, UE package version, GPU, resolution, OS, Python version, and exact config values.

Definition of done:

- One humanoid can be spawned.
- Camera observations return usable arrays.
- At least one movement command changes the agent position.
- A minimal `reset()` and `step()` loop can run for 10 steps.
- Failures and workarounds are written down with exact commands and config values.

Recommended repo addition:

- Add `docs/local_setup_log.md` for machine-specific notes that should not be treated as benchmark protocol.

## Iteration 1: Profile UnrealCV+/Gym Speed

Question: is SimWorld practical for online RL, offline trajectory generation, or mostly evaluation?

Files to read:

- `simworld/communicator/unrealcv.py`
- `simworld/communicator/communicator.py`
- `simworld/utils/video_recorder.py`
- `examples/gym_interface_demo.ipynb`
- `docs/source/components/ue_detail.rst`

Implementation plan:

1. Create a small profiler script, for example `scripts/profile_unrealcv_loop.py`.
2. Measure command latency for `get_location`, `get_orientation`, `get_camera_observation`, `humanoid_step_forward`, `humanoid_rotate`, and `tick`.
3. Measure steps per second with no image observations.
4. Measure steps per second with RGB observations at multiple resolutions.
5. Measure steps per second with depth or object masks if needed.
6. Repeat with 1, 2, 5, 10, and 20 humanoids.
7. Repeat in async mode and sync mode.
8. Save results as JSON or CSV under `runs/profiles/`.

Suggested output schema:

```json
{
  "timestamp": "2026-05-18T00:00:00",
  "machine": "local-gaming-pc",
  "ue_map": "/Game/Maps/demo_1",
  "mode": "sync",
  "tick_interval": 0.05,
  "resolution": [640, 480],
  "num_agents": 2,
  "observation": "rgb",
  "steps": 100,
  "mean_step_seconds": 0.0,
  "p95_step_seconds": 0.0,
  "steps_per_second": 0.0
}
```

Definition of done:

- A table shows realistic throughput for structured-only, RGB, depth, and object-mask settings.
- The guide states whether this machine is suitable for online RL, offline rollouts, or evaluation-only usage.
- The benchmark plan names a default observation mode and resolution.

Likely repo modifications:

- Add a reusable timing helper under `simworld/utils/profiling.py`.
- Add `scripts/profile_unrealcv_loop.py`.
- Add `runs/profiles/.gitkeep` and keep large generated CSV/JSON files out of git unless intentionally versioned.

## Iteration 2: Local Model And API Support

Question: can constrained agents run through local LLMs or OpenAI-compatible local endpoints?

Files to read:

- `simworld/llm/base_llm.py`
- `simworld/llm/a2a_llm.py`
- `simworld/assets_rp/utils/input_parser.py`
- `simworld/config/default.yaml`
- `config/example.yaml`
- `examples/local_action_planner.ipynb`

Current finding:

- `BaseLLM` accepts `provider="local"` and a custom `url`.
- `A2ALLM.generate_instructions()` rejects `local` even though its parent supports it.
- `InputParser` bypasses `BaseLLM` and directly creates `OpenAI(api_key=os.getenv("OPENAI_API_KEY"))`.

Implementation plan:

1. Make `A2ALLM` support OpenAI-compatible local endpoints for text-only structured JSON output.
2. Decide how to handle local VLM support separately, because image input and parsed structured output vary by local server.
3. Refactor `InputParser` to accept an LLM client or config values instead of hard-coding OpenAI.
4. Add a small script that sends a constrained parser prompt to a local endpoint and validates the JSON schema.
5. Add example config values for local endpoints.

Example config shape:

```yaml
user:
  llm_model_path: "Qwen2.5-7B-Instruct"
  llm_url: "http://127.0.0.1:8000/v1"
  llm_provider: "local"
```

Definition of done:

- `BaseLLM(..., provider="local", url="http://127.0.0.1:8000/v1")` can complete a text request.
- `A2ALLM(..., provider="local", url=...)` can return a valid `HighLevelActionSpace` for a simple plan.
- Natural-language asset placement parsing can use the same provider abstraction.
- The docs explicitly separate "local text LLM works" from "local VLM works."

Likely repo modifications:

- Update `simworld/llm/a2a_llm.py`.
- Update `simworld/assets_rp/utils/input_parser.py`.
- Add `examples/local_llm_endpoint_demo.py` or a notebook.
- Add docs for OpenAI-compatible local endpoints.

## Iteration 3: LLM/VLM Agent Setup

Question: which observations should agents receive, and when is a VLM actually necessary?

Files to read:

- `simworld/local_planner/local_planner.py`
- `simworld/local_planner/action_space.py`
- `simworld/local_planner/prompt/prompt.py`
- `simworld/map/map.py`
- `simworld/communicator/communicator.py`
- `examples/local_action_planner.ipynb`
- `examples/camera.ipynb`

Implementation plan:

1. Define observation profiles:
   `state_only`, `scene_graph`, `rgb`, `rgb_depth`, and `privileged_debug`.
2. Implement a small observation builder that returns consistent dictionaries for each profile.
3. Add a text-only agent path that uses position, direction, map, task state, and known object metadata.
4. Add a VLM path that includes current image and optionally the previous image.
5. Log every model input and action output during rollouts.
6. Test one task with text-only observations and one task with RGB observations.

Suggested observation schema:

```python
{
    "agent_id": 0,
    "position": [0.0, 0.0],
    "yaw": 90.0,
    "direction": [0.0, 1.0],
    "task": {"goal": "pick up the red cup"},
    "nearby_objects": [],
    "messages": [],
    "image": None
}
```

Definition of done:

- A text-only agent can navigate using structured state and map information.
- A VLM-capable agent can receive images through one consistent code path.
- Rollout logs make it clear what the model saw and why the action was chosen.

Likely repo modifications:

- Add `simworld/observations/` or `simworld/benchmark/observations.py`.
- Add a rollout logger under `simworld/utils/rollout_logging.py`.
- Add an example that compares text-only and RGB observation profiles.

## Iteration 4: Natural-Language Scene Editing Model

Question: which model parses scene edits, and how configurable is it?

Files to read:

- `simworld/assets_rp/AssetsRP.py`
- `simworld/assets_rp/utils/input_parser.py`
- `simworld/assets_rp/utils/assets_rp_utils.py`
- `examples/asset_rp.ipynb`
- `examples/world_generation.ipynb`

Current finding:

- Scene edit parsing is currently specialized to asset retrieval and placement.
- `InputParser` hard-codes OpenAI client construction.
- Placement output is JSON for later world generation, not obviously live in-simulation editing.

Implementation plan:

1. Rename the mental model from "scene editing" to "asset retrieval and placement" unless live editing is added.
2. Refactor parser provider configuration as described in Iteration 2.
3. Add deterministic parsing mode with temperature `0`.
4. Add validation for the four parser keys: `asset_to_place`, `reference_asset`, `relation`, and `surrounding_assets`.
5. Save raw model response, parsed JSON, selected reference asset, selected target asset, and final placement JSON.
6. Create a tiny frozen asset-placement fixture for regression testing.

Definition of done:

- The parser can run with OpenAI, OpenRouter, or an OpenAI-compatible local endpoint for text parsing.
- Invalid parser output fails loudly in construction mode and never silently corrupts benchmark scenes.
- Every generated scene edit can be reproduced from prompt, model name, config, seed, and asset library version.

Likely repo modifications:

- Update `simworld/assets_rp/utils/input_parser.py`.
- Add `simworld/assets_rp/schema.py`.
- Add a deterministic fixture under `tests/fixtures/assets_rp/` if a test suite is introduced.

## Iteration 5: Text-To-3D Asset Generation Pipeline

Question: is text-to-3D actually available in this checkout, and can another model be substituted?

Current finding:

- This repository does not expose Hunyuan3D by name.
- Existing code retrieves from bundled/local asset images and writes placement JSON.
- Importing arbitrary `.pak` assets is documented in `docs/source/customization/make_your_own_pak.rst`.

Implementation plan:

1. Treat text-to-3D as absent until a separate package, API, or UE asset pipeline is identified.
2. Define an interface before choosing a generator.
3. The interface should take a text prompt and produce a versioned asset artifact plus metadata.
4. Add adapters later for Hunyuan3D, external APIs, or manually curated assets.
5. Keep generation out of benchmark evaluation by default.

Suggested interface:

```python
class AssetGenerator:
    def generate(self, prompt: str, output_dir: str) -> "GeneratedAsset":
        ...
```

Suggested metadata:

```json
{
  "prompt": "red ceramic mug with white handle",
  "generator": "manual-or-hunyuan3d",
  "generator_version": "unknown",
  "created_at": "2026-05-18T00:00:00",
  "asset_path": "assets/generated/red_mug/",
  "ue_asset_path": null,
  "license": "unknown",
  "quality_notes": []
}
```

Definition of done:

- The repo has a documented answer for whether text-to-3D is present.
- Benchmark construction can reference generated assets by stable IDs.
- Evaluation episodes never depend on live text-to-3D generation.

Likely repo modifications:

- Add `simworld/assets_generation/README.md`.
- Add metadata schema for generated or imported assets.
- Add docs explaining how generated assets become Unreal-compatible `.pak` files.

## Iteration 6: Live Vs Prebuilt Assets And Scenes

Question: should scene changes happen during simulation or only during construction?

Files to read:

- `simworld/communicator/communicator.py`
- `simworld/assets_rp/AssetsRP.py`
- `simworld/utils/data_exporter.py`
- `examples/world_generation.ipynb`
- `examples/asset_rp.ipynb`

Current finding:

- `Communicator.spawn_object()` can spawn an asset if a UE blueprint path already exists.
- `Communicator.generate_world()` renders a JSON world by spawning objects.
- `AssetsRetrieverPlacer.generate_assets_manually()` writes placement JSON, which points toward pre-generation.

Implementation plan:

1. Define two modes: `construction_mode` and `evaluation_mode`.
2. In construction mode, allow procedural generation, scene editing, asset retrieval, and possibly text-to-3D.
3. In evaluation mode, load frozen scene JSON, frozen asset libraries, fixed seeds, and fixed map paths.
4. Add a manifest file for every benchmark scene.
5. Add a validator that checks required assets and map files exist before an evaluation run starts.

Suggested benchmark scene manifest:

```yaml
scene_id: two_agent_mug_v001
map_path: /Game/Maps/demo_1
roads_json: data/example_city/demo_city_1/roads.json
world_json: data/example_city/demo_city_1/progen_world.json
asset_library: data/ue_assets.json
seed: 42
construction:
  prompts: []
  generated_assets: []
evaluation:
  allow_live_scene_edits: false
  allow_live_asset_generation: false
```

Definition of done:

- Benchmark scenes are reproducible from manifests.
- Live scene mutation is disabled for leaderboard-style evaluation.
- Construction-time tools are still available for creating new tasks.

Likely repo modifications:

- Add `benchmarks/scenes/`.
- Add `simworld/benchmark/scene_manifest.py`.
- Add `scripts/validate_scene_manifest.py`.

## Iteration 7: Action Planner Boundary

Question: is the local planner part of the evaluated agent or fixed environment infrastructure?

Files to read:

- `simworld/local_planner/action_space.py`
- `simworld/local_planner/local_planner.py`
- `simworld/local_planner/prompt/prompt.py`
- `docs/source/components/agent_system.rst`

Implementation plan:

1. Define three benchmark tracks:
   `structured_action`, `high_level_language_action`, and `low_level_primitive_action`.
2. In `structured_action`, the evaluated agent chooses from a fixed schema and the environment executes it.
3. In `high_level_language_action`, the evaluated agent emits natural language and the parser may be fixed infrastructure.
4. In `low_level_primitive_action`, the evaluated agent controls movement/interaction primitives directly.
5. Log both the agent output and the executed simulator command.
6. Make planner use explicit in every metric report.

Recommended benchmark policy:

- Start with `structured_action` for clean comparisons.
- Add `high_level_language_action` as a separate track once parser reliability is measured.
- Use `low_level_primitive_action` only for agents designed for fine-grained embodied control.

Definition of done:

- Every run records which action track was used.
- Reports distinguish model reasoning failures from planner parsing failures and simulator execution failures.
- No benchmark result hides a strong planner inside the environment without disclosure.

Likely repo modifications:

- Extend `HighLevelAction` for benchmark-specific actions.
- Add action schemas under `simworld/benchmark/actions.py`.
- Add action trace logging.

## Iteration 8: Async Vs Sync Mode

Question: which mode is reproducible enough for benchmark evaluation?

Files to read:

- `simworld/communicator/unrealcv.py`
- `docs/source/components/ue_detail.rst`
- `docs/source/components/traffic_system.rst`
- `simworld/traffic/controller/traffic_controller.py`

Implementation plan:

1. Write a tiny deterministic task with one spawned humanoid and one target.
2. Run it 10 times in sync mode with the same seed and fixed actions.
3. Run it 10 times in async mode with the same seed and fixed actions.
4. Compare final positions, collision counts, timestamps, and command latencies.
5. Repeat with traffic disabled and traffic enabled.
6. Decide the default evaluation mode.

Expected policy:

- Use sync mode for leaderboard-style benchmark evaluation.
- Use async mode for open-ended simulations where real-time independent agents matter.
- Treat traffic async constraints as a known risk until measured.

Definition of done:

- Reproducibility is measured, not assumed.
- The default evaluation runner uses sync mode unless a task explicitly opts out.
- Async results are labeled as async in logs and reports.

Likely repo modifications:

- Add `scripts/check_sync_reproducibility.py`.
- Add `simworld/benchmark/runner.py` with explicit `mode` and `tick_interval`.
- Add reproducibility fields to rollout logs.

## Iteration 9: Unreal Engine Bottlenecks And Scalability

Question: what limits the number of agents: rendering, physics, sensors, traffic, Python communication, or model inference?

Files to read:

- `simworld/communicator/communicator.py`
- `simworld/communicator/unrealcv.py`
- `simworld/traffic/controller/traffic_controller.py`
- `simworld/traffic/manager/vehicle_manager.py`
- `simworld/traffic/manager/pedestrian_manager.py`
- `simworld/llm/base_llm.py`
- `simworld/llm/a2a_llm.py`

Implementation plan:

1. Separate simulator stepping from model inference.
2. Profile environment-only rollouts with scripted actions.
3. Profile local LLM text inference without UE.
4. Profile VLM inference without UE.
5. Profile full closed-loop episodes.
6. Vary agent counts: 2, 5, 10, 20, and more than 20 if stable.
7. Vary background traffic and pedestrians.
8. Vary sensor count and resolution.

Definition of done:

- There is a bottleneck table for each experimental setting.
- The initial benchmark chooses an agent count that is stable on local hardware.
- The roadmap states what must improve before larger-N experiments are credible.

Likely repo modifications:

- Add profiling scripts that can disable models, images, traffic, or physics-dependent actions independently.
- Add machine-readable run summaries.

## Iteration 10: Initial Two-Agent Communication Task

Question: can we design a task where success is impossible without communication?

Benchmark idea:

- Agent A sees or knows the target object.
- Agent B has the ability or location needed to retrieve it.
- Neither agent alone has enough information to succeed.

Minimal version:

- Two mugs exist: red mug and blue mug.
- Agent A is told "the target is the red mug near the bus stop."
- Agent B can reach the mugs but does not know which mug is target.
- Agent A cannot reach or pick up the mug.
- Agent B must ask, receive, or infer the target through messages.

Files to modify or add:

- `simworld/benchmark/tasks/two_agent_reference.py`
- `simworld/benchmark/messages.py`
- `simworld/benchmark/rewards.py`
- `simworld/benchmark/runner.py`

Implementation plan:

1. Start without UE by building a tiny state-machine mock task to validate scoring.
2. Add message objects with sender, receiver, content, timestep, and optional referenced object IDs.
3. Add a scripted silent baseline that should fail.
4. Add a scripted communicating baseline that should pass.
5. Port the task to SimWorld objects, using fixed object names and spawn positions.
6. Add metrics for success, steps, message count, clarification count, wrong-object attempts, and collisions.

Definition of done:

- Silent baseline fails for the intended reason.
- Communicating baseline succeeds.
- The task logs enough information to replay the common-ground formation process.

## Iteration 11: Referential Ambiguity Task

Question: can agents disambiguate objects, locations, ownership, and perspective?

Benchmark idea:

- Create multiple similar objects with conflicting references such as "my mug," "your mug," "the chair to my left," and "the red cup near me."
- The correct referent depends on speaker identity, listener perspective, prior dialogue, or pointing history.

Implementation plan:

1. Define object metadata: ID, type, color, owner, location, nearby landmarks.
2. Define agent-relative language: left, right, near me, behind you, closest to the table.
3. Add a dialogue history field to observations.
4. Create tasks where the same phrase maps to different objects depending on speaker.
5. Score the selected object ID, not just task success.
6. Add distractor objects and measure clarification behavior.

Suggested metrics:

- `referent_accuracy`
- `clarification_rate`
- `wrong_pickup_count`
- `dialogue_turns_to_resolution`
- `perspective_error_count`
- `history_dependency_success_rate`

Definition of done:

- A scripted oracle can solve the task using metadata.
- A no-dialogue or no-history baseline fails on ambiguous cases.
- Logs show which reference expression was resolved to which object ID.

## Iteration 12: Divergent Agents And Perceptual Asymmetry

Question: can the environment give agents different fields of view, sensors, scene-graph access, memory, or model backbones?

Files to read:

- `simworld/communicator/unrealcv.py`
- `simworld/communicator/communicator.py`
- `simworld/agent/humanoid.py`
- `simworld/local_planner/local_planner.py`

Implementation plan:

1. Define an `AgentProfile` with observation permissions.
2. Give one agent RGB only and another structured state only.
3. Give one agent privileged target metadata and another movement ability.
4. Give agents different camera resolutions or FOV if UE camera commands are reliable.
5. Allow heterogeneous policies: scripted, local LLM, cloud LLM, VLM.
6. Log exactly what each agent could observe.

Example profile:

```yaml
agent_id: 0
name: describer
can_move: false
can_pick_up: false
observations:
  rgb: true
  depth: false
  object_mask: false
  scene_graph: true
  private_goal: true
policy:
  provider: local
  model: Qwen2.5-7B-Instruct
```

Definition of done:

- Agents in the same episode can receive different observation dictionaries.
- Metrics record which asymmetries were active.
- At least one task requires combining asymmetric information to succeed.

## Proposed Benchmark Package Shape

Once the first few iterations are stable, organize benchmark-specific code separately from core SimWorld modules:

```text
simworld/
  benchmark/
    __init__.py
    actions.py
    agent_profile.py
    messages.py
    observations.py
    rewards.py
    runner.py
    scene_manifest.py
    tasks/
      __init__.py
      two_agent_reference.py
      referential_ambiguity.py
scripts/
  profile_unrealcv_loop.py
  check_sync_reproducibility.py
  validate_scene_manifest.py
benchmarks/
  scenes/
  configs/
  results/
docs/
  simworld_benchmark_learning_roadmap.md
```

Keep these boundaries:

- `simworld/communicator` should remain the UE command boundary.
- `simworld/local_planner` should remain planner infrastructure, not benchmark policy code.
- `simworld/benchmark` should own benchmark tasks, observations, rewards, action tracks, runner configuration, and logs.
- `benchmarks/` should hold task manifests, scene manifests, and small reproducible configs.

## Suggested First Three Code Changes

If the goal is to learn while making useful progress, start here:

1. Add local provider support to `A2ALLM`.
   This is small, directly useful, and teaches the LLM/planner boundary.

2. Add a profiling script for the Gym-like loop.
   This answers the biggest feasibility question before over-investing in task design.

3. Add a minimal `simworld/benchmark` skeleton with observations, messages, and a scripted two-agent task mock.
   This lets benchmark design proceed even before every UE detail is stable.

## Research Log Template

For each iteration, keep one short entry:

```md
## YYYY-MM-DD: Iteration Name

Goal:

Setup:

- Commit:
- Config:
- UE map:
- Machine:
- Mode:

What I ran:

Results:

What surprised me:

Decision:

Next action:
```

## Key Decisions To Make Explicit

These choices should appear in any benchmark paper, README, or leaderboard:

- Does the evaluated agent include the local planner, or is the planner fixed infrastructure?
- Are actions structured, high-level language, or low-level primitives?
- Are scenes frozen before evaluation?
- Are text-to-3D and natural-language scene editing disabled during evaluation?
- Is the runner synchronous or asynchronous?
- Which observations are allowed for each agent?
- Are agents homogeneous or divergent?
- Are model latency and failed parses counted in the score?
- Are communication messages private, broadcast, range-limited, or grounded by physical proximity?
- Are results averaged over seeds, maps, personas, and object layouts?

## Working Assumption For The First Benchmark

Start with a small, reproducible benchmark:

- Two humanoid agents.
- Frozen scene.
- Sync mode.
- Structured action track.
- Text-only observations plus optional RGB track.
- No live asset generation.
- No live scene editing.
- Scripted silent and communicating baselines.
- Metrics for success, referent accuracy, dialogue turns, collisions, latency, and variance.

After that works, expand one axis at a time: more ambiguous objects, more agents, richer sensors, VLM policies, traffic, larger maps, and async open-ended simulations.
