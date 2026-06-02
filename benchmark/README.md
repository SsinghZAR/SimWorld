# SimWorld UnrealCV+/Gym Speed Benchmarks

This folder contains simple CLI tooling for profiling SimWorld step throughput.
Launch the Unreal Engine backend yourself, then run the profiler against the
UnrealCV port.

## Quick Dry Run

Validate the benchmark matrix and output writers without a running UE backend:

```bash
python benchmark/profile_unrealcv_loop.py \
  --dry-run \
  --preset rl \
  --scenario-name dry_run_demo_1 \
  --map-uri /Game/Maps/demo_1.umap
```

## Live Smoke Test

Start the UE backend first:

```bash
./SimWorld.sh /Game/Maps/demo_1.umap
```

Then run a short structured-only profile:

```bash
python benchmark/profile_unrealcv_loop.py \
  --preset quick \
  --scenario-name demo_1_smoke \
  --map-uri /Game/Maps/demo_1.umap \
  --roads-json data/example_city/demo_city_1/roads.json \
  --ue-launch-profile rendered \
  --gpu-verified \
  --gpu-name "RTX 4090"
```

## Reusable Matrix Args

Use comma-separated values to build a matrix:

```bash
python benchmark/profile_unrealcv_loop.py \
  --scenario-name demo_1_rl \
  --map-uri /Game/Maps/demo_1.umap \
  --roads-json data/example_city/demo_city_1/roads.json \
  --num-agents 1,2,4,8 \
  --observation-profiles structured_only,rgb,depth,object_mask \
  --resolutions 320x240,640x480 \
  --modes sync \
  --step-kinds sensor_tick,gym_wrapper \
  --steps 100 \
  --warmup-steps 10 \
  --gpu-verified
```

Important profiles:

- `structured_only`: positions and orientations only.
- `rgb`, `depth`, `object_mask`, `rgb_depth`: one camera per agent.
- `shared_rgb`: one RGB camera shared across agents, useful for separating
  per-agent sensor cost from world simulation cost.
- `sensor_tick`: observations plus sync ticks, the primary RL throughput proxy.
- `gym_wrapper`: current movement wrapper behavior, including wrapper sleeps.
- `command_latency`: individual UnrealCV command timings.
- `robot`: Spot-like robot motion/camera profile.

## Environment Scenarios

Benchmark the same matrix across these setup tiers:

- Base city: `/Game/Maps/demo_1.umap` with `data/example_city/demo_city_1/roads.json`.
- Alternate base city: `/Game/Maps/demo_2.umap` with `data/example_city/demo_city_2/roads.json`.
- Procedural city: launch `/Game/Maps/empty.umap`, render a generated
  `progen_world.json`, and pass `--world-json`.
- Traffic city: run traffic separately and describe it with `--traffic-profile`.
- Heavy pak city: launch an installed `.pak` map such as Tokyo or Industrial Area.

For optional `.pak` scenes, set `--navigation-graph-available false` unless you
have matching `roads.json` data for waypoint or traffic benchmarks.

## GPU Verification

Rendered and offscreen image benchmarks should be GPU-backed. Verify the UE
process with `nvidia-smi` or platform-equivalent telemetry, then pass:

```bash
--gpu-verified --gpu-name "GPU name" --rhi "DirectX12/Vulkan/etc"
```

Do not use image profiles for `-nullrhi` or no-render launches. Those should be
structured-only runs.

## Outputs

Results are appended to:

- `runs/profiles/<scenario>.jsonl`
- `runs/profiles/<scenario>.csv`

Generated profile outputs are ignored by git.
