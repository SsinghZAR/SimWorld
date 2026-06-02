# UnrealCV+/Gym Speed Profile

This document is the report template for deciding whether SimWorld is usable for
online RL, offline rollout generation, or evaluation-only usage.

## Benchmark Tooling

Use `benchmark/profile_unrealcv_loop.py` from the repository root. It writes
JSONL and CSV summaries under `runs/profiles/`.

Primary dimensions:

- Environment: base city, generated city, traffic-heavy city, optional heavy
  `.pak` city, and optional robot city.
- Agent count: `1, 2, 4, 8, 16`, with `20` if stable.
- Observation: `structured_only`, `rgb`, `depth`, `object_mask`, `rgb_depth`,
  and `shared_rgb`.
- Resolution: `320x240`, `640x480`, `1280x720`.
- Mode: `sync` for RL decisions, `async` for comparison.
- Step kind: `sensor_tick`, `gym_wrapper`, `command_latency`, and optional
  `robot`.

## Run Metadata To Record

- Machine, OS, CPU, RAM.
- GPU model, driver version, VRAM, and whether GPU use was verified.
- UE package version and exact launch command.
- Map URI and any required `.pak` files.
- `roads.json` and `progen_world.json` paths when available.
- Traffic/background-load description.
- Whether a navigation graph is available for the scene.

## Recommended Scenarios

1. `base_demo_1`: `/Game/Maps/demo_1.umap` with
   `data/example_city/demo_city_1/roads.json`.
2. `base_demo_2`: `/Game/Maps/demo_2.umap` with
   `data/example_city/demo_city_2/roads.json`.
3. `procedural_city_small`: `/Game/Maps/empty.umap` plus a small generated
   `progen_world.json`.
4. `procedural_city_large`: `/Game/Maps/empty.umap` plus a larger generated
   `progen_world.json`.
5. `traffic_city`: demo or procedural city with fixed vehicle/pedestrian counts.
6. `heavy_pak_urban`: installed Marketplace-scale city map, such as Tokyo,
   Industrial Area, Downtown West, or Modular Victorian City.

## Feasibility Criteria

Use sync-mode results for the RL decision:

- `online_rl_ready`: structured observations sustain about `50+` steps/sec for
  one agent and degrade smoothly with more agents.
- `image_online_rl_plausible`: RGB at the default resolution sustains about
  `10-20+` steps/sec without high p95 latency spikes.
- `offline_rollout_preferred`: image observations work but throughput is only a
  few steps/sec or multi-agent scaling is steep.
- `evaluation_only`: step latency is too high or unstable for online learning.

If `gym_wrapper` is far slower than `sensor_tick`, optimize non-blocking action
wrappers before judging the simulator itself.
