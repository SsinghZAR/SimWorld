# Venue Meetup V0 — Implementation Checklist

## Purpose

This document lists what is needed to implement the **V0 embodied venue meetup scenario** in SimWorld and run a small evaluation. The goal is to separate:

1. **What SimWorld must already provide or support**, and
2. **What we need to implement ourselves as the benchmark wrapper / scenario layer.**

V0 scenario summary:

> Visitor agents are placed in an unfamiliar SimWorld district. Each agent receives a coarse map, a private constraint, group chat history, and first-person visual frames. Agents must explore, inspect venues, communicate findings, and physically converge on the best available venue.

---

## 1. SimWorld features we need to verify

These are not necessarily things we build from scratch, but they must exist or be usable through SimWorld's API for the scenario to work.

### 1.1 Multi-agent spawning

We need to confirm SimWorld can:

- Spawn multiple humanoid agents in the same scene.
- Assign each agent a distinct starting location.
- Keep agents active in the same shared world.
- Step agents independently or sequentially in a controlled rollout loop.

Minimum for V0:

- 2 agents working.
- Then scale test to 3 and 5 agents.

---

### 1.2 First-person visual observations

We need each agent to receive only its own current camera frame as the local observation.

Need to verify:

- Can retrieve first-person RGB frame for each agent.
- Camera view changes with movement and looking actions.
- Multiple agents can each have separate camera observations.
- Frames can be passed to a VLM agent or saved to logs.

Important V0 constraint:

- No text scene description.
- No object list.
- No scene graph.
- No coordinates.
- No automatic venue labels.

---

### 1.3 Movement and looking actions

SimWorld should already support movement and camera orientation. We need to verify the usable action API.

Need to confirm available actions such as:

- move forward / backward
- turn left / right
- look up / down
- rotate camera/body
- move toward visible target or waypoint, if supported
- stop / wait

V0 does not need low-level motor-control realism. It does need camera changes and physical movement through the scene.

---

### 1.4 Venue-like assets and landmark assets

We need to confirm that the base SimWorld asset set can represent a small urban district with candidate venues and landmarks.

Needed assets or approximations:

- cafes / restaurants / pubs / shops / hotel lobbies / station entrances
- public landmarks such as fountain, bridge, clock tower, bus stop, statue, market, canal, station
- signs or facade cues visible from first-person view
- entrances and side entrances
- simple occluders or route blockers

If the built-in assets are insufficient, we need to either:

- reuse existing generic buildings as venue slots,
- add a small custom asset set,
- or generate/import assets before running scenarios.

---

### 1.5 Scene editing / map construction

We need to verify how practical it is to build small fixed map templates.

Need to check whether SimWorld supports:

- loading predefined small maps,
- placing buildings/landmarks at specified slots,
- randomising assets across slots,
- adding/removing obstacles or closed entrances,
- changing signs or facade elements,
- saving/loading scenario seeds.

V0 does not require fully procedural city generation. It needs fixed templates with randomised node/slot assignments.

---

### 1.6 Collision, navigation, and venue arrival detection

We need to know whether SimWorld exposes enough state to detect where agents are.

Need to verify:

- Can query agent pose/location internally for scoring.
- Can define venue regions / trigger zones.
- Can detect whether an agent is inside or near a venue.
- Can detect if agents are at the same venue.
- Can detect timeout / stuck / no movement conditions.

Agents should not receive this state directly. It is only for the evaluator.

---

### 1.7 Headless or efficient rendering mode

Small evaluation runs can be graphical, but we should check whether SimWorld can run efficiently.

Need to verify:

- Can run with rendering enabled and capture frames.
- Can run at acceptable speed for 2-5 agents.
- Can disable unnecessary rendering/UI while still collecting agent frames.
- Approximate steps per second for a small district.

For V0, this is just a feasibility check, not an optimisation target.

---

## 2. Things we need to implement ourselves

These are benchmark-specific components that sit on top of SimWorld.

### 2.1 Scenario metadata layer

We need a structured metadata representation for each generated scenario.

Each scenario should store:

```yaml
scenario_id: string
map_template_id: string
seed: int
venues:
  - venue_id: string
    slot_id: string
    venue_type: cafe | pub | restaurant | hotel_lobby | shop | station_entrance | public_square
    properties:
      open: bool
      reachable: bool
      capacity: int
      accessible: bool
      shelter: bool
      food_drink: bool
      quiet_score: float
      crowding_score: float
    entrances:
      - entrance_id: string
        status: open | blocked | stairs_only | accessible
landmarks:
  - landmark_id: string
    slot_id: string
    type: fountain | station | bridge | bus_stop | clock_tower | statue | market
agents:
  - agent_id: string
    spawn_slot: string
    private_constraint: string
requirements:
  hard: []
  soft: []
weights: {}
```

This metadata is hidden from agents and used for:

- generation,
- scoring,
- logging,
- normalised best-available score.

---

### 2.2 Fixed graph templates

We need to define a small number of reusable graph/map templates.

Each template should include:

- venue slots,
- landmark slots,
- spawn slots,
- route slots,
- possible occlusion/blockage slots,
- venue trigger regions,
- approximate coarse-map representation.

V0 target:

- 1 template for first smoke test.
- 3 templates for initial small evaluation.

Example templates:

- central square layout,
- station street layout,
- canal / bridge layout.

---

### 2.3 Slot randomisation

For each episode, the generator should randomise:

- venue type assigned to each venue slot,
- landmark type assigned to each landmark slot,
- venue properties,
- entrance status,
- route/obstacle status,
- agent spawn locations,
- private constraints,
- requirement weights.

Generation constraints:

- 4-6 candidate venues.
- At least 2 plausible venues before inspection.
- At least 1 false-positive venue.
- At least 1 important fact discoverable only through visual inspection.
- No single agent can identify the best venue from its starting view alone.
- The best venue is reachable.
- Communication is required for high performance.

---

### 2.4 Coarse map representation

We need to create the coarse map given to agents at episode start.

Possible formats:

1. Text-only schematic description.
2. Image/schematic map.
3. Both text and schematic image.

Because the task is meant to be visually grounded, the map should probably be a simple schematic image plus minimal explanatory text.

The coarse map may include:

- rough district layout,
- major streets/intersections,
- public landmarks,
- broad venue clusters,
- approximate route topology,
- agent starting area.

It must not include:

- current venue status,
- open/closed information,
- accessibility details,
- crowding/noise,
- exact entrance conditions,
- blocked routes,
- which venue is optimal.

Implementation task:

- Generate or hand-author one coarse map per template.
- Optionally annotate spawn area without giving exact coordinates.

---

### 2.5 Private constraint sampler

Each agent should receive a private constraint or preference.

Examples:

- needs step-free access,
- prefers a quiet venue,
- needs food/drink available,
- needs shelter,
- needs to stay near transit,
- has a time/distance preference,
- wants to avoid crowded places.

Implementation task:

- Define hard and soft constraint types.
- Define weights.
- Ensure constraints interact with generated venue properties.
- Ensure no single agent has all requirements.

---

### 2.6 Group chat action

We need an explicit group communication action.

Action:

```python
communicate_to_group(message: str)
```

Requirements:

- Message is broadcast to all agents.
- Message is appended to shared group chat history.
- Message count and token count are logged.
- Communication consumes an action/turn.

V0 does not need private messages or calls.

---

### 2.7 Inspect action

This is the main custom action.

Action examples:

```python
inspect(target)
inspect_visible_object(target_description)
inspect_current_entrance()
inspect_current_venue()
```

Design requirement:

- Inspect should not simply narrate hidden metadata to the agent.
- It should be visually grounded.

Preferred V0 behaviour:

- Check whether the agent is near enough and oriented toward the target.
- Focus/zoom/orient the camera toward the relevant object or entrance.
- Return an updated first-person frame.
- Internally log which venue/property was inspected.

Need to decide:

- Whether `inspect` can target only visible objects.
- Whether invalid inspect actions fail silently or return a generic failure.
- Whether inspect consumes one full action.

---

### 2.8 Agent prompt/context wrapper

We need a standard prompt/context format for agents.

Each agent receives:

- role: visitor,
- task objective,
- private constraint,
- coarse map,
- latest first-person frame,
- group chat history,
- valid action schema.

The prompt should emphasise:

- do not assume venue properties without inspection,
- communicate useful findings to the group,
- use the coarse map and landmarks to localise,
- coordinate to find the best feasible venue,
- physically converge once a good venue is selected.

---

### 2.9 Episode controller

The episode controller handles the rollout.

Responsibilities:

- Load/generate scenario.
- Spawn agents.
- Provide initial context.
- Retrieve each agent's current frame.
- Query each agent policy/VLM.
- Parse chosen action.
- Apply action to SimWorld or group chat.
- Advance timestep.
- Check termination.
- Save logs.
- Compute metrics.

V0 can use sequential agent turns rather than fully async execution.

---

### 2.10 Logging

Minimum per-step log:

```json
{
  "scenario_id": "...",
  "timestep": 0,
  "agent_id": "agent_1",
  "action_type": "communicate|move|look|inspect|wait",
  "action_text": "...",
  "message_text": "...",
  "token_count": 0,
  "agent_pose_internal": "hidden_from_agent",
  "current_frame_path": "...",
  "venue_region_internal": "hidden_from_agent",
  "inspect_target_internal": "hidden_from_agent",
  "timestamp": "..."
}
```

Episode-level log:

```json
{
  "scenario_id": "...",
  "map_template_id": "...",
  "seed": 123,
  "num_agents": 3,
  "final_venue_id": "...",
  "final_venue_requirement_score": 0.8,
  "best_available_venue_score": 1.0,
  "normalised_best_available_score": 0.8,
  "arrival_score": 1.0,
  "episode_score": 0.8,
  "venues_inspected": 3,
  "total_candidate_venues": 5,
  "message_count": 12,
  "token_count": 950,
  "timed_out": false
}
```

---

### 2.11 Scoring implementation

Primary metrics:

1. **Final venue requirement score**

```text
weighted satisfied requirements / total requirement weight
```

2. **Normalised best-available score**

```text
final venue score / best venue score in generated scenario
```

3. **Arrival / convergence score**

```text
agents at final venue / total agents
```

Episode score:

```text
normalised best-available score × arrival/convergence score
```

Diagnostics:

- exploration coverage,
- group message count,
- total token count.

---

## 3. Minimal small evaluation run

This is the smallest run that would test whether the environment works end-to-end.

### 3.1 Smoke test

Goal: verify the environment wrapper works.

Configuration:

- 1 map template.
- 4 candidate venues.
- 2 visitor agents.
- 2 private constraints.
- 1 false-positive venue.
- 1 best available venue.
- fixed random seed.
- short step limit.

Expected output:

- agents spawn correctly,
- first-person frames retrieved,
- group chat works,
- movement/look actions work,
- inspect action works,
- final venue detected,
- score computed,
- logs saved.

---

### 3.2 Small evaluation run

Goal: test basic performance and metric stability.

Configuration:

- 3 map templates.
- 10 generated episodes per template.
- N = 2 and N = 3 agents.
- visitor-only agents.
- group chat enabled.
- first-person visual observations only.
- coarse map enabled.

Total:

```text
3 templates × 10 episodes × 2 N-settings = 60 episodes
```

Metrics:

- episode score,
- final venue requirement score,
- normalised best-available score,
- arrival/convergence score,
- exploration coverage,
- message count,
- token count.

---

### 3.3 Minimal ablation run

Only run after the small evaluation works.

Ablations:

1. Main setting.
2. No communication.
3. No coarse map.
4. Shared constraints.
5. Full shared information.

Suggested initial size:

```text
1 template × 5 episodes × 2 N-settings × 5 conditions = 50 episodes
```

Purpose:

- Confirm no-communication mostly fails.
- Check whether coarse map improves coordination.
- Check whether private constraints are a meaningful bottleneck.
- Estimate an upper bound with full shared information.

---

## 4. Open implementation questions

These need to be checked in SimWorld or decided during implementation.

### SimWorld verification

- How do we retrieve per-agent first-person frames?
- Can multiple agent cameras be queried per step?
- What movement/look actions are exposed?
- Can we define venue trigger regions?
- Can we place/randomise venue and landmark assets at fixed slots?
- Can signs/facades be changed per scenario?
- Can we run small multi-agent scenes at acceptable speed?

### Benchmark design decisions

- Should coarse maps be schematic images, text, or both?
- Should inspect return only a focused frame, or also a minimal non-semantic success/failure signal?
- How high-level should movement actions be?
- Should final venue be inferred only from physical convergence, or should agents have an explicit commit action?
- What is the initial set of venue properties and requirement weights?

---

## 5. Minimum implementation order

Recommended order:

1. Verify SimWorld multi-agent camera retrieval.
2. Build one fixed map template with venue/landmark slots.
3. Implement scenario metadata and scoring from static metadata.
4. Implement group chat action.
5. Implement movement/look wrapper.
6. Implement inspect action.
7. Implement episode controller and logging.
8. Run a 2-agent smoke test with fixed scenario.
9. Add slot randomisation.
10. Run small evaluation with N = 2 and N = 3.

---

## 6. Non-goals for V0

V0 does not need:

- local/resident agent types,
- private messages or phone calls,
- mixed-motive deception,
- full procedural city generation,
- large-N scaling,
- fine-grained social preference modelling,
- LLM-judged communication metrics as primary scores.

These can be added after the basic embodied venue meetup task is working.
