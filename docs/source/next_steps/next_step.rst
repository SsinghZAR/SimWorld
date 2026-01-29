Next Steps Roadmap
===================

This page provides more details on several key directions on SimWorld's roadmap.
We are happy to share our Unreal project files and assets to all contributors for internal development and research purposes.
If you're interested in collaborating on any item below, please reach out to the corresponding contact listed for that project.

.. contents::
   :local:
   :depth: 1


.. _next-steps-comprehensive-agent-framework:

Comprehensive Agent Framework
-----------------------------

We plan to build a **general, modular agent framework** for autonomous agents in SimWorld, including:

- **Standardized agent modules** (perception, memory, reasoning, and learning) that can be flexibly composed (e.g., dynamic cheat sheets, CoT, reflection)
- **Gym-compatible interfaces** for RL training across a wide range of embodied tasks
- **Systematic ablations across environments** to understand what actually matters for success in long-horizon embodied tasks

If you're interested in contributing designs or use cases for this framework, please reach out to **lingjun@ucsd.edu**.

.. _next-steps-codegen-scenes:

Code Generation for Scenes
--------------------------

We are exploring **AI-powered coding agents** that programmatically generate rich scenarios and cities:

- A **scene DSL / API** that compiles to SimWorld maps, assets, traffic rules, and scripted events
- **LLM tooling chains** that turn prompts or task specs into executable scene code with validation and preview
- **Safety and quality checks** (asset budgets, collision-free placements, playability tests)
- **Curated seeds and benchmarks** to evaluate diversity, controllability, and realism of generated content

If you have use cases or evaluation ideas for scene code generation, please contact **x8ye@ucsd.edu**.

.. _next-steps-interactive-layout-editor:

Interactive Layout Editor
-------------------------

We plan to build a **web-based layout editor** for real-time city visualization and editing:

- Live **map canvas** with layers for roads, zoning, traffic lights, and spawn points
- **Asset palette and snapping** for roads, buildings, props, and scripted triggers with constraint-aware placement
- **Co-editing and versioning** so teams can iterate together and diff/export layouts into UE or SimWorld gym wrappers
- **Simulation-aware validation** (navmesh coverage, connectivity, spawn density, performance budget estimates)

If you are interested in frontend or visualization contributions, please reach out to **x8ye@ucsd.edu**.

.. _next-steps-nl-to-ue-actions:

Arbitrary Natural Language → UE Actions
---------------------------------------

SimWorld already exposes a rich set of low-level Unreal Engine actions (e.g., move, rotate, interact, pick up).
The next step is to support **mapping free-form natural language instructions to executable UE actions/tools**, for example:

- "Walk to the coffee shop on the left, then sit down at the table by the window."
- "Spawn ten pedestrians crossing the main street and record a 20-second video."

This involves:

- Designing an extensible **action schema / tool specification** for UE actions
- Training / prompting llm local planners that ground language into these tools
- Providing debugging and visualization tools for action traces

If you are working on language-to-action or tool-use agents and would like to build on SimWorld, please contact **lingjun@ucsd.edu**.


.. _next-steps-rl-training-pipeline:

RL Training Pipeline for SimWorld
--------------------

We plan to provide a **unified RL training pipeline** for diverse embodied tasks (e.g., DeliveryBench) in SimWorld, including:

- Gym-like environment wrappers
- Standard observation and reward interfaces for embodied tasks
- Reference training scripts (e.g., PPO, SAC, multi-agent RL)

This will make it straightforward to run **large-scale RL experiments** across diverse embodied tasks, and to derive insights that can guide the design of new RL algorithms.

If you are interested in RL research and exploration in embodied simulation settings, please reach out at **lingjun@ucsd.edu**.


.. _next-steps-city-scale-multi-agent:

City-Scale Multi-Agent Simulation
---------------------------------

One of SimWorld's long-term goals is to support **city-scale multi-agent simulation** with 1K+ concurrent agents in the same city,
covering pedestrians, vehicles, service robots, and other interactive entities.

Key directions include:

- Scalable simulation backends and load balancing across machines
- Rich social and physical interaction patterns between agents
- Tools for logging, visualization, and analysis of large-scale behaviors

This direction is especially relevant for research on **emergent behavior, social dynamics, and large-scale coordination**.
If you are interested in pushing city-scale simulations or have industrial use cases, please contact **jir015@ucsd.edu**.

.. _next-steps-video-to-scene:

Video-to-Scene Generation
-------------------------

We aim to support **video-to-scene** pipelines that convert real videos into simulation-ready UE scenes:

- **Camera pose and intrinsics estimation** plus multi-view geometry / SLAM for structure recovery
- **Object detection, tracking, and 3D reconstruction** to infer dynamic actors and static layout
- **Asset mapping** to replace reconstructed meshes with SimWorld-ready assets and materials
- **Temporal consistency and evaluation** tools to check fidelity, scale, and replayability of generated scenes

Reference material:

- Vid2Sim: https://metadriverse.github.io/vid2sim/
- Video2Game: https://video2game.github.io/

If you work on video-to-3D or can share datasets, please contact **x8ye@ucsd.edu**.


.. _next-steps-mujoco:

MuJoCo Integration
------------------

We are prototyping **MuJoCo as an optional physics backend** to complement UE for high-frequency control:

- **Interchange layer** to mirror SimWorld agents, sensors, and actions in MuJoCo while keeping scene semantics
- **Time-sync and co-sim bridges** so planners can mix UE visuals with MuJoCo dynamics when needed
- **Benchmark tasks** (manipulation, legged locomotion, aerial) to compare fidelity and performance across backends

Reference material:

- MuJoCo-Unity plugin example: https://github.com/OpenHUTB/mujoco_plugin
- MuJoCo in Unity docs: https://mujoco.readthedocs.io/en/latest/unity.html
- URoboViz (3D robot visualization): https://github.com/HoangGiang93/URoboViz
- Unreal Robotics Lab: https://arxiv.org/html/2504.14135v2

If you want to help shape the MuJoCo bridge or contribute tasks, please reach out to **x8ye@ucsd.edu**.


.. _next-steps-expand-agent-action-objects:

Expanded Agents, Actions, and Interactable Objects
--------------------------------------------------

We plan to expand the **embodied ecosystem** across agents, action spaces, and interactables:

- New agent types (e.g., **drones, service robots, manipulation platforms**) with standardized capability profiles
- **Richer action schemas** (continuous and symbolic) with compatibility across UE tools, RL wrappers, and planners
- **Broader interactable set** (doors, elevators, appliances, IoT props) with consistent affordances and state machines
- **Evaluation suites** to measure coverage, compositionality, and cross-agent interoperability

If you are interested in defining new agent/action specs or supplying assets, please contact **x8ye@ucsd.edu**.

