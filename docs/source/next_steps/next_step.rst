Next Steps Roadmap
===================

This page provides more details about several key directions on SimWorld's roadmap.  
If you are interested in collaborating on any of the following items, feel free to contact **lingjun@ucsd.edu**.

.. contents::
   :local:
   :depth: 1


.. _next-steps-plugin-system:

Plugin System
-------------

The SimWorld plugin system is designed to make it easy to **extend the platform without modifying the core codebase**.
With plugins, users can:

- Register custom environment types, agents, or tools
- Package UE content and Python logic into reusable modules
- Share plugins within a team or research community

For a practical example of how to package custom content as plugins, see:

- ``docs/source/customization/make_your_own_pak.rst``
- The corresponding section in the README: ``Make Your SimWorld``

If you are building your own SimWorld extensions and would like deeper integration support, please reach out at **lingjun@ucsd.edu**.


.. _next-steps-comprehensive-agent-framework:

Comprehensive Agent Framework
-----------------------------

We plan to provide a **unified training and evaluation framework** for autonomous agents in SimWorld, including:

- Standardized agent interfaces (policy, memory, tool-use, multi-agent communication)
- Plug-and-play support for both RL agents and LLM/VLM-based agents
- Evaluation benchmarks for navigation, social interaction, planning, and long-horizon tasks

The goal is to make it easy to run **reproducible experiments** across different algorithms and tasks with a single framework.
If you are interested in contributing designs or use cases for this framework, please email **lingjun@ucsd.edu**.


.. _next-steps-nl-to-ue-actions:

Arbitrary Natural Language → UE Actions
---------------------------------------

SimWorld already exposes a rich set of low-level Unreal Engine actions (e.g., move, rotate, interact, pick up).
The next step is to support **mapping free-form natural language instructions to executable UE actions/tools**, for example:

- "Walk to the coffee shop on the left, then sit down at the table by the window."
- "Spawn ten pedestrians crossing the main street and record a 20-second video."

This involves:

- Designing an extensible **action schema / tool specification** for UE actions
- Training / prompting models that ground language into these tools
- Providing debugging and visualization tools for action traces

If you are working on language-to-action or tool-use agents and would like to build on SimWorld, please contact **lingjun@ucsd.edu**.


.. _next-steps-rl-training-pipeline:

RL Training Pipeline
--------------------

To support reinforcement learning research in SimWorld, we plan to provide a **unified RL training pipeline**, including:

- Gym-like / Gymnasium-compatible environment wrappers
- Standard observation and reward interfaces for common tasks
- Reference training scripts (e.g., PPO, SAC, multi-agent RL)
- Logging, evaluation, and checkpoint management utilities

This will make it straightforward to scale from **toy examples to large-scale RL experiments** in realistic city-scale environments.
If you are interested in RL benchmarks or integration with existing RL libraries, please reach out at **lingjun@ucsd.edu**.


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
If you are interested in pushing city-scale simulations or have industrial use cases, please contact **lingjun@ucsd.edu**.

