Venue Meetup Benchmark
======================

UE-grounded multi-agent benchmark for **social information sharing**: agents must
pool private observations to meet at a unique group-feasible venue. Locomotion is
a controlled variable; the dependent variables are communicative.

Every agent observation also includes a deterministic shop-closing clock. By
default shops close at 18:00 and each synchronized action turn consumes one
simulated minute, making inspection, communication, and movement share the same
finite time budget. The initial time is derived from the episode turn cap.

Package path: ``benchmark/venue_meetup/``. See also the in-tree
``benchmark/venue_meetup/README.md``.

Research question
-----------------

Does an agent integrate partner reports (group information), and does it share
facts that matter to the partner's private needs (other-regarding message
passing)? Outcome scoring rewards meeting at the group-feasible venue; process
metrics diagnose how agents shared information.

Information partition and hidden properties
-------------------------------------------

* **Public to agents:** coarse map, candidate venues, own private hard requirement
  (in ``main``), ego camera, inbox, known inspect facts.
* **Hidden:** venue ground-truth traits until a successful ``INSPECT``; partner
  constraints (unless ablation); optimum id; scoring soft weights.
* ``--info-partition spatial``: inspect only own-zone venues; cross-zone facts
  require communication.
* ``--info-partition none``: no zone gating on inspect (easier upper bound).
* ``skill_check`` (V2 DnD attribute rolls) is **not** implemented.

Actions and synchronous turns
-----------------------------

Agents act in synchronous turns. Each turn is one structured action:
``NAVIGATE``, ``INSPECT``, ``COMMUNICATE``, optional ``STEP_FORWARD`` /
``TURN_AROUND``, or ``WAIT``.

``COMMUNICATE`` may include an optional free-text ``message`` and/or structured
``shared_facts`` claims of the form ``[{venue_id, trait, value}, ...]`` for
traits the sender personally inspected.

Teleport vs walk navigation
---------------------------

* **Teleport (default):** ``NAVIGATE`` places the agent at the venue meeting
  point. Prefer this for social/reference evaluation.
* **``--walk``:** physical traversal using **graph-backed layout routes**
  (sidewalks, crossings, bridges) when a district layout is present; otherwise
  the **legacy obstacle-aware free-space planner**. Keep navigation diagnostics
  separate from social scores.

Map templates
-------------

Use these real template IDs. The three Rosebank scale tiers are the canonical
size comparison; the earlier layouts remain useful compatibility baselines:

.. list-table::
   :header-rows: 1
   :widths: 35 20 25 20

   * - Template ID
     - Scale
     - Structure
     - Default budget
   * - ``central_square_v0``
     - Plaza
     - 4 venues (ring)
     - 32 turns
   * - ``station_quarter_medium_v1``
     - ~350–500 m
     - 8 venues, 4 blocks
     - 64 turns
   * - ``riverside_market_large_v1``
     - ~700–900 m
     - 12 venues, 6+ blocks, barrier + bridges
     - 128 turns
   * - ``rosebank_grid_3x3_v0``
     - 9 blocks
     - 4 venues, 3 landmarks
     - 96 turns
   * - ``rosebank_grid_5x5_v0``
     - 25 blocks
     - 8 venues, 6 landmarks
     - 160 turns
   * - ``rosebank_grid_7x7_v0``
     - 49 blocks
     - 12 venues, 6 landmarks
     - 256 turns
   * - ``rosebank_grid_9x9_v0``
     - 81 blocks
     - 36-venue visual/navigation stress map
     - 384 turns

``station_street_v0`` and ``canal_bridge_v0`` are **legacy metadata aliases of
central-square geometry**. Do not treat them as independent maps.

Layout authoring model
----------------------

District templates are deterministic Python-authored ``DistrictLayout`` specs
(streets, blocks, frontages, landmarks, spawns, walk graph). The Rosebank family
uses one size-aware planner for its 3x3, 5x5, 7x7, and retained 9x9 forms. Actors
spawn into the existing SimWorld base map via the asset catalog.

Hidden-profile invariants
-------------------------

With ``--hidden-profile``:

* Exactly one group-feasible optimum.
* No single agent can identify it from own-zone inspects plus own constraint.
* Partner-only decisive facts exist so other-regarding sharing is measurable.

**Current limitation:** the generator supports **2 agents** only. This page does
not claim 3-agent hidden-profile support.

Scoring vs process metrics
--------------------------

* **Outcome (``episode_score``):** venue quality x arrival / convergence.
* **Process metrics (``social_metrics.json``):**

  * **Structured ``shared_facts`` (exact):** claims attached to messages are
    evaluated exactly against the sender's first-hand inspection records and
    scenario decision facts. Categories include first-hand supported,
    unsupported, contradictory, duplicate/redundant, and partner-relevant
    claims (plus exact sharing completeness when applicable).
  * **Legacy free-text (heuristic):** venue/trait co-mention extraction for
    sharing completeness, other-regarding ratio, redundancy, necessity
    (``must_pool``), and related diagnostics. These remain approximate and are
    retained for comparison with the structured path.

Ablations: ``main``, ``no_communication``, ``no_coarse_map``,
``shared_constraints``, ``full_shared_information`` (``--ablation-matrix``).

Setup and commands
------------------

1. Install the SimWorld Python environment for this repository.
2. For live runs, start the UE SimWorld backend with UnrealCV reachable
   (default ``127.0.0.1:9000``).
3. Configure provider credentials via environment variables expected by the
   client. Defaults: ``--provider minimax``, ``--model MiniMax-M3``. Do not put
   secrets on the CLI.

.. code-block:: bash

   # Dry-run (no UE)
   python -m benchmark.venue_meetup.run_venue_eval \
     --dry-run --small-eval --hidden-profile --num-agents 2 \
     --info-partition spatial --seeds 7

   # Live scripted smoke
   python -m benchmark.venue_meetup.run_venue_eval \
     --hidden-profile --info-partition spatial --policy scripted --seeds 7

   # Live VLM
   python -m benchmark.venue_meetup.run_venue_eval \
     --hidden-profile --info-partition spatial --policy minimax --seeds 7

   # Walk mode
   python -m benchmark.venue_meetup.run_venue_eval \
     --template-id station_quarter_medium_v1 --walk --policy scripted --seeds 7

Artifact layout and manifest
----------------------------

.. code-block:: text

   runs/venue_meetup/<run_name>/
     run_manifest.json
     summary.json
     <template_id>/<scenario_id>/<ablation>/
       scenario_*.json, trajectory, social_metrics, metadata, ...

``run_manifest.json`` is written at the run root **before** cases execute. It
includes schema version, timestamp, git commit (or ``null``), sanitized CLI/config
args, discoverable runtime/package versions, template/scenario ids, seeds, agent
counts, ablations, and navigation mode.

Trajectory renderer
-------------------

.. code-block:: bash

   python -m benchmark.venue_meetup.render_trajectory --run-dir <case_dir>
   python -m benchmark.venue_meetup.render_trajectory --run-dir <case_dir> --map-only

Known limitations
-----------------

* Ego camera is third-person (the agent sees its own back).
* UE preflight has not been claimed for the new medium/large maps here; do not
  treat this page as evidence of completed UE validation.
* The 3x3/5x5/7x7 scale tiers require per-tier live preflight after geometry or
  packaged-asset changes.
* Default provider assumptions favor MiniMax-compatible APIs.
* V2 ``skill_check`` is absent; hidden profile is 2-agent only.
* Structured ``shared_facts`` metrics are exact claim checks; legacy free-text
  process metrics remain heuristic.
