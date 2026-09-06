# Targeted, timed Venue Meetup

Approved design (2026-09-06): two agents, four physical information sources per
venue, independent multi-tick actions on a shared clock, and varied private needs.
Time is the only action cost. No intermediate reward, score, efficiency label, or
strategic feedback is sent to the agents. Operational results and the clock remain
visible. Final scoring checks a valid meetup by closing, inclusive of the deadline.

## Requirements (stage 3)

Use six existing categories first: accessibility, food/drink, shelter, quietness,
low crowding, and enough seating for both visitors. Sample one or two explicit
hard needs per agent, with a distinct need for each partner. All venues must be
open and have an unobstructed approach. Keep exactly one group-feasible venue,
an optimum-zone decoy satisfying its observer but failing the partner, and at
least two partner-zone traps. Randomize identities, needs, and irrelevant facts;
do not encode feasibility in interaction IDs, colors, or availability. These
structural checks do not prove that an uninformed agent cannot guess correctly.
Dietary and monetary constraints are deferred until matching menu data exists.

## Interactions (stage 1)

- Hours notice: open state and posted common closing time (1 tick).
- Access information point: accessible/reachable (2 ticks).
- Menu/services board: food_drink (2 ticks).
- Meeting-area information point: capacity, shelter, quiet, uncrowded (3 ticks).

Each point has its own actor, label, position, visibility mask, and stable ID.
Only selected evidence is revealed at completion; memory merges partial findings.
Unknown is not false. Failures reveal no facts. Repeat checks are deterministic.
Existing spatial permissions remain. Staff dialogue is deferred.

## Timing (stage 2)

One tick is 30 simulated seconds. Default start 17:30, close 18:00; these are
independent of the API-call safety cap. Travel uses authored walkable route length
at 40 m/tick, rounded up. Communication (512 characters), wait, and fine movement
cost one tick. Busy agents receive no policy calls and cannot interrupt actions.
Independent actions progress together. Evidence and messages arrive at completion.
No unfinished action completes after closing. Public activity contains only the
acting agent's state, not its partner's destination or evidence.

## Delivery gates

1. Pure deterministic interaction/profile/scheduler tests and leakage checks.
2. Compatibility tests for the explicitly selected legacy protocol.
3. Live interaction visibility and walking smoke, with screenshots.
4. Scripted solvability check, then two-agent 5x5 MiniMax episode, saved transcript
   and timestamped movement replay. Treat the first budget as calibration, not
   a validated benchmark difficulty tier.

Implementation status: complete and smoke-tested. The default CLI protocol is
targeted; legacy is explicitly selectable. Prompts and durations share
configuration. Clock, source registry, UE source adapter, scheduler, timed
navigation and varied profiles are separate modules rather than additions to
the legacy environment monolith.

## Validation / artifacts (2026-09-06)

- 390 offline tests passed; compilation, import normalization and whitespace
  checks passed. Profile checks cover 24 seeds on each of 3x3, 5x5 and 7x7.
- Final live 5x5 visibility sweep passed all 32 sources. Minimum mask size 535
  pixels, threshold 20. Screenshots/report: `runs/venue_meetup/targeted_final_preflight_seed7/`.
- Observation-only scripted 5x5 walk: score 1.0, 41 ticks, 17:50:30 meetup.
  Run: `runs/venue_meetup/targeted_v1_scripted_walk_seed7/`.
- Final MiniMax-M3 5x5 walk, seed 7, two agents: score 1.0, valid D2 skyscraper
  lobby meetup at 17:48:00, 36 ticks, 38 model decisions, 18 messages, five venues
  inspected, no provider/parse failures. Run:
  `runs/venue_meetup/targeted_v1_minimax_walk_seed7_final/`.
- The earlier MiniMax run is preserved under `targeted_v1_minimax_walk_seed7`.
  It exposed a within-region navigation status/timing edge case, corrected and
  regression-tested before the final run. The final arrival reports
  `NAVIGATE_OK` and consumes all seven advertised travel ticks.

This establishes an operational smoke-tested protocol, not calibrated benchmark
difficulty. The final MiniMax run retained 12 simulated minutes, so budget/seed
and communication-ablation sweeps remain necessary before research claims.
Staff dialogue, dietary/price data and custom-authored sign assets remain outside
this first implementation. Information panels use neutral prototype geometry
with role-only camera annotations; their contents remain gated readable evidence.
