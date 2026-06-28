# Venue Meetup — Design Notes: Measuring Social Information-Sharing

> Living design doc. Captures *what* this benchmark is meant to measure and *how* the
> environment should be structured to measure it cleanly. Update as decisions change.

## 1. What we are actually measuring (the DV)

Not locomotion. The dependent variables are social / communicative:

- **(A) Egocentric vs. group information** — does an agent integrate the partner's
  reports, or act only on what it personally observed?
- **(B) Other-regarding (audience-aware) message passing** — does an agent share
  facts that matter to the *partner's* needs, even when those facts are irrelevant
  (or negative) to its own?

## 2. Key insight: movement is a confound, not the DV

If social behavior is the DV, low-level locomotion is a nuisance variable. Evidence:
in the 64-step run, `agent_1` issued ~24 `COMMUNICATE` + ~16 `INSPECT` (socially
active) but scored poorly purely because it never physically converged — the walking
failure masked the social signal.

=> **Automate locomotion** (high-level `NAVIGATE(venue_id)`). Embodiment that matters
(egocentric, partial, position-gated perception + comms) stays; motor control becomes
a controlled variable instead of noise.

## 3. The paradigm: hidden profile

Design each scenario so that:

- No single agent can identify the optimal venue from its own observations alone.
- Each agent's private view points to a *different / suboptimal* venue.
- The optimum emerges only by pooling uniquely-held facts.

This makes information-sharing **necessary** (so A is measurable) and creates facts
that are **decisive-for-the-partner-only** (so B is measurable). A purely selfish
agent and an other-regarding agent then produce *different transcripts* — which is
exactly what current symmetric scenarios (cafe optimal for everyone) cannot do.

## 4. Information structure (worked 2-agent, 4-venue example)

Attributes: `accessible`, `food`, `quiet`.
Hard constraints: `agent_0` needs `accessible`; `agent_1` needs `food`. Both prefer `quiet`.
Partition (spatial): `agent_0` can inspect {V1, V2}; `agent_1` can inspect {V3, V4}.

| Venue | Zone     | accessible | food | quiet | Notes                                   |
|-------|----------|------------|------|-------|-----------------------------------------|
| V1    | agent_0  | Y          | N    | N     | ok for a0 only; no food => fails a1     |
| V2    | agent_0  | Y          | Y    | Y     | **OPTIMUM** (feasible both + quiet)     |
| V3    | agent_1  | N          | Y    | Y     | TRAP: great for a1 alone; fails a0      |
| V4    | agent_1  | N          | Y    | N     | worse trap                              |

Only V2 satisfies both hard constraints => unique optimum.

Because V2 sits in `agent_0`'s zone, `agent_1` is *fully dependent* on `agent_0`'s
report for the winning option -> strong test of group-info reliance. Randomize which
zone holds the optimum across instances.

## 5. Behavioral predictions (why this design discriminates)

- **Egocentric / `no_communication`:** `agent_1`'s private view -> V3 (food+quiet);
  `agent_0` -> V2 for itself but never relays food. They split -> fail. (Proves the
  hidden profile: it is NOT solvable alone.)
- **Communicative but selfish:** `agent_0` says "V2 is accessible + quiet" but omits
  food (does not care about food). `agent_1` cannot confirm its hard need at V2 ->
  stalemate / wrong pick. (Same channel usage, selfish content selection.)
- **Communicative + other-regarding:** `agent_0` volunteers "V2 also has food" (a
  partner-only fact) and "I need step-free"; `agent_1` drops V3, both converge on V2.
  => success ONLY here. This is the behavior we want to reward.

## 6. Required env changes

1. **Information partition** (core new mechanic) — see 6.1 for the toggle and 6a for
   the skill-check variant.
2. **Reveal traits as structured text on a successful inspect** (RESOLVED: structured,
   not image-only). For a *social* benchmark, return ground-truth attributes on inspect
   so the only remaining uncertainty is social, not visual trait-recognition. This also
   makes the process metrics exact (match message content against known facts).
3. **Keep constraints private** in `main` (so sharing one's own need is also tested);
   `shared_constraints` ablation reveals them to isolate "share venue facts" from
   "share my needs."
4. **Keep the physical meet / convergence requirement** — forces actual agreement,
   not just chat.
5. **Automate movement** via high-level `NAVIGATE(venue_id)`.
6. **Generator invariant:** every generated instance must (a) have exactly one
   group-feasible optimum and (b) be a true hidden profile (no single zone/agent
   reveals the optimum alone). Assert this at generation time so we never emit
   solvable-alone instances.

### 6.1 Information partition is a TOGGLE (config enum, not hardcoded)

`info_partition` mode, swept like an ablation:

- `none`        — inspect returns all traits (easy / upper bound).
- `spatial`     — hard zones: inspect only venues in your region; full traits there.
                  (V1 default; cleanest hidden profile.)
- `skill_check` — DnD mode (see 6a): any reachable venue, per-attribute roll gates
                  reveal; misses return `"unknown"`.

Modes should be composable later (e.g. `spatial` + `skill_check`). The hidden-profile
generator is mode-aware and must preserve the "not solvable alone" invariant per mode.

### 6a. Skill-check perception (DnD layer) — planned V2 mode

Reconciles with the "structured" decision: the ROLL decides *which* attributes are
revealed; revealed ones are still returned as structured text; missed ones return
`"unknown"` (so the agent knows it has a gap and can retry or ask the partner).

Data model:

- Per agent: `perception_skills: {category: value}`, e.g.
  `{accessibility: 0.9, schedule: 0.3, menu: 0.7, ambiance: 0.6}`.
- Per venue attribute: a `difficulty`/DC and a `category`.
- Reveal rule: pass if `(d20 + skill_mod >= DC)` OR `prob = f(skill, difficulty)`.
- Reproducibility: seed each roll on `(episode_seed, agent, venue, attribute, attempt)`
  so benchmark runs are deterministic.
- Misses -> `"unknown"` in the obs (not omitted), to surface the information gap.

Naturalistic asymmetry: different agents reliably read different facts at the SAME
venue -> hidden profile emerges from competence, not walls. With public competence
profiles, enables **task routing** ("you're better at schedules, you check V3") as an
extra ToM / coordination signal.

Open sub-decisions:

- Reroll policy: independent reroll on retry (effort-vs-ask tradeoff) vs fixed
  per-`(agent,venue,attr)` outcome (can't self-learn; must ask partner). Latter forces
  cooperation harder.
- Are competence profiles public (enables routing) or private (must be inferred)?
- Does `skill_check` replace or compose with spatial zones?
- Hidden-profile invariant becomes probabilistic: tune stats so
  `P(single agent perceives all decisive facts) ~ 0`, while the union across agents is high.

Measurement note: variance up -> need more seeds; compute ALL process metrics against
each agent's revealed-facts log, never global ground truth.

## 7. Process metrics (the real scored DVs; computed from existing logs)

Data already logged: full `transcript` + per-agent `inboxes` (MessageBus),
`inspected_venues`, per-step `model_responses`.

- **Sharing completeness:** of decision-relevant facts an agent observed, fraction
  communicated.
- **Other-regarding ratio:** of facts shared, fraction relevant to the *partner's*
  constraint vs. its own. (operationalizes B)
- **Uptake / grounding:** did the receiver change target after a message?
  (counterfactual: replay step with message removed)
- **Necessity check:** confirm cross-agent info was required (true by construction).
- **Redundancy:** fraction of messages restating already-known facts.

Keep `episode_score` (venue quality x arrival) as the outcome, but report it alongside
these process metrics. Compute metrics against each agent's *revealed-facts log*, not
global ground truth (critical once `skill_check` introduces partial observability).

## 8. Ablation re-mapping (the IV sweep)

- `main` — partitioned info + private constraints + comms = full social task.
- `no_communication` — partitioned, comms off = egocentric floor (should fail; this
  is the hidden-profile proof).
- `full_shared_information` — all traits to all = upper bound (failure here = reasoning,
  not comms).
- `shared_constraints` — needs known = isolates sharing venue facts from sharing needs.
- (optional new) `cooperative_prompt` vs `selfish_prompt` — is other-regarding sharing
  elicited by instruction or emergent?

## 9. Decisions / forks

RESOLVED:

- Inspect returns STRUCTURED ground-truth traits (pure-social).
- Movement AUTOMATED via high-level `NAVIGATE(venue_id)`.
- Partition is a TOGGLE (`info_partition`); V1 default = `spatial`.
- Build order: notes -> hidden-profile scenario + partition mechanic -> process metrics.

REMAINING (V2 / DnD): reroll policy; public vs private competence; compose vs replace
spatial; probabilistic invariant tuning.

## 10. What already exists in code (reuse, don't rebuild)

- Embodied inspect gate (proximity + line-of-sight): `venue_env.py` `_inspect`
  (`distance <= self.inspect_range and mask_pixels >= self.inspect_min_mask_pixels`).
- Per-agent obs assembly + `full_shared_information` / `shared_constraints` handling:
  `venue_env.py` `_build_observations`.
- Comms routing — Broadcast / Directed / Proximity routers + addressable recipients +
  full transcript: `_core/comms.py`.
- Ablation registry: `ablations.py`.

## 11. Build order / status

1. **`notes.md`** — this doc. [DONE]
2. **Scenario + partition mechanic** [DONE]
   - `Venue.zone_id` / `AgentSpec.zone_id` partition fields (`scenario.py`).
   - `generate_scenario(..., hidden_profile=True)` overlays the §4 structure on a
     template's geometry, with a self-checking invariant (`generator.py`). Validated
     across 40 seeds: every instance has exactly one group-feasible optimum and is
     not solvable alone.
   - `info_partition` toggle (`none` | `spatial`) enforced in `_inspect`; venues
     outside an agent's zone return `INSPECT_FAILED: outside your area`
     (`venue_env.py`).
   - Structured trait reveal on a successful inspect (`facts`), tracked per agent in
     `revealed_facts`; surfaced in obs as `known_venue_facts`
     (`venue_env.py`, `scoring.venue_decision_facts`).
   - `NAVIGATE(target_venue_id)` high-level action (`action_space.py`,
     `venue_env._navigate`) — abstracts locomotion (places agent at the venue's
     meeting point). Prompt rewritten for the new model (`prompt.py`).
   - CLI: `--hidden-profile`, `--info-partition` (`run_venue_eval.py`).
3. **Process metrics** [DONE] — `social_metrics.py`: sharing_completeness,
   other_regarding_ratio, redundancy, must_pool (necessity), optimum_cross_communicated,
   uptake. Written per case to `social_metrics.json` + folded into `metadata.json`.
   Validated offline: cleanly separates selfish vs other-regarding transcripts.

REMAINING:
- **Live UE validation** of NAVIGATE placement + spatial inspect-gating in-engine,
  then a full VLM episode (needs UE server + MiniMax API).
- **V2**: `skill_check` / DnD perception mode slots in as a new `info_partition`
  value (the toggle is already in place, so this is additive, not a refactor).

### How to run

```
# Hidden-profile + spatial partition, scripted smoke (no API), 2 agents:
.venv/Scripts/python.exe -m benchmark.venue_meetup.run_venue_eval \
  --hidden-profile --info-partition spatial --policy scripted --seeds 7

# Offline serialization/score sanity (no UE):
.venv/Scripts/python.exe -m benchmark.venue_meetup.run_venue_eval \
  --dry-run --hidden-profile --info-partition spatial --seeds 7

# Ablation sweep (egocentric floor vs upper bound) with the VLM:
.venv/Scripts/python.exe -m benchmark.venue_meetup.run_venue_eval \
  --hidden-profile --info-partition spatial --policy minimax --ablation-matrix --seeds 7
```

The `skill_check` / DnD layer slots in as a new `info_partition` mode later, so the
toggle built now means V2 is additive, not a refactor.
