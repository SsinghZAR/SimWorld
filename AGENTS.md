# SimWorld Project Instructions

This repository contains the SimWorld Python client and the Venue Meetup
benchmark. Before changing or running the benchmark, read
[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md). It records the local Unreal Engine
package location, the supported maps, launch commands, smoke tests, artifact
locations, and known runtime constraints.

For Venue Meetup work, treat the following documents and modules as the source
of truth:

- `venue_meetup_v0_implementation_checklist.md` — original implementation plan.
- `benchmark/venue_meetup/notes.md` — experiment design and hidden-profile
  rationale.
- `benchmark/venue_meetup/cleanup.md` — remaining cleanup and city-scale plan.
- `benchmark/venue_meetup/README.md` — benchmark interface and outputs.
- `benchmark/venue_meetup/district_scene.py` and `scene_builder.py` — runtime
  UE scene construction.

Keep generated artifacts under `runs/` (which is ignored). Preserve the
benchmark's distinction between social-evaluation results and physical
navigation diagnostics. Do not place credentials in commands, scripts, or
commits.
