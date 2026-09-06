"""Post-episode movement overlays on the public Venue Meetup coarse map."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - cv2 ships with eval extras.
    raise RuntimeError("OpenCV (cv2) is required to render trajectory minimaps.") from exc

from benchmark.venue_meetup.coarse_map import (
    coarse_map_extent,
    render_coarse_map,
    world_to_map_pixel,
)
from benchmark.venue_meetup.scenario import Scenario, scenario_from_dict

AGENT_BGR: tuple[tuple[int, int, int], ...] = (
    (220, 95, 25),    # agent_0 - blue
    (45, 45, 220),    # agent_1 - red
    (0, 150, 230),    # agent_2 - amber
    (70, 170, 70),    # agent_3 - green
)


def _point(value: Any) -> tuple[float, float] | None:
    """Normalize a serialized 2D/3D point, rejecting malformed values."""

    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _append_distinct(points: list[tuple[float, float]], point: tuple[float, float]) -> None:
    if not points or math.dist(points[-1], point) > 0.01:
        points.append(point)


def movement_history(
    scenario: dict[str, Any],
    trajectory: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return per-agent, per-turn movement traces from evaluator logs.

    New trajectories contain measured intra-turn points in
    ``movement_paths_internal``. Older trajectories fall back to their final
    per-turn positions so archived episodes remain renderable.
    """

    starts = {
        str(agent["agent_id"]): _point(agent.get("position"))
        for agent in scenario.get("agents", [])
        if isinstance(agent, dict) and agent.get("agent_id")
    }
    histories: dict[str, list[dict[str, Any]]] = {
        agent_id: [] for agent_id, start in starts.items() if start is not None
    }
    last_positions = {
        agent_id: start for agent_id, start in starts.items() if start is not None
    }

    for turn_index, entry in enumerate(trajectory):
        info = entry.get("info") if isinstance(entry.get("info"), dict) else {}
        traces = info.get("movement_paths_internal") if isinstance(info, dict) else {}
        traces = traces if isinstance(traces, dict) else {}
        positions = info.get("positions_internal") if isinstance(info, dict) else {}
        positions = positions if isinstance(positions, dict) else {}
        actions = info.get("actions") if isinstance(info, dict) else {}
        actions = actions if isinstance(actions, dict) else {}
        turns = entry.get("turns") if isinstance(entry.get("turns"), dict) else {}
        step = int(entry.get("step", turn_index))

        for agent_id in histories:
            points: list[tuple[float, float]] = []
            _append_distinct(points, last_positions[agent_id])
            raw_trace = traces.get(agent_id)
            if isinstance(raw_trace, list):
                for raw_point in raw_trace:
                    normalized = _point(raw_point)
                    if normalized is not None:
                        _append_distinct(points, normalized)

            final_position = _point(positions.get(agent_id))
            if final_position is not None:
                _append_distinct(points, final_position)
            last_positions[agent_id] = points[-1]

            action = actions.get(agent_id) if isinstance(actions.get(agent_id), dict) else {}
            turn = turns.get(agent_id) if isinstance(turns.get(agent_id), dict) else {}
            choice = turn.get("choice", (action.get("turn") or {}).get("choice"))
            navigate_mode = action.get("mode", info.get("navigation_mode"))
            if choice == 5 and navigate_mode != "walk":
                movement_kind = "teleport"
            else:
                movement_kind = "physical"
            histories[agent_id].append(
                {
                    "step": step,
                    "points": points,
                    "kind": movement_kind,
                    "result": str(action.get("result", "")),
                    "moved": len(points) > 1,
                }
            )
    return histories


def _text(
    image: Any,
    label: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.45,
    color: tuple[int, int, int] = (30, 30, 30),
    thickness: int = 1,
) -> None:
    """Draw outlined text that remains legible over the map."""

    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = (max(4, min(origin[0], image.shape[1] - text_width - 4)),
              max(text_height + 3, min(origin[1], image.shape[0] - 5)))
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _dashed_line(
    image: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 3,
    dash_px: float = 12.0,
) -> None:
    """Draw a dashed line used specifically for abstracted teleport movement."""

    distance = math.dist(start, end)
    if distance <= 0.0:
        return
    dx = (end[0] - start[0]) / distance
    dy = (end[1] - start[1]) / distance
    cursor = 0.0
    while cursor < distance:
        finish = min(cursor + dash_px, distance)
        p0 = (int(round(start[0] + dx * cursor)), int(round(start[1] + dy * cursor)))
        p1 = (int(round(start[0] + dx * finish)), int(round(start[1] + dy * finish)))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)
        cursor += dash_px * 1.75


def _draw_history(
    base: Any,
    *,
    histories: dict[str, list[dict[str, Any]]],
    scenario: Scenario,
    upto_turn: int | None,
) -> Any:
    """Draw histories through ``upto_turn`` over a copy of the coarse map."""

    image = base.copy()
    size = int(image.shape[0])
    extent = coarse_map_extent(scenario)
    project = lambda point: world_to_map_pixel(point, size=size, extent=extent)

    for agent_index, (agent_id, steps) in enumerate(sorted(histories.items())):
        color = AGENT_BGR[agent_index % len(AGENT_BGR)]
        visible = steps if upto_turn is None else steps[: upto_turn + 1]
        if not visible:
            continue
        start_world = visible[0]["points"][0]
        start = project(start_world)
        cv2.circle(image, start, 8, color, 2, cv2.LINE_AA)
        _text(image, f"{agent_id} start", (start[0] + 8, start[1] - 8), color=color)

        final_point = start
        for step_data in visible:
            points = step_data["points"]
            for point_index in range(1, len(points)):
                p0 = project(points[point_index - 1])
                p1 = project(points[point_index])
                if step_data["kind"] == "teleport":
                    _dashed_line(image, p0, p1, color)
                else:
                    cv2.line(image, p0, p1, color, 3, cv2.LINE_AA)
            if step_data["moved"]:
                final_point = project(points[-1])
                cv2.circle(image, final_point, 4, color, -1, cv2.LINE_AA)
                _text(image, str(step_data["step"] + 1), (final_point[0] + 5, final_point[1] - 5), scale=0.36, color=color)

        cv2.drawMarker(image, final_point, color, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)

    return image


def _title_and_legend(
    image: Any,
    *,
    steps: int,
    upto_turn: int | None,
    clock_state: dict[str, Any] | None = None,
) -> None:
    """Add post-run context and a compact two-agent legend."""

    width = int(image.shape[1])
    cv2.rectangle(image, (0, 0), (width, 39), (255, 255, 255), -1)
    completed = upto_turn + 1 if upto_turn is not None else steps
    if clock_state:
        total_turns = completed + int(clock_state.get("turns_remaining", 0))
        unit = "tick" if "tick_seconds" in clock_state else "turn"
        label = (
            f"Movement minimap | {unit} {completed}/{total_turns} | "
            f"{clock_state.get('current_time')} | {clock_state.get('minutes_remaining')} min to close"
        )
        scale = 0.48
    else:
        label = f"Movement minimap - turn {completed}/{steps}"
        scale = 0.62
    _text(image, label, (18, 25), scale=scale, color=(20, 20, 20), thickness=1)

    left = max(12, width - 228)
    cv2.rectangle(image, (left, 116), (width - 12, 185), (248, 248, 248), -1)
    cv2.rectangle(image, (left, 116), (width - 12, 185), (120, 120, 120), 1)
    for index, agent_id in enumerate(("agent_0", "agent_1")):
        y = 136 + index * 19
        cv2.line(image, (left + 10, y), (left + 38, y), AGENT_BGR[index], 3, cv2.LINE_AA)
        _text(image, agent_id, (left + 46, y + 4), scale=0.4, color=AGENT_BGR[index])
    _dashed_line(image, (left + 117, 136), (left + 155, 136), (70, 70, 70), thickness=2, dash_px=7)
    _text(image, "teleport", (left + 161, 140), scale=0.34, color=(70, 70, 70))
    unit_label = "numbers = elapsed ticks" if clock_state and "tick_seconds" in clock_state else "numbers = movement turn"
    _text(image, unit_label, (left + 10, 178), scale=0.34, color=(80, 80, 80))


def _activity_overlay(image: Any, info: dict[str, Any]) -> None:
    """Evaluator-only replay labels; never part of an agent observation."""

    activities = info.get("activities_internal")
    if not activities:
        return
    names = {0: "WAIT", 1: "STEP", 2: "TURN", 3: "INSPECT", 4: "COMMUNICATE", 5: "NAVIGATE"}
    cv2.rectangle(image, (0, 39), (image.shape[1], 83), (250, 250, 250), -1)
    for index, (agent, state) in enumerate(sorted(activities.items())):
        if state.get("status") == "busy":
            description = f"{names.get(state.get('choice'), 'ACTION')} - {state['ticks_remaining']} ticks left"
        else:
            description = info.get("actions", {}).get(agent, {}).get("result", "ready")
        _text(image, f"{agent}: {description}", (18, 56 + index * 19), scale=.4, color=AGENT_BGR[index])


def _coarse_map_path(run_dir: Path, scenario: Scenario) -> Path:
    """Resolve or recreate the public coarse-map image for a case directory."""

    candidates = [run_dir / f"{scenario.scenario_id}_coarse_map.png"]
    if scenario.coarse_map_path:
        supplied = Path(scenario.coarse_map_path)
        candidates.append(supplied if supplied.is_absolute() else Path.cwd() / supplied)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return render_coarse_map(scenario, candidates[0])


def render_trajectory_minimap(
    run_dir: Path,
    *,
    fps: float = 2.0,
    write_video: bool = True,
) -> dict[str, Path]:
    """Render static and animated agent paths for one finished case."""

    scenario_payload = json.loads((run_dir / "scenario_hidden.json").read_text(encoding="utf-8"))
    trajectory = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    scenario = scenario_from_dict(scenario_payload)
    base_path = _coarse_map_path(run_dir, scenario)
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if base is None:
        raise RuntimeError(f"Could not load coarse map {base_path}")
    if base.shape[0] != base.shape[1]:
        raise ValueError(f"Coarse map must be square, got {base.shape[1]}x{base.shape[0]}")

    histories = movement_history(scenario_payload, trajectory)
    final = _draw_history(base, histories=histories, scenario=scenario, upto_turn=None)
    final_clock = (
        trajectory[-1].get("info", {}).get("closing_clock")
        if trajectory
        else None
    )
    _title_and_legend(
        final,
        steps=len(trajectory),
        upto_turn=None,
        clock_state=final_clock,
    )
    png_path = run_dir / "trajectory_minimap.png"
    if trajectory:
        _activity_overlay(final, trajectory[-1].get("info", {}))
    if not cv2.imwrite(str(png_path), final):
        raise RuntimeError(f"Could not write trajectory minimap {png_path}")

    outputs = {"trajectory_minimap": png_path}
    if not write_video:
        return outputs

    mp4_path = run_dir / "trajectory_minimap.mp4"
    size = int(base.shape[0])
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create trajectory minimap video {mp4_path}")
    try:
        for turn_index in range(len(trajectory)):
            frame = _draw_history(base, histories=histories, scenario=scenario, upto_turn=turn_index)
            clock_state = trajectory[turn_index].get("info", {}).get("closing_clock")
            _title_and_legend(
                frame,
                steps=len(trajectory),
                upto_turn=turn_index,
                clock_state=clock_state,
            )
            _activity_overlay(frame, trajectory[turn_index].get("info", {}))
            writer.write(frame)
        for _ in range(max(1, int(fps * 2))):
            writer.write(final)
    finally:
        writer.release()
    outputs["trajectory_minimap_video"] = mp4_path
    return outputs
