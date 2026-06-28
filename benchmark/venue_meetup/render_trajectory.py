"""Top-down trajectory renderer for Venue Meetup runs.

Reads a finished run's ``scenario_hidden.json`` (venue geometry) and
``trajectory.json`` (true per-step agent positions logged under
``info.positions_internal``) and renders a bird's-eye view that makes agent
locomotion legible: where each agent walked, where it was physically blocked,
and that it halts at a building's collision footprint instead of passing
through it.

Outputs ``topdown.png`` (all paths) and ``topdown.mp4`` (paths revealed step by
step) next to the trajectory file. Uses only numpy + OpenCV, which the eval
pipeline already depends on for video, so no extra packages are required.

Usage::

    python -m benchmark.venue_meetup.render_trajectory --run-dir <case_dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - cv2 ships with the eval extras.
    raise RuntimeError("OpenCV (cv2) is required to render trajectories.") from exc


AGENT_BGR = [
    (255, 196, 0),    # agent_0 - cyan/blue
    (180, 80, 255),   # agent_1 - magenta
    (0, 215, 255),    # agent_2 - amber
    (120, 220, 120),  # agent_3 - green
]


def rgb_to_bgr(rgb: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert an RGB triple (as stored in scenario JSON) to OpenCV BGR."""

    r, g, b = (int(channel) for channel in rgb)
    return (b, g, r)


class TopDownView:
    """Maps world centimetres to image pixels with north = up, east = right."""

    def __init__(self, points: list[tuple[float, float]], *, size: int = 1024, margin_cm: float = 1500.0):
        xs = [p[0] for p in points] or [0.0]
        ys = [p[1] for p in points] or [0.0]
        self.min_x, self.max_x = min(xs) - margin_cm, max(xs) + margin_cm
        self.min_y, self.max_y = min(ys) - margin_cm, max(ys) + margin_cm
        span = max(self.max_x - self.min_x, self.max_y - self.min_y, 1.0)
        cx = (self.min_x + self.max_x) / 2.0
        cy = (self.min_y + self.max_y) / 2.0
        self.min_x, self.max_x = cx - span / 2.0, cx + span / 2.0
        self.min_y, self.max_y = cy - span / 2.0, cy + span / 2.0
        self.size = size
        self.scale = size / span

    def px(self, x: float, y: float) -> tuple[int, int]:
        """World (x east, y north) -> pixel (col, row), flipping y for image space."""

        col = (x - self.min_x) * self.scale
        row = (self.max_y - y) * self.scale
        return int(round(col)), int(round(row))

    def length(self, cm: float) -> int:
        """Convert a world length to pixels."""

        return max(1, int(round(cm * self.scale)))


def _blend_circle(img: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int], alpha: float) -> None:
    """Draw a translucent filled circle in place."""

    overlay = img.copy()
    cv2.circle(overlay, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def _dashed_circle(img: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int], *, dashes: int = 40) -> None:
    """Draw a dashed circle outline (region marker)."""

    for k in range(dashes):
        if k % 2:
            continue
        a0 = 2 * math.pi * k / dashes
        a1 = 2 * math.pi * (k + 1) / dashes
        p0 = (int(center[0] + radius * math.cos(a0)), int(center[1] - radius * math.sin(a0)))
        p1 = (int(center[0] + radius * math.cos(a1)), int(center[1] - radius * math.sin(a1)))
        cv2.line(img, p0, p1, color, 2, lineType=cv2.LINE_AA)


def _text(img: np.ndarray, label: str, org: tuple[int, int], *, scale: float = 0.5, color: tuple[int, int, int] = (40, 40, 40), thickness: int = 1) -> None:
    """Draw text with a light outline for readability over any background."""

    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_static_scene(view: TopDownView, scenario: dict[str, Any]) -> np.ndarray:
    """Render the fixed backdrop: grid, venues (collision blobs + regions), landmarks."""

    img = np.full((view.size, view.size, 3), 248, dtype=np.uint8)

    # Light grid every 10 m plus bold axes through the plaza origin.
    step_cm = 1000.0
    gx = math.floor(view.min_x / step_cm) * step_cm
    while gx <= view.max_x:
        col, _ = view.px(gx, 0.0)
        cv2.line(img, (col, 0), (col, view.size), (232, 232, 232), 1, cv2.LINE_AA)
        gx += step_cm
    gy = math.floor(view.min_y / step_cm) * step_cm
    while gy <= view.max_y:
        _, row = view.px(0.0, gy)
        cv2.line(img, (0, row), (view.size, row), (232, 232, 232), 1, cv2.LINE_AA)
        gy += step_cm
    ox, oy = view.px(0.0, 0.0)
    cv2.line(img, (ox, 0), (ox, view.size), (210, 210, 210), 1, cv2.LINE_AA)
    cv2.line(img, (0, oy), (view.size, oy), (210, 210, 210), 1, cv2.LINE_AA)
    _text(img, "N", (ox - 6, 18), scale=0.6, color=(120, 120, 120))
    _text(img, "E", (view.size - 22, oy - 8), scale=0.6, color=(120, 120, 120))

    for landmark in scenario.get("landmarks", []):
        lx, ly, *_ = landmark["position"]
        center = view.px(lx, ly)
        color = rgb_to_bgr(landmark.get("mask_color_rgb", [120, 120, 120]))
        cv2.drawMarker(img, center, color, cv2.MARKER_TRIANGLE_UP, 18, 2, cv2.LINE_AA)
        _text(img, landmark.get("landmark_type", "landmark"), (center[0] + 10, center[1]), scale=0.42, color=(110, 110, 110))

    for venue in scenario.get("venues", []):
        vx, vy, *_ = venue["position"]
        pivot = view.px(vx, vy)
        color = rgb_to_bgr(venue.get("mask_color_rgb", [120, 120, 120]))
        region = venue.get("region", {})
        rcenter = region.get("center", [vx, vy])
        meet = view.px(rcenter[0], rcenter[1])
        # Approximate collision footprint: a disk from the pivot out to the
        # plaza-side meeting point, i.e. the face an approaching agent hits.
        reach_cm = math.hypot(vx - rcenter[0], vy - rcenter[1])
        _blend_circle(img, pivot, view.length(max(reach_cm, 250.0)), color, alpha=0.30)
        cv2.circle(img, pivot, view.length(max(reach_cm, 250.0)), color, 2, cv2.LINE_AA)
        cv2.drawMarker(img, pivot, color, cv2.MARKER_SQUARE, 14, 2, cv2.LINE_AA)
        # Arrival/convergence region centred on the plaza-side meeting point.
        _dashed_circle(img, meet, view.length(region.get("radius", 1200.0)), color)
        _text(img, venue.get("venue_id", "venue").replace("venue_", ""), (pivot[0] + 10, pivot[1] + 4), scale=0.44, color=tuple(int(c * 0.6) for c in color))

    return img


def _agent_paths(scenario: dict[str, Any], trajectory: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Assemble per-agent point lists + per-step blocked flags from a run."""

    spawns = {agent["agent_id"]: (agent["position"][0], agent["position"][1]) for agent in scenario.get("agents", [])}
    paths: dict[str, dict[str, Any]] = {
        agent_id: {"points": [spawn], "blocked": [False]} for agent_id, spawn in spawns.items()
    }
    for entry in trajectory:
        info = entry.get("info", {})
        positions = info.get("positions_internal", {})
        actions = info.get("actions", {})
        for agent_id, path in paths.items():
            pos = positions.get(agent_id)
            if pos is None:
                continue
            path["points"].append((pos[0], pos[1]))
            result = (actions.get(agent_id) or {}).get("result", "")
            path["blocked"].append("BLOCKED" in str(result))
    return paths


def _draw_paths(img: np.ndarray, view: TopDownView, paths: dict[str, dict[str, Any]], *, upto: int | None = None) -> None:
    """Overlay agent trajectories (optionally only through step ``upto``)."""

    for index, (agent_id, path) in enumerate(sorted(paths.items())):
        color = AGENT_BGR[index % len(AGENT_BGR)]
        pts = path["points"]
        blocked = path["blocked"]
        last = len(pts) - 1 if upto is None else min(upto, len(pts) - 1)
        for k in range(1, last + 1):
            cv2.line(img, view.px(*pts[k - 1]), view.px(*pts[k]), color, 2, cv2.LINE_AA)
        for k in range(0, last + 1):
            center = view.px(*pts[k])
            cv2.circle(img, center, 3, color, -1, cv2.LINE_AA)
            if blocked[k]:
                cv2.drawMarker(img, center, (0, 0, 230), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
        start = view.px(*pts[0])
        cv2.circle(img, start, 7, color, 2, cv2.LINE_AA)
        _text(img, agent_id, (start[0] + 9, start[1] - 6), scale=0.46, color=color)
        head = view.px(*pts[last])
        cv2.drawMarker(img, head, color, cv2.MARKER_STAR, 16, 2, cv2.LINE_AA)


def _draw_arrow(img: np.ndarray, view: TopDownView, x: float, y: float, yaw_deg: float, color: tuple[int, int, int], *, length_cm: float = 700.0) -> None:
    """Draw a short facing arrow from a world point."""

    x2 = x + math.cos(math.radians(yaw_deg)) * length_cm
    y2 = y + math.sin(math.radians(yaw_deg)) * length_cm
    cv2.arrowedLine(img, view.px(x, y), view.px(x2, y2), color, 2, cv2.LINE_AA, tipLength=0.35)


def _venue_traits(properties: dict[str, Any]) -> str:
    """Compact human-readable trait tags for a venue."""

    tags = ["OPEN" if properties.get("open") else "CLOSED"]
    tags.append("step-free" if properties.get("accessible") else "stairs-only")
    if properties.get("food_drink"):
        tags.append("food/drink")
    if properties.get("quiet_score", 0.0) >= 0.65:
        tags.append("quiet")
    if properties.get("crowding_score", 0.0) >= 0.5:
        tags.append("busy")
    tags.append(f"cap{properties.get('capacity', '?')}")
    if properties.get("near_transit"):
        tags.append("transit")
    return " - ".join(tags)


def render_map(scenario: dict[str, Any], out_path: Path, *, size: int = 1000, panel_w: int = 660) -> Path:
    """Render an annotated top-down map of a scenario (no trajectory needed)."""

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for venue in scenario.get("venues", []):
        vx, vy = venue["position"][0], venue["position"][1]
        rcenter = venue.get("region", {}).get("center", [vx, vy])
        reach = math.hypot(vx - rcenter[0], vy - rcenter[1])
        # Include the full collision disk extent so big buildings are not clipped.
        points.extend([(vx + reach, vy), (vx - reach, vy), (vx, vy + reach), (vx, vy - reach)])
    for landmark in scenario.get("landmarks", []):
        points.append((landmark["position"][0], landmark["position"][1]))
    for agent in scenario.get("agents", []):
        points.append((agent["position"][0], agent["position"][1]))
    view = TopDownView(points, size=size)
    scene = _draw_static_scene(view, scenario)

    # Plaza "meeting" zone + per-venue meeting points.
    origin = view.px(0.0, 0.0)
    for venue in scenario.get("venues", []):
        rcenter = venue.get("region", {}).get("center", [0, 0])
        meet = view.px(rcenter[0], rcenter[1])
        color = rgb_to_bgr(venue.get("mask_color_rgb", [120, 120, 120]))
        cv2.drawMarker(scene, meet, color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    cv2.circle(scene, origin, 6, (60, 60, 60), -1, cv2.LINE_AA)
    _text(scene, "plaza centre (agents meet ~22 m out at each facade)", (origin[0] + 10, origin[1] + 18), scale=0.42, color=(60, 60, 60))

    for index, agent in enumerate(scenario.get("agents", [])):
        ax, ay, *_ = agent["position"]
        color = AGENT_BGR[index % len(AGENT_BGR)]
        center = view.px(ax, ay)
        cv2.circle(scene, center, 8, color, -1, cv2.LINE_AA)
        cv2.circle(scene, center, 8, (30, 30, 30), 1, cv2.LINE_AA)
        _draw_arrow(scene, view, ax, ay, agent.get("yaw_deg", 0.0), color, length_cm=900.0)
        _text(scene, f"{agent['agent_id']} spawn", (center[0] + 11, center[1] - 8), scale=0.46, color=color)

    canvas = np.full((size, size + panel_w, 3), 255, dtype=np.uint8)
    canvas[:, :size] = scene
    cv2.line(canvas, (size, 0), (size, size), (210, 210, 210), 1)

    x0 = size + 18
    y = 30
    _text(canvas, "central_square - venue meetup map", (x0, y), scale=0.6, color=(20, 20, 20))
    y += 14
    _text(canvas, "world frame: north=+y(up), east=+x(right); metres", (x0, y + 8), scale=0.4, color=(110, 110, 110))
    y += 34
    for venue in scenario.get("venues", []):
        color = rgb_to_bgr(venue.get("mask_color_rgb", [120, 120, 120]))
        vx, vy, *_ = venue["position"]
        rcenter = venue.get("region", {}).get("center", [vx, vy])
        ring = math.hypot(vx, vy) / 100.0
        slot = venue.get("slot_id", "").split("_")[0]
        status = (venue.get("entrances") or [{}])[0].get("status", "?")
        cv2.rectangle(canvas, (x0, y - 11), (x0 + 16, y + 3), color, -1)
        cv2.rectangle(canvas, (x0, y - 11), (x0 + 16, y + 3), (60, 60, 60), 1)
        _text(canvas, f"{venue['venue_id'].replace('venue_', '')}  [{venue.get('venue_type')}, {slot}]", (x0 + 24, y), scale=0.5, color=(20, 20, 20))
        y += 20
        _text(canvas, f"ring {ring:.0f} m - meet ({rcenter[0]/100:.0f}, {rcenter[1]/100:.0f}) m - entrance {status}", (x0 + 24, y), scale=0.42, color=(90, 90, 90))
        y += 18
        _text(canvas, _venue_traits(venue.get("properties", {})), (x0 + 24, y), scale=0.42, color=(90, 90, 90))
        y += 28

    y += 6
    _text(canvas, "Landmarks (localization cues only):", (x0, y), scale=0.48, color=(20, 20, 20))
    y += 22
    for landmark in scenario.get("landmarks", []):
        lx, ly, *_ = landmark["position"]
        color = rgb_to_bgr(landmark.get("mask_color_rgb", [120, 120, 120]))
        cv2.drawMarker(canvas, (x0 + 8, y - 4), color, cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
        _text(canvas, f"{landmark.get('landmark_type')}  ({lx/100:.0f}, {ly/100:.0f}) m", (x0 + 24, y), scale=0.44, color=(90, 90, 90))
        y += 22

    y += 6
    _text(canvas, "Agents:", (x0, y), scale=0.48, color=(20, 20, 20))
    y += 22
    for index, agent in enumerate(scenario.get("agents", [])):
        color = AGENT_BGR[index % len(AGENT_BGR)]
        ax, ay, *_ = agent["position"]
        cv2.circle(canvas, (x0 + 8, y - 4), 6, color, -1, cv2.LINE_AA)
        _text(canvas, f"{agent['agent_id']} ({ax/100:.0f}, {ay/100:.0f}) m, yaw {agent.get('yaw_deg', 0):.0f}", (x0 + 24, y), scale=0.42, color=(90, 90, 90))
        y += 18
        _text(canvas, f"needs: {agent.get('private_constraint', '')}"[:64], (x0 + 24, y), scale=0.4, color=(120, 120, 120))
        y += 24

    cv2.imwrite(str(out_path), canvas)
    return out_path


def _legend(img: np.ndarray, title: str, subtitle: str) -> None:
    """Stamp a title bar describing the figure."""

    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (255, 255, 255), -1)
    cv2.line(img, (0, 40), (img.shape[1], 40), (220, 220, 220), 1)
    _text(img, title, (12, 18), scale=0.6, color=(30, 30, 30), thickness=1)
    _text(img, subtitle, (12, 34), scale=0.42, color=(90, 90, 90))


def render(run_dir: Path, *, size: int = 1024, fps: float = 2.0) -> dict[str, Path]:
    """Render ``topdown.png`` and ``topdown.mp4`` for a finished run case dir."""

    scenario = json.loads((run_dir / "scenario_hidden.json").read_text(encoding="utf-8"))
    trajectory = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    paths = _agent_paths(scenario, trajectory)

    points: list[tuple[float, float]] = []
    for venue in scenario.get("venues", []):
        points.append((venue["position"][0], venue["position"][1]))
    for landmark in scenario.get("landmarks", []):
        points.append((landmark["position"][0], landmark["position"][1]))
    for path in paths.values():
        points.extend(path["points"])
    view = TopDownView(points, size=size)

    base = _draw_static_scene(view, scenario)
    success = bool(trajectory[-1].get("info", {}).get("success")) if trajectory else False
    steps = len(trajectory)
    subtitle = (
        "solid disk = building collision footprint - dashed ring = arrival region - "
        "X = blocked by obstacle/entrance - star = final position"
    )
    title = f"Venue Meetup top-down - {steps} steps - converged: {success}"

    final = base.copy()
    _draw_paths(final, view, paths)
    _legend(final, title, subtitle)
    png_path = run_dir / "topdown.png"
    cv2.imwrite(str(png_path), final)

    mp4_path = run_dir / "topdown.mp4"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    max_points = max((len(path["points"]) for path in paths.values()), default=1)
    for frame_idx in range(max_points):
        frame = base.copy()
        _draw_paths(frame, view, paths, upto=frame_idx)
        _legend(frame, title, f"step {frame_idx}/{max_points - 1}  -  {subtitle}")
        writer.write(frame)
    for _ in range(int(fps * 2)):  # hold the final frame for a beat.
        writer.write(final)
    writer.release()

    return {"png": png_path, "mp4": mp4_path}


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Case dir containing scenario_hidden.json + trajectory.json")
    parser.add_argument("--map-only", action="store_true", help="Render only an annotated scenario map (no trajectory needed).")
    parser.add_argument("--template-seed", type=int, help="Build the map straight from the central_square template at this seed.")
    parser.add_argument("--out", type=Path, help="Output path for --map-only (defaults to <run-dir>/map.png).")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()

    if args.map_only or args.template_seed is not None:
        if args.template_seed is not None:
            from benchmark.venue_meetup.templates.central_square import build_fixed_scenario

            scenario = build_fixed_scenario(args.template_seed).compact()
            default_out = Path(f"central_square_seed{args.template_seed}_map.png")
        else:
            scenario = json.loads((args.run_dir / "scenario_hidden.json").read_text(encoding="utf-8"))
            default_out = args.run_dir / "map.png"
        out = args.out or default_out
        print(f"map: {render_map(scenario, out, size=args.size)}")
        return

    if args.run_dir is None:
        parser.error("--run-dir is required unless --template-seed is given")
    outputs = render(args.run_dir, size=args.size, fps=args.fps)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
