"""Shared UnrealCV camera helpers for Venue Meetup visual previews."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from benchmark.venue_meetup.scene_builder import AGENT_BLUEPRINT
from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.config import Config
from simworld.utils.vector import Vector


@dataclass(frozen=True, slots=True)
class HiddenCamera:
    """Camera identifiers for one hidden humanoid camera carrier."""

    actor_name: str
    camera_id: int


def backend_reachable(
    ip: str,
    port: int,
    timeout_seconds: float = 2.0,
) -> bool:
    """Return whether an UnrealCV TCP endpoint accepts a connection."""

    try:
        with socket.create_connection((ip, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def set_preview_lighting(communicator: Communicator) -> None:
    """Apply the proven empty-map sun angle on a best-effort basis."""

    try:
        objects = tuple(communicator.unrealcv.get_objects())
        sun = next(
            (name for name in objects if name.lower().startswith("directionallight")),
            None,
        )
        if sun is not None:
            communicator.unrealcv.set_orientation((-50.0, 180.0, 180.0), sun)
    except Exception:
        return


def spawn_hidden_camera(
    communicator: Communicator,
    *,
    position: tuple[float, float, float],
    direction: tuple[float, float],
    resolution: tuple[int, int],
) -> HiddenCamera:
    """Spawn and hide a humanoid while retaining its attached camera."""

    agent = Humanoid(
        position=Vector(position[0], position[1]),
        direction=Vector(*direction),
        communicator=communicator,
        config=Config(),
    )
    communicator.spawn_agent(
        agent,
        name=None,
        position=position,
        model_path=AGENT_BLUEPRINT,
        type="humanoid",
    )
    actor_name = communicator.get_humanoid_name(agent.id)
    with communicator.unrealcv.lock:
        communicator.unrealcv.client.request(f"vset /object/{actor_name}/hide")
    communicator.unrealcv.set_camera_resolution(agent.camera_id, resolution)
    return HiddenCamera(actor_name=actor_name, camera_id=agent.camera_id)


def capture_hidden_camera(
    communicator: Communicator,
    camera: HiddenCamera,
    *,
    position: tuple[float, float, float],
    yaw_deg: float,
    actor_pitch_deg: float,
    direct_camera_pitch_deg: float | None = None,
    fov_deg: float,
    frame_gamma: float,
    output_path: Path,
    mask_path: Path,
) -> None:
    """Move a hidden camera carrier and save lit plus object-mask frames."""

    unrealcv = communicator.unrealcv
    unrealcv.enable_controller(camera.actor_name, 0)
    unrealcv.set_location(list(position), camera.actor_name)
    unrealcv.set_orientation(
        (actor_pitch_deg, yaw_deg, 0.0),
        camera.actor_name,
    )
    if direct_camera_pitch_deg is not None:
        # A near-vertical humanoid spring arm offsets the scene substantially.
        # Direct camera rotation is opt-in so ordinary street previews retain
        # the packaged third-person framing while survey views stay centered.
        unrealcv.set_camera_rotation(
            camera.camera_id,
            (direct_camera_pitch_deg, yaw_deg, 0.0),
        )
    unrealcv.set_camera_fov(camera.camera_id, fov_deg)
    # The attached spring arm eases after a large teleport. Advance several
    # deterministic frames so each capture reaches its requested pose.
    for _frame in range(8):
        unrealcv.tick()
        time.sleep(0.03)

    frame = communicator.get_camera_observation(
        camera.camera_id,
        "lit",
        mode="direct",
    )
    if frame is None or not frame.size:
        raise RuntimeError(f"Camera returned no frame for {output_path.name}")
    if abs(frame_gamma - 1.0) > 1e-3:
        lookup = (
            (np.arange(256) / 255.0) ** frame_gamma * 255.0
        ).clip(0, 255).astype(np.uint8)
        frame = lookup[frame]
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Unable to write {output_path}")

    mask = communicator.get_camera_observation(
        camera.camera_id,
        "object_mask",
        mode="direct",
    )
    if mask is None or not mask.size or not cv2.imwrite(str(mask_path), mask):
        raise RuntimeError(f"Unable to write {mask_path}")


__all__ = [
    "HiddenCamera",
    "backend_reachable",
    "capture_hidden_camera",
    "set_preview_lighting",
    "spawn_hidden_camera",
]
