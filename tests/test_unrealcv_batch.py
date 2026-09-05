"""Command serialization tests for bounded UnrealCV blueprint batches."""

from __future__ import annotations

from threading import Lock

import pytest

from simworld.communicator.unrealcv import BlueprintObjectSpec, UnrealCV


class _BatchClient:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def request_batch(self, commands: list[str]) -> list[str]:
        self.batches.append(list(commands))
        return ["ok"] * len(commands)


def _unrealcv_stub() -> tuple[UnrealCV, _BatchClient]:
    unrealcv = object.__new__(UnrealCV)
    client = _BatchClient()
    unrealcv.client = client
    unrealcv.lock = Lock()
    return unrealcv, client


def test_spawn_bp_assets_batch_serializes_complete_ordered_actor_setup() -> None:
    unrealcv, client = _unrealcv_stub()
    specs = (
        BlueprintObjectSpec(
            prefab_path="/Game/Fake/A.A_C",
            name="GEN_BP_A",
            location=(1e-12, 20.0, 30.0),
            rotation=(0.0, 90.0, 0.0),
            scale=(0.5, 0.6, 0.7),
            collision=False,
            movable=False,
            color=(10, 20, 30),
        ),
        BlueprintObjectSpec(
            prefab_path="/Game/Fake/B.B_C",
            name="GEN_BP_B",
            location=(-40.0, 50.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        ),
    )

    responses = unrealcv.spawn_bp_assets_batch(specs, batch_size=1)

    assert len(client.batches) == 2
    assert client.batches[0] == [
        "vset /objects/spawn_bp_asset /Game/Fake/A.A_C GEN_BP_A",
        "vset /object/GEN_BP_A/location 0.0000 20.0000 30.0000",
        "vset /object/GEN_BP_A/rotation 0.0 90.0 0.0",
        "vset /object/GEN_BP_A/scale 0.5 0.6 0.7",
        "vset /object/GEN_BP_A/color 10 20 30",
        "vset /object/GEN_BP_A/collision false",
        "vset /object/GEN_BP_A/object_mobility False",
    ]
    assert client.batches[1][-2:] == [
        "vset /object/GEN_BP_B/collision true",
        "vset /object/GEN_BP_B/object_mobility True",
    ]
    assert responses == ["ok"] * 13


def test_spawn_bp_assets_batch_rejects_non_positive_batch_size() -> None:
    unrealcv, _client = _unrealcv_stub()

    with pytest.raises(ValueError, match="batch_size must be positive"):
        unrealcv.spawn_bp_assets_batch((), batch_size=0)
