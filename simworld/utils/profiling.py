"""Small profiling helpers for SimWorld benchmark scripts."""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark matrix entry."""

    scenario_name: str
    map_uri: str | None
    num_agents: int
    observation_profile: str
    resolution: tuple[int, int]
    mode: str
    step_kind: str
    tick_interval: float
    steps: int
    warmup_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


def now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def parse_csv(value: str) -> list[str]:
    """Parse a comma-separated CLI value into non-empty tokens."""

    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    """Parse a comma-separated list of integers."""

    return [int(item) for item in parse_csv(value)]


def parse_resolutions(value: str) -> list[tuple[int, int]]:
    """Parse CLI resolutions like ``320x240,640x480``."""

    resolutions = []
    for item in parse_csv(value):
        try:
            width, height = item.lower().split("x", maxsplit=1)
            resolutions.append((int(width), int(height)))
        except ValueError as exc:
            raise ValueError(f"Invalid resolution '{item}', expected WIDTHxHEIGHT") from exc
    return resolutions


def resolution_label(resolution: tuple[int, int]) -> str:
    """Format a resolution tuple for filenames and CSV output."""

    return f"{resolution[0]}x{resolution[1]}"


def percentile(values: list[float], pct: float) -> float:
    """Return a nearest-rank percentile for a non-empty list."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def summarize_durations(durations: list[float]) -> dict[str, float]:
    """Summarize step or command durations in seconds."""

    if not durations:
        return {
            "count": 0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p95_seconds": 0.0,
            "p99_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
            "steps_per_second": 0.0,
        }

    total = sum(durations)
    return {
        "count": len(durations),
        "mean_seconds": statistics.fmean(durations),
        "median_seconds": statistics.median(durations),
        "p95_seconds": percentile(durations, 0.95),
        "p99_seconds": percentile(durations, 0.99),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "steps_per_second": len(durations) / total if total > 0 else 0.0,
    }


def machine_metadata() -> dict[str, Any]:
    """Collect lightweight host metadata that is useful for comparisons."""

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def nvidia_smi_metadata() -> dict[str, Any]:
    """Return basic NVIDIA GPU telemetry if ``nvidia-smi`` is available."""

    query = "name,driver_version,memory.total,memory.used,utilization.gpu"
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False}

    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        name, driver, total_mb, used_mb, utilization = parts
        gpus.append(
            {
                "name": name,
                "driver_version": driver,
                "memory_total_mb": int(total_mb),
                "memory_used_mb": int(used_mb),
                "utilization_gpu_percent": int(utilization),
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def benchmark_record(
    case: BenchmarkCase,
    *,
    status: str,
    durations: list[float] | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable result record."""

    record = {
        "timestamp": now_iso(),
        "status": status,
        **asdict(case),
        "resolution": list(case.resolution),
        "resolution_label": resolution_label(case.resolution),
        "summary": summarize_durations(durations or []),
        "machine": machine_metadata(),
    }
    if error:
        record["error"] = error
    if extra:
        record.update(extra)
    return record


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append benchmark records as JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv_summary(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append compact benchmark summaries to CSV."""

    rows = []
    for record in records:
        summary = record.get("summary", {})
        metadata = record.get("metadata", {})
        rows.append(
            {
                "timestamp": record.get("timestamp"),
                "status": record.get("status"),
                "scenario_name": record.get("scenario_name"),
                "map_uri": record.get("map_uri"),
                "num_agents": record.get("num_agents"),
                "observation_profile": record.get("observation_profile"),
                "resolution": record.get("resolution_label"),
                "mode": record.get("mode"),
                "step_kind": record.get("step_kind"),
                "steps": record.get("steps"),
                "mean_seconds": summary.get("mean_seconds"),
                "p95_seconds": summary.get("p95_seconds"),
                "p99_seconds": summary.get("p99_seconds"),
                "steps_per_second": summary.get("steps_per_second"),
                "ue_launch_profile": metadata.get("ue_launch_profile"),
                "gpu_verified": metadata.get("gpu_verified"),
                "navigation_graph_available": metadata.get("navigation_graph_available"),
                "error": record.get("error"),
            }
        )
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
