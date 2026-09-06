"""Human-readable post-run transcript; never used as agent feedback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_chat_log(case_dir: Path, trajectory: list[dict]) -> Path:
    """Write delivered messages once, with simulation-time delivery labels."""

    transcript = next((row.get("info", {}).get("comms", {}).get("transcript", [])
                       for row in reversed(trajectory)
                       if row.get("info", {}).get("comms", {}).get("transcript")), [])
    clocks = {}
    for row in trajectory:
        info = row.get("info", {})
        clock = info.get("closing_clock", {})
        if "tick_seconds" in clock:
            clocks[info.get("step")] = clock.get("current_time")
        else:
            observations = row.get("observations", {})
            first = next(iter(observations.values()), {})
            clocks[row.get("step")] = first.get("closing_clock", {}).get("current_time")
    lines = ["# Delivered episode chat", "",
             "Post-run transcript. Times indicate message delivery, not the start of speaking.", ""]
    for message in transcript:
        tick = message.get("step")
        lines.extend([f"## {clocks.get(tick) or f'Turn {tick}'} — {message.get('sender')}", ""])
        lines.extend("> " + line for line in str(message.get("content", "")).splitlines())
        lines.append("")
    path = case_dir / "chat_log.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    trajectory = json.loads((args.run_dir / "trajectory.json").read_text(encoding="utf-8"))
    print(write_chat_log(args.run_dir, trajectory))


if __name__ == "__main__":
    main()
