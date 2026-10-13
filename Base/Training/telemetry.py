"""Read actual trainer output and TensorBoard events without changing trainers."""
from __future__ import annotations

import math
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def log_tail(path: str, lines: int = 300) -> str:
    if not path or not Path(path).is_file():
        return ""
    with Path(path).open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 256_000))
        text = handle.read().decode("utf-8", errors="replace")
    return "\n".join(ANSI.sub("", text).replace("\r", "\n").splitlines()[-lines:])


def read_telemetry(workspace: str, target: str, log_file: str = "", completed: bool = False, total_epochs: int = 0, window_start: float | None = None, window_end: float | None = None) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    points: dict[str, dict[int, dict]] = {}
    epochs = []
    event_dir = Path(workspace) / "train" / ("gpt" if target == "gpt" else "sovits")
    errors = []
    for file in event_dir.rglob("events.out.tfevents.*"):
        if "eval" in file.parts:
            continue
        try:
            accumulator = EventAccumulator(str(file), size_guidance={"scalars": 1000})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                values = [v for v in accumulator.Scalars(tag) if (window_start is None or v.wall_time >= window_start) and (window_end is None or v.wall_time <= window_end)]
                if tag == "epoch":
                    epochs.extend(int(v.value) + 1 for v in values)
                if "loss" not in tag.lower():
                    continue
                series = points.setdefault(tag, {})
                for value in values:
                    if math.isfinite(value.value):
                        series[value.step] = {"step":value.step, "value":value.value, "time":value.wall_time}
        except Exception as exc:
            errors.append(str(exc))
    preferred = ("total_loss_epoch", "total_loss_step", "total_loss") if target == "gpt" else ("loss/g/total", "loss/g/mel")
    tag = next((name for name in preferred if name in points), next(iter(points), None))
    history = sorted(points.get(tag, {}).values(), key=lambda item:item["step"])
    text = log_tail(log_file)
    if target == "sovits":
        epochs.extend(int(v) for v in re.findall(r"(?:Train Epoch:|====> Epoch:)\s*(\d+)", text))
    else:
        epochs.extend(int(v)+1 for v in re.findall(r"Epoch\s+(\d+)/\d+", text))
    current = total_epochs if completed else max(epochs, default=0)
    return {"current_epoch":current, "total_epochs":total_epochs,
            "current_loss":history[-1]["value"] if history else None,
            "loss_tag":tag, "loss_history":history,
            "loss_series":{name:sorted(series.values(),key=lambda p:p["step"]) for name,series in points.items()},
            "telemetry_error":"; ".join(errors) if errors else None}
