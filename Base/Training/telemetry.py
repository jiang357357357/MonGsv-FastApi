"""Read epoch means and step samples without inventing missing measurements."""
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


def _metric_log(path: str) -> str:
    """Keep the initial loader size as well as recent progress in long logs."""
    if not path or not Path(path).is_file():
        return ""
    with Path(path).open("rb") as handle:
        head = handle.read(128_000)
        if Path(path).stat().st_size > len(head):
            handle.seek(max(len(head), Path(path).stat().st_size - 256_000))
            head += b"\n" + handle.read()
    return ANSI.sub("", head.decode("utf-8", errors="replace")).replace("\r", "\n")


def _sovits_steps_per_epoch(text: str) -> int | None:
    # Only trust a loader progress fraction attached to a named training epoch.
    for line in text.splitlines():
        if "Train Epoch:" not in line:
            continue
        fractions = re.findall(r"(\d+)/(\d+)\s+\[", line)
        if fractions and int(fractions[-1][1]) > 0:
            return int(fractions[-1][1])
    return None


def _latest(series: dict, key: int, point: dict) -> None:
    if key not in series or series[key]["time"] <= point["time"]:
        series[key] = point


def _chart_points(values: list[dict]) -> list[dict]:
    if len(values) <= 1000:
        return values
    return [values[index * (len(values) - 1) // 999] for index in range(1000)]


def read_telemetry(workspace: str, target: str, log_file: str = "", completed: bool = False, total_epochs: int = 0, window_start: float | None = None, window_end: float | None = None) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    points: dict[str, dict[int, dict]] = {}
    epoch_means: dict[int, dict] = {}
    epochs = []
    event_dir = Path(workspace) / "train" / ("gpt" if target == "gpt" else "sovits")
    errors = []
    for file in event_dir.rglob("events.out.tfevents.*"):
        if "eval" in file.parts:
            continue
        try:
            accumulator = EventAccumulator(str(file), size_guidance={"scalars": 0})
            accumulator.Reload()
            scalars = {
                tag: [v for v in accumulator.Scalars(tag) if math.isfinite(v.value)
                      and (window_start is None or v.wall_time >= window_start)
                      and (window_end is None or v.wall_time <= window_end)]
                for tag in accumulator.Tags().get("scalars", [])
            }
            epoch_markers = {v.step: int(v.value) + (1 if target == "gpt" else 0) for v in scalars.get("epoch", [])}
            epochs.extend(epoch_markers.values())
            mean_tag = "total_loss_epoch" if target == "gpt" else "loss/g/total_epoch"
            for value in scalars.get(mean_tag, []):
                epoch = epoch_markers.get(value.step) if target == "gpt" else value.step
                if epoch is not None and epoch > 0:
                    _latest(epoch_means, epoch, {"epoch":epoch, "value":value.value, "time":value.wall_time,
                                               "step":value.step if target == "gpt" else None, "kind":"mean"})
                    epochs.append(epoch)
            for tag, values in scalars.items():
                if "loss" not in tag.lower():
                    continue
                series = points.setdefault(tag, {})
                for value in values:
                    _latest(series, value.step, {"step":value.step, "value":value.value, "time":value.wall_time})
        except Exception as exc:
            errors.append(str(exc))
    preferred = ("total_loss_epoch", "total_loss_step", "total_loss") if target == "gpt" else ("loss/g/total", "loss/g/mel")
    tag = next((name for name in preferred if name in points), next((name for name in points if name != "loss/g/total_epoch"), None))
    history = sorted(points.get(tag, {}).values(), key=lambda item:item["step"])
    text = _metric_log(log_file)
    if target == "sovits":
        epochs.extend(int(v) for v in re.findall(r"(?:Train Epoch:|====> Epoch:)\s*(\d+)", text))
    else:
        epochs.extend(int(v)+1 for v in re.findall(r"Epoch\s+(\d+)/\d+", text))
    current = total_epochs if completed else max(epochs, default=0)
    rows: dict[int, dict] = {}
    steps_per_epoch = _sovits_steps_per_epoch(text) if target == "sovits" else None
    if steps_per_epoch:
        for value in history:
            epoch = value["step"] // steps_per_epoch + 1
            if not total_epochs or epoch <= total_epochs:
                rows[epoch] = {**value, "epoch":epoch, "kind":"sample"}
                current = max(current, epoch)
    # Log means can recover explicitly recorded epochs if an event file is missing.
    for match in re.finditer(r"Epoch Loss: epoch=(\d+) step=(\d+) mean=([^\s]+)", text):
        if window_start is not None or window_end is not None:
            continue  # These console lines have no wall time to apply the requested window.
        epoch, step = int(match[1]), int(match[2])
        try:
            value = float(match[3])
        except ValueError:
            continue
        if math.isfinite(value) and epoch not in epoch_means:
            epoch_means[epoch] = {"epoch":epoch, "value":value, "time":0, "step":step, "kind":"mean"}
            current = max(current, epoch)
    rows.update(epoch_means)
    if total_epochs:
        current = min(current, total_epochs)
    epoch_history = [rows.get(epoch, {"epoch":epoch, "value":None, "step":None, "time":None, "kind":"missing"})
                     for epoch in range(1, current + 1)]
    recorded = [row for row in epoch_history if row["value"] is not None]
    kinds = {row["kind"] for row in recorded}
    latest_epoch = recorded[-1] if recorded else None
    return {"current_epoch":current, "total_epochs":total_epochs,
            "current_loss":latest_epoch["value"] if latest_epoch else history[-1]["value"] if history else None,
            "current_loss_epoch":latest_epoch["epoch"] if latest_epoch else None,
            "current_loss_kind":latest_epoch["kind"] if latest_epoch else None,
            "epoch_loss_history":epoch_history,
            "epoch_loss_source":"mixed" if len(kinds)>1 else "epoch_mean" if "mean" in kinds else "step_sample" if "sample" in kinds else "unavailable",
            "missing_epochs":[row["epoch"] for row in epoch_history if row["value"] is None],
            "loss_tag":tag, "loss_history":_chart_points(history),
            "loss_series":{name:_chart_points(sorted(series.values(),key=lambda p:p["step"])) for name,series in points.items()},
            "telemetry_error":"; ".join(errors) if errors else None}
