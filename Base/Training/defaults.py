"""Official training presets with one save at the final epoch."""
from functools import lru_cache

VERSIONS = ("v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4", "v5dev", "v5turbo")
LORA_VERSIONS = {"v3", "v4", "v5dev", "v5turbo"}
SOURCE_URL = "https://github.com/RVC-Boss/GPT-SoVITS/blob/main/webui.py"


@lru_cache(maxsize=1)
def hardware_memory() -> tuple[float, bool, bool]:
    import psutil
    import torch

    if torch.cuda.is_available():
        import re
        devices = []
        for i in range(torch.cuda.device_count()):
            # Match config.py's GPU eligibility, precision, and +0.4 GiB rounding.
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3 + 0.4
            major, minor = torch.cuda.get_device_capability(i)
            sm = major + minor / 10
            series16 = bool(re.search(r"16\d{2}", torch.cuda.get_device_name(i))) and sm == 7.5
            if memory < 4 or sm < 6.1:
                devices.append((0.0, False, False))
            else:
                devices.append((memory, True, sm != 6.1 and not series16))
        if any(device[1] for device in devices):
            return min(device[0] for device in devices), True, any(device[2] for device in devices)
    return psutil.virtual_memory().total / 1024**3, False, False


def training_defaults(version: str) -> dict:
    if version not in VERSIONS:
        raise ValueError(f"不支持的模型版本: {version}")
    memory, gpu, half = hardware_memory()
    lora = version in LORA_VERSIONS
    return {
        "sovits_batch_size": max(1, int(memory // (8 if lora else 2))) if gpu else max(1, int(memory / 4)),
        "gpt_batch_size": max(1, int(memory // 2)) if gpu else max(1, int(memory / 4)),
        "sovits_total_epoch": 2 if lora else 8,
        "gpt_total_epoch": 15,
        "sovits_save_every_epoch": 2 if lora else 8,
        "gpt_save_every_epoch": 15,
        "sovits_learning_rate": 0.0001,
        "gpt_learning_rate": 0.01,
        "lora_rank": 32,
        "grad_checkpoint": False,
        "fp16": half,
        "log_interval": 100,
    }


def fill_training_defaults(options, version: str):
    """Resolve training defaults and always save once at each stage's final epoch."""
    values = training_defaults(version)
    updates = {key: value for key, value in values.items() if getattr(options, key, None) is None}
    for target in ("gpt", "sovits"):
        epochs = getattr(options, f"{target}_total_epoch", None) or values[f"{target}_total_epoch"]
        updates[f"{target}_save_every_epoch"] = epochs
    updates["save_every_epoch"] = None
    return options.model_copy(update=updates)
