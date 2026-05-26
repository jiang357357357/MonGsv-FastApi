__all__ = ["ASRManager", "SpeakerManager", "VADManager", "SileroVAD"]


def __getattr__(name):
    if name == "ASRManager":
        from .asr import ASRManager
        return ASRManager
    if name == "SpeakerManager":
        from .speaker import SpeakerManager
        return SpeakerManager
    if name in {"VADManager", "SileroVAD"}:
        from .vad import VADManager, SileroVAD
        return {"VADManager": VADManager, "SileroVAD": SileroVAD}[name]
    raise AttributeError(name)
