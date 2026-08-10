import os
from dataclasses import dataclass

from Code.FastApi.Base.ASR.services import voice_service


DEFAULT_SPEAKER_THRESHOLD = 0.75
DEFAULT_MIN_SPEAKER_AUDIO_MS = 1000


@dataclass(frozen=True)
class SpeakerGateDecision:
    allowed: bool
    code: str
    message: str
    speaker_info: dict | None = None


def speaker_id_from_start(data: dict) -> str:
    """Use the authenticated upstream user id as the registered speaker id."""
    value = data.get("speaker_id", data.get("user_id", ""))
    return str(value or "").strip()


def configured_threshold() -> float:
    try:
        value = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", str(DEFAULT_SPEAKER_THRESHOLD)))
    except (TypeError, ValueError):
        value = DEFAULT_SPEAKER_THRESHOLD
    return max(0.0, min(1.0, value))


def configured_min_audio_ms() -> int:
    try:
        value = int(os.getenv("SPEAKER_MIN_AUDIO_MS", str(DEFAULT_MIN_SPEAKER_AUDIO_MS)))
    except (TypeError, ValueError):
        value = DEFAULT_MIN_SPEAKER_AUDIO_MS
    return max(250, value)


async def verify_current_speaker(audio, sample_rate: int, speaker_id: str) -> SpeakerGateDecision:
    if not speaker_id:
        return SpeakerGateDecision(False, "VOICEPRINT_REQUIRED", "缺少当前用户声纹标识")

    duration_ms = int(len(audio) * 1000 / sample_rate)
    min_audio_ms = configured_min_audio_ms()
    if duration_ms < min_audio_ms:
        return SpeakerGateDecision(
            False,
            "VOICEPRINT_AUDIO_TOO_SHORT",
            f"有效语音不足 {min_audio_ms}ms，无法可靠验证当前用户",
        )

    from asgiref.sync import sync_to_async

    try:
        info = await sync_to_async(voice_service.verify_registered_speaker_from_array)(
            audio,
            speaker_id,
            sample_rate,
            configured_threshold(),
        )
    except Exception as exc:
        print(f"[SpeakerGate] 声纹验证失败，拒绝 ASR: {exc}")
        return SpeakerGateDecision(False, "VOICEPRINT_ERROR", "声纹验证失败，已拒绝语音识别")

    if not info.get("is_registered"):
        return SpeakerGateDecision(
            False,
            "VOICEPRINT_NOT_REGISTERED",
            "当前用户尚未注册声纹",
            info,
        )
    if not info.get("is_match"):
        return SpeakerGateDecision(
            False,
            "VOICEPRINT_MISMATCH",
            "说话人不是当前用户，已忽略该段语音",
            info,
        )
    return SpeakerGateDecision(True, "VOICEPRINT_MATCH", "声纹验证通过", info)
