from Code.FastApi.Base.Inference.service import InferenceRequest, InferenceService
from Code.runtime_env import ensure_gpt_sovits_english_runtime


def test_english_runtime_resources_available_locally():
    assert ensure_gpt_sovits_english_runtime(auto_download_nltk=False)


def test_english_request_requires_runtime_check():
    service = object.__new__(InferenceService)
    request = InferenceRequest(
        text="When a person uses work to meet the light.",
        text_language="en",
        ref_audio_path="dummy.wav",
        prompt_language="ja",
    )

    assert service._request_uses_english_text(request)


def test_auto_request_with_english_requires_runtime_check():
    service = object.__new__(InferenceService)
    request = InferenceRequest(
        text="今日は sunny.",
        text_language="auto",
        ref_audio_path="dummy.wav",
        prompt_language="ja",
    )

    assert service._request_uses_english_text(request)


def test_japanese_request_does_not_require_english_runtime_check():
    service = object.__new__(InferenceService)
    request = InferenceRequest(
        text="こんにちは。",
        text_language="ja",
        ref_audio_path="dummy.wav",
        prompt_language="ja",
    )

    assert not service._request_uses_english_text(request)
