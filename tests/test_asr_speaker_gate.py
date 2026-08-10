import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

from Code.FastApi.Base.ASR.consumers.final import ASRFinalWebSocketHandler
from Code.FastApi.Base.ASR.consumers.speaker_gate import (
    SpeakerGateDecision,
    personal_speaker_id,
    speaker_id_from_start,
)
from Code.FastApi.Base.ASR.services import voice_service
from Code.FastApi.Base.ASR.services.speaker_db import SpeakerDatabase
from Code.FastApi.Base.ASR.services.voice_service import VoiceService


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


class FakeASR:
    def __init__(self):
        self.transcribe_calls = 0

    def transcribe_array(self, audio, sample_rate):
        self.transcribe_calls += 1
        return {"text": "验证通过。"}

    def punctuate(self, text):
        return text


class FakeCollection:
    def __init__(self, speaker_id="user-1", embedding=None):
        self.speaker_id = speaker_id
        self.embedding = embedding if embedding is not None else [1.0, 0.0]

    def get(self, ids, include=None):
        if ids != [self.speaker_id]:
            return {"ids": [], "embeddings": [], "metadatas": []}
        return {
            "ids": [self.speaker_id],
            "embeddings": [self.embedding],
            "metadatas": [{"name": "Current User"}],
        }


class FakeVAD:
    def __init__(self, segments):
        self.segments = segments

    def detect(self, audio_path):
        return self.segments


class SpeakerDatabaseTests(unittest.TestCase):
    def test_verify_only_compares_requested_speaker(self):
        database = SpeakerDatabase.__new__(SpeakerDatabase)
        database.collection = FakeCollection()

        matched = database.verify("user-1", np.array([1.0, 0.0]), threshold=0.75)
        missing = database.verify("other-user", np.array([1.0, 0.0]), threshold=0.75)

        self.assertTrue(matched["is_match"])
        self.assertTrue(matched["is_registered"])
        self.assertFalse(missing["is_match"])
        self.assertFalse(missing["is_registered"])

    def test_chroma_round_trip_verifies_exact_registered_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = SpeakerDatabase(Path(temp_dir) / "speaker_db")
            try:
                database.register("user-1", "Current User", np.array([1.0, 0.0]))

                matched = database.verify("user-1", np.array([0.99, 0.01]), threshold=0.75)
                other = database.verify("other-user", np.array([0.99, 0.01]), threshold=0.75)

                self.assertTrue(matched["is_match"])
                self.assertEqual(matched["speaker_id"], "user-1")
                self.assertFalse(other["is_registered"])
            finally:
                database.client._system.stop()


class PersonalSpeakerIdentityTests(unittest.TestCase):
    def test_streaming_sessions_always_use_personal_owner(self):
        with patch.dict(os.environ, {"PERSONAL_SPEAKER_ID": "owner-voiceprint"}):
            self.assertEqual(personal_speaker_id(), "owner-voiceprint")
            self.assertEqual(
                speaker_id_from_start({"speaker_id": "another-user", "user_id": "ignored"}),
                "owner-voiceprint",
            )


class VoiceServiceGateTests(unittest.TestCase):
    def test_short_effective_speech_is_rejected_before_speaker_model(self):
        service = object.__new__(VoiceService)
        service._vad = FakeVAD([[100, 700]])
        service._speaker = None
        service._speaker_db = None
        service._asr = None

        result = service.authorize_audio_for_speaker("unused.wav", "user-1", 0.75)

        self.assertEqual(result["status"], "speaker_audio_too_short")


class FinalSpeakerGateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_asr = voice_service._asr
        self.fake_asr = FakeASR()
        voice_service._asr = self.fake_asr
        self.audio = np.full(32000, 0.1, dtype=np.float32)

    async def asyncTearDown(self):
        voice_service._asr = self.previous_asr

    async def test_rejected_speaker_never_runs_asr_or_returns_text(self):
        handler = ASRFinalWebSocketHandler(require_speaker_gate=True)
        handler.expected_speaker_id = "user-1"
        websocket = FakeWebSocket()
        rejected = SpeakerGateDecision(
            False,
            "VOICEPRINT_MISMATCH",
            "说话人不是当前用户",
            {"speaker_id": "user-1", "similarity": 0.41, "is_match": False},
        )

        with patch(
            "Code.FastApi.Base.ASR.consumers.final.verify_current_speaker",
            new=AsyncMock(return_value=rejected),
        ):
            await handler._finalize_pcm(websocket, (self.audio * 32768).astype(np.int16).tobytes(), "test")

        self.assertEqual(self.fake_asr.transcribe_calls, 0)
        self.assertFalse(any(message.get("type") == "result" for message in websocket.messages))
        self.assertTrue(any(message.get("code") == "VOICEPRINT_MISMATCH" for message in websocket.messages))

    async def test_matched_speaker_runs_asr_and_returns_verified_result(self):
        handler = ASRFinalWebSocketHandler(require_speaker_gate=True)
        handler.expected_speaker_id = "user-1"
        websocket = FakeWebSocket()
        accepted = SpeakerGateDecision(
            True,
            "VOICEPRINT_MATCH",
            "声纹验证通过",
            {
                "speaker_id": "user-1",
                "name": "Current User",
                "similarity": 0.92,
                "is_match": True,
            },
        )

        with patch(
            "Code.FastApi.Base.ASR.consumers.final.verify_current_speaker",
            new=AsyncMock(return_value=accepted),
        ):
            await handler._finalize_pcm(websocket, (self.audio * 32768).astype(np.int16).tobytes(), "test")

        results = [message for message in websocket.messages if message.get("type") == "result"]
        self.assertEqual(self.fake_asr.transcribe_calls, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "验证通过。")
        self.assertTrue(results[0]["speaker_verified"])

    async def test_plain_mode_runs_asr_without_speaker_verification(self):
        handler = ASRFinalWebSocketHandler(require_speaker_gate=False)
        websocket = FakeWebSocket()
        verify_mock = AsyncMock()

        with patch(
            "Code.FastApi.Base.ASR.consumers.final.verify_current_speaker",
            new=verify_mock,
        ):
            await handler._finalize_pcm(
                websocket,
                (self.audio * 32768).astype(np.int16).tobytes(),
                "test",
            )

        results = [message for message in websocket.messages if message.get("type") == "result"]
        self.assertEqual(self.fake_asr.transcribe_calls, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["speaker_verified"])
        verify_mock.assert_not_awaited()

    async def test_connection_payload_distinguishes_plain_and_voiceprint_modes(self):
        plain_socket = FakeWebSocket()
        gated_socket = FakeWebSocket()

        await ASRFinalWebSocketHandler(False).handle_connect(plain_socket)
        await ASRFinalWebSocketHandler(True).handle_connect(gated_socket)

        self.assertEqual(plain_socket.messages[0]["protocol"], "vad-final-v1")
        self.assertEqual(plain_socket.messages[0]["speaker_gate"], "disabled")
        self.assertEqual(gated_socket.messages[0]["protocol"], "vad-final-speaker-gate-v1")
        self.assertEqual(gated_socket.messages[0]["speaker_gate"], "required")


if __name__ == "__main__":
    unittest.main()
