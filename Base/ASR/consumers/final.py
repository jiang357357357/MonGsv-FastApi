import json
import math
import os

import numpy as np
from Code.FastApi.Base.ASR.services import voice_service

PUNCTUATION_CHARS = set("，。！？；：、,.!?;:")
VAD_CHUNK_MS = 200
PREROLL_MS = 1200
END_SILENCE_CHUNKS = 9
SPEECH_NOISE_THRESHOLD = 0.6
MIN_SPEECH_DURATION_MS = 250
MIN_FINAL_BYTES = 3200
LOW_VOLUME_LEVEL = 0.015
CLIPPING_LEVEL = 0.98


class ASRFinalWebSocketHandler:
    """VAD endpointing + final ASR.

    This endpoint is meant for dialog systems: it emits one final result when a
    speech segment ends, instead of sending streaming partial captions.
    """

    def __init__(self):
        self.sample_rate = 16000
        self.audio_buffer = bytearray()
        self.raw_pcm = bytearray()
        self.pre_speech_pcm = bytearray()
        self.speech_pcm = bytearray()
        self.vad_cache = {}
        self.is_speaking = False
        self.silence_count = 0
        self.vad_process_count = 0
        self.vad_hit_count = 0
        self.segment_index = 0
        self.accumulated_text = ""
        self.speech_ms = 0
        self.silence_ms = 0
        self.noise_level = 0.0
        self.low_volume_chunks = 0
        self.last_endpoint_silence_ms = 0
        self._reset_vad_config()

    def _reset_vad_config(self):
        self.vad_chunk_ms = VAD_CHUNK_MS
        self.vad_chunk_bytes = self._ms_to_pcm_bytes(self.vad_chunk_ms)
        self.preroll_ms = PREROLL_MS
        self.preroll_bytes = self._ms_to_pcm_bytes(self.preroll_ms)
        self.end_silence_ms = VAD_CHUNK_MS * END_SILENCE_CHUNKS
        self.end_silence_chunks = END_SILENCE_CHUNKS
        self.speech_noise_threshold = SPEECH_NOISE_THRESHOLD
        self.min_speech_duration_ms = MIN_SPEECH_DURATION_MS

    def _apply_vad_config(self, data: dict):
        vad = data.get("vad") if isinstance(data.get("vad"), dict) else {}
        if "chunk_ms" in vad:
            self.vad_chunk_ms = int(vad["chunk_ms"])
        if "preroll_ms" in vad:
            self.preroll_ms = int(vad["preroll_ms"])
        if "speech_noise_threshold" in vad:
            self.speech_noise_threshold = float(vad["speech_noise_threshold"])
        if "min_speech_duration_ms" in vad:
            self.min_speech_duration_ms = int(vad["min_speech_duration_ms"])

        end_silence_ms = vad.get("end_silence_ms", data.get("end_silence_ms"))
        if end_silence_ms is not None:
            self.end_silence_ms = int(end_silence_ms)

        self.vad_chunk_bytes = self._ms_to_pcm_bytes(self.vad_chunk_ms)
        self.preroll_bytes = self._ms_to_pcm_bytes(self.preroll_ms)
        self.end_silence_chunks = math.ceil(self.end_silence_ms / self.vad_chunk_ms)

    def _vad_config_payload(self):
        return {
            "chunk_ms": self.vad_chunk_ms,
            "end_silence_ms": self.end_silence_ms,
            "end_silence_chunks": self.end_silence_chunks,
            "speech_noise_threshold": self.speech_noise_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "preroll_ms": self.preroll_ms,
        }

    def _reset_session_state(self):
        self.audio_buffer = bytearray()
        self.raw_pcm = bytearray()
        self.pre_speech_pcm = bytearray()
        self.speech_pcm = bytearray()
        self.vad_cache = {}
        self.is_speaking = False
        self.silence_count = 0
        self.vad_process_count = 0
        self.vad_hit_count = 0
        self.segment_index = 0
        self.accumulated_text = ""
        self.speech_ms = 0
        self.silence_ms = 0
        self.noise_level = 0.0
        self.low_volume_chunks = 0
        self.last_endpoint_silence_ms = 0

    async def handle_connect(self, websocket):
        self._reset_session_state()
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "VAD final STT 已就绪",
            "protocol": "vad-final-v1",
            "audio_format": {
                "sample_rate": self.sample_rate,
                "channels": 1,
                "sample_format": "s16le",
            },
        })

    async def handle_audio(self, websocket, bytes_data):
        if not bytes_data:
            return
        if len(bytes_data) % 2 != 0:
            await self._send_warning(
                websocket,
                "AUDIO_FORMAT_UNSUPPORTED",
                "音频必须是 16kHz/mono/signed int16/little-endian 裸 PCM",
            )
            bytes_data = bytes_data[:-1]
            if not bytes_data:
                return
        self.raw_pcm.extend(bytes_data)
        self.audio_buffer.extend(bytes_data)
        while len(self.audio_buffer) >= self.vad_chunk_bytes:
            chunk = bytes(self.audio_buffer[:self.vad_chunk_bytes])
            del self.audio_buffer[:self.vad_chunk_bytes]
            await self._process_vad(websocket, chunk, is_final=False)

    async def handle_text(self, websocket, text_data):
        data = json.loads(text_data)
        cmd = data.get("command")
        if cmd == "start":
            await self._on_start(websocket, data)
        elif cmd == "stop":
            await self._on_stop(websocket)
        elif cmd == "reset":
            self._reset_session_state()
            await websocket.send_json({"type": "status", "message": "已重置"})

    async def _on_start(self, websocket, data: dict):
        print("[WS-ASR-FINAL] 开始录音")
        self._reset_session_state()
        self._reset_vad_config()
        self._apply_vad_config(data)
        print(f"[WS-ASR-FINAL] VAD配置: {self._vad_config_payload()}")
        await websocket.send_json({
            "type": "status",
            "message": "开始录音",
            "vad": self._vad_config_payload(),
        })

    async def _on_stop(self, websocket):
        raw_duration = len(self.raw_pcm) / 2 / self.sample_rate if self.raw_pcm else 0
        print(f"[WS-ASR-FINAL] 停止录音: raw={len(self.raw_pcm)}B/{raw_duration:.2f}s "
              f"vad_calls={self.vad_process_count} vad_hits={self.vad_hit_count} "
              f"pending={len(self.audio_buffer)}B speech={len(self.speech_pcm)}B")
        if self.audio_buffer:
            chunk = bytes(self.audio_buffer)
            self.audio_buffer = bytearray()
            await self._process_vad(websocket, chunk, is_final=True)

        if self.speech_pcm:
            await self._finalize_current_segment(websocket, source="stop-segment")
        elif not self.accumulated_text and len(self.raw_pcm) >= MIN_FINAL_BYTES:
            await self._finalize_pcm(websocket, bytes(self.raw_pcm), source="raw-fallback")
        elif not self.accumulated_text:
            print("[WS-ASR-FINAL] 没有可识别音频，返回空结果")
            await self._send_warning(websocket, "NO_SPEECH", "未检测到有效人声")
            await self._send_commit_hint(websocket, "manual_stop", False)

        await websocket.send_json({
            "type": "status",
            "message": "录音结束",
            "final_text": self.accumulated_text,
        })
        self.vad_cache = {}
        self.is_speaking = False
        self.silence_count = 0

    async def _process_vad(self, websocket, audio_data: bytes, is_final: bool = False):
        if not audio_data:
            return
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_array) == 0:
            return
        audio_float = audio_array.astype(np.float32) / 32768.0
        audio_state = self._get_audio_state(audio_float)

        from asgiref.sync import sync_to_async
        vad_result = await sync_to_async(voice_service.vad.detect_streaming)(
            audio_float,
            self.vad_cache,
            is_final=is_final,
            chunk_size=self.vad_chunk_ms,
            speech_noise_thres=self.speech_noise_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
        )
        is_speech = vad_result["is_speech"]
        segments = vad_result.get("segments", [])
        endpoint = self._has_endpoint(segments)
        self.vad_process_count += 1
        if is_speech:
            self.vad_hit_count += 1
            self.speech_ms += self.vad_chunk_ms
            self.silence_ms = 0
        else:
            self.silence_ms += self.vad_chunk_ms
            self.speech_ms = 0
            self._update_noise_level(audio_state["input_level"])
        if self.vad_process_count % 10 == 1 or is_speech or is_final:
            print(f"[WS-ASR-FINAL] VAD #{self.vad_process_count} speech={is_speech} "
                  f"speaking={self.is_speaking} silence={self.silence_count} "
                  f"endpoint={endpoint} final={is_final} segments={segments}")

        if audio_state["input_level"] < LOW_VOLUME_LEVEL:
            self.low_volume_chunks += 1
        else:
            self.low_volume_chunks = 0
        if self.low_volume_chunks == 10:
            await self._send_warning(websocket, "LOW_VOLUME", "输入音量过低")

        await websocket.send_json(audio_state)
        await websocket.send_json({
            "type": "voice_activity",
            "is_speech": is_speech,
            "silence_ms": self.silence_ms,
            "speech_ms": self.speech_ms,
        })

        if is_speech:
            if not self.is_speaking:
                self._start_speech_segment()
            self.speech_pcm.extend(audio_data)
            self.silence_count = 0
            return

        if not self.is_speaking:
            self._append_preroll(audio_data)
            return

        self.speech_pcm.extend(audio_data)
        self.silence_count += 1
        if self.silence_count >= self.end_silence_chunks or is_final:
            await self._finalize_current_segment(websocket, source="silence-end")

    def _start_speech_segment(self):
        print("[WS-ASR-FINAL] >>> 检测到语音")
        self.is_speaking = True
        self.silence_count = 0
        self.speech_pcm = bytearray()
        if self.pre_speech_pcm:
            preroll = bytes(self.pre_speech_pcm)
            self.speech_pcm.extend(preroll)
            print(f"[WS-ASR-FINAL] 带入前置音频: {len(preroll)}B/{len(preroll) / 2 / self.sample_rate:.2f}s")
        self.pre_speech_pcm = bytearray()

    async def _finalize_current_segment(self, websocket, source: str):
        pcm_data = bytes(self.speech_pcm)
        self.speech_pcm = bytearray()
        self.is_speaking = False
        if source == "silence-end":
            self.last_endpoint_silence_ms = self.silence_count * self.vad_chunk_ms
        self.silence_count = 0
        self.pre_speech_pcm = bytearray()
        self.vad_cache = {}
        await self._finalize_pcm(websocket, pcm_data, source=source)

    async def _finalize_pcm(self, websocket, pcm_data: bytes, source: str):
        speech_array = np.frombuffer(pcm_data, dtype=np.int16)
        if len(speech_array) == 0:
            print(f"[WS-ASR-FINAL] {source} 为空，跳过")
            return
        duration = len(speech_array) / self.sample_rate
        if len(pcm_data) < MIN_FINAL_BYTES:
            print(f"[WS-ASR-FINAL] 语音过短 ({duration:.2f}s)，跳过")
            await self._send_warning(websocket, "NO_SPEECH", "语音过短，未检测到有效人声")
            return

        speech_float = speech_array.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(speech_float ** 2))
        energy_db = 20 * np.log10(rms + 1e-10)
        if energy_db < -50:
            print(f"[WS-ASR-FINAL] 能量过低 ({energy_db:.1f}dB)，跳过")
            await self._send_warning(websocket, "LOW_VOLUME", "输入音量过低")
            return

        print(f"[WS-ASR-FINAL] 最终确认[{source}]: {len(pcm_data)}B, {duration:.2f}s, {energy_db:.1f}dB")

        from asgiref.sync import sync_to_async
        result = await sync_to_async(voice_service.asr.transcribe_array)(speech_float, self.sample_rate)
        text = result.get("text", "").strip()
        if self._should_punctuate(text):
            punctuated = await sync_to_async(voice_service.asr.punctuate)(text)
            if punctuated:
                text = punctuated
        if not text:
            print(f"[WS-ASR-FINAL] {source} 未识别到文本，跳过 result")
            await self._send_warning(websocket, "NO_SPEECH", "未识别到有效文本")
            return

        self.segment_index += 1
        self.accumulated_text = (self.accumulated_text + " " + text).strip()

        speaker_info = None
        if os.getenv("ENABLE_ASR_SPEAKER", "").lower() in {"1", "true", "yes"}:
            try:
                threshold = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.75"))
                speaker_info = await sync_to_async(voice_service.identify_speaker_from_array)(
                    speech_float, self.sample_rate, threshold,
                )
            except Exception as exc:
                print(f"[WS-ASR-FINAL] 声纹识别失败（跳过）: {exc}")

        await websocket.send_json({
            "type": "result",
            "text": text,
            "accumulated": self.accumulated_text,
            "is_interim": False,
            "sentence_end": True,
            "segment_index": self.segment_index,
            "source": source,
            "duration": duration,
            "speaker_id": speaker_info.get("speaker_id") if speaker_info else None,
            "speaker_name": speaker_info.get("name") if speaker_info else None,
            "speaker_similarity": speaker_info.get("similarity") if speaker_info else None,
            "speaker_is_known": speaker_info.get("is_known") if speaker_info else None,
        })
        await self._send_commit_hint(websocket, self._commit_reason(source, text), True)
        print(f"[WS-ASR-FINAL] final#{self.segment_index}: '{text}', accumulated: '{self.accumulated_text}'")

    def _append_preroll(self, audio_data: bytes):
        self.pre_speech_pcm.extend(audio_data)
        if len(self.pre_speech_pcm) > self.preroll_bytes:
            del self.pre_speech_pcm[:len(self.pre_speech_pcm) - self.preroll_bytes]

    def _get_audio_state(self, audio_float):
        if len(audio_float) == 0:
            input_level = 0.0
            peak = 0.0
        else:
            rms = float(np.sqrt(np.mean(audio_float ** 2)))
            peak = float(np.max(np.abs(audio_float)))
            input_level = min(1.0, rms * 8.0)
        return {
            "type": "audio_state",
            "input_level": round(input_level, 4),
            "noise_level": round(self.noise_level, 4),
            "clipping": peak >= CLIPPING_LEVEL,
        }

    def _update_noise_level(self, input_level: float):
        if self.noise_level <= 0:
            self.noise_level = input_level
        else:
            self.noise_level = self.noise_level * 0.9 + input_level * 0.1

    async def _send_warning(self, websocket, code: str, message: str):
        await websocket.send_json({
            "type": "warning",
            "code": code,
            "message": message,
        })

    async def _send_commit_hint(self, websocket, reason: str, should_commit: bool):
        await websocket.send_json({
            "type": "commit_hint",
            "reason": reason,
            "should_commit": should_commit,
            "final_text": self.accumulated_text,
            "vad": {
                "end_silence_ms": self.end_silence_ms,
                "actual_silence_ms": self.last_endpoint_silence_ms if reason == "silence" else self.silence_ms,
            },
        })

    def _ms_to_pcm_bytes(self, ms: int) -> int:
        return int(self.sample_rate * 2 * ms / 1000)

    @staticmethod
    def _commit_reason(source: str, text: str) -> str:
        if source == "stop-segment":
            return "manual_stop"
        if source == "raw-fallback":
            return "manual_stop"
        if source == "silence-end":
            return "silence"
        if text and text[-1] in PUNCTUATION_CHARS:
            return "sentence_end"
        return "silence"

    @staticmethod
    def _has_endpoint(segments) -> bool:
        return any(len(seg) >= 2 and seg[0] == -1 and seg[1] != -1 for seg in segments)

    @staticmethod
    def _should_punctuate(text: str) -> bool:
        return bool(text) and not any(ch in PUNCTUATION_CHARS for ch in text)
