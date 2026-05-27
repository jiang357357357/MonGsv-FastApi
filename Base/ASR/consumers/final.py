import json
import os

import numpy as np
from Code.FastApi.Base.ASR.services import voice_service

PUNCTUATION_CHARS = set("，。！？；：、,.!?;:")
VAD_CHUNK_MS = 200
VAD_CHUNK_BYTES = 6400
PREROLL_MS = 1200
PREROLL_BYTES = int(16000 * 2 * PREROLL_MS / 1000)
END_SILENCE_CHUNKS = 9
MIN_FINAL_BYTES = 3200


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

    async def handle_connect(self, websocket):
        self._reset_session_state()
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "VAD final 识别已就绪",
        })

    async def handle_audio(self, websocket, bytes_data):
        if not bytes_data:
            return
        self.raw_pcm.extend(bytes_data)
        self.audio_buffer.extend(bytes_data)
        while len(self.audio_buffer) >= VAD_CHUNK_BYTES:
            chunk = bytes(self.audio_buffer[:VAD_CHUNK_BYTES])
            del self.audio_buffer[:VAD_CHUNK_BYTES]
            await self._process_vad(websocket, chunk, is_final=False)

    async def handle_text(self, websocket, text_data):
        data = json.loads(text_data)
        cmd = data.get("command")
        if cmd == "start":
            await self._on_start(websocket)
        elif cmd == "stop":
            await self._on_stop(websocket)
        elif cmd == "reset":
            self._reset_session_state()
            await websocket.send_json({"type": "status", "message": "已重置"})

    async def _on_start(self, websocket):
        print("[WS-ASR-FINAL] 开始录音")
        self._reset_session_state()
        await websocket.send_json({"type": "status", "message": "开始录音"})

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

        from asgiref.sync import sync_to_async
        vad_result = await sync_to_async(voice_service.vad.detect_streaming)(
            audio_float,
            self.vad_cache,
            is_final=is_final,
            chunk_size=VAD_CHUNK_MS,
            speech_noise_thres=0.6,
            min_speech_duration_ms=250,
        )
        is_speech = vad_result["is_speech"]
        segments = vad_result.get("segments", [])
        endpoint = self._has_endpoint(segments)
        self.vad_process_count += 1
        if is_speech:
            self.vad_hit_count += 1
        if self.vad_process_count % 10 == 1 or is_speech or is_final:
            print(f"[WS-ASR-FINAL] VAD #{self.vad_process_count} speech={is_speech} "
                  f"speaking={self.is_speaking} silence={self.silence_count} "
                  f"endpoint={endpoint} final={is_final} segments={segments}")

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
        if self.silence_count >= END_SILENCE_CHUNKS or is_final:
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
            return

        speech_float = speech_array.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(speech_float ** 2))
        energy_db = 20 * np.log10(rms + 1e-10)
        if energy_db < -50:
            print(f"[WS-ASR-FINAL] 能量过低 ({energy_db:.1f}dB)，跳过")
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
        print(f"[WS-ASR-FINAL] final#{self.segment_index}: '{text}', accumulated: '{self.accumulated_text}'")

    def _append_preroll(self, audio_data: bytes):
        self.pre_speech_pcm.extend(audio_data)
        if len(self.pre_speech_pcm) > PREROLL_BYTES:
            del self.pre_speech_pcm[:len(self.pre_speech_pcm) - PREROLL_BYTES]

    @staticmethod
    def _has_endpoint(segments) -> bool:
        return any(len(seg) >= 2 and seg[0] == -1 and seg[1] != -1 for seg in segments)

    @staticmethod
    def _should_punctuate(text: str) -> bool:
        return bool(text) and not any(ch in PUNCTUATION_CHARS for ch in text)
