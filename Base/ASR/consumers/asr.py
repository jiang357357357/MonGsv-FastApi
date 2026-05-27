import json
import os

import numpy as np
from Code.FastApi.Base.ASR.services import voice_service

PUNCTUATION_CHARS = set("，。！？；：、,.!?;:")
VAD_CHUNK_MS = 200
VAD_CHUNK_BYTES = 6400
PREROLL_MS = 1200
PREROLL_BYTES = int(16000 * 2 * PREROLL_MS / 1000)
STREAM_CHUNK_BYTES = 19200
SILENCE_LIMIT = 20
MIN_FINAL_BYTES = 3200


class ASRWebSocketHandler:
    def __init__(self):
        self.audio_buffer = bytearray()
        self.raw_pcm = bytearray()
        self.pre_speech_pcm = bytearray()
        self.sample_rate = 16000
        self.vad_cache = {}
        self.is_speaking = False
        self.silence_count = 0
        self.accumulated_text = ""
        self.vad_process_count = 0
        self.asr_cache = {}
        self.asr_pending = bytearray()
        self.speech_pcm = bytearray()
        self.last_interim = ""
        self.vad_hit_count = 0

    def _reset_session_state(self):
        self.audio_buffer = bytearray()
        self.raw_pcm = bytearray()
        self.pre_speech_pcm = bytearray()
        self.vad_cache = {}
        self.is_speaking = False
        self.silence_count = 0
        self.accumulated_text = ""
        self.vad_process_count = 0
        self.vad_hit_count = 0
        self.asr_cache = {}
        self.asr_pending = bytearray()
        self.speech_pcm = bytearray()
        self.last_interim = ""

    async def handle_connect(self, websocket):
        self._reset_session_state()
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "2-pass 流式识别已就绪",
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

    async def _on_start(self, websocket):
        print("[WS-ASR] 开始录音 (2-pass)")
        self._reset_session_state()
        await websocket.send_json({"type": "status", "message": "开始录音"})

    async def _on_stop(self, websocket):
        raw_duration = len(self.raw_pcm) / 2 / self.sample_rate if self.raw_pcm else 0
        print(f"[WS-ASR] 停止录音: raw={len(self.raw_pcm)}B/{raw_duration:.2f}s "
              f"vad_calls={self.vad_process_count} vad_hits={self.vad_hit_count} "
              f"pending={len(self.audio_buffer)}B speech={len(self.speech_pcm)}B")
        if self.audio_buffer:
            chunk = bytes(self.audio_buffer)
            self.audio_buffer = bytearray()
            await self._process_vad(websocket, chunk, is_final=True)

        if len(self.raw_pcm) >= MIN_FINAL_BYTES:
            self.asr_pending = bytearray()
            await self._finalize_pcm(
                websocket,
                bytes(self.raw_pcm),
                source="raw-final",
                replace_accumulated=True,
            )
        elif self.speech_pcm:
            self.asr_pending = bytearray()
            await self._finalize_speech_segment(websocket)
        elif not self.accumulated_text:
            print("[WS-ASR] 没有可识别音频，返回空结果")
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
            audio_float, self.vad_cache, is_final=is_final, chunk_size=VAD_CHUNK_MS,
            speech_noise_thres=0.6, min_speech_duration_ms=250,
        )
        is_speech = vad_result["is_speech"]
        segments = vad_result.get("segments", [])
        self.vad_process_count += 1
        if is_speech:
            self.vad_hit_count += 1
        if self.vad_process_count % 10 == 1 or is_speech or is_final:
            print(f"[WS-ASR] VAD #{self.vad_process_count} speech={is_speech} "
                  f"speaking={self.is_speaking} silence={self.silence_count} "
                  f"final={is_final} segments={segments}")

        if is_speech:
            if not self.is_speaking:
                print("[WS-ASR] >>> 检测到语音")
                self.is_speaking = True
                self.silence_count = 0
                self.asr_cache = {}
                self.asr_pending = bytearray()
                self.speech_pcm = bytearray()
                self.last_interim = ""
                if self.pre_speech_pcm:
                    preroll = bytes(self.pre_speech_pcm)
                    self.asr_pending.extend(preroll)
                    self.speech_pcm.extend(preroll)
                    print(f"[WS-ASR] 带入前置音频: {len(preroll)}B/{len(preroll) / 2 / self.sample_rate:.2f}s")
                self.pre_speech_pcm = bytearray()
            self.asr_pending.extend(audio_data)
            self.speech_pcm.extend(audio_data)
            if len(self.asr_pending) >= STREAM_CHUNK_BYTES:
                await self._flush_interim(websocket)
        else:
            if not self.is_speaking:
                self._append_preroll(audio_data)
            else:
                self.asr_pending.extend(audio_data)
                self.speech_pcm.extend(audio_data)
                self.silence_count += 1
                if self.silence_count >= SILENCE_LIMIT:
                    dur = len(self.speech_pcm) / 2 / self.sample_rate
                    if dur < 0.3:
                        print(f"[WS-ASR] 语音过短 ({dur:.2f}s)，丢弃")
                        self.is_speaking = False
                        self.silence_count = 0
                        self.speech_pcm = bytearray()
                        self.asr_pending = bytearray()
                    else:
                        print(f"[WS-ASR] <<< 语音结束 ({self.silence_count} 次静音)")
                        self.vad_cache = {}
                        self.pre_speech_pcm = bytearray()
                        self.asr_pending = bytearray()
                        await self._finalize_speech_segment(websocket)
                        self.is_speaking = False
                        self.silence_count = 0
                        self.speech_pcm = bytearray()

    def _append_preroll(self, audio_data: bytes):
        self.pre_speech_pcm.extend(audio_data)
        if len(self.pre_speech_pcm) > PREROLL_BYTES:
            del self.pre_speech_pcm[:len(self.pre_speech_pcm) - PREROLL_BYTES]

    async def _flush_interim(self, websocket):
        chunk = bytes(self.asr_pending)
        self.asr_pending = bytearray()
        audio_array = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0 if len(audio_array) > 0 else np.array([], dtype=np.float32)

        from asgiref.sync import sync_to_async
        result = await sync_to_async(voice_service.asr.transcribe_streaming)(audio_float, self.asr_cache, is_final=False)
        text = result.get("text", "").strip()
        if text:
            self.last_interim = text
            print(f"[WS-ASR] interim: '{text}'")
            await websocket.send_json({
                "type": "result",
                "text": text,
                "accumulated": text,
                "is_interim": True,
                "sentence_end": result.get("sentence_end", False),
            })

    async def _finalize_speech_segment(self, websocket):
        await self._finalize_pcm(websocket, bytes(self.speech_pcm), source="vad-segment")

    def _should_punctuate(self, text: str) -> bool:
        return bool(text) and not any(ch in PUNCTUATION_CHARS for ch in text)

    async def _finalize_pcm(self, websocket, pcm_data: bytes, source: str, replace_accumulated: bool = False):
        speech_array = np.frombuffer(pcm_data, dtype=np.int16)
        if len(speech_array) == 0:
            print(f"[WS-ASR] {source} 为空，跳过")
            return
        speech_float = speech_array.astype(np.float32) / 32768.0
        duration = len(speech_array) / self.sample_rate
        rms = np.sqrt(np.mean(speech_float ** 2))
        energy_db = 20 * np.log10(rms + 1e-10)

        if energy_db < -50:
            print(f"[WS-ASR] 能量过低 ({energy_db:.1f}dB)，跳过")
            return

        print(f"[WS-ASR] 最终确认[{source}]: {len(pcm_data)}B, {duration:.2f}s, {energy_db:.1f}dB")

        from asgiref.sync import sync_to_async
        result = await sync_to_async(voice_service.asr.transcribe_array)(speech_float, self.sample_rate)
        text = result.get("text", "").strip()
        if not text:
            text = self.last_interim
        if self._should_punctuate(text):
            punctuated = await sync_to_async(voice_service.asr.punctuate)(text)
            if punctuated:
                text = punctuated

        if replace_accumulated:
            self.accumulated_text = text
        else:
            self.accumulated_text = (self.accumulated_text + " " + text).strip()

        speaker_info = None
        try:
            threshold = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.75"))
            speaker_info = await sync_to_async(voice_service.identify_speaker_from_array)(
                speech_float, self.sample_rate, threshold,
            )
        except Exception as e:
            print(f"[WS-ASR] 声纹识别失败（跳过）: {e}")

        await websocket.send_json({
            "type": "result",
            "text": text,
            "accumulated": self.accumulated_text,
            "is_interim": False,
            "sentence_end": True,
            "speaker_id": speaker_info.get("speaker_id") if speaker_info else None,
            "speaker_name": speaker_info.get("name") if speaker_info else None,
            "speaker_similarity": speaker_info.get("similarity") if speaker_info else None,
            "speaker_is_known": speaker_info.get("is_known") if speaker_info else None,
        })

        print(f"[WS-ASR] final: '{text}', accumulated: '{self.accumulated_text}'")
