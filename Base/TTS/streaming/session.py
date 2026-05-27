import asyncio
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from Code.FastApi.Base.Inference.service import InferenceConfig, InferenceRequest
from Code.FastApi.Base.TTS.streaming.segmenter import TextSegmenter


def audio_to_pcm_s16le(audio: np.ndarray) -> bytes:
    array = np.asarray(audio)
    if array.ndim > 1:
        array = np.mean(array, axis=1)
    if array.dtype == np.int16:
        return array.astype("<i2", copy=False).tobytes()
    array = array.astype(np.float32, copy=False)
    if array.size and np.max(np.abs(array)) > 1.5:
        array = array / 32768.0
    array = np.clip(array, -1.0, 1.0)
    return (array * 32767.0).astype("<i2").tobytes()


@dataclass
class TTSRequestContext:
    request_id: str
    text_language: str
    ref_audio_path: str
    prompt_text: str
    prompt_language: str
    config: InferenceConfig


class TTSStreamSession:
    def __init__(self, websocket, inference_service, context: TTSRequestContext):
        self.websocket = websocket
        self.inference_service = inference_service
        self.context = context
        self.segmenter = TextSegmenter()
        self.segment_queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue(maxsize=16)
        self.cancelled = asyncio.Event()
        self.finished = False
        self.seq = 0
        self.worker_task: asyncio.Task | None = None
        self.flush_task: asyncio.Task | None = None
        self._flush_version = 0

    async def start(self):
        self.worker_task = asyncio.create_task(self._tts_worker())

    async def push_text(self, text: str):
        await self._enqueue_segments(self.segmenter.push(text))
        self._schedule_timeout_flush()

    async def flush(self):
        await self._enqueue_segments(self.segmenter.flush())

    async def finish(self):
        self.finished = True
        await self.flush()
        await self.segment_queue.put(None)
        if self.worker_task:
            await self.worker_task

    async def cancel(self):
        self.cancelled.set()
        if self.flush_task:
            self.flush_task.cancel()
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def _enqueue_segments(self, segments: list[str]):
        for segment in segments:
            self.seq += 1
            await self.segment_queue.put((self.seq, segment))

    def _schedule_timeout_flush(self):
        self._flush_version += 1
        version = self._flush_version
        if self.flush_task:
            self.flush_task.cancel()
        self.flush_task = asyncio.create_task(self._timeout_flush(version))

    async def _timeout_flush(self, version: int):
        try:
            await asyncio.sleep(self.segmenter.flush_timeout)
            if version != self._flush_version or self.finished or self.cancelled.is_set():
                return
            if self.segmenter.should_timeout_flush():
                await self.flush()
        except asyncio.CancelledError:
            pass

    async def _tts_worker(self):
        while not self.cancelled.is_set():
            item = await self.segment_queue.get()
            if item is None:
                break
            seq, text = item
            try:
                await self._stream_segment(seq, text)
            except Exception as exc:
                await self.websocket.send_json({
                    "type": "error",
                    "request_id": self.context.request_id,
                    "seq": seq,
                    "message": str(exc),
                })
                break
        if not self.cancelled.is_set():
            await self.websocket.send_json({"type": "end", "request_id": self.context.request_id})

    async def _stream_segment(self, seq: int, text: str):
        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def run_tts():
            try:
                request = InferenceRequest(
                    text=text,
                    text_language=self.context.text_language,
                    ref_audio_path=self.context.ref_audio_path,
                    prompt_text=self.context.prompt_text,
                    prompt_language=self.context.prompt_language,
                    config=self.context.config.copy(deep=True),
                    return_base64=False,
                )
                for sample_rate, audio in self.inference_service.stream_inference(request):
                    pcm = audio_to_pcm_s16le(audio)
                    if pcm:
                        loop.call_soon_threadsafe(audio_queue.put_nowait, ("audio", (sample_rate, pcm)))
                loop.call_soon_threadsafe(audio_queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(audio_queue.put_nowait, ("error", exc))

        thread = threading.Thread(target=run_tts, name=f"tts-stream-{self.context.request_id}-{seq}", daemon=True)
        thread.start()

        sample_rate = None
        total_bytes = 0
        audio_started = False
        while True:
            kind, payload = await audio_queue.get()
            if kind == "audio":
                sample_rate, pcm = payload
                if not audio_started:
                    await self.websocket.send_json({
                        "type": "audio_start",
                        "request_id": self.context.request_id,
                        "seq": seq,
                        "text": text,
                        "sample_rate": sample_rate,
                        "format": "pcm_s16le",
                        "channels": 1,
                    })
                    audio_started = True
                total_bytes += len(pcm)
                await self.websocket.send_bytes(pcm)
            elif kind == "error":
                raise payload
            elif kind == "done":
                break

        await self.websocket.send_json({
            "type": "audio_end",
            "request_id": self.context.request_id,
            "seq": seq,
            "sample_rate": sample_rate,
            "bytes": total_bytes,
        })
        print(f"[WS-TTS] seq={seq} text={text!r} bytes={total_bytes} sr={sample_rate}")
