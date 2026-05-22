import json

import numpy as np
from Code.FastApi.Base.ASR.services import voice_service


class VADWebSocketHandler:
    def __init__(self):
        self.buffer = []

    async def handle_connect(self, websocket):
        self.buffer = []
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "VAD 检测已就绪",
        })

    async def handle_audio(self, websocket, bytes_data):
        self.buffer.append(bytes_data)
        total = sum(len(c) for c in self.buffer)
        if total >= 4096:
            await self._process(websocket)

    async def handle_text(self, websocket, text_data):
        data = json.loads(text_data)
        if data.get("command") == "reset":
            self.buffer = []
            await websocket.send_json({"type": "status", "message": "buffer reset"})

    async def _process(self, websocket):
        audio_data = b"".join(self.buffer)
        self.buffer = []
        audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        from asgiref.sync import sync_to_async
        result = await sync_to_async(voice_service.silero_vad.detect)(audio_float)

        await websocket.send_json({
            "type": "vad",
            "is_speech": result["is_speech"],
            "confidence": result["confidence"],
        })
