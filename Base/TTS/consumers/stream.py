import json
import asyncio
from typing import Any

from Code.FastApi.Base.Inference.service import InferenceConfig
from Code.FastApi.Base.TTS.streaming.protocol import TTSProtocolError, message_type, require_request_id
from Code.FastApi.Base.TTS.streaming.session import TTSRequestContext, TTSStreamSession
from Code.FastApi.Domain.Role.Services import RoleService


class TTSStreamWebSocketHandler:
    def __init__(self, inference_service):
        self.inference_service = inference_service
        self.session: TTSStreamSession | None = None

    async def handle_connect(self, websocket):
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "TTS 流式合成已就绪",
        })

    async def handle_text(self, websocket, text_data: str):
        try:
            message = json.loads(text_data)
            msg_type = message_type(message)
            if msg_type == "start":
                await self._on_start(websocket, message)
            elif msg_type in {"text_delta", "text"}:
                await self._require_session(message).push_text(str(message.get("text") or ""))
            elif msg_type == "flush":
                await self._require_session(message).flush()
            elif msg_type == "finish":
                await self._require_session(message).finish()
            elif msg_type == "cancel":
                request_id = require_request_id(message)
                await self._cancel_session()
                await websocket.send_json({"type": "cancelled", "request_id": request_id})
            else:
                raise TTSProtocolError(f"不支持的消息类型: {msg_type}")
        except Exception as exc:
            request_id = ""
            try:
                request_id = str(json.loads(text_data).get("request_id") or "")
            except Exception:
                pass
            await websocket.send_json({
                "type": "error",
                "request_id": request_id,
                "message": str(exc),
            })

    async def handle_disconnect(self):
        await self._cancel_session()

    async def _on_start(self, websocket, message: dict[str, Any]):
        await self._cancel_session()
        await websocket.send_json({
            "type": "status",
            "request_id": require_request_id(message),
            "message": "加载角色与模型",
        })
        context = await asyncio.to_thread(self._build_context, message)
        self.session = TTSStreamSession(websocket, self.inference_service, context)
        await self.session.start()
        await websocket.send_json({
            "type": "ready",
            "request_id": context.request_id,
            "format": "pcm_s16le",
            "channels": 1,
        })

    def _build_context(self, message: dict[str, Any]) -> TTSRequestContext:
        request_id = require_request_id(message)
        role_id = int(message.get("role_id") or 0)
        emotion = str(message.get("emotion") or "").strip()
        if role_id <= 0:
            raise ValueError("role_id 必须大于 0")
        if not emotion:
            raise ValueError("请选择情感")

        role_service = RoleService()
        role = role_service.get_role(role_id)
        world_id = message.get("world_id")
        version = str(message.get("version") or "").strip()
        if world_id is not None and role.world_id != int(world_id):
            raise ValueError("角色不属于当前世界")
        if version and role.version != version:
            raise ValueError("角色不属于当前版本")
        if not role.gpt_model_path or not role.sov_model_path:
            raise ValueError("当前角色缺少 GPT 或 SoVITS 模型路径")

        emotions = role_service.list_role_emotions(role_id)
        selected_emotion = next(
            (item for item in emotions if str(item.get("name", "")).strip() == emotion),
            None,
        )
        if selected_emotion is None:
            raise ValueError(f"当前角色没有情感配置: {emotion}")

        ref_audio_path = str(selected_emotion.get("music_url") or "").strip()
        if not ref_audio_path:
            raise ValueError("当前情感缺少参考音频")
        text_language = str(message.get("text_language") or "zh")
        prompt_text = str(selected_emotion.get("text") or role.prompt_text or "")
        prompt_language = str(selected_emotion.get("text_language") or role.language or text_language)

        model_info = self.inference_service.get_model_info() if hasattr(self.inference_service, "get_model_info") else {}
        models_loaded = bool(
            model_info.get("models_loaded")
            and model_info.get("gpt_path") == role.gpt_model_path
            and model_info.get("sovits_path") == role.sov_model_path
        )
        if not models_loaded and not self.inference_service.load_models(role.gpt_model_path, role.sov_model_path):
            raise ValueError("模型加载失败")

        return TTSRequestContext(
            request_id=request_id,
            text_language=text_language,
            ref_audio_path=ref_audio_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            config=InferenceConfig(
                how_to_cut=str(message.get("how_to_cut") or "不切"),
                top_k=int(message.get("top_k") or 20),
                top_p=float(message.get("top_p") or 0.6),
                temperature=float(message.get("temperature") or 0.6),
                speed=float(message.get("speed") or 1.0),
                sample_steps=int(message.get("sample_steps") or 8),
                if_sr=self._bool_value(message.get("if_sr"), False),
                ref_free=self._bool_value(message.get("ref_free"), False),
                if_freeze=self._bool_value(message.get("if_freeze"), False),
                pause_second=float(message.get("pause_second") or 0.3),
                streaming_mode=True,
                parallel_infer=False,
                split_bucket=False,
            ),
        )

    def _require_session(self, message: dict[str, Any]) -> TTSStreamSession:
        request_id = require_request_id(message)
        if self.session is None:
            raise TTSProtocolError("请先发送 start")
        if self.session.context.request_id != request_id:
            raise TTSProtocolError("request_id 与当前会话不一致")
        return self.session

    async def _cancel_session(self):
        if self.session is not None:
            await self.session.cancel()
            self.session = None

    @staticmethod
    def _bool_value(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}
