from typing import Any


class TTSProtocolError(ValueError):
    pass


def require_request_id(message: dict[str, Any]) -> str:
    request_id = str(message.get("request_id") or "").strip()
    if not request_id:
        raise TTSProtocolError("缺少 request_id")
    return request_id


def message_type(message: dict[str, Any]) -> str:
    value = str(message.get("type") or "").strip()
    if not value:
        raise TTSProtocolError("缺少消息 type")
    return value
