import time


class TextSegmenter:
    """Buffer LLM deltas and cut them into TTS-friendly segments."""

    HARD_PUNCTUATION = set("。！？!?")
    SOFT_PUNCTUATION = set("，；、,;")

    def __init__(
        self,
        first_max_chars: int = 30,
        max_chars: int = 80,
        min_chars: int = 6,
        flush_timeout: float = 1.0,
    ):
        self.first_max_chars = first_max_chars
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.flush_timeout = flush_timeout
        self.buffer = ""
        self.segment_count = 0
        self.last_update = 0.0

    def push(self, text: str) -> list[str]:
        text = text or ""
        if not text:
            return []
        self.buffer += text
        self.last_update = time.monotonic()
        return self.pop_ready(force=False)

    def pop_ready(self, force: bool = False) -> list[str]:
        segments: list[str] = []
        while True:
            segment = self._pop_one(force=force)
            if not segment:
                break
            segments.append(segment)
            force = False
        return segments

    def should_timeout_flush(self) -> bool:
        if not self.buffer.strip() or not self.last_update:
            return False
        if len(self.buffer.strip()) < self.min_chars:
            return False
        return time.monotonic() - self.last_update >= self.flush_timeout

    def flush(self) -> list[str]:
        return self.pop_ready(force=True)

    def _pop_one(self, force: bool = False) -> str:
        text = self.buffer.lstrip()
        if not text:
            self.buffer = ""
            return ""

        limit = self.first_max_chars if self.segment_count == 0 else self.max_chars
        hard_idx = self._find_first_punctuation(text, self.HARD_PUNCTUATION)
        if hard_idx >= 0 and (hard_idx + 1 >= self.min_chars or force):
            return self._consume(text, hard_idx + 1)

        if len(text) >= limit:
            soft_idx = self._find_last_punctuation(text[:limit], self.SOFT_PUNCTUATION | self.HARD_PUNCTUATION)
            if soft_idx + 1 >= self.min_chars:
                return self._consume(text, soft_idx + 1)
            return self._consume(text, limit)

        if force:
            return self._consume(text, len(text))
        return ""

    def _consume(self, text: str, end: int) -> str:
        segment = text[:end].strip()
        self.buffer = text[end:]
        if segment:
            self.segment_count += 1
        return segment

    @staticmethod
    def _find_first_punctuation(text: str, chars: set[str]) -> int:
        for idx, char in enumerate(text):
            if char in chars:
                return idx
        return -1

    @staticmethod
    def _find_last_punctuation(text: str, chars: set[str]) -> int:
        for idx in range(len(text) - 1, -1, -1):
            if text[idx] in chars:
                return idx
        return -1
