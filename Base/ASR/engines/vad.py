import torch
import numpy as np
from funasr import AutoModel


class SileroVAD:
    def __init__(self):
        print("[VAD] 加载 Silero VAD 模型...")
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()
        self.model.eval()
        print("[VAD] Silero VAD 加载完成")

    def detect(self, audio_float, sample_rate=16000, threshold=0.5):
        try:
            chunk_size = 512 if sample_rate == 16000 else 256
            confidences = []
            tensor = torch.from_numpy(audio_float)
            for i in range(0, len(tensor) - chunk_size + 1, chunk_size):
                chunk = tensor[i:i + chunk_size]
                with torch.no_grad():
                    conf = self.model(chunk, sample_rate).item()
                confidences.append(conf)
            if not confidences:
                return {"is_speech": False, "confidence": 0.0}
            confidence = max(confidences)
            return {"is_speech": confidence >= threshold, "confidence": round(confidence, 3)}
        except Exception as e:
            print(f"[VAD] Silero 检测失败: {e}")
            return {"is_speech": False, "confidence": 0.0}


class VADManager:
    def __init__(self):
        self.model_id = "fsmn-vad"
        print(f"[VAD] 加载 FSMN VAD 模型: {self.model_id}")
        self.model = AutoModel(model=self.model_id, model_revision="v2.0.4")
        print("[VAD] FSMN VAD 加载完成")

    def detect_streaming(self, audio_data, cache, is_final=False, chunk_size=200,
                         speech_noise_thres=0.6, min_speech_duration_ms=250):
        try:
            res = self.model.generate(
                input=audio_data,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                speech_noise_thres=speech_noise_thres,
                min_speech_duration_ms=min_speech_duration_ms,
                disable_pbar=True,
            )
            segments = res[0].get("value", []) if res and len(res) > 0 else []
            is_speech = len(segments) > 0
            is_ongoing = any(seg[1] == -1 for seg in segments if len(seg) >= 2)
            return {"is_speech": is_speech, "is_ongoing": is_ongoing, "segments": segments}
        except Exception as e:
            print(f"[VAD] 流式检测失败: {e}")
            return {"is_speech": True, "segments": []}

    def detect(self, audio_path, max_end_silence_time=800, speech_noise_thres=0.4,
               min_speech_duration_ms=200):
        print(f"[VAD] 检测: threshold={speech_noise_thres}, min_dur={min_speech_duration_ms}ms")
        res = self.model.generate(
            input=audio_path,
            max_end_silence_time=max_end_silence_time,
            speech_noise_thres=speech_noise_thres,
            min_speech_duration_ms=min_speech_duration_ms,
            disable_pbar=True,
        )
        segments = res[0]["value"]
        print(f"[VAD] 检测到 {len(segments)} 个语音片段")
        return segments
