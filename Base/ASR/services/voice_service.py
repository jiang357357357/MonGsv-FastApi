import os
import tempfile

import soundfile as sf


class VoiceService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._vad = None
            cls._instance._silero_vad = None
            cls._instance._speaker = None
            cls._instance._asr = None
            cls._instance._speaker_db = None
        return cls._instance

    def __init__(self):
        pass

    @property
    def vad(self):
        if self._vad is None:
            from Code.FastApi.Base.ASR.engines import VADManager
            print("[VoiceService] 加载 VADManager...")
            self._vad = VADManager()
        return self._vad

    @property
    def silero_vad(self):
        if self._silero_vad is None:
            from Code.FastApi.Base.ASR.engines import SileroVAD
            print("[VoiceService] 加载 SileroVAD...")
            self._silero_vad = SileroVAD()
        return self._silero_vad

    @property
    def speaker(self):
        if self._speaker is None:
            from Code.FastApi.Base.ASR.engines import SpeakerManager
            print("[VoiceService] 加载 SpeakerManager...")
            self._speaker = SpeakerManager()
        return self._speaker

    @property
    def asr(self):
        if self._asr is None:
            from Code.FastApi.Base.ASR.engines import ASRManager
            print("[VoiceService] 加载 ASRManager...")
            self._asr = ASRManager()
        return self._asr

    @property
    def speaker_db(self):
        if self._speaker_db is None:
            from Code.FastApi.Base.ASR.services.speaker_db import SpeakerDatabase
            print("[VoiceService] 加载 SpeakerDatabase...")
            self._speaker_db = SpeakerDatabase()
        return self._speaker_db

    def process_audio(self, audio_path):
        segments = self.vad.detect(audio_path)
        print(f"[VoiceService] VAD 检测到 {len(segments)} 个片段")
        if len(segments) == 0:
            return {"text": "", "speaker_info": None, "segments": segments, "status": "no_speech"}
        asr_result = self.asr.transcribe(audio_path)
        if isinstance(asr_result, str):
            text = asr_result
        else:
            text = asr_result.get("text", "")
        print(f"[VoiceService] ASR 结果: {text[:80]}...")
        return {"text": text, "speaker_info": None, "segments": segments, "status": "success"}

    def identify_speaker_from_audio(self, audio_path, threshold=0.75):
        embedding = self.speaker.get_embedding(audio_path)
        return self.speaker_db.identify(embedding, threshold)

    def identify_speaker_from_array(self, audio_array, sample_rate=16000, threshold=0.75):
        base_dir = None
        from pathlib import Path
        try:
            from Code.FastApi.Base.monconfig import MonConfig
            config = MonConfig()
            base_dir = config.workspace_root() or Path.cwd()
        except Exception:
            base_dir = Path.cwd()
        temp_dir = base_dir / "Data" / "Temp" / "asr"
        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = temp_dir / f"spk_{os.getpid()}.wav"
        try:
            sf.write(str(tmp_path), audio_array, sample_rate)
            return self.identify_speaker_from_audio(str(tmp_path), threshold)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def verify_speaker(self, audio_path1, audio_path2):
        emb1 = self.speaker.get_embedding(audio_path1)
        emb2 = self.speaker.get_embedding(audio_path2)
        similarity = self.speaker.compare(emb1, emb2)
        threshold = 0.75
        return {"similarity": similarity, "is_same": similarity >= threshold, "threshold": threshold}

    def process_audio_with_diarization(self, audio_path, language="auto", cluster_threshold=0.75):
        from Code.FastApi.Base.ASR.services.diarization import DiarizationService
        diarizer = DiarizationService(self)
        return diarizer.process_file(audio_path, language, cluster_threshold)


voice_service = VoiceService()
