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

    def process_audio_for_speaker(self, audio_path, speaker_id, threshold=0.75):
        """Fail closed unless the audio matches the explicitly selected user."""
        authorization = self.authorize_audio_for_speaker(audio_path, speaker_id, threshold)
        if authorization["status"] != "authorized":
            return authorization

        speaker_info = authorization["speaker_info"]
        segments = authorization["segments"]
        asr_result = self.asr.transcribe(audio_path)
        text = asr_result if isinstance(asr_result, str) else asr_result.get("text", "")
        print(f"[VoiceService] 声纹通过，ASR 结果: {text[:80]}...")
        return {
            "text": text,
            "speaker_info": speaker_info,
            "segments": segments,
            "status": "success",
        }

    def authorize_audio_for_speaker(self, audio_path, speaker_id, threshold=0.75):
        """Run VAD and exact-speaker verification without invoking ASR."""
        speaker_id = str(speaker_id or "").strip()
        if not speaker_id:
            return {
                "text": "",
                "speaker_info": None,
                "segments": [],
                "status": "speaker_required",
            }

        segments = self.vad.detect(audio_path)
        print(f"[VoiceService] VAD 检测到 {len(segments)} 个片段")
        if len(segments) == 0:
            return {"text": "", "speaker_info": None, "segments": segments, "status": "no_speech"}

        try:
            min_audio_ms = max(250, int(os.getenv("SPEAKER_MIN_AUDIO_MS", "1000")))
        except (TypeError, ValueError):
            min_audio_ms = 1000
        speech_duration_ms = sum(
            max(0, int(segment[1]) - int(segment[0]))
            for segment in segments
            if len(segment) >= 2 and segment[1] >= 0
        )
        if speech_duration_ms < min_audio_ms:
            print(
                f"[VoiceService] 有效人声过短: {speech_duration_ms}ms < {min_audio_ms}ms"
            )
            return {
                "text": "",
                "speaker_info": None,
                "segments": segments,
                "status": "speaker_audio_too_short",
            }

        speaker_info = self.verify_registered_speaker_from_audio(audio_path, speaker_id, threshold)
        if not speaker_info.get("is_registered"):
            return {
                "text": "",
                "speaker_info": speaker_info,
                "segments": segments,
                "status": "speaker_not_registered",
            }
        if not speaker_info.get("is_match"):
            return {
                "text": "",
                "speaker_info": speaker_info,
                "segments": segments,
                "status": "speaker_mismatch",
            }
        return {
            "text": "",
            "speaker_info": speaker_info,
            "segments": segments,
            "status": "authorized",
        }

    def identify_speaker_from_audio(self, audio_path, threshold=0.75):
        embedding = self.speaker.get_embedding(audio_path)
        return self.speaker_db.identify(embedding, threshold)

    def identify_speaker_from_array(self, audio_array, sample_rate=16000, threshold=0.75):
        tmp_path = self._write_temp_speaker_audio(audio_array, sample_rate)
        try:
            return self.identify_speaker_from_audio(str(tmp_path), threshold)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def verify_registered_speaker_from_audio(self, audio_path, speaker_id, threshold=0.75):
        embedding = self.speaker.get_embedding(audio_path)
        return self.speaker_db.verify(speaker_id, embedding, threshold)

    def verify_registered_speaker_from_array(
        self,
        audio_array,
        speaker_id,
        sample_rate=16000,
        threshold=0.75,
    ):
        tmp_path = self._write_temp_speaker_audio(audio_array, sample_rate)
        try:
            return self.verify_registered_speaker_from_audio(str(tmp_path), speaker_id, threshold)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _write_temp_speaker_audio(self, audio_array, sample_rate):
        from pathlib import Path

        try:
            from Code.FastApi.Base.monconfig import MonConfig
            config = MonConfig()
            base_dir = config.workspace_root() or Path.cwd()
        except Exception:
            base_dir = Path.cwd()
        temp_dir = base_dir / "Data" / "Temp" / "asr"
        temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="spk_",
            suffix=".wav",
            dir=temp_dir,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        sf.write(str(tmp_path), audio_array, sample_rate)
        return tmp_path

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
