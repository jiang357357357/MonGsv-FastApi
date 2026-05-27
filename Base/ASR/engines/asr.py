import os
import sys
import traceback
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import soundfile
import torch
from scipy import signal
from funasr import AutoModel
from tqdm import tqdm


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
STREAMING_SAMPLE_RATE = 16000


def resolve_inputs(input_path: str) -> Tuple[List[Tuple[str, str]], str]:
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [(os.path.basename(input_path), input_path)], os.path.basename(input_path)
    if os.path.isdir(input_path):
        files = []
        for name in sorted(os.listdir(input_path)):
            fp = os.path.join(input_path, name)
            if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                files.append((name, fp))
        return files, os.path.basename(input_path.rstrip("/\\"))
    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def write_output_file(output_folder: str, output_file_name: str, recognition_results: List[Dict[str, str]]) -> str:
    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.abspath(f"{output_folder}/{output_file_name}.list")
    lines = ["{audio_path}|{speaker}|{language}|{text}".format(**r) for r in recognition_results]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ASR] 标注文件: {path}")
    return path


class StreamingParaformerEngine:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        print(f"[ASR] 加载流式模型: paraformer-zh-streaming (设备: {device_name})")
        self.model = AutoModel(
            model="paraformer-zh-streaming",
            device=device,
            disable_update=True,
        )
        print(f"[ASR] 流式模型加载完成 ({device_name})")
        self.chunk_size = [0, 10, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32, copy=False)
        if sample_rate != STREAMING_SAMPLE_RATE:
            divisor = gcd(sample_rate, STREAMING_SAMPLE_RATE)
            audio = signal.resample_poly(
                audio,
                STREAMING_SAMPLE_RATE // divisor,
                sample_rate // divisor,
            ).astype(np.float32, copy=False)
        return audio

    def transcribe_streaming(self, audio_data: np.ndarray, cache: dict, is_final: bool = False) -> dict:
        try:
            res = self.model.generate(
                input=audio_data, cache=cache, is_final=is_final,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
                disable_pbar=True,
            )
            if res and len(res) > 0:
                text = res[0].get("text", "").strip()
                sentence_end = res[0].get("sentence_end", False)
            else:
                text = ""
                sentence_end = False
            return {"text": text, "is_final": is_final, "sentence_end": sentence_end}
        except Exception as e:
            print(f"[ASR] 流式识别失败: {e}")
            return {"text": "", "is_final": is_final, "sentence_end": False}

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        try:
            audio_array = self._prepare_audio(audio_array, sample_rate)
            res = self.model.generate(input=audio_array, batch_size=1, disable_pbar=True)
            text = res[0].get("text", "").strip() if res and len(res) > 0 else ""
            return {"text": text}
        except Exception as e:
            print(f"[ASR] 数组识别失败: {e}")
            return {"text": ""}

    def transcribe(self, audio_path: str) -> dict:
        try:
            speech, sr = soundfile.read(audio_path)
            speech = self._prepare_audio(speech, sr)
            cache = {}
            stride = self.chunk_size[1] * 960
            total = int(len(speech) / stride) + 1
            full = ""
            for i in range(total):
                chunk = speech[i * stride : (i + 1) * stride]
                r = self.transcribe_streaming(chunk, cache, is_final=(i == total - 1))
                if r["text"]:
                    full += r["text"]
            return {"text": full}
        except Exception as e:
            print(f"[ASR] 文件识别失败: {e}")
            return {"text": ""}

    def batch_transcribe(self, input_path: str, language: str = "zh") -> Tuple[str, List[Dict[str, str]]]:
        input_files, output_name = resolve_inputs(input_path)
        results: List[Dict[str, str]] = []
        for fname, fpath in tqdm(input_files):
            try:
                print(f"\n[ASR] {fname}")
                r = self.transcribe(fpath)
                results.append({
                    "audio_path": fpath,
                    "speaker": output_name,
                    "language": language.upper(),
                    "text": r.get("text", ""),
                })
            except Exception:
                print(traceback.format_exc())
        return output_name, results


class PunctuationEngine:
    def __init__(self):
        print("[ASR] 加载中文标点模型...")
        root = self._find_root()
        if root:
            sys.path.insert(0, root)
        from tools.asr.funasr_asr import ensure_snapshot

        model_path = ensure_snapshot(
            "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            "tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            ["configuration.json", "model.pt", "tokens.json"],
        )
        self.model = AutoModel(
            model=model_path,
            model_revision="v2.0.4",
            disable_update=True,
        )
        print("[ASR] 中文标点模型加载完成")

    def _find_root(self) -> Optional[str]:
        for parent in Path(__file__).resolve().parents:
            if (parent / "GPT_SoVITS").exists():
                return str(parent)
        return None

    def punctuate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        try:
            result = self.model.generate(input=text, disable_pbar=True)
            if result and len(result) > 0:
                return result[0].get("text", text).strip()
        except Exception as exc:
            print(f"[ASR] 标点恢复失败: {exc}")
        return text


class LegacyFunasrEngine:
    def __init__(self):
        print("[ASR] 加载传统 Paraformer-large 引擎...")
        root = self._find_root()
        if root:
            sys.path.insert(0, root)
        from tools.asr.funasr_asr import create_model, transcribe_with_model, recognize_with_model
        self._create_model = create_model
        self._transcribe_with_model = transcribe_with_model
        self._recognize_with_model = recognize_with_model
        self._model_cache: dict = {}
        print("[ASR] 传统引擎就绪")

    def _find_root(self) -> Optional[str]:
        for parent in Path(__file__).resolve().parents:
            if (parent / "GPT_SoVITS").exists():
                return str(parent)
        return None

    def _get_model(self, language: str):
        if language not in self._model_cache:
            self._model_cache[language] = self._create_model(language, use_cache=False)
        return self._model_cache[language]

    def transcribe(self, audio_path: str, language: str = "zh") -> dict:
        try:
            model = self._get_model(language)
            text = self._transcribe_with_model(model, audio_path)
            return {"text": text}
        except Exception as e:
            print(f"[ASR] 传统引擎识别失败: {e}")
            return {"text": ""}

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000, language: str = "zh") -> dict:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            soundfile.write(tmp.name, audio_array, sample_rate)
            return self.transcribe(tmp.name, language)
        finally:
            if os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except PermissionError as exc:
                    print(f"[ASR] 临时文件删除失败（跳过）: {exc}")

    def batch_transcribe(self, input_path: str, language: str = "zh") -> Tuple[str, List[Dict[str, str]]]:
        model = self._get_model(language)
        return self._recognize_with_model(model, input_path, language)


class FasterWhisperEngine:
    def __init__(self):
        print("[ASR] 加载 Faster-Whisper 引擎...")
        root = self._find_root()
        if root:
            sys.path.insert(0, root)
        from tools.asr import fasterwhisper_asr
        self._module = fasterwhisper_asr
        print("[ASR] Faster-Whisper 引擎就绪")

    def _find_root(self) -> Optional[str]:
        for parent in Path(__file__).resolve().parents:
            if (parent / "GPT_SoVITS").exists():
                return str(parent)
        return None

    def create_model(self, model_size: str = "large-v3", precision: str = "float16"):
        return self._module.create_model(model_size, precision)

    def transcribe_with_model(self, model, file_path: str, language: str = "auto",
                              funasr_fallback: Optional[callable] = None) -> Tuple[str, str]:
        return self._module.transcribe_with_model(model, file_path, language,
                                                   funasr_fallback=funasr_fallback)

    def transcribe(self, audio_path: str, language: str = "auto") -> dict:
        try:
            import soundfile
            speech, sr = soundfile.read(audio_path)
            return self.transcribe_array(speech, sr, language)
        except Exception as e:
            print(f"[ASR] Whisper 文件识别失败: {e}")
            return {"text": ""}

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000, language: str = "auto") -> dict:
        try:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                import soundfile
                soundfile.write(tmp.name, audio_array, sample_rate)
                model = self.create_model()
                _, text = self.transcribe_with_model(model, tmp.name, language)
                return {"text": text}
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
        except Exception as e:
            print(f"[ASR] Whisper 数组识别失败: {e}")
            return {"text": ""}

    def batch_transcribe(self, input_path: str, language: str = "auto",
                         model_size: str = "large-v3", precision: str = "float16",
                         funasr_fallback: Optional[callable] = None) -> Tuple[str, List[Dict[str, str]]]:
        model = self.create_model(model_size, precision)
        return self._module.recognize_with_model(model, input_path, language,
                                                  funasr_fallback=funasr_fallback)


class ASRManager:
    """统一 ASR 引擎

    自动选择后端：
    - funasr：Paraformer-large + VAD + 标点，高精度离线识别
    - paraformer-zh-streaming：流式 + 非流式，中文为主
    - faster_whisper：多语言
    """

    ENGINE_STREAMING = "streaming"
    ENGINE_FUNASR = "funasr"
    ENGINE_WHISPER = "whisper"

    def __init__(self, engine: str = ENGINE_FUNASR):
        self._engine_type = engine
        self._streaming: Optional[StreamingParaformerEngine] = None
        self._funasr: Optional[LegacyFunasrEngine] = None
        self._whisper: Optional[FasterWhisperEngine] = None
        self._punctuation: Optional[PunctuationEngine] = None

    @property
    def streaming(self) -> StreamingParaformerEngine:
        if self._streaming is None:
            self._streaming = StreamingParaformerEngine()
        return self._streaming

    @property
    def funasr(self) -> LegacyFunasrEngine:
        if self._funasr is None:
            self._funasr = LegacyFunasrEngine()
        return self._funasr

    @property
    def whisper(self) -> FasterWhisperEngine:
        if self._whisper is None:
            self._whisper = FasterWhisperEngine()
        return self._whisper

    @property
    def punctuation(self) -> PunctuationEngine:
        if self._punctuation is None:
            self._punctuation = PunctuationEngine()
        return self._punctuation

    def select_engine(self, language: str = "zh", model_type: str = "") -> str:
        if model_type == "faster_whisper":
            return self.ENGINE_WHISPER
        if model_type == "funasr":
            return self.ENGINE_FUNASR
        if self._engine_type:
            return self._engine_type
        if language in ("zh", "yue"):
            return self.ENGINE_FUNASR
        return self.ENGINE_WHISPER

    def transcribe_streaming(self, audio_data: np.ndarray, cache: dict, is_final: bool = False) -> dict:
        return self.streaming.transcribe_streaming(audio_data, cache, is_final)

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000,
                         language: str = "zh", engine: str = "") -> dict:
        e = engine or self.select_engine(language)
        if e == self.ENGINE_WHISPER:
            return self.whisper.transcribe_array(audio_array, sample_rate, language)
        if e == self.ENGINE_FUNASR:
            return self.funasr.transcribe_array(audio_array, sample_rate, language)
        return self.streaming.transcribe_array(audio_array, sample_rate)

    def transcribe(self, audio_path: str, language: str = "zh", engine: str = "") -> dict:
        e = engine or self.select_engine(language)
        if e == self.ENGINE_WHISPER:
            return self.whisper.transcribe(audio_path, language)
        if e == self.ENGINE_FUNASR:
            return self.funasr.transcribe(audio_path, language)
        return self.streaming.transcribe(audio_path)

    def batch_transcribe(self, input_path: str, language: str = "zh",
                         engine: str = "") -> Tuple[str, List[Dict[str, str]]]:
        e = engine or self.select_engine(language)
        if e == self.ENGINE_WHISPER:
            return self.whisper.batch_transcribe(input_path, language)
        if e == self.ENGINE_FUNASR:
            return self.funasr.batch_transcribe(input_path, language)
        return self.streaming.batch_transcribe(input_path, language)

    def punctuate(self, text: str) -> str:
        return self.punctuation.punctuate(text)
