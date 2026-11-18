"""Protect the actual Nano selection, portable file loading and model cache."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from tools.asr import funasr_asr
from Code.FastApi.Base.ASR.engines.asr import FunASREngine
from Code.FastApi.Base.DataPreparation.asr_recognition.service import ASRConfig, ASRRecognitionService, ASRRequest


class FunASRNanoTests(unittest.TestCase):
    def test_default_backend_loads_nano_and_keeps_classic_explicit(self):
        sentinel = object()
        with patch.object(funasr_asr, 'create_fun_asr_nano_model', return_value=sentinel) as load:
            self.assertIs(funasr_asr.create_model('zh', use_cache=False), sentinel)
            load.assert_called_once()
        self.assertEqual(funasr_asr.normalize_backend(), 'fun-asr-nano')
        self.assertEqual(funasr_asr.normalize_backend('paraformer'), 'classic')

    def test_complete_local_snapshot_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in funasr_asr.FUN_ASR_NANO_REQUIRED_FILES:
                file = folder / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            with patch.object(funasr_asr, 'FUN_ASR_NANO_MODEL_DIR', folder), \
                 patch('huggingface_hub.snapshot_download') as download:
                self.assertEqual(funasr_asr.ensure_fun_asr_nano_snapshot(), folder)
                download.assert_not_called()

    def test_missing_snapshot_download_uses_pinned_revision_and_checks_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(funasr_asr, 'FUN_ASR_NANO_MODEL_DIR', Path(directory)), \
                 patch('huggingface_hub.snapshot_download') as download:
                with self.assertRaises(FileNotFoundError):
                    funasr_asr.ensure_fun_asr_nano_snapshot()
                self.assertEqual(download.call_args.kwargs['revision'], funasr_asr.FUN_ASR_NANO_REVISION)
                self.assertEqual(download.call_args.kwargs['repo_id'], 'FunAudioLLM/Fun-ASR-Nano-2512')

    def test_local_model_and_vad_use_native_implementation_on_cpu(self):
        folder = Path('/portable/asr/Fun-ASR-Nano-2512')
        with patch.object(funasr_asr, 'ensure_fun_asr_nano_snapshot', return_value=folder), \
             patch.object(funasr_asr, 'ensure_snapshot', return_value='/portable/asr/vad'), \
             patch.object(funasr_asr, 'AutoModel') as constructor:
            funasr_asr.create_fun_asr_nano_model('cpu')
        options = constructor.call_args.kwargs
        self.assertEqual(options['model'], str(folder))
        self.assertEqual(options['vad_model'], '/portable/asr/vad')
        self.assertFalse(options['trust_remote_code'])
        self.assertEqual(options['llm_conf']['llm_dtype'], 'fp32')

    def test_engine_requests_nano_and_reuses_its_model(self):
        engine = object.__new__(FunASREngine)
        engine._create_model = Mock(side_effect=[object(), object()])
        engine._model_cache = {}
        self.assertIs(engine._get_model('zh'), engine._get_model('zh'))
        engine._create_model.assert_called_once_with('zh', use_cache=False, backend='fun-asr-nano')
        engine._get_model('yue')
        engine._create_model.assert_called_with('yue', use_cache=False, backend='classic')

    def test_batch_training_asr_uses_nano_and_reports_the_actual_backend(self):
        model = Mock()
        model.generate.return_value = [{'text': '这是一段测试录音。'}]
        root = str(Path(__file__).resolve().parents[3])
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'sample.wav'
            sf.write(audio, np.zeros(1600, dtype=np.float32), 16000)
            with patch.object(funasr_asr, 'create_fun_asr_nano_model', return_value=model) as load:
                service = ASRRecognitionService(root)
                response = asyncio.run(service.process(ASRRequest(input_path=str(audio),
                    output_dir=str(Path(directory) / 'result'), config=ASRConfig())))
                self.assertTrue(response.success, response.message)
                self.assertEqual(response.recognition_results[0]['text'], '这是一段测试录音。')
                self.assertEqual(service.get_model_info()['model_descriptor'], 'fun-asr-nano/zh')
                self.assertEqual(service.get_supported_models()['funasr']['model_id'], 'FunAudioLLM/Fun-ASR-Nano-2512')
                load.assert_called_once()
                self.assertTrue(Path(response.output_file).is_file())


if __name__ == '__main__':
    unittest.main()
