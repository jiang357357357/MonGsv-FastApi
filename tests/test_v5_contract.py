"""Compatibility checks for V5 requests and exported LoRA checkpoint headers."""
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

from Code.FastApi.Base.Gateway.schemas import RoleEmotionSynthesisRequest
from Code.FastApi.Base.Inference.service import InferenceConfig, InferenceRequest, InferenceService
from Code.FastApi.Base.Inference.model_manager import ModelManager
from Code.FastApi.Base.Training.sovits_training.service import SoVITSTrainingService


class V5ContractTests(unittest.TestCase):
    def test_dataset_request_schemas_accept_actual_v5_variants(self):
        from Code.FastApi.Base.DatasetFormatting.audio_features.server import AudioFeaturesConfigModel
        for version in ("v5dev", "v5turbo"):
            self.assertEqual(AudioFeaturesConfigModel(version=version).version, version)
        with self.assertRaises(ValueError):
            AudioFeaturesConfigModel(version="v5")

    def test_accel_import_preserves_host_runtime_settings(self):
        script = '''import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = False
torch.ones(2, 2).matmul(torch.ones(2, 2))
from pathlib import Path
from Code.FastApi.Base.gpt_sovits_env import setup_gpt_sovits_paths
setup_gpt_sovits_paths(Path.cwd())
import GPT_SoVITS.Accel
assert torch.get_num_threads() == 2
assert torch.get_num_interop_threads() == 1
assert not torch.backends.cudnn.benchmark
'''
        subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[3])

    def test_omitted_sampling_parameters_reach_model_defaults(self):
        payload = RoleEmotionSynthesisRequest(role_id=1, emotion="test", text="测试文本")
        request = InferenceRequest(text=payload.text, config=InferenceConfig(sample_steps=payload.sample_steps))
        inputs = object.__new__(InferenceService)._build_tts_inputs(request, "reference.wav")
        self.assertIsNone(inputs["sample_steps"])
        self.assertIsNone(inputs["cfg_rate"])
        self.assertFalse(inputs["use_cuda_graph"])
        self.assertFalse(inputs["use_flash_attention"])

    def test_explicit_sampling_parameters_are_preserved(self):
        payload = RoleEmotionSynthesisRequest(role_id=1, emotion="test", text="测试文本", sample_steps=8, cfg_rate=1.1)
        request = InferenceRequest(text=payload.text, config=InferenceConfig(sample_steps=payload.sample_steps, cfg_rate=payload.cfg_rate))
        inputs = object.__new__(InferenceService)._build_tts_inputs(request, "reference.wav")
        self.assertEqual((inputs["sample_steps"], inputs["cfg_rate"]), (8, 1.1))

    def test_lora_version_comes_from_header_without_version_in_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "role.pth"
            manager = object.__new__(ModelManager)
            for header, version in [(b"07", "v5dev"), (b"08", "v5turbo")]:
                model.write_bytes(header + b"test-header")
                self.assertEqual(manager._detect_model_version(str(model)), version)

    def test_v5_training_uses_v5_generator_without_legacy_discriminator(self):
        service = object.__new__(SoVITSTrainingService)
        service.gpt_sovits_root = str(Path(__file__).resolve().parents[3])
        for version in ("v5dev", "v5turbo"):
            generator, discriminator = service._resolve_pretrained_s2_paths(version, "", "")
            self.assertTrue(generator.endswith(f"gsv-v5-pretrained/s2G{version}.pth"))
            self.assertEqual(discriminator, "")
        with self.assertRaises(ValueError):
            service._resolve_pretrained_s2_paths("v5", "", "")


if __name__ == "__main__":
    unittest.main()
