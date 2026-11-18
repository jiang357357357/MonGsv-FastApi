"""Verify official training parameters and final-only saving across all versions."""
import json
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace

import yaml

from Code.FastApi.Base.Gateway.schemas import TrainingWorkbenchRequest, TrainingWorkflowOptions
from Code.FastApi.Base.Training.defaults import fill_training_defaults, training_defaults
from Code.FastApi.Base.Training.gpt_training.service import GPTTrainingConfig, GPTTrainingRequest, GPTTrainingService
from Code.FastApi.Base.Training.sovits_training.service import SoVITSTrainingConfig, SoVITSTrainingRequest, SoVITSTrainingService

ROOT = Path(__file__).resolve().parents[3]
# Official epochs and batch sizes for a 32 GiB GPU; saving follows the user's policy.
OFFICIAL = {
    "v1": (8, 16), "v2": (8, 16), "v2Pro": (8, 16), "v2ProPlus": (8, 16),
    "v3": (2, 4), "v4": (2, 4), "v5dev": (2, 4), "v5turbo": (2, 4),
}


class OfficialTrainingDefaultTests(unittest.TestCase):
    def setUp(self):
        self.memory = patch("Code.FastApi.Base.Training.defaults.hardware_memory", return_value=(32.0, True, True))
        self.memory.start()
        self.addCleanup(self.memory.stop)

    def test_all_versions_keep_official_training_parameters_and_save_only_at_the_end(self):
        with tempfile.TemporaryDirectory() as directory:
            for version, (epochs, batch) in OFFICIAL.items():
                with self.subTest(version=version):
                    options = TrainingWorkbenchRequest(project_name="role", world_name="world", version=version)
                    self.assertEqual((options.sovits_total_epoch, options.gpt_total_epoch), (epochs, 15))
                    self.assertEqual((options.sovits_batch_size, options.gpt_batch_size), (batch, 16))
                    self.assertEqual((options.sovits_save_every_epoch, options.gpt_save_every_epoch), (epochs, 15))
                    self.assertFalse(options.grad_checkpoint)
                    workspace = Path(directory) / version
                    workspace.mkdir()
                    gpt_service = object.__new__(GPTTrainingService)
                    sovits_service = object.__new__(SoVITSTrainingService)
                    gpt_service.gpt_sovits_root = sovits_service.gpt_sovits_root = str(ROOT)
                    gpt = GPTTrainingRequest(exp_name="role", exp_root=directory, config=GPTTrainingConfig(version=version))
                    sovits = SoVITSTrainingRequest(exp_name="role", exp_root=directory, config=SoVITSTrainingConfig(version=version))
                    gpt_path = gpt_service._create_config_file(gpt, str(workspace))
                    sovits_path = sovits_service._create_config_file(sovits, str(workspace))
                    g = yaml.safe_load(Path(gpt_path).read_text())
                    s = json.loads(Path(sovits_path).read_text())
                    self.assertEqual((g["train"]["epochs"], g["train"]["batch_size"], g["train"]["save_every_n_epoch"]), (15, 16, 15))
                    self.assertEqual((s["train"]["epochs"], s["train"]["batch_size"], s["train"]["save_every_epoch"]), (epochs, batch, epochs))
                    self.assertFalse(s["train"]["grad_ckpt"])
                    self.assertEqual((g["optimizer"]["lr"], g["optimizer"]["warmup_steps"]), (0.01, 2000))
                    self.assertEqual(s["train"]["learning_rate"], 0.0001)

    def test_automatic_batch_uses_gpu_or_cpu_memory_and_never_becomes_zero(self):
        for memory, gpu, expected in [(16, True, (8, 2)), (8, True, (4, 1)), (2, True, (1, 1)), (32, False, (8, 8))]:
            with self.subTest(memory=memory, gpu=gpu), patch(
                "Code.FastApi.Base.Training.defaults.hardware_memory", return_value=(memory, gpu, gpu)
            ):
                defaults = training_defaults("v5dev")
                self.assertEqual((defaults["gpt_batch_size"], defaults["sovits_batch_size"]), expected)
                self.assertEqual(defaults["fp16"], gpu)

    def test_old_requests_use_final_saving_and_keep_other_training_parameters(self):
        values = dict(project_name="role", world_name="world", version="v5dev", gpt_batch_size=4,
                      sovits_batch_size=2, gpt_total_epoch=20, sovits_total_epoch=12,
                      grad_checkpoint=True, fp16=False, save_every_epoch=1)
        request = TrainingWorkbenchRequest(**values)
        restored = TrainingWorkbenchRequest.model_validate_json(request.model_dump_json())
        self.assertEqual((restored.gpt_total_epoch, restored.sovits_total_epoch), (20, 12))
        self.assertEqual((restored.gpt_batch_size, restored.sovits_batch_size), (4, 2))
        self.assertEqual((restored.gpt_save_every_epoch, restored.sovits_save_every_epoch), (20, 12))
        self.assertIsNone(restored.save_every_epoch)
        self.assertTrue(restored.grad_checkpoint)
        self.assertFalse(restored.fp16)
        overridden = TrainingWorkbenchRequest(**values, gpt_save_every_epoch=3, sovits_save_every_epoch=2)
        self.assertEqual((overridden.gpt_save_every_epoch, overridden.sovits_save_every_epoch), (20, 12))

    def test_custom_epoch_counts_reach_generated_configs_even_with_stale_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            for version in OFFICIAL:
                for epochs in (1, 7, 13):
                    with self.subTest(version=version, epochs=epochs):
                        workspace = Path(directory) / f"{version}-{epochs}"
                        workspace.mkdir()
                        for config_type, request_type, service_type, suffix, key in (
                            (GPTTrainingConfig, GPTTrainingRequest, GPTTrainingService, ".yaml", "save_every_n_epoch"),
                            (SoVITSTrainingConfig, SoVITSTrainingRequest, SoVITSTrainingService, ".json", "save_every_epoch"),
                        ):
                            config = config_type(version=version, total_epoch=epochs, save_every_epoch=1)
                            self.assertEqual(config.save_every_epoch, epochs)
                            # A copied/mutated config must not reintroduce a periodic save interval.
                            config = config.model_copy(update={"save_every_epoch": 1})
                            service = object.__new__(service_type)
                            service.gpt_sovits_root = str(ROOT)
                            request = request_type(exp_name="role", exp_root=directory, config=config)
                            path = Path(service._create_config_file(request, str(workspace)))
                            data = yaml.safe_load(path.read_text()) if suffix == ".yaml" else json.loads(path.read_text())
                            self.assertEqual(data["train"][key], epochs)
                            self.assertEqual(data["train"]["epochs"], epochs)
                            self.assertTrue(data["train"]["if_save_every_weights"])

    def test_legacy_workflows_use_the_same_defaults(self):
        for version, (epochs, _) in OFFICIAL.items():
            with self.subTest(version=version):
                legacy = fill_training_defaults(TrainingWorkflowOptions(), version)
                self.assertEqual((legacy.sovits_total_epoch, legacy.sovits_save_every_epoch, legacy.gpt_save_every_epoch), (epochs, epochs, 15))

    def test_v2_uses_official_v2_gpt_checkpoint_and_other_versions_keep_their_base(self):
        service = object.__new__(GPTTrainingService)
        service.gpt_sovits_root = str(ROOT)
        self.assertTrue(service._resolve_pretrained_s1("v2", "").endswith("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"))
        self.assertTrue(service._resolve_pretrained_s1("v1", "").endswith("s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"))
        for version in ("v2Pro", "v2ProPlus", "v3", "v4", "v5dev", "v5turbo"):
            self.assertTrue(service._resolve_pretrained_s1(version, "").endswith("s1v3.ckpt"))
        self.assertEqual(service._resolve_pretrained_s1("v2", "custom.ckpt"), "custom.ckpt")

    def test_gateway_passes_final_save_intervals_to_actual_service_configs(self):
        from Code.FastApi.Base.Gateway.service_manager import ServiceManager
        with patch.object(ServiceManager, "load_all_services", return_value=None):
            from Code.FastApi.Base.Gateway import unified_gateway as gateway
        for version, (epochs, _) in OFFICIAL.items():
            with self.subTest(version=version):
                options = TrainingWorkflowOptions()
                service = SimpleNamespace(start_training=AsyncMock(return_value=SimpleNamespace(success=True)))
                with patch.object(gateway, "_ensure_service", return_value=service):
                    for target, expected in (("gpt", 15), ("sovits", epochs)):
                        asyncio.run(gateway._launch_training_target(target, "role", "workspace", version, options, "gpt-out", "sovits-out"))
                        self.assertEqual(service.start_training.call_args.args[0].config.save_every_epoch, expected)


if __name__ == "__main__":
    unittest.main()
