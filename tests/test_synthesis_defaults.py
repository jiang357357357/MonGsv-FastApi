"""The UI contract must reach inference unchanged and reject out-of-range API requests."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from Code.FastApi.Base.Gateway.schemas import RoleEmotionSynthesisRequest
from Code.FastApi.Base.Inference.service import InferenceConfig, InferenceRequest, InferenceService


class SynthesisDefaultsTests(unittest.TestCase):
    def test_webui_range_boundaries_survive_both_request_layers(self):
        for values in [dict(top_k=1, top_p=0, temperature=0, cfg_rate=0, pause_second=0.1),
                       dict(top_k=100, top_p=1, temperature=1, cfg_rate=2, pause_second=0.5)]:
            payload = RoleEmotionSynthesisRequest(role_id=1, emotion='test', text='测试', **values)
            config = InferenceConfig(**{key: getattr(payload, key) for key in values})
            for key, value in values.items():
                self.assertEqual(getattr(config, key), value)

    def test_out_of_range_parameters_are_rejected_before_inference(self):
        invalid = [('top_k', 0), ('top_k', 101), ('top_p', -0.01), ('top_p', 1.01),
                   ('temperature', -0.01), ('temperature', 1.01), ('cfg_rate', -0.01),
                   ('cfg_rate', 2.01), ('pause_second', 0), ('pause_second', 0.51),
                   ('temperature', float('nan')), ('cfg_rate', float('inf'))]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValidationError):
                    RoleEmotionSynthesisRequest(role_id=1, emotion='test', text='测试', **{key: value})
                with self.assertRaises(ValidationError):
                    InferenceConfig(**{key: value})

    def test_only_official_sampling_choices_are_accepted(self):
        for steps in [4, 8, 16, 32]:
            self.assertEqual(InferenceConfig(sample_steps=steps).sample_steps, steps)
            self.assertEqual(RoleEmotionSynthesisRequest(role_id=1, emotion='test', text='测试', sample_steps=steps).sample_steps, steps)
        for steps in [0, 1, 7, 64, 128, 8.5]:
            with self.subTest(steps=steps):
                with self.assertRaises(ValidationError):
                    InferenceConfig(sample_steps=steps)
                with self.assertRaises(ValidationError):
                    RoleEmotionSynthesisRequest(role_id=1, emotion='test', text='测试', sample_steps=steps)

    def test_v4_fallback_and_explicit_choices_reach_pipeline(self):
        service = object.__new__(InferenceService)
        service.model_version = 'v4'
        request = InferenceRequest(text='测试')
        self.assertEqual(service._build_tts_inputs(request, 'ref.wav')['sample_steps'], 8)
        # The actual pipeline version remains usable when only the pipeline is available.
        service.model_version = None
        service.tts_pipeline = SimpleNamespace(configs=SimpleNamespace(version='v4'))
        self.assertEqual(service._build_tts_inputs(request, 'ref.wav')['sample_steps'], 8)
        for steps in [4, 8, 16, 32]:
            request.config.sample_steps = steps
            self.assertEqual(service._build_tts_inputs(request, 'ref.wav')['sample_steps'], steps)

    def test_role_endpoint_preserves_v4_default_and_zero_sampling_values(self):
        from Code.FastApi.Base.Gateway import unified_gateway as gateway
        role = SimpleNamespace(id=1, world_id=2, version='v4', gpt_model_path='gpt.ckpt',
                               sov_model_path='sovits.pth', prompt_text='参考', language='zh')
        roles = SimpleNamespace(get_role=lambda _: role,
            list_role_emotions=lambda _: [{'name': '日常', 'music_url': 'ref.wav', 'text': '参考'}])
        service = SimpleNamespace(inference=AsyncMock(return_value={'success': True}),
            get_model_info=lambda: {'models_loaded': True, 'gpt_path': 'gpt.ckpt', 'sovits_path': 'sovits.pth'})
        with patch.object(gateway, '_ensure_service', return_value=service), patch.object(gateway, 'RoleService', return_value=roles):
            asyncio.run(gateway.synthesize_role_emotion(RoleEmotionSynthesisRequest(
                role_id=1, world_id=2, version='v4', emotion='日常', text='试听', top_p=0, temperature=0, pause_second=0.1)))
        inference_service = object.__new__(InferenceService)
        inference_service.model_version = 'v4'
        inputs = inference_service._build_tts_inputs(service.inference.call_args.args[0], 'ref.wav')
        self.assertEqual(inputs['sample_steps'], 8)
        self.assertEqual(inputs['top_p'], 0)
        self.assertEqual(inputs['temperature'], 0)
        self.assertEqual(inputs['fragment_interval'], 0.1)


if __name__ == '__main__':
    unittest.main()
