import ast
import importlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from Code.FastApi.Base.Gateway.service_manager import ServiceManager


GATEWAY_PATH = Path(__file__).resolve().parents[1] / "Base" / "Gateway" / "unified_gateway.py"


class TranscribeRouteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = GATEWAY_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.functions = {
            node.name: node
            for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def _post_path(self, function_name: str) -> str | None:
        function = self.functions[function_name]
        for decorator in function.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "post"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
            ):
                return decorator.args[0].value
        return None

    def _speaker_gate_argument(self, function_name: str) -> bool | None:
        function = self.functions[function_name]
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "_inference_transcribe":
                continue
            for keyword in node.keywords:
                if keyword.arg == "require_speaker_gate" and isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
        return None

    def test_plain_route_explicitly_disables_speaker_gate(self):
        self.assertEqual(self._post_path("inference_transcribe"), "/inference/transcribe")
        self.assertIs(self._speaker_gate_argument("inference_transcribe"), False)

    def test_voiceprint_route_explicitly_enables_speaker_gate(self):
        self.assertEqual(
            self._post_path("inference_transcribe_with_voiceprint"),
            "/inference/transcribe/voiceprint",
        )
        self.assertIs(
            self._speaker_gate_argument("inference_transcribe_with_voiceprint"),
            True,
        )

    def test_shared_transcriber_only_authorizes_when_gate_is_required(self):
        function = self.functions["_inference_transcribe"]
        function_source = ast.get_source_segment(self.source, function) or ""

        self.assertIn("_authorize_transcription_speaker", function_source)
        self.assertIn("if require_speaker_gate", function_source)


class TranscribeRouteBehaviorTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(ServiceManager, "load_all_services", return_value=None):
            cls.gateway = importlib.import_module(
                "Code.FastApi.Base.Gateway.unified_gateway"
            )

    def _upload(self):
        return self.gateway.UploadFile(
            filename="role.wav",
            file=io.BytesIO(b"test-audio"),
        )

    def _service(self):
        result = SimpleNamespace(
            success=True,
            message="识别完成",
            recognition_results=[{"text": "角色台词", "language": "zh"}],
            processing_time=0.1,
        )
        return SimpleNamespace(process=AsyncMock(return_value=result))

    async def test_plain_route_never_calls_voiceprint_authorization(self):
        with (
            patch.object(self.gateway, "_ensure_service", return_value=self._service()),
            patch.object(
                self.gateway,
                "_authorize_transcription_speaker",
                side_effect=AssertionError("plain transcription must not verify voiceprint"),
            ) as authorize_mock,
        ):
            response = await self.gateway.inference_transcribe(
                audio_file=self._upload(),
                audio_path="",
                language="zh",
                model_type="funasr",
                model_size="large",
                precision="float32",
                user=None,
            )

        authorize_mock.assert_not_called()
        self.assertEqual(response["text"], "角色台词")
        self.assertFalse(response["speaker_verified"])
        self.assertIsNone(response["speaker"])

    async def test_voiceprint_route_authorizes_before_transcription(self):
        authorization = {
            "status": "authorized",
            "speaker_info": {
                "speaker_id": "personal-owner",
                "similarity": 0.93,
                "is_match": True,
            },
        }
        with (
            patch.object(self.gateway, "_ensure_service", return_value=self._service()),
            patch.object(
                self.gateway,
                "_authorize_transcription_speaker",
                return_value=authorization,
            ) as authorize_mock,
        ):
            response = await self.gateway.inference_transcribe_with_voiceprint(
                audio_file=self._upload(),
                audio_path="",
                language="zh",
                model_type="funasr",
                model_size="large",
                precision="float32",
                user=None,
            )

        authorize_mock.assert_called_once()
        self.assertTrue(response["speaker_verified"])
        self.assertEqual(response["speaker"]["speaker_id"], "personal-owner")


if __name__ == "__main__":
    unittest.main()
