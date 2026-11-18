"""Reference audio must remain available after training moved into isolated runs."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Code.FastApi.Domain.Role.Models.role_info import RoleInfo
from Code.FastApi.Domain.Role.Services.role_service import RoleService


class EmotionSlicedAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "Model/world/role/v5dev"
        self.train = self.root / "Train/Projects/world/role/v5dev"
        self.model.mkdir(parents=True)
        self.train.mkdir(parents=True)
        self.service = RoleService.__new__(RoleService)
        self.service._model_root = lambda: self.root / "Model"
        self.service._train_projects_root = lambda: self.root / "Train/Projects"

    def clip(self, relative):
        path = self.train / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF-reference-audio")
        return path

    def workspace(self, **query):
        return self.service.list_role_workspaces(
            role_name="role", world_name="world", base_version="v5dev", **query
        )[0]

    def test_newest_run_audio_is_listed_and_empty_runs_do_not_hide_it(self):
        self.clip("dataset/sliced/legacy.wav")
        self.clip("runs/training_20261018/dataset/sliced/older.wav")
        latest = self.clip("runs/training_20261019/dataset/sliced/latest.wav")
        (self.train / "runs/training_20261020/dataset/sliced").mkdir(parents=True)
        latest.with_name("annotation.list").write_text("not audio")
        workspace = self.workspace()
        self.assertEqual(workspace["sliced_files"], [str(latest)])
        self.assertEqual(workspace["sliced_dir"], str(latest.parent))
        self.assertEqual(self.workspace(include_files=False)["sliced_files"], [])

    def test_legacy_source_and_cached_audio_remain_available(self):
        for relative in ("dataset/sliced/legacy.wav", "runs/one/source/sliced/source.wav", "runs/two/dataset/5-wav32k/cached.wav"):
            with self.subTest(relative=relative):
                path = self.clip(relative)
                self.assertEqual(self.workspace()["sliced_files"], [str(path)])
                self.assertEqual(self.service._resolve_role_audio_source(self.model, self.train, str(path)), path)
                path.unlink()

    def test_role_id_finds_clips_when_display_name_differs_from_directory(self):
        clip = self.clip("runs/training_20261019/dataset/sliced/voice.wav")
        self.service._write_json(self.model / "meta/role.json", {"display_name": "角色 v5dev"})
        role_id = self.service._role_id("world", "role", "v5dev")
        workspaces = self.service.list_role_workspaces(role_id=role_id)
        self.assertEqual(len(workspaces), 1)
        self.assertEqual(workspaces[0]["role_id"], role_id)
        self.assertEqual(workspaces[0]["sliced_files"], [str(clip)])
        self.assertEqual(self.service.list_role_workspaces(role_id=role_id + 1), [])

    def test_selected_run_clip_is_copied_into_the_new_emotion(self):
        path = self.clip("runs/training_20261019/dataset/sliced/voice.wav")
        entry = SimpleNamespace(model_root=self.model, train_root=self.train,
                                info=RoleInfo(id=1, name="role", world_name="world", version="v5dev"))
        self.service._require_role_entry = lambda _: entry
        result = self.service.upsert_role_emotion(
            1, "开心", "你好", text_language="zh", audio_source_path=str(path)
        )
        saved = Path(result["audio_file"])
        self.assertEqual(saved.parent, self.model / "emotion")
        self.assertEqual(saved.read_bytes(), path.read_bytes())
        self.assertEqual(result["emotions"][0]["name"], "开心")

    def test_audio_from_other_roles_versions_or_outside_resources_is_rejected(self):
        for relative in ("../v5turbo/runs/one/dataset/sliced/wrong.wav", "../../other/v5dev/runs/one/dataset/sliced/wrong.wav", "../../../../../outside.wav"):
            with self.subTest(relative=relative):
                path = self.clip(relative).resolve()
                with self.assertRaisesRegex(ValueError, "当前角色和版本"):
                    self.service._resolve_role_audio_source(self.model, self.train, str(path))


if __name__ == "__main__":
    unittest.main()
