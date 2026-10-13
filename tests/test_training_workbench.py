"""Protect checkpoint selection, isolated retraining and stop/history behavior."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from Code.FastApi.Base.Gateway.schemas import TrainingWorkbenchRequest
from Code.FastApi.Base.Gateway.training_workbench import PreviewRequest, TrainingWorkbench, checkpoint_files, validate_resume


def request(**overrides):
    return TrainingWorkbenchRequest(project_name="role",world_name="world",version="v5dev",**overrides)


class CheckpointTests(unittest.TestCase):
    def test_history_loss_is_limited_to_the_selected_run(self):
        from torch.utils.tensorboard import SummaryWriter
        from Code.FastApi.Base.Training.telemetry import read_telemetry
        with tempfile.TemporaryDirectory() as directory:
            with SummaryWriter(str(Path(directory)/"train/sovits")) as writer:
                writer.add_scalar("loss/g/total",1.0,1,walltime=100)
                writer.add_scalar("loss/g/total",2.0,2,walltime=200)
                writer.add_scalar("loss/g/total",3.0,3,walltime=300)
            values=read_telemetry(directory,"sovits",window_start=150,window_end=250)
            self.assertEqual([(v["step"],v["value"]) for v in values["loss_history"]],[(2,2.0)])

    def test_gpu_and_training_parameters_reject_invalid_values(self):
        self.assertEqual(request(gpu_numbers="0,1").gpu_numbers,"0-1")
        for fields in ({"gpu_numbers":"0,0"},{"gpu_numbers":"gpu0"},{"gpt_total_epoch":0},{"sovits_batch_size":0},{"lora_rank":0}):
            with self.assertRaises(ValueError): request(**fields)

    def test_resume_requires_matching_version_rank_and_more_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            canonical=Path(directory)/"v5dev";workspace=canonical/"runs/previous"
            (workspace/"train/gpt/ckpt").mkdir(parents=True)
            (workspace/"dataset/logs_s2_v5dev_lora_32").mkdir(parents=True)
            torch.save({"epoch":1},workspace/"train/gpt/ckpt/epoch=1.ckpt")
            torch.save({"iteration":2},workspace/"dataset/logs_s2_v5dev_lora_32/G_1.pth")
            validate_resume(workspace,canonical,request(gpt_total_epoch=3,sovits_total_epoch=3))
            for candidate,params in ((workspace,request(gpt_total_epoch=2,sovits_total_epoch=3)),(workspace,request(gpt_total_epoch=3,sovits_total_epoch=3,lora_rank=4)),(Path(directory)/"v5turbo",request())):
                with self.assertRaises(ValueError):validate_resume(candidate,canonical,params)


class WorkbenchLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/"Resources";self.root.mkdir()
        async def user():return None
        self.gateway=SimpleNamespace(get_current_user=user,training_workflow_lock=asyncio.Lock())
        self.patch=patch("Code.FastApi.Base.Gateway.training_workbench.resources_root",return_value=self.root)
        self.patch.start()
        self.workbench=TrainingWorkbench(self.gateway)

    async def asyncTearDown(self):
        self.patch.stop();self.temp.cleanup()

    def record(self,state="queued"):
        return {"workflow_id":"test_run","status":state,"phase":state,"world_name":"world","project_name":"role","version":"v5dev",
                "created_at":"2026-10-13T00:00:00","targets":[],"current_target":None,"stop_requested":False}

    async def test_cancelling_before_background_task_starts_marks_stopped(self):
        record=self.record();self.workbench.runs["test_run"]=record
        task=asyncio.create_task(asyncio.sleep(600));self.workbench.tasks["test_run"]=task
        result=await self.workbench.stop("test_run")
        self.assertEqual(result["status"],"stopped");self.assertTrue(task.cancelled())

    async def test_preview_persists_base64_audio_without_temporary_file_path(self):
        import base64
        async def infer(_):return SimpleNamespace(success=True,audio_data=base64.b64encode(b"RIFF-test-wave").decode(),audio_path=None)
        self.gateway._ensure_service=lambda _:SimpleNamespace(load_models=lambda *_:True,inference=infer)
        self.gateway._to_payload=lambda result:vars(result)
        workspace=self.root/"run";workspace.mkdir()
        record=dict(self.record("completed"),project_root=str(workspace),options={"language":"zh"},
                    preview={"gpt_path":"gpt.ckpt","sovits_path":"sovits.pth","emotions":[{"name":"日常","path":"ref.wav","text":"文本","language":"zh"}]})
        self.workbench.runs[record["workflow_id"]]=record
        result=await self.workbench.preview(record["workflow_id"],PreviewRequest(text="试听",emotion="日常"))
        self.assertEqual(Path(result["audio_path"]).read_bytes(),b"RIFF-test-wave")
        self.assertEqual(record["previews"][-1]["path"],result["audio_path"])

    async def test_restart_keeps_completed_history_and_marks_monitoring_interrupted(self):
        done=self.record("completed");self.workbench.save(done)
        running=dict(self.record("running"),workflow_id="active");self.workbench.save(running)
        loaded=TrainingWorkbench(self.gateway)
        self.assertEqual(loaded.runs["test_run"]["status"],"completed")
        self.assertEqual(loaded.runs["active"]["status"],"interrupted")

    async def test_fresh_cached_features_do_not_include_previous_checkpoints(self):
        source=self.root/"source";dataset=source/"dataset";dataset.mkdir(parents=True)
        for name in ("3-bert","4-cnhubert","5-wav32k"):(dataset/name).mkdir()
        (dataset/"2-name2text.txt").write_text("voice.wav\tphonemes\twords\t文本\n")
        (dataset/"6-name2semantic.tsv").write_text("item_name\tsemantic_audio\nvoice.wav\t1 2 3\n")
        old=dataset/"logs_s2_v5dev_lora_32";old.mkdir();torch.save({"iteration":12},old/"G_1.pth")
        fresh=self.root/"fresh"
        record=dict(self.record("running"),preprocessing={})
        await self.workbench.prepare(record,request(source_version="v5dev",source_workspace=str(source)),{"dataset_root":str(fresh/"dataset")})
        self.assertEqual(checkpoint_files(fresh,"sovits","v5dev"),[])
        self.assertTrue((fresh/"dataset/4-cnhubert").is_symlink())
        self.assertFalse((fresh/"dataset/6-name2semantic.tsv").is_symlink())
        self.assertEqual(torch.load(old/"G_1.pth",weights_only=False)["iteration"],12)


if __name__ == "__main__":unittest.main()
