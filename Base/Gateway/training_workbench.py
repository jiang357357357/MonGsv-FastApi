"""Training workbench: isolated runs, explicit resume, history and previews."""
from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from Code.FastApi.Base.Gateway.resources import normalize_path_segment, resource_layout, resources_root
from Code.FastApi.Base.Gateway.schemas import TrainingWorkbenchRequest
from Code.FastApi.Base.Training.telemetry import log_tail, read_telemetry
from Code.FastApi.Base.Training.defaults import SOURCE_URL, VERSIONS, training_defaults

STAGES = ("audio_slice", "asr_recognition", "text_processing", "audio_features", "semantic_encoding")
FEATURES = ("2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k", "6-name2semantic.tsv", "7-sv_cn")
TERMINAL = {"completed", "failed", "stopped", "interrupted"}


def feature_workspace(canonical: Path) -> Path:
    candidates = [canonical, *sorted((canonical/"runs").glob("*"),key=lambda p:p.stat().st_mtime_ns,reverse=True)]
    return next((p for p in candidates if all((p/"dataset"/name).exists() for name in FEATURES[:4])),canonical)


class PreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    emotion: str
    seed: int = 2027


def checkpoint_files(workspace: Path, target: str, version: str, rank: int = 32) -> list[Path]:
    if target == "gpt":
        paths = list((workspace / "train/gpt/ckpt").glob("*.ckpt"))
    else:
        suffix = f"logs_s2_{version}_lora_{rank}" if version in {"v3", "v4", "v5dev", "v5turbo"} else f"logs_s2_{version}"
        paths = list((workspace / "dataset" / suffix).glob("G_*.pth"))
    return sorted(paths, key=lambda p:p.stat().st_mtime_ns)


def validate_resume(workspace: Path, canonical: Path, request: TrainingWorkbenchRequest) -> None:
    if not workspace.is_relative_to(canonical) or not (workspace / "dataset").is_dir():
        raise ValueError("续训目录必须属于当前角色和模型版本")
    import torch
    for target, enabled, total in (("gpt",request.train_gpt,request.gpt_total_epoch), ("sovits",request.train_sovits,request.sovits_total_epoch)):
        if not enabled:
            continue
        files = checkpoint_files(workspace, target, request.version, request.lora_rank)
        if not files:
            raise ValueError(f"选择的目录没有 {target.upper()} 训练 checkpoint；请使用从底模重训")
        checkpoint = torch.load(files[-1], map_location="cpu", weights_only=False, mmap=True)
        epoch = int(checkpoint["epoch"])+1 if target == "gpt" else int(checkpoint["iteration"])
        if total <= epoch:
            raise ValueError(f"{target.upper()} 已完成 {epoch} 轮，续训的累计总轮数必须大于 {epoch}")


class TrainingWorkbench:
    def __init__(self, gateway: Any):
        self.gateway = gateway
        self.store = resources_root().parent / "Data/Training/workflows"
        self.store.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, dict] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.telemetry_cache: dict[tuple, tuple[float,dict]] = {}
        for path in self.store.glob("*.json"):
            try:
                record = json.loads(path.read_text())
                if record["status"] not in TERMINAL:
                    record.update(status="interrupted", error="后端已重启，任务监控中断；可选择对应 checkpoint 续训", current_target=None)
                self.runs[record["workflow_id"]] = record
            except (ValueError, KeyError):
                continue
        self.router = APIRouter(prefix="/training/workbench", tags=["training-workbench"], dependencies=[Depends(gateway.get_current_user)])
        self.router.add_api_route("/start", self.start, methods=["POST"])
        self.router.add_api_route("/defaults", self.defaults, methods=["GET"])
        self.router.add_api_route("/workspaces", self.workspaces, methods=["GET"])
        self.router.add_api_route("/runs", self.history, methods=["GET"])
        self.router.add_api_route("/runs/{workflow_id}", self.status, methods=["GET"])
        self.router.add_api_route("/runs/{workflow_id}/stop", self.stop, methods=["POST"])
        self.router.add_api_route("/runs/{workflow_id}/logs", self.logs, methods=["GET"])
        self.router.add_api_route("/runs/{workflow_id}/preview", self.preview, methods=["POST"])

    async def defaults(self) -> dict:
        return {"source": SOURCE_URL, "save_policy": "final_only", "versions": {version: training_defaults(version) for version in VERSIONS}}

    def save(self, record: dict) -> None:
        record["updated_at"] = datetime.now().isoformat()
        destination = self.store / (record["workflow_id"] + ".json")
        temp = destination.with_suffix(".tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
        temp.replace(destination)

    def require(self, workflow_id: str) -> dict:
        if workflow_id not in self.runs:
            raise HTTPException(404, "训练记录不存在")
        return self.runs[workflow_id]

    async def workspaces(self, world_name: str, project_name: str) -> dict:
        try:
            layout = resource_layout(project_name, "v2Pro", world_name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        versions = []
        for directory in sorted(Path(layout["train_root"]).parent.iterdir()) if Path(layout["train_root"]).parent.exists() else []:
            if not directory.is_dir():
                continue
            cached = feature_workspace(directory)
            dataset = cached / "dataset"
            versions.append({"version":directory.name, "workspace":str(cached), "canonical_workspace":str(directory),
                             "text_lines":len((dataset/"2-name2text.txt").read_text().splitlines()) if (dataset/"2-name2text.txt").is_file() else 0,
                             "has_features":all((dataset/name).exists() for name in FEATURES[:4]),
                             "has_semantic":(dataset/"6-name2semantic.tsv").is_file()})
        resumes = []
        for entry in versions:
            directory = Path(entry["canonical_workspace"])
            for workspace in [directory, *sorted((directory/"runs").glob("*"))]:
                if not workspace.is_dir():
                    continue
                gpt = checkpoint_files(workspace, "gpt", entry["version"])
                sovits_ranks = []
                for folder in (workspace/"dataset").glob(f"logs_s2_{entry['version']}*"):
                    if list(folder.glob("G_*.pth")):
                        match = re.search(r"_lora_(\d+)$", folder.name)
                        sovits_ranks.append(int(match[1]) if match else 0)
                if gpt or sovits_ranks:
                    resumes.append({"workspace":str(workspace), "version":entry["version"],
                                    "gpt":bool(gpt), "sovits":bool(sovits_ranks), "lora_ranks":sovits_ranks,
                                    "label":f"{entry['version']} / {workspace.name}"})
        return {"versions":versions, "resume_choices":resumes}

    async def start(self, options: str = Form(...), audio_files: list[UploadFile] | None = File(default=None)) -> dict:
        try:
            request = TrainingWorkbenchRequest.model_validate_json(options)
            normalize_path_segment(request.world_name)
            normalize_path_segment(request.project_name)
            if not request.version or not (request.train_gpt or request.train_sovits):
                raise ValueError("请选择模型版本及至少一个训练目标")
            import torch
            if not torch.cuda.is_available() or any(int(gpu) >= torch.cuda.device_count() for gpu in request.gpu_numbers.split("-")):
                raise ValueError("选择的 GPU 不可用")
            layout = resource_layout(request.project_name, request.version, request.world_name)
            canonical = Path(layout["train_root"])
            if request.role_mode == "new" and (canonical.parent.exists() or Path(layout["model_root"]).parent.exists()):
                raise ValueError("角色已存在，请选择“已有角色 / 新增版本或复训”")
            if request.role_mode == "existing" and not (canonical.parent.exists() or Path(layout["model_root"]).parent.exists()):
                raise ValueError("已有角色不存在")
            if any(r["status"] not in TERMINAL and r["world_name"] == request.world_name and r["project_name"] == request.project_name and r["version"] == request.version for r in self.runs.values()):
                raise ValueError("当前角色版本已有排队或运行中的训练任务")
            workflow_id = "training_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            if request.training_mode == "resume":
                if not request.resume_workspace:
                    raise ValueError("请选择续训 checkpoint 所在的训练记录")
                workspace = Path(request.resume_workspace).resolve()
                validate_resume(workspace, canonical.resolve(), request)
                if any(r["status"] not in TERMINAL and r["project_root"] == str(workspace) for r in self.runs.values()):
                    raise ValueError("这个训练目录正在使用中")
            else:
                workspace = canonical / "runs" / workflow_id
                if request.preprocessing_mode == "cached":
                    source_canonical = canonical.parent / (request.source_version or request.version)
                    source_workspace = Path(request.source_workspace).resolve() if request.source_workspace else feature_workspace(source_canonical)
                    if not source_workspace.is_relative_to(source_canonical.resolve()):
                        raise ValueError("缓存来源目录必须属于当前角色和所选数据版本")
                    request.source_workspace = str(source_workspace)
                    source = source_workspace / "dataset"
                    if not all((source/name).exists() for name in FEATURES[:4]):
                        raise ValueError("数据来源缺少缓存特征，请选择完整流程或已有标注")
                    if ((request.source_version or request.version) == "v1") != (request.version == "v1"):
                        raise ValueError("v1 与其他版本的文本特征不同，请使用已有标注重新提取")
                elif request.preprocessing_mode == "existing":
                    if not Path(request.input_audio_dir).is_dir() or not Path(request.list_file).is_file():
                        raise ValueError("请提供有效的切分音频目录和 .list 标注")
                elif not audio_files and not Path(request.input_audio_dir).is_dir():
                    raise ValueError("请上传音频或提供有效的原始音频目录")
            # Uploads are staged before returning; background tasks never retain closed UploadFile handles.
            if audio_files:
                names=[Path(file.filename or "uploaded_audio.bin").name for file in audio_files]
                if len(set(names)) != len(names):
                    raise ValueError("上传的音频文件名重复，请先重命名")
                request.input_audio_dir = str(workspace/"source/raw")
                self.gateway._save_uploaded_audio_files(request.input_audio_dir, audio_files)
            layout.update(train_root=str(workspace), dataset_root=str(workspace/"dataset"),
                          model_sliced_dir=str(workspace/"source/sliced"),
                          gpt_model_dir=str(workspace/"export/GPT"), sovits_model_dir=str(workspace/"export/SoVITS"))
            order = ["gpt", "sovits"] if request.training_order == "gpt_first" else ["sovits", "gpt"]
            targets = [target for target in order if getattr(request,"train_"+target)]
            record = {"workflow_id":workflow_id, "project_name":request.project_name, "world_name":request.world_name,
                      "version":request.version, "project_root":str(workspace), "model_root":layout["model_root"],
                      "created_at":datetime.now().isoformat(), "status":"queued", "phase":"queued", "current_target":None,
                      "order":targets, "targets":[{"target":t,"status":"pending","job_id":None,"launch":None,"details":None,"error":None} for t in targets],
                      "preprocessing":{stage:"pending" for stage in STAGES}, "dataset":{}, "options":request.model_dump(),
                      "error":None, "stop_requested":False, "preview":None}
            self.runs[workflow_id] = record
            self.save(record)
            self.tasks[workflow_id] = asyncio.create_task(self.run(record, request, layout), name=workflow_id)
            self.tasks[workflow_id].add_done_callback(lambda _:self.tasks.pop(workflow_id,None))
            return {"success":True, "training_workflow":record, "preprocess_steps":[]}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def stage(self, record: dict, stage: str, status: str) -> None:
        record["current_stage"] = stage
        record["preprocessing"][stage] = status
        self.save(record)
        if record["stop_requested"]:
            raise asyncio.CancelledError()

    async def prepare(self, record: dict, request: TrainingWorkbenchRequest, layout: dict) -> None:
        if request.training_mode == "resume":
            record["preprocessing"] = dict.fromkeys(STAGES,"skipped")
            return
        callback = lambda stage,status:self.stage(record,stage,status)
        if request.preprocessing_mode != "cached":
            if request.preprocessing_mode == "existing":
                record["preprocessing"]["audio_slice"] = record["preprocessing"]["asr_recognition"] = "skipped"
                await self.gateway._run_formatting_only_workflow(
                    request.project_name, request.input_audio_dir, request.list_file, "", request.version,
                    request.world_name, layout_override=layout, progress_callback=callback)
            else:
                await self.gateway._run_preprocessing_workflow(
                    request.project_name, request.input_audio_dir, "", request.language, request.version,
                    request.world_name, layout_override=layout, progress_callback=callback)
            return
        source = Path(request.source_workspace) / "dataset"
        dataset = Path(layout["dataset_root"])
        dataset.mkdir(parents=True, exist_ok=True)
        for name in FEATURES:
            path = source/name
            if path.is_dir():
                try:
                    (dataset/name).symlink_to(path.resolve(), target_is_directory=True)
                except OSError:
                    shutil.copytree(path,dataset/name)
            elif path.is_file():
                shutil.copy2(path,dataset/name)
        record["preprocessing"] = dict.fromkeys(STAGES,"skipped")
        if request.version != (request.source_version or request.version) or (request.train_gpt and not (dataset/"6-name2semantic.tsv").exists()):
            rows = [line.split("\t") for line in (dataset/"2-name2text.txt").read_text().splitlines()]
            if not all(len(row)==4 for row in rows):
                raise ValueError("缓存文本特征格式不正确")
            annotations = dataset/"input.list"
            annotations.write_text("\n".join(f"{dataset/'5-wav32k'/row[0]}|{request.project_name}|{request.language}|{row[3]}" for row in rows)+"\n")
            callback("semantic_encoding","running")
            result = await self.gateway.execute_format_task("semantic_encoding",self.gateway._ensure_service("semantic_encoding"),str(annotations),str(dataset/"5-wav32k"),str(dataset),request.project_name,request.version)
            self.gateway._require_step_success("semantic_encoding",result)
            callback("semantic_encoding","completed")

    async def run(self, record: dict, request: TrainingWorkbenchRequest, layout: dict) -> None:
        try:
            async with self.gateway.training_workflow_lock:
                record.update(status="running",phase="preprocessing")
                self.save(record)
                await self.prepare(record,request,layout)
                if record["stop_requested"]:
                    raise asyncio.CancelledError()
                record["dataset"] = self.gateway._validate_training_dataset(layout["train_root"],request.version,request.train_gpt,request.train_sovits)
                record["phase"] = "training"
                for target in record["targets"]:
                    record["current_target"] = target["target"]
                    target["status"] = "starting"
                    self.save(record)
                    service,result = await self.gateway._launch_training_target(
                        target["target"], request.project_name+"-"+request.version+"-"+record["workflow_id"],
                        layout["train_root"], request.version, request, layout["gpt_model_dir"],layout["sovits_model_dir"])
                    if not result.success or not result.job_id:
                        raise ValueError(result.message)
                    target.update(status="running",job_id=result.job_id,launch=self.gateway._to_payload(result))
                    self.save(record)
                    final = await self.gateway._wait_for_training_job(service,result.job_id)
                    target.update(status=final["status"],details=final,error=final.get("error_message"))
                    self.save(record)
                    if final["status"] == "stopped":
                        raise asyncio.CancelledError()
                    if final["status"] != "completed":
                        raise ValueError(final.get("error_message") or "训练失败")
                    directory = Path(layout["gpt_model_dir"] if target["target"]=="gpt" else layout["sovits_model_dir"])
                    models = list(directory.glob("*.ckpt" if target["target"]=="gpt" else "*.pth"))
                    if not models:
                        raise ValueError("训练结束但没有导出模型")
                    target["model"] = str(max(models,key=lambda p:p.stat().st_mtime_ns))
                self.publish(record,request)
                record.update(status="completed",phase="completed",current_target=None)
        except asyncio.CancelledError:
            current = next((t for t in record["targets"] if t["target"]==record["current_target"]),None)
            if current and current.get("job_id"):
                service = self.gateway.service_manager.get_service(current["target"]+"_training")
                await asyncio.to_thread(service.stop_training,current["job_id"])
                current["status"] = "stopped"
            record.update(status="stopped",phase="stopped",current_target=None)
            for stage,status in record["preprocessing"].items():
                if status == "running":
                    record["preprocessing"][stage] = "stopped"
        except Exception as exc:
            record.update(status="failed",error=str(exc))
            for target in record["targets"]:
                if target["status"] in {"starting","running"}:
                    target.update(status="failed",error=str(exc))
            if record.get("current_stage") and record["phase"] == "preprocessing":
                record["preprocessing"][record["current_stage"]] = "failed"
            record["current_target"] = None
        finally:
            record["finished_at"] = datetime.now().isoformat()
            self.save(record)

    def publish(self, record: dict, request: TrainingWorkbenchRequest) -> None:
        root = Path(record["model_root"])
        models = {target["target"]:Path(target["model"]) for target in record["targets"]}
        for target in ("gpt","sovits"):
            if target not in models:
                folder = root/("GPT" if target=="gpt" else "SoVITS")
                candidates = list(folder.glob("*.ckpt" if target=="gpt" else "*.pth"))
                if candidates:
                    models[target] = max(candidates,key=lambda p:p.stat().st_mtime_ns)
                elif target == "gpt":
                    service = self.gateway._ensure_service("gpt_training")
                    models[target] = Path(service._resolve_pretrained_s1(request.version,""))
                else:
                    service = self.gateway._ensure_service("sovits_training")
                    models[target] = Path(service._resolve_pretrained_s2_paths(request.version,"","")[0])
        if request.publish_models:
            for target,path in models.items():
                folder = root/("GPT" if target=="gpt" else "SoVITS")
                folder.mkdir(parents=True,exist_ok=True)
                destination = folder/path.name
                if path.resolve() != destination.resolve():
                    shutil.copy2(path,destination)
            (root/"meta").mkdir(parents=True,exist_ok=True)
            meta = root/"meta/role.json"
            data = json.loads(meta.read_text()) if meta.exists() else {}
            data.update(role_name=request.project_name, world_name=request.world_name, base_version=request.version,
                        language=request.language,display_name=data.get("display_name") or f"{request.project_name} {request.version}")
            meta.write_text(json.dumps(data,ensure_ascii=False,indent=2))
        source = Path(resource_layout(request.project_name,request.source_version or request.version,request.world_name)["model_root"])/"emotion"
        emotions = []
        if source.is_dir():
            for file in source.glob("*.wav"):
                match = re.match(r"\[([^]]+)\]\[([^]]+)\](.*)",file.stem)
                if match:
                    emotions.append({"name":match[1],"language":match[2],"text":match[3],"path":str(file)})
                    if request.publish_models:
                        (root/"emotion").mkdir(exist_ok=True)
                        destination = root/"emotion"/file.name
                        if file.resolve()!=destination.resolve() and not destination.exists():
                            shutil.copy2(file,destination)
        if not emotions:
            import soundfile as sf
            dataset = Path(record["project_root"])/"dataset"
            for row in (dataset/"2-name2text.txt").read_text().splitlines():
                parts = row.split("\t")
                wav = dataset/"5-wav32k"/parts[0]
                if len(parts)==4 and wav.is_file() and 3<=sf.info(wav).duration<=10:
                    emotions=[{"name":"默认","language":request.language,"text":parts[3],"path":str(wav)}]
                    if request.publish_models:
                        (root/"emotion").mkdir(exist_ok=True)
                        safe_text=re.sub(r'[\\/:*?"<>|]',"，",parts[3]).encode("utf-8")[:200].decode("utf-8",errors="ignore")
                        shutil.copy2(wav,root/"emotion"/f"[默认][{request.language}]{safe_text}.wav")
                    break
        record["preview"] = {"gpt_path":str(models["gpt"]),"sovits_path":str(models["sovits"]),"emotions":emotions}

    def snapshot(self, record: dict, telemetry: bool = True) -> dict:
        result = json.loads(json.dumps(record,default=str))
        self.refresh_emotions(result)
        for target in result["targets"]:
            if not target.get("launch"):
                continue
            if record["status"] not in TERMINAL and target.get("job_id"):
                service = self.gateway.service_manager.get_service(target["target"]+"_training")
                if service:
                    status = service.get_training_status(target["job_id"])
                    target["details"] = self.gateway._to_payload(status) if status else target["details"]
            if telemetry:
                log_file = (target.get("details") or {}).get("log_file","")
                key=(result["project_root"],target["target"],log_file,target["status"])
                cached=self.telemetry_cache.get(key)
                if not cached or time.monotonic()-cached[0]>2:
                    # Resume curves include the preceding steps from the same checkpoint history.
                    started=datetime.fromisoformat(result["created_at"]).timestamp() if result["options"].get("training_mode") != "resume" else None
                    finished=datetime.fromisoformat(result["finished_at"]).timestamp() if result.get("finished_at") else None
                    values=read_telemetry(result["project_root"],target["target"],log_file,target["status"]=="completed",result["options"][target["target"]+"_total_epoch"],started,finished)
                    self.telemetry_cache[key]=(time.monotonic(),values)
                target["telemetry"]=self.telemetry_cache[key][1]
        return result

    @staticmethod
    def refresh_emotions(record: dict) -> None:
        if not record.get("preview") or not record.get("model_root"):
            return
        emotions=[]
        directory=Path(record["model_root"])/"emotion"
        for file in sorted(directory.glob("*")):
            match=re.match(r"\[([^]]+)\]\[([^]]+)\](.*)",file.stem)
            if file.is_file() and file.suffix.lower() in {".wav",".mp3",".flac",".m4a",".ogg"} and match:
                emotions.append({"name":match[1],"language":match[2],"text":match[3],"path":str(file)})
        if directory.is_dir():
            record["preview"]["emotions"]=emotions

    async def history(self) -> dict:
        return {"runs":[self.snapshot(record,False) for record in sorted(self.runs.values(),key=lambda r:r["created_at"],reverse=True)[:100]]}

    async def shutdown(self) -> None:
        tasks=list(self.tasks.values())
        for record in self.runs.values():
            if record["status"] not in TERMINAL:
                await self.stop(record["workflow_id"])
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks,return_exceptions=True),timeout=30)
            except asyncio.TimeoutError:
                pass

    async def status(self, workflow_id: str) -> dict:
        return await asyncio.to_thread(self.snapshot,self.require(workflow_id))

    async def logs(self, workflow_id: str, target: str = "sovits") -> dict:
        record=self.require(workflow_id)
        item=next((t for t in record["targets"] if t["target"]==target),None)
        path=((item or {}).get("details") or {}).get("log_file","")
        if not path and item and item.get("launch"):
            path=((item["launch"].get("status") or {}).get("log_file","") or "")
        return {"target":target,"log_file":path,"text":log_tail(path)}

    async def stop(self, workflow_id: str) -> dict:
        record=self.require(workflow_id)
        if record["status"] not in TERMINAL:
            record["stop_requested"]=True
            task=self.tasks.get(workflow_id)
            if task and record["phase"] != "preprocessing":
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                if record["status"] not in TERMINAL:
                    record.update(status="stopped",phase="stopped",current_target=None,finished_at=datetime.now().isoformat())
            self.save(record)
        return self.snapshot(record,False)

    async def preview(self, workflow_id: str, payload: PreviewRequest) -> dict:
        record=self.require(workflow_id)
        if record["status"] != "completed" or not record.get("preview"):
            raise HTTPException(400,"请先完成训练")
        self.refresh_emotions(record)
        emotion=next((e for e in record["preview"]["emotions"] if e["name"]==payload.emotion),None)
        if not emotion:
            raise HTTPException(400,"请选择可用的参考情绪")
        if self.gateway.training_workflow_lock.locked():
            raise HTTPException(409,"GPU 正在训练，请在训练结束后生成试听")
        from Code.FastApi.Base.Inference.service import InferenceConfig,InferenceRequest
        async with self.gateway.training_workflow_lock:
            service=self.gateway._ensure_service("inference")
            if not service.load_models(record["preview"]["gpt_path"],record["preview"]["sovits_path"]):
                raise HTTPException(400,"训练权重加载失败")
            result=await service.inference(InferenceRequest(text=payload.text.strip(),text_language=record["options"]["language"],
                ref_audio_path=emotion["path"],prompt_text=emotion["text"],prompt_language=emotion["language"],
                config=InferenceConfig(seed=payload.seed,use_cuda_graph=False),return_base64=True))
            if not result.success:
                raise HTTPException(400,result.error_details or result.message)
            output=Path(record["project_root"])/"preview"
            output.mkdir(exist_ok=True)
            audio=output/(datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".wav")
            audio.write_bytes(base64.b64decode(result.audio_data))
            record.setdefault("previews",[]).append({"text":payload.text,"emotion":payload.emotion,"seed":payload.seed,"path":str(audio)})
            self.save(record)
            result.audio_path=str(audio)
            return self.gateway._to_payload(result)
