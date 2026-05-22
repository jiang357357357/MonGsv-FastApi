import numpy as np
from pathlib import Path
from datetime import datetime


class SpeakerDatabase:
    def __init__(self, db_path=None):
        if db_path is None:
            try:
                from Code.FastApi.Base.monconfig import MonConfig
                config = MonConfig()
                workspace_root = config.workspace_root() or Path.cwd()
                base_dir = workspace_root
            except Exception:
                base_dir = Path.cwd()
            db_path = base_dir / "Data" / "speaker_db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="speakers",
            metadata={"hnsw:space": "cosine"},
        )
        count = self.collection.count()
        print(f"[SpeakerDB] 加载完成: {self.db_path}, 已注册 {count} 人")

    def register(self, speaker_id, name, embedding):
        if isinstance(embedding, np.ndarray):
            embedding = embedding.flatten().tolist()
        existing = self.collection.get(ids=[speaker_id])
        if existing["ids"]:
            self.collection.update(
                ids=[speaker_id],
                embeddings=[embedding],
                metadatas=[{"name": name, "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}],
            )
            print(f"[SpeakerDB] 更新说话人: {speaker_id} ({name})")
        else:
            self.collection.add(
                ids=[speaker_id],
                embeddings=[embedding],
                metadatas=[{"name": name, "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}],
            )
            print(f"[SpeakerDB] 注册说话人: {speaker_id} ({name})")
        return True

    def unregister(self, speaker_id):
        existing = self.collection.get(ids=[speaker_id])
        if not existing["ids"]:
            print(f"[SpeakerDB] 说话人 {speaker_id} 不存在")
            return False
        self.collection.delete(ids=[speaker_id])
        print(f"[SpeakerDB] 注销说话人: {speaker_id}")
        return True

    def identify(self, embedding, threshold=0.75):
        if self.collection.count() == 0:
            return {"speaker_id": None, "name": "Unknown", "similarity": 0.0, "is_known": False}
        if isinstance(embedding, np.ndarray):
            embedding = embedding.flatten().tolist()
        results = self.collection.query(query_embeddings=[embedding], n_results=1)
        if not results["ids"][0]:
            return {"speaker_id": None, "name": "Unknown", "similarity": 0.0, "is_known": False}
        distance = results["distances"][0][0]
        similarity = max(0.0, 1.0 - distance)
        best_id = results["ids"][0][0]
        best_name = results["metadatas"][0][0].get("name", "Unknown")
        if similarity >= threshold:
            print(f"[SpeakerDB] 识别成功: {best_name} ({similarity:.4f})")
            return {"speaker_id": best_id, "name": best_name, "similarity": round(similarity, 4), "is_known": True}
        print(f"[SpeakerDB] 未识别: 最高 {similarity:.4f} < {threshold}")
        return {"speaker_id": None, "name": "Unknown", "similarity": round(similarity, 4), "is_known": False}

    def list_speakers(self):
        results = self.collection.get()
        speakers = []
        for i, sid in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            speakers.append({
                "speaker_id": sid,
                "name": meta.get("name", ""),
                "registered_at": meta.get("registered_at", ""),
            })
        return speakers

    def get_speaker(self, speaker_id):
        results = self.collection.get(ids=[speaker_id])
        if not results["ids"]:
            return None
        meta = results["metadatas"][0]
        return {"speaker_id": speaker_id, "name": meta.get("name", ""), "registered_at": meta.get("registered_at", "")}
