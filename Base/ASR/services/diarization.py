import os
import tempfile

import numpy as np
import soundfile as sf


def cosine_similarity(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0


def cluster_embeddings(embeddings, threshold=0.75):
    if not embeddings:
        return []
    n = len(embeddings)
    centroids = []
    labels = [-1] * n
    for i in range(n):
        best_label = -1
        best_sim = 0.0
        for label_idx, centroid in enumerate(centroids):
            sim = cosine_similarity(embeddings[i], centroid)
            if sim > best_sim:
                best_sim = sim
                best_label = label_idx
        if best_label >= 0 and best_sim >= threshold:
            labels[i] = best_label
            count = sum(1 for l in labels[:i] if l == best_label)
            centroids[best_label] = (centroids[best_label] * count + embeddings[i]) / (count + 1)
        else:
            labels[i] = len(centroids)
            centroids.append(embeddings[i].copy())
    return labels


class DiarizationService:
    def __init__(self, voice_service):
        self.vad = voice_service.vad
        self.asr = voice_service.asr
        self.speaker = voice_service.speaker
        self.speaker_db = voice_service.speaker_db

    def process_file(self, audio_path, language="auto", cluster_threshold=0.75):
        segments = self.vad.detect(audio_path)
        if not segments:
            print("[Diarization] 未检测到语音")
            return {"segments": [], "num_speakers": 0, "status": "no_speech"}
        print(f"[Diarization] VAD {len(segments)} 个片段")
        speech, sr = sf.read(audio_path)
        seg_results = []
        for i, (start_ms, end_ms) in enumerate(segments):
            start_sample = int(start_ms * sr / 1000)
            end_sample = int(end_ms * sr / 1000) if end_ms > 0 else len(speech)
            seg_audio = speech[start_sample:end_sample]
            if len(seg_audio) < sr * 0.3:
                continue
            asr_res = self.asr.transcribe_array(seg_audio, sr)
            text = asr_res.get("text", "").strip()
            if not text:
                continue
            embedding = None
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, seg_audio, sr)
                embedding = self.speaker.get_embedding(tmp_path)
            except Exception as e:
                print(f"[Diarization] embedding 提取失败 (段 {i}): {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            seg_results.append({"text": text, "embedding": embedding, "start_ms": start_ms, "end_ms": end_ms})
        if not seg_results:
            return {"segments": [], "num_speakers": 0, "status": "no_speech"}
        embeddings = [r["embedding"] for r in seg_results if r["embedding"] is not None]
        if len(embeddings) >= 2:
            cluster_labels = cluster_embeddings(embeddings, cluster_threshold)
        elif len(embeddings) == 1:
            cluster_labels = [0]
        else:
            cluster_labels = [-1] * len(seg_results)
        speaker_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        output_segments = []
        emb_idx = 0
        seen_clusters = {}
        for r in seg_results:
            if r["embedding"] is not None:
                label_idx = cluster_labels[emb_idx]
                emb_idx += 1
            else:
                label_idx = -1
            label = speaker_labels[label_idx] if label_idx >= 0 else "?"
            seen_clusters[label_idx] = True
            speaker_id = None
            speaker_name = None
            if label_idx >= 0 and r["embedding"] is not None:
                try:
                    spk_info = self.speaker_db.identify(r["embedding"], cluster_threshold)
                    if spk_info.get("is_known"):
                        speaker_id = spk_info["speaker_id"]
                        speaker_name = spk_info["name"]
                except Exception as e:
                    print(f"[Diarization] 声纹匹配失败: {e}")
            output_segments.append({
                "speaker": label,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "start_ms": r["start_ms"],
                "end_ms": r["end_ms"],
                "text": r["text"],
            })
        print(f"[Diarization] 完成: {len(output_segments)} 句, {len(seen_clusters)} 人")
        return {"segments": output_segments, "num_speakers": len(seen_clusters), "status": "success"}
