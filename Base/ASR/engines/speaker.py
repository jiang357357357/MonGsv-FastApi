import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class SpeakerManager:
    def __init__(self):
        self.model_id = "iic/speech_eres2net_sv_zh-cn_16k-common"
        print(f"[Speaker] 加载声纹模型: {self.model_id}")
        self.pipeline = pipeline(Tasks.speaker_verification, model=self.model_id)
        print("[Speaker] 声纹模型加载完成")

    def get_embedding(self, audio_path):
        res = self.pipeline(audio_path)
        return res["spk_embedding"]

    def compare(self, emb1, emb2):
        v1 = emb1.flatten()
        v2 = emb2.flatten()
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return float(dot_product / (norm_v1 * norm_v2))
