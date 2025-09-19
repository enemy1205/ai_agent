import os
import io
import json
import base64
import time
import uuid
from typing import Dict, Iterator, Tuple, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

from wespeaker.cli.hub import Hub
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


def _load_or_download(model_name_or_path: str) -> str:
    if model_name_or_path in Hub.Assets:
        model_dir = Hub.get_model(model_name_or_path)
    else:
        model_dir = model_name_or_path
    return model_dir


def _load_model_pt(model_name_or_path: str) -> torch.nn.Module:
    model_dir = _load_or_download(model_name_or_path)
    required_files = ["config.yaml", "avg_model.pt"]
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file} not found in {model_dir}")
    with open(os.path.join(model_dir, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = get_speaker_model(config["model"])(**config["model_args"])
    load_checkpoint(model, os.path.join(model_dir, "avg_model.pt"))
    model.eval()
    return model


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


class LocalSpeaker:
    """
    极简语音注册/识别接口：
    - 无 VAD 处理（假设上游已完成 VAD 切分）
    - 输入为 base64 音频数据（支持常见容器/编码，交由 torchaudio 解码）
    - 注册向量持久化到本地目录；识别时遍历该目录进行比对
    仅保留核心 API：register、recognize，以及必要的子功能。
    """

    def __init__(self, model_name_or_dir: str, db_dir: str):
        self.model: torch.nn.Module = _load_model_pt(model_name_or_dir)
        self.device: torch.device = torch.device("cuda:0")
        self.model = self.model.to(self.device)

        self.resample_rate: int = 16000
        self.wavform_norm: bool = False
        self.window_type: str = "hamming"

        self.db_dir: str = os.path.abspath(db_dir)
        _ensure_dir(self.db_dir)

    # ----------------------- 可选参数设置 -----------------------
    def set_device(self, device: str) -> None:
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

    def set_resample_rate(self, resample_rate: int) -> None:
        self.resample_rate = resample_rate

    def set_wavform_norm(self, wavform_norm: bool) -> None:
        self.wavform_norm = wavform_norm

    def set_window_type(self, window_type: str) -> None:
        self.window_type = window_type

    # ----------------------- 特征与嵌入 -----------------------
    def _compute_fbank(
        self,
        wavform: torch.Tensor,
        sample_rate: int,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        cmn: bool = True,
    ) -> torch.Tensor:
        feat = kaldi.fbank(
            wavform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            sample_frequency=sample_rate,
            window_type=self.window_type,
        )
        if cmn:
            feat = feat - torch.mean(feat, 0)
        return feat

    def _extract_embedding_from_pcm(self, pcm: torch.Tensor, sample_rate: int) -> Optional[torch.Tensor]:
        if pcm is None or pcm.numel() == 0:
            return None
        pcm = pcm.to(torch.float)
        if pcm.size(0) > 1:
            pcm = pcm.mean(dim=0, keepdim=True)
        if sample_rate != self.resample_rate:
            pcm = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate
            )(pcm)
        feats = self._compute_fbank(
            pcm, sample_rate=self.resample_rate, cmn=True
        ).unsqueeze(0)
        feats = feats.to(self.device)
        with torch.no_grad():
            outputs = self.model(feats)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
        embedding = outputs[0].to(torch.device("cpu"))
        return embedding

    def extract_embedding_from_base64(self, audio_base64: str) -> Optional[torch.Tensor]:
        if not audio_base64:
            return None
        # 兼容 data URI 前缀
        if "," in audio_base64 and audio_base64.strip().lower().startswith("data:"):
            audio_base64 = audio_base64.split(",", 1)[1]
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=False)
        except Exception:
            # 某些客户端会对 base64 进行 URL 安全替换
            audio_base64 = audio_base64.replace("-", "+").replace("_", "/")
            audio_bytes = base64.b64decode(audio_base64, validate=False)

        import soundfile as sf
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        pcm = torch.from_numpy(data.T)  # [C, T]
        return self._extract_embedding_from_pcm(pcm, sr)


    # ----------------------- 相似度 -----------------------
    @staticmethod
    def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> float:
        score = torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2))
        return (score.item() + 1.0) / 2.0  # [-1,1] -> [0,1]

    # ----------------------- 本地数据库 -----------------------
    def _iter_db_embeddings(self) -> Iterator[Tuple[str, np.ndarray]]:
        for root, _, files in os.walk(self.db_dir):
            for filename in files:
                if filename.lower().endswith(".npy"):
                    path = os.path.join(root, filename)
                    try:
                        vec = np.load(path)
                        # 读取同名 .json 获取 name
                        name = os.path.splitext(filename)[0]
                        meta_path = path[:-4] + ".json"
                        if os.path.exists(meta_path):
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                                name = meta.get("name", name)
                        yield name, vec
                    except Exception:
                        continue

    def _save_embedding(self, name: str, embedding: torch.Tensor) -> str:
        safe = _safe_filename(name)
        speaker_dir = os.path.join(self.db_dir, safe)
        _ensure_dir(speaker_dir)
        uid = uuid.uuid4().hex[:8]
        stem = f"{safe}_{uid}"
        npy_path = os.path.join(speaker_dir, stem + ".npy")
        meta_path = os.path.join(speaker_dir, stem + ".json")
        np.save(npy_path, embedding.detach().cpu().numpy().astype(np.float32))
        meta = {
            "name": name,
            "saved_at": int(time.time()),
            "model_device": str(self.device),
            "resample_rate": self.resample_rate,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        return npy_path

    # ----------------------- 核心 API -----------------------
    def register(self, name: str, audio_base64: str) -> Dict[str, str]:
        if not name:
            raise ValueError("name 不能为空")
        embedding = self.extract_embedding_from_base64(audio_base64)
        if embedding is None:
            raise RuntimeError("无法从音频中提取说话人向量")
        saved_path = self._save_embedding(name, embedding)
        return {"name": name, "path": saved_path}

    def recognize(self, audio_base64: str) -> Dict[str, Optional[float]]:
        query = self.extract_embedding_from_base64(audio_base64)
        if query is None:
            return {"name": None, "confidence": 0.0}

        best_score: float = 0.0
        best_name: Optional[str] = None

        for name, vec in self._iter_db_embeddings():
            db_tensor = torch.from_numpy(vec)
            score = self.cosine_similarity(query, db_tensor)
            if score > best_score:
                best_score = score
                best_name = name

        return {"name": best_name, "confidence": float(best_score)}


