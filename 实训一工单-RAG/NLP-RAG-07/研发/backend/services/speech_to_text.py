# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from backend.config import settings


class SpeechToTextService:
    def __init__(self) -> None:
        self.model_path = settings.asr_model_path
        self.model_name = settings.asr_model_name
        self.device = settings.asr_device
        self.compute_type = settings.asr_compute_type
        self._model = None
        self.backend = "unavailable"

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        model_ref = self.model_path or self.model_name
        self._model = WhisperModel(
            model_size_or_path=model_ref,
            device=self.device,
            compute_type=self.compute_type,
        )
        self.backend = f"faster_whisper:{model_ref}"

    def transcribe(self, audio_path: str) -> Dict[str, object]:
        try:
            self._load_model()
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "语音识别依赖不可用。请先安装 faster-whisper，并准备 ffmpeg。"
            ) from exc

        segments, info = self._model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            language=None,
            condition_on_previous_text=False,
        )
        parts: List[str] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)
        transcript = " ".join(parts).strip()
        return {
            "text": transcript,
            "language": getattr(info, "language", "unknown"),
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
            "backend": self.backend,
            "source_path": str(Path(audio_path)),
        }
