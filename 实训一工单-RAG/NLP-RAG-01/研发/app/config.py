"""Application configuration for the PDF RAG system."""

# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


def _find_first_matching_dir(base: Path, required_files: Iterable[str]) -> str:
    if not base.exists():
        return ""
    required = list(required_files)
    for candidate in [base, *base.rglob("*")]:
        if candidate.is_dir() and all((candidate / file_name).exists() for file_name in required):
            return str(candidate)
    return ""


@dataclass
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    artifact_dir: Path = field(init=False)
    report_dir: Path = field(init=False)
    pdf_path: Path = field(init=False)
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    generation_top_n: int = 3
    max_context_chars: int = 320
    max_new_tokens: int = 220
    collection_name: str = "prospectus_chunks"
    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://127.0.0.1:19531"))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai_compatible"))
    llm_api_url: str = field(default_factory=lambda: os.getenv("LLM_API_URL", "http://127.0.0.1:8002/v1/chat/completions"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "Qwen2.5-0.5B-Instruct"))
    embedding_model_path: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_PATH", ""))
    llm_local_model_path: str = field(default_factory=lambda: os.getenv("LLM_LOCAL_MODEL_PATH", ""))
    asr_model_path: str = field(default_factory=lambda: os.getenv("ASR_MODEL_PATH", ""))
    asr_model_name: str = field(default_factory=lambda: os.getenv("ASR_MODEL_NAME", "small"))
    asr_device: str = field(default_factory=lambda: os.getenv("ASR_DEVICE", "auto"))
    asr_compute_type: str = field(default_factory=lambda: os.getenv("ASR_COMPUTE_TYPE", "int8"))
    ocr_lang: str = field(default_factory=lambda: os.getenv("OCR_LANG", "ch"))
    max_ocr_pages_per_pdf: int = field(default_factory=lambda: int(os.getenv("MAX_OCR_PAGES_PER_PDF", "20")))
    large_pdf_page_threshold: int = field(default_factory=lambda: int(os.getenv("LARGE_PDF_PAGE_THRESHOLD", "200")))
    host: str = field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "30")))

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.model_dir = self.project_root / "model"
        self.artifact_dir = self.project_root / "artifacts"
        self.report_dir = self.project_root / "reports"
        pdf_candidates = sorted(self.data_dir.glob("*.pdf"))
        self.pdf_path = pdf_candidates[0] if pdf_candidates else self.data_dir / "招股说明书1.pdf"
        if not self.embedding_model_path:
            self.embedding_model_path = _find_first_matching_dir(
                self.model_dir / "embedding",
                ["config.json", "tokenizer.json"],
            )
        if not self.llm_local_model_path:
            self.llm_local_model_path = _find_first_matching_dir(
                self.model_dir / "llm",
                ["config.json", "tokenizer.json"],
            )
        if not self.asr_model_path:
            self.asr_model_path = _find_first_matching_dir(
                self.model_dir / "asr",
                ["config.json"],
            )


settings = Settings()
