"""Application configuration for the PDF RAG system."""

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _find_first_matching_dir(base: Path, required_files: Iterable[str]) -> str:
    if not base.exists():
        return ""
    required = list(required_files)
    for candidate in [base, *base.rglob("*")]:
        if candidate.is_dir() and all((candidate / file_name).exists() for file_name in required):
            return str(candidate)
    return ""


_load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    artifact_dir: Path = field(init=False)
    report_dir: Path = field(init=False)
    pdf_path: Path = field(init=False)
    pdf_paths: list[Path] = field(init=False)

    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    generation_top_n: int = field(default_factory=lambda: int(os.getenv("GENERATION_TOP_N", "3")))
    max_context_chars: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "420")))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "260")))
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "prospectus_chunks_04"))

    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://127.0.0.1:19531"))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai_compatible"))
    llm_api_url: str = field(default_factory=lambda: os.getenv("LLM_API_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", ""))
    llm_fallback_api_url: str = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_API_URL", "http://127.0.0.1:8002/v1/chat/completions")
    )
    llm_fallback_api_key: str = field(default_factory=lambda: os.getenv("LLM_FALLBACK_API_KEY", ""))
    llm_fallback_model_name: str = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_MODEL_NAME", "Qwen2.5-0.5B-Instruct")
    )
    llm_request_timeout: int = field(default_factory=lambda: int(os.getenv("LLM_REQUEST_TIMEOUT", "30")))

    pdf_vlm_provider: str = field(default_factory=lambda: os.getenv("PDF_VLM_PROVIDER", "openai_compatible"))
    pdf_vlm_api_url: str = field(default_factory=lambda: os.getenv("PDF_VLM_API_URL", ""))
    pdf_vlm_api_key: str = field(default_factory=lambda: os.getenv("PDF_VLM_API_KEY", ""))
    pdf_vlm_model_name: str = field(default_factory=lambda: os.getenv("PDF_VLM_MODEL_NAME", ""))
    pdf_vlm_local_first: bool = field(default_factory=lambda: os.getenv("PDF_VLM_LOCAL_FIRST", "1") == "1")
    pdf_vlm_fallback_enabled: bool = field(default_factory=lambda: os.getenv("PDF_VLM_FALLBACK_ENABLED", "1") == "1")
    pdf_vlm_request_timeout: int = field(default_factory=lambda: int(os.getenv("PDF_VLM_REQUEST_TIMEOUT", "60")))
    pdf_vlm_max_pages: int = field(default_factory=lambda: int(os.getenv("PDF_VLM_MAX_PAGES", "40")))
    pdf_vlm_min_text_chars: int = field(default_factory=lambda: int(os.getenv("PDF_VLM_MIN_TEXT_CHARS", "80")))
    pdf_vlm_table_trigger_chars: int = field(default_factory=lambda: int(os.getenv("PDF_VLM_TABLE_TRIGGER_CHARS", "120")))
    pdf_vlm_image_trigger_count: int = field(default_factory=lambda: int(os.getenv("PDF_VLM_IMAGE_TRIGGER_COUNT", "6")))
    pdf_vlm_render_scale: float = field(default_factory=lambda: float(os.getenv("PDF_VLM_RENDER_SCALE", "1.35")))
    pdf_vlm_retry_count: int = field(default_factory=lambda: int(os.getenv("PDF_VLM_RETRY_COUNT", "3")))
    pdf_vlm_retry_backoff: float = field(default_factory=lambda: float(os.getenv("PDF_VLM_RETRY_BACKOFF", "2.0")))
    pdf_vlm_strict_mode: bool = field(default_factory=lambda: os.getenv("PDF_VLM_STRICT_MODE", "1") == "1")
    reranker_model_path: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL_PATH", ""))
    reranker_top_n: int = field(default_factory=lambda: int(os.getenv("RERANKER_TOP_N", "5")))
    reranker_candidate_limit: int = field(default_factory=lambda: int(os.getenv("RERANKER_CANDIDATE_LIMIT", "20")))
    reranker_enabled: bool = field(default_factory=lambda: os.getenv("RERANKER_ENABLED", "1") == "1")

    embedding_model_path: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_PATH", ""))
    llm_local_model_path: str = field(default_factory=lambda: os.getenv("LLM_LOCAL_MODEL_PATH", ""))
    asr_model_path: str = field(default_factory=lambda: os.getenv("ASR_MODEL_PATH", ""))

    asr_model_name: str = field(default_factory=lambda: os.getenv("ASR_MODEL_NAME", "small"))
    asr_device: str = field(default_factory=lambda: os.getenv("ASR_DEVICE", "auto"))
    asr_compute_type: str = field(default_factory=lambda: os.getenv("ASR_COMPUTE_TYPE", "int8"))

    ocr_lang: str = field(default_factory=lambda: os.getenv("OCR_LANG", "ch"))
    max_ocr_pages_per_pdf: int = field(default_factory=lambda: int(os.getenv("MAX_OCR_PAGES_PER_PDF", "20")))
    large_pdf_page_threshold: int = field(default_factory=lambda: int(os.getenv("LARGE_PDF_PAGE_THRESHOLD", "200")))

    pdf_parser_backend: str = field(default_factory=lambda: os.getenv("PDF_PARSER_BACKEND", "auto"))
    pdf_parser_python: str = field(default_factory=lambda: os.getenv("PDF_PARSER_PYTHON", ""))
    pdf_parser_conda_env: str = field(default_factory=lambda: os.getenv("PDF_PARSER_CONDA_ENV", ""))
    pdf_parser_timeout: int = field(default_factory=lambda: int(os.getenv("PDF_PARSER_TIMEOUT", "600")))
    docling_enabled: bool = field(default_factory=lambda: os.getenv("DOCLING_ENABLED", "0") == "1")

    hybrid_dense_weight: float = field(default_factory=lambda: float(os.getenv("HYBRID_DENSE_WEIGHT", "0.58")))
    hybrid_lexical_weight: float = field(default_factory=lambda: float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.30")))
    hybrid_overlap_weight: float = field(default_factory=lambda: float(os.getenv("HYBRID_OVERLAP_WEIGHT", "0.12")))
    bm25_k1: float = field(default_factory=lambda: float(os.getenv("BM25_K1", "1.5")))
    bm25_b: float = field(default_factory=lambda: float(os.getenv("BM25_B", "0.75")))
    multi_query_enabled: bool = field(default_factory=lambda: os.getenv("MULTI_QUERY_ENABLED", "1") == "1")
    multi_query_max_queries: int = field(default_factory=lambda: int(os.getenv("MULTI_QUERY_MAX_QUERIES", "4")))
    multi_query_top_k: int = field(default_factory=lambda: int(os.getenv("MULTI_QUERY_TOP_K", "12")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    retrieval_candidate_limit: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "20")))
    llm_enhancement_enabled: bool = field(default_factory=lambda: os.getenv("LLM_ENHANCEMENT_ENABLED", "1") == "1")
    llm_enhancement_max_pages: int = field(default_factory=lambda: int(os.getenv("LLM_ENHANCEMENT_MAX_PAGES", "80")))
    llm_table_analysis_enabled: bool = field(default_factory=lambda: os.getenv("LLM_TABLE_ANALYSIS_ENABLED", "1") == "1")
    enable_redaction: bool = field(default_factory=lambda: os.getenv("ENABLE_REDACTION", "0") == "1")
    answer_include_table_markdown: bool = field(
        default_factory=lambda: os.getenv("ANSWER_INCLUDE_TABLE_MARKDOWN", "0") == "1"
    )

    host: str = field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "30")))

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.model_dir = self.project_root / "model"
        self.artifact_dir = self.project_root / "artifacts"
        self.report_dir = self.project_root / "reports"
        pdf_candidates = sorted(self.data_dir.glob("*.pdf"))
        self.pdf_paths = pdf_candidates
        self.pdf_path = pdf_candidates[0] if pdf_candidates else self.data_dir / "prospectus.pdf"

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
