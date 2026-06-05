from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    language: Literal["zh", "en", "auto"] = "auto"
    use_llm: bool = True
    corpus: Literal["default", "uploaded"] = "default"


class SourceChunk(BaseModel):
    chunk_id: str
    page_number: int
    logical_page: Optional[str] = None
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryIntent(BaseModel):
    language: Literal["zh", "en"]
    intent: str
    rewritten_query: str
    rewrite_strategy: Literal["simple", "expanded", "decomposed"] = "simple"
    target_company: str = ""
    ambiguities: List[str] = Field(default_factory=list)
    sub_questions: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    field_keys: List[str] = Field(default_factory=list)
    preferred_sections: List[str] = Field(default_factory=list)
    query_tags: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    citations: List[SourceChunk]
    intent: QueryIntent
    latency_ms: int
    retrieval_mode: str
    grounded: bool


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    backend: str


class AudioQueryResponse(BaseModel):
    transcript: TranscriptionResponse
    result: QueryResponse


class IngestResponse(BaseModel):
    status: str
    chunks: int
    collection_name: str
    mode: str = ""


class ResetIndexResponse(BaseModel):
    status: str
    collection_name: str
    cleared_artifacts: bool = True


class UploadPdfResponse(BaseModel):
    status: str
    filename: str
    chunks: int
    collection_name: str


class HealthResponse(BaseModel):
    status: str
    pdf_exists: bool
    milvus_uri: str
    embedding_backend: str
    llm_provider: str
    llm_api_url: str = ""
    llm_model_name: str = ""
    llm_fallback_api_url: str = ""
    llm_fallback_model_name: str = ""
    vector_store: str
    uploaded_pdf_active: bool = False
    uploaded_pdf_name: str = ""
    pdf_parser_backend: str = ""
    pdf_vlm_enabled: bool = False
    pdf_vlm_model_name: str = ""
    multi_query_enabled: bool = False
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    reranker_enabled: bool = False
    reranker_backend: str = ""
    reranker_model_path: str = ""
