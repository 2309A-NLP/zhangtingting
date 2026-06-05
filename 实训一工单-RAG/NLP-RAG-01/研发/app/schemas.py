from __future__ import annotations

# 工单: 人工智能NLP-RAG-基于PDF文档的问答系统

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
    ambiguities: List[str] = Field(default_factory=list)
    sub_questions: List[str] = Field(default_factory=list)


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
    vector_store: str
    uploaded_pdf_active: bool = False
    uploaded_pdf_name: str = ""
