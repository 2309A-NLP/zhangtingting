from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from backend.conversation.models import ConversationTurn


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    language: Literal["zh", "en", "auto"] = "auto"
    use_llm: bool = True
    corpus: Literal["default", "uploaded"] = "default"
    session_id: Optional[str] = None
    enable_conversation: bool = True


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
    question_type: str = "fact_text"
    target_company: str = ""
    ambiguities: List[str] = Field(default_factory=list)
    sub_questions: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    field_keys: List[str] = Field(default_factory=list)
    preferred_sections: List[str] = Field(default_factory=list)
    query_tags: List[str] = Field(default_factory=list)
    preferred_block_types: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    citations: List[SourceChunk]
    intent: QueryIntent
    latency_ms: int
    retrieval_mode: str
    grounded: bool
    rewritten_query: str = ""
    resolved_company: str = ""
    resolved_profile: str = ""
    used_history: bool = False


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    backend: str


class AudioQueryResponse(BaseModel):
    transcript: TranscriptionResponse
    result: QueryResponse


class ConversationStateResponse(BaseModel):
    session_id: str
    current_company: str = ""
    current_profile_id: str = ""
    current_topic: str = ""
    current_question_type: str = ""
    current_subject: str = ""
    last_rewritten_query: str = ""
    last_answer_summary: str = ""
    history_turns: List[ConversationTurn] = Field(default_factory=list)


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
    default_collection_name: str = ""
    default_collection_count: int = 0
    uploaded_collection_name: str = ""
    uploaded_collection_count: int = 0
    runtime_fallback_active: bool = False
