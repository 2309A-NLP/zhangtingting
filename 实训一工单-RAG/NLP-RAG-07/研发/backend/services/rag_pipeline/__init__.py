# ������ţ��˹����� NLP-RAG-ͼ�����ݽ����������Ż�
from __future__ import annotations

"""RAG Pipeline service."""

from backend.services.rag_pipeline._client import RAGPipeline
from backend.services.rag_pipeline._scoring import (
    answerability_bonus,
    answer_context_bonus,
    sort_company_routed_matches,
    rerank_for_answerability,
    apply_company_routing,
    select_answer_contexts,
)
from backend.services.rag_pipeline._chunks import (
    build_main_chunks,
    build_pdf_intelligence_chunks,
    build_enhanced_chunks,
    build_structured_chunk,
    build_candidate_from_page,
)
from backend.services.rag_pipeline._vlm import (
    build_vlm_chunks,
    load_pdf_vlm_items,
)
from backend.services.rag_pipeline.rag_utils import (
    looks_like_mojibake,
    sanitize_vlm_context,
    normalize_company_name,
    get_company_aliases,
    write_redacted_export,
    write_pdf_vlm_failure,
    is_valid_enhanced_item,
    is_low_value_context,
    prune_low_value_contexts,
    resolve_target_pdfs,
    main_manifest_path,
    enhance_manifest_path,
    parsed_cache_path,
    redacted_cache_path,
    default_pdf_paths,
)

__all__ = [
    "RAGPipeline",
    "answerability_bonus",
    "answer_context_bonus",
    "sort_company_routed_matches",
    "rerank_for_answerability",
    "apply_company_routing",
    "select_answer_contexts",
    "build_main_chunks",
    "build_pdf_intelligence_chunks",
    "build_enhanced_chunks",
    "build_structured_chunk",
    "build_candidate_from_page",
    "build_vlm_chunks",
    "load_pdf_vlm_items",
    "looks_like_mojibake",
    "sanitize_vlm_context",
    "normalize_company_name",
    "get_company_aliases",
    "write_redacted_export",
    "write_pdf_vlm_failure",
    "is_valid_enhanced_item",
    "is_low_value_context",
    "prune_low_value_contexts",
    "resolve_target_pdfs",
    "main_manifest_path",
    "enhance_manifest_path",
    "parsed_cache_path",
    "redacted_cache_path",
    "default_pdf_paths",
]
