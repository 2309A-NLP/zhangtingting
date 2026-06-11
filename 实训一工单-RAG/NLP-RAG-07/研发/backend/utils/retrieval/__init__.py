# Retrieval utilities re-exports for backend.utils

from backend.utils.retrieval.constants import (
    GENERIC_STOPWORDS,
    MARKER_RE,
    PHRASE_HINTS,
    COMPANY_RE,
)
from backend.utils.retrieval.scoring import (
    SimpleBM25Index,
    build_query_tokens,
    keyword_score,
    keyword_overlap_score,
    normalize_score_map,
    cosine_similarity,
    extract_company_aliases,
    extract_focus_terms,
    compute_focus_signal,
    compute_company_signal,
    infer_question_type,
    infer_query_tags,
    compute_page_position_penalty,
    compute_answer_boost,
)
from backend.utils.retrieval.text_normalize import (
    normalize_text,
    strip_company_query_prefixes,
    split_sentences,
)

__all__ = [
    'GENERIC_STOPWORDS', 'MARKER_RE', 'PHRASE_HINTS', 'COMPANY_RE',
    'SimpleBM25Index', 'build_query_tokens', 'keyword_score', 'keyword_overlap_score',
    'normalize_score_map', 'cosine_similarity', 'extract_company_aliases',
    'extract_focus_terms', 'compute_focus_signal', 'compute_company_signal',
    'infer_question_type', 'infer_query_tags', 'compute_page_position_penalty',
    'compute_answer_boost', 'normalize_text', 'strip_company_query_prefixes', 'split_sentences',
]
