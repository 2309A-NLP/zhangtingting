from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.services.embedding import EmbeddingService
from app.services.llm_client import LLMClient


DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "stage2_precise_extraction_rewire_test"
DEFAULT_TEXT_TOP_K = 7
DEFAULT_VISUAL_TOP_K = 3
DEFAULT_FUSED_TOP_K = 10
DEFAULT_RERANK_TOP_K = 5
DEFAULT_RRF_K = 60


@dataclass
class EvidenceItem:
    evidence_id: str
    evidence_type: str
    page_number: int
    title: str
    text: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal retrieval test: rewrite + dense + BM25 + RRF + rerank + LLM.")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR), help="Artifact directory")
    parser.add_argument("--text-collection", type=str, default="", help="Milvus text collection override")
    parser.add_argument("--visual-collection", type=str, default="", help="Milvus visual collection override")
    parser.add_argument("--dense-text-k", type=int, default=DEFAULT_TEXT_TOP_K, help="Dense text retrieval top-k")
    parser.add_argument("--dense-visual-k", type=int, default=DEFAULT_VISUAL_TOP_K, help="Dense visual retrieval top-k")
    parser.add_argument("--bm25-text-k", type=int, default=DEFAULT_TEXT_TOP_K, help="BM25 text retrieval top-k")
    parser.add_argument("--bm25-visual-k", type=int, default=DEFAULT_VISUAL_TOP_K, help="BM25 visual retrieval top-k")
    parser.add_argument("--fused-top-k", type=int, default=DEFAULT_FUSED_TOP_K, help="RRF fused top-k")
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K, help="Final reranked top-k")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF k constant")
    parser.add_argument("--disable-llm", action="store_true", help="Do not call LLM")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def preview(text: str, limit: int = 220) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def build_query_tokens(query: str) -> list[str]:
    normalized = normalize_text(query).lower()
    segments = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_.%-]+", normalized)
    tokens: list[str] = []
    for segment in segments:
        tokens.append(segment)
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            for n in (4, 3, 2):
                if len(segment) >= n:
                    tokens.extend(segment[i : i + n] for i in range(len(segment) - n + 1))
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip()
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    segments = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_.%-]+", normalized)
    tokens: list[str] = []
    for segment in segments:
        tokens.append(segment)
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            for n in (2, 3):
                if len(segment) >= n:
                    tokens.extend(segment[i : i + n] for i in range(len(segment) - n + 1))
    return tokens


class SimpleBM25Index:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.term_frequencies = [Counter(tokens) for tokens in corpus_tokens]
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        doc_freq: Counter[str] = Counter()
        for tokens in self.corpus_tokens:
            doc_freq.update(set(tokens))
        total_docs = len(self.corpus_tokens)
        return {
            token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def score(self, query_tokens: list[str]) -> list[float]:
        if not query_tokens or not self.corpus_tokens:
            return [0.0] * len(self.corpus_tokens)
        query_tf = Counter(query_tokens)
        scores = [0.0] * len(self.corpus_tokens)
        for index, doc_tf in enumerate(self.term_frequencies):
            doc_len = self.doc_lengths[index] or 1
            norm = self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-6))
            total = 0.0
            for token, qtf in query_tf.items():
                tf = doc_tf.get(token, 0)
                if tf <= 0:
                    continue
                idf = self.idf.get(token, 0.0)
                total += idf * ((tf * (self.k1 + 1)) / max(tf + norm, 1e-6)) * qtf
            scores[index] = total
        return scores


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def build_text_evidence(chunk: dict[str, Any]) -> EvidenceItem:
    source_pages = list(chunk.get("source_pages") or [])
    page_number = int(source_pages[0]) if source_pages else 0
    chunk_id = str(chunk.get("chunk_id") or "")
    text = str(chunk.get("text") or "")
    return EvidenceItem(
        evidence_id=f"text:{chunk_id}",
        evidence_type="text",
        page_number=page_number,
        title=chunk_id,
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "source_pages": source_pages,
            "markers": list(chunk.get("markers") or []),
            "doc_name": str(chunk.get("doc_name") or ""),
        },
    )


def build_visual_evidence(row: dict[str, Any]) -> EvidenceItem:
    marker_id = str(row.get("marker_id") or row.get("visual_id") or "")
    page_number = int(row.get("page_number") or 0)
    summary_text = str(row.get("summary_text") or "")
    search_text = str(row.get("search_text") or "")
    visual_type = str(row.get("visual_type") or "")
    merged_text = "\n".join(
        part for part in [f"visual_type={visual_type}", summary_text, search_text] if normalize_text(part)
    )
    return EvidenceItem(
        evidence_id=f"visual:{marker_id}",
        evidence_type="visual",
        page_number=page_number,
        title=marker_id,
        text=merged_text,
        metadata={
            "marker_id": marker_id,
            "visual_type": visual_type,
            "summary_text": summary_text,
            "search_text": search_text,
            "minio_path": str(row.get("minio_path") or ""),
            "doc_name": str(row.get("doc_name") or ""),
        },
    )


def load_text_chunks(artifact_dir: Path) -> list[EvidenceItem]:
    path = artifact_dir / "stage3_text_chunking" / "text_chunks.jsonl"
    rows = load_jsonl(path)
    return [build_text_evidence(row) for row in rows]


def load_visual_rows(artifact_dir: Path) -> list[EvidenceItem]:
    persisted = artifact_dir / "stage4_vectorized_visuals" / "visual_vector_index.persisted.jsonl"
    fallback = artifact_dir / "stage4_vectorized_visuals" / "visual_vector_index.jsonl"
    path = persisted if persisted.exists() else fallback
    rows = load_jsonl(path)
    return [build_visual_evidence(row) for row in rows]


def rewrite_query(query: str) -> dict[str, Any]:
    client = LLMClient(
        provider=settings.llm_provider,
        api_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model_name=settings.llm_model_name,
        fallback_api_url=settings.llm_fallback_api_url,
        fallback_api_key=settings.llm_fallback_api_key,
        fallback_model_name=settings.llm_fallback_model_name,
    )
    prompt = (
        "请将下面的问题改写成更利于检索的一个查询句。\n"
        "要求：保留公司名、年份、数字、图表/表格指向、关键词，不要回答问题，不要扩写无关信息。\n"
        '只返回 JSON：{"rewritten_query":"..."}。\n\n'
        f"问题：{query}"
    )
    try:
        content = client._call_primary_then_fallback(
            prompt=prompt,
            system_prompt="你是一个只做检索查询改写的助手。",
            max_tokens=220,
        )
    except Exception as exc:
        return {"original_query": query, "rewritten_query": query, "rewrite_source": f"fallback_error:{exc}"}

    if content:
        match = re.search(r"\{.*\}", content, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
                rewritten = normalize_text(str(payload.get("rewritten_query") or ""))
                if rewritten:
                    return {
                        "original_query": query,
                        "rewritten_query": rewritten,
                        "rewrite_source": "llm",
                    }
            except Exception:
                pass
    return {"original_query": query, "rewritten_query": query, "rewrite_source": "fallback_original"}


def search_dense_text(query: str, collection_name: str, top_k: int) -> list[dict[str, Any]]:
    from pymilvus import Collection, connections

    connections.connect(alias="minimal_text_dense", uri=settings.milvus_uri)
    collection = Collection(name=collection_name, using="minimal_text_dense")
    collection.load()
    embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
    query_vector = embedder.embed_query(query)
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {}},
        limit=max(1, top_k),
        output_fields=[
            "doc_name",
            "page_start",
            "page_end",
            "source_pages_json",
            "markers_json",
            "text",
        ],
    )
    rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(results[0], start=1):
        entity = hit.entity
        rows.append(
            {
                "evidence_id": f"text:{str(hit.id)}",
                "rank": rank,
                "score": float(hit.score),
                "chunk_id": str(hit.id),
                "page_start": int(entity.get("page_start") or 0),
                "page_end": int(entity.get("page_end") or 0),
                "text": str(entity.get("text") or ""),
                "doc_name": str(entity.get("doc_name") or ""),
                "source_pages": json.loads(str(entity.get("source_pages_json") or "[]")),
                "markers": json.loads(str(entity.get("markers_json") or "[]")),
            }
        )
    return rows


def search_dense_visual(query: str, collection_name: str, top_k: int) -> list[dict[str, Any]]:
    from pymilvus import Collection, connections

    connections.connect(alias="minimal_visual_dense", uri=settings.milvus_uri)
    collection = Collection(name=collection_name, using="minimal_visual_dense")
    collection.load()
    embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
    query_vector = embedder.embed_query(query)
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {}},
        limit=max(1, top_k),
        output_fields=[
            "doc_name",
            "page_number",
            "visual_type",
            "marker_id",
            "summary_text",
            "search_text",
            "minio_path",
        ],
    )
    rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(results[0], start=1):
        entity = hit.entity
        rows.append(
            {
                "evidence_id": f"visual:{str(entity.get('marker_id') or hit.id)}",
                "rank": rank,
                "score": float(hit.score),
                "marker_id": str(entity.get("marker_id") or hit.id),
                "page_number": int(entity.get("page_number") or 0),
                "visual_type": str(entity.get("visual_type") or ""),
                "summary_text": str(entity.get("summary_text") or ""),
                "search_text": str(entity.get("search_text") or ""),
                "minio_path": str(entity.get("minio_path") or ""),
                "doc_name": str(entity.get("doc_name") or ""),
            }
        )
    return rows


def search_bm25(query: str, corpus: list[EvidenceItem], top_k: int) -> list[dict[str, Any]]:
    tokens = [tokenize_for_bm25(item.text) for item in corpus]
    index = SimpleBM25Index(tokens)
    query_tokens = tokenize_for_bm25(query)
    scores = index.score(query_tokens)
    rows: list[dict[str, Any]] = []
    for item, score in zip(corpus, scores):
        rows.append(
            {
                "evidence_id": item.evidence_id,
                "score": float(score),
                "page_number": item.page_number,
                "title": item.title,
                "text": item.text,
                "metadata": item.metadata,
                "evidence_type": item.evidence_type,
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows[: max(1, top_k)]


def rrf_fuse(lists: list[list[dict[str, Any]]], evidence_lookup: dict[str, EvidenceItem], rrf_k: int, top_k: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for result_list in lists:
        for rank, row in enumerate(result_list, start=1):
            evidence_id = str(row.get("evidence_id") or "")
            if not evidence_id or evidence_id not in evidence_lookup:
                continue
            rrf_score = 1.0 / (rrf_k + rank)
            item = merged.setdefault(
                evidence_id,
                {
                    "evidence_id": evidence_id,
                    "rrf_score": 0.0,
                    "retrieval_hits": [],
                    "evidence_type": evidence_lookup[evidence_id].evidence_type,
                    "page_number": evidence_lookup[evidence_id].page_number,
                    "title": evidence_lookup[evidence_id].title,
                    "text": evidence_lookup[evidence_id].text,
                    "metadata": evidence_lookup[evidence_id].metadata,
                },
            )
            item["rrf_score"] += rrf_score
            item["retrieval_hits"].append(
                {
                    "rank": rank,
                    "score": float(row.get("score") or 0.0),
                }
            )
    fused = list(merged.values())
    fused.sort(key=lambda item: float(item.get("rrf_score") or 0.0), reverse=True)
    return fused[: max(1, top_k)]


def rerank_evidences(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
    query_vector = embedder.embed_query(query)
    reranked: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text") or "")
        evidence_vector = embedder.embed_query(text)
        semantic = cosine_similarity(query_vector, evidence_vector)
        lexical = len(set(build_query_tokens(query)) & set(build_query_tokens(text))) / max(1, len(set(build_query_tokens(query))))
        final_score = semantic * 0.7 + lexical * 0.3 + float(row.get("rrf_score") or 0.0) * 0.2
        enriched = dict(row)
        enriched["rerank_score"] = final_score
        reranked.append(enriched)
    reranked.sort(key=lambda item: float(item.get("rerank_score") or 0.0), reverse=True)
    return reranked


def build_llm_contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for row in rows:
        contexts.append(
            {
                "page_number": int(row.get("page_number") or 0),
                "logical_page": "",
                "text": str(row.get("text") or ""),
                "metadata": {
                    "page_type": str(row.get("evidence_type") or ""),
                    "chunk_id": str(row.get("metadata", {}).get("chunk_id") or ""),
                    "visual_id": str(row.get("metadata", {}).get("marker_id") or ""),
                    "source_pages": list(row.get("metadata", {}).get("source_pages") or []),
                },
            }
        )
    return contexts


def build_llm_prompt(query: str, rewritten_query: str, rows: list[dict[str, Any]]) -> str:
    evidence_blocks: list[str] = []
    for index, row in enumerate(rows, start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[证据{index}] type={row.get('evidence_type')} page={int(row.get('page_number') or 0)} title={row.get('title') or ''}",
                    str(row.get("text") or ""),
                ]
            )
        )
    return (
        "你是一个严格基于招股书检索证据回答问题的助手。\n"
        "只能依据给定证据回答，不允许使用外部知识，不允许猜测。\n"
        "如果证据不足，就明确回答：未检索到足够证据。\n"
        "不要输出内部实现说明，不要输出 chunk_id、marker_id、minio_path 等内部字段。\n"
        "请优先结合所有证据之间的相互印证关系作答，尤其注意图表证据和正文证据是否在表达同一件事。\n"
        "答案末尾必须附：引用页码：页码1、页码2。\n\n"
        f"原始问题：{query}\n"
        f"检索改写：{rewritten_query}\n\n"
        f"证据：\n{chr(10).join(evidence_blocks)}\n\n"
        "请直接给出最终答案。"
    )


def call_llm(prompt: str) -> tuple[str | None, dict[str, Any]]:
    client = LLMClient(
        provider=settings.llm_provider,
        api_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model_name=settings.llm_model_name,
        fallback_api_url=settings.llm_fallback_api_url,
        fallback_api_key=settings.llm_fallback_api_key,
        fallback_model_name=settings.llm_fallback_model_name,
    )
    content = client._call_primary_then_fallback(
        prompt=prompt,
        system_prompt="你是一个严格基于证据回答问题的助手。",
        max_tokens=settings.max_new_tokens,
    )
    return (client._clean_answer_text(content) if content else None), dict(getattr(client, "last_call_details", {}) or {})


def print_result_block(title: str, rows: list[dict[str, Any]], score_key: str) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("No results.")
        return
    for index, row in enumerate(rows, start=1):
        print(
            f"[{index}] type={row.get('evidence_type') or ''} page={int(row.get('page_number') or 0)} "
            f"title={row.get('title') or ''} {score_key}={float(row.get(score_key) or 0.0):.4f}"
        )
        print(f"    preview={preview(str(row.get('text') or ''))}")


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    text_collection = str(args.text_collection or settings.text_vector_collection_name)
    visual_collection = str(args.visual_collection or settings.visual_vector_collection_name)

    text_corpus = load_text_chunks(artifact_dir)
    visual_corpus = load_visual_rows(artifact_dir)
    evidence_lookup = {item.evidence_id: item for item in [*text_corpus, *visual_corpus]}

    rewrite = rewrite_query(args.query)
    retrieval_query = str(rewrite.get("rewritten_query") or args.query)

    dense_text_raw = search_dense_text(retrieval_query, text_collection, args.dense_text_k)
    dense_visual_raw = search_dense_visual(retrieval_query, visual_collection, args.dense_visual_k)
    bm25_text_raw = search_bm25(retrieval_query, text_corpus, args.bm25_text_k)
    bm25_visual_raw = search_bm25(retrieval_query, visual_corpus, args.bm25_visual_k)

    dense_rows = []
    for row in dense_text_raw:
        item = evidence_lookup.get(str(row["evidence_id"]))
        if not item:
            continue
        dense_rows.append(
            {
                "evidence_id": row["evidence_id"],
                "evidence_type": "text",
                "page_number": item.page_number,
                "title": item.title,
                "text": item.text,
                "metadata": item.metadata,
                "score": float(row["score"]),
            }
        )
    for row in dense_visual_raw:
        item = evidence_lookup.get(str(row["evidence_id"]))
        if not item:
            continue
        dense_rows.append(
            {
                "evidence_id": row["evidence_id"],
                "evidence_type": "visual",
                "page_number": item.page_number,
                "title": item.title,
                "text": item.text,
                "metadata": item.metadata,
                "score": float(row["score"]),
            }
        )

    fused_rows = rrf_fuse(
        [dense_rows, bm25_text_raw + bm25_visual_raw],
        evidence_lookup,
        args.rrf_k,
        args.fused_top_k,
    )
    reranked_rows = rerank_evidences(retrieval_query, fused_rows)[: max(1, args.rerank_top_k)]

    print(f"query={args.query}")
    print(f"rewritten_query={retrieval_query}")
    print(f"rewrite_source={rewrite.get('rewrite_source') or ''}")
    print(f"artifact_dir={artifact_dir}")
    print(f"text_collection={text_collection}")
    print(f"visual_collection={visual_collection}")

    print_result_block("Dense Retrieval", dense_rows, "score")
    print_result_block("BM25 Retrieval", bm25_text_raw + bm25_visual_raw, "score")
    print_result_block("RRF Fused Top10", fused_rows, "rrf_score")
    print_result_block("Reranked Top5", reranked_rows, "rerank_score")

    if args.disable_llm:
        return

    prompt = build_llm_prompt(args.query, retrieval_query, reranked_rows)
    answer, call_details = call_llm(prompt)
    print("\n=== LLM Answer ===")
    if answer:
        print(answer)
    else:
        print("Unavailable: LLM returned no response.")

    print("\n=== LLM Call Details ===")
    print(json.dumps(call_details, ensure_ascii=False, indent=2))

    print("\n=== Final Evidence To LLM ===")
    for index, row in enumerate(reranked_rows, start=1):
        print(
            f"[{index}] type={row.get('evidence_type') or ''} page={int(row.get('page_number') or 0)} "
            f"title={row.get('title') or ''}"
        )
        print(str(row.get("text") or ""))


if __name__ == "__main__":
    main()
