from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
except ImportError as exc:  # pragma: no cover - developer dependency guard
    raise SystemExit(
        "RAGAS is not installed. Install dev dependencies first: pip install -r requirements.dev.txt"
    ) from exc

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from app.api.dependencies import get_role_service
from app.chat.context_builder import ContextBuilder
from app.chat.llm_client import LLMClient
from app.chat.role_guard import RoleGuard
from app.core.config import get_settings
from app.db.milvus_client import close_milvus, init_milvus
from app.db.mysql_client import close_mysql, get_mysql_session_factory, init_mysql
from app.db.redis_client import close_redis, init_redis


class SentenceTransformerEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_name_or_path: str, device: str) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name_or_path, device=device)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector.tolist()] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Eval file must be a JSON list.")
    return raw


def build_group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    metric_names = {"faithfulness", "answer_relevancy", "context_recall", "context_precision"}

    for row in rows:
        group_value = str(row.get(key, "unknown"))
        grouped.setdefault(group_value, []).append(row)

    summary: dict[str, dict[str, float]] = {}
    for group_value, group_rows in grouped.items():
        metrics: dict[str, float] = {}
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in group_rows
                if isinstance(row.get(metric), (int, float))
                and float(row[metric]) == float(row[metric])
            ]
            if values:
                metrics[metric] = sum(values) / len(values)
        metrics["count"] = float(len(group_rows))
        summary[group_value] = metrics
    return summary


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


async def build_row(
    *,
    item: dict[str, Any],
    user_id: str,
    role_id: str,
    role_name: str | None,
    session_id: str,
    top_k: int,
    temperature: float,
) -> dict[str, Any]:
    settings = get_settings()
    session_factory = get_mysql_session_factory()
    role_service = get_role_service()
    context_builder = ContextBuilder()
    llm_client = LLMClient()
    role_guard = RoleGuard()

    async with session_factory() as db_session:
        role = await role_service.resolve_role(
            db_session,
            user_id=user_id,
            role_id=role_id,
            role_name=role_name,
        )
        system_prompt = role_guard.build_system_prompt(
            role_name=role.name,
            role_category=role.category,
            system_prompt=role.system_prompt,
        )
        built_context = await context_builder.build(
            db_session=db_session,
            user_id=user_id,
            role_id=role.role_id,
            role_name=role.name,
            role_category=role.category,
            system_prompt=system_prompt,
            query=str(item["question"]),
            session_id=session_id,
            top_k=top_k,
        )
        result = await llm_client.complete(
            messages=built_context.messages,
            temperature=temperature,
        )

    return {
        "role_id": role.role_id,
        "role_name": role.name,
        "question": str(item["question"]),
        "ground_truth": str(item["expected_answer"]),
        "rewritten_query": built_context.rewritten_query,
        "contexts": [source.text for source in built_context.context_sources],
        "answer": result.content,
        "retrieval_debug": built_context.retrieval_debug,
        "retrieved_context_sources": [
            {
                "doc_id": source.doc_id,
                "chunk_id": source.chunk_id,
                "source": source.source,
                "score": source.score,
            }
            for source in built_context.context_sources
        ],
        "expected_context": item.get("expected_context", []),
    }


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    eval_items = load_eval_items(Path(args.eval_set))

    await init_mysql()
    await init_redis()
    await init_milvus()

    try:
        rows = []
        for index, item in enumerate(eval_items, start=1):
            row = await build_row(
                item=item,
                user_id=args.user_id,
                role_id=item.get("role_id", args.role_id),
                role_name=item.get("role_name"),
                session_id=f"ragas-eval-{uuid4().hex}",
                top_k=args.top_k,
                temperature=args.temperature,
            )
            row["row_index"] = index
            rows.append(row)

        dataset = Dataset.from_list(rows)
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=args.judge_model,
                base_url=args.judge_base_url,
                api_key=args.judge_api_key,
                default_headers={"Authorization": f"Bearer {args.judge_api_key}"},
            )
        )
        embeddings = SentenceTransformerEmbeddings(
            model_name_or_path=args.embedding_model_path or settings.embedding_model_name,
            device=args.embedding_device or settings.embedding_device,
        )

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=evaluator_llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=True,
        )

        df = result.to_pandas()
        summary = df.mean(numeric_only=True).to_dict()
        ragas_rows = df.to_dict(orient="records")
        row_records: list[dict[str, Any]] = []
        for index, ragas_row in enumerate(ragas_rows):
            merged_row = dict(ragas_row)
            if index < len(rows):
                source_row = rows[index]
                for key, value in source_row.items():
                    merged_row.setdefault(key, value)
            row_records.append(merged_row)

        output = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
            "role_summary": build_group_summary(row_records, "role_id"),
            "rows": row_records,
        }
        output = sanitize_for_json(output)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"saved: {output_path}")

        gates = {
            "faithfulness": args.min_faithfulness,
            "answer_relevancy": args.min_answer_relevancy,
            "context_recall": args.min_context_recall,
            "context_precision": args.min_context_precision,
        }
        failed = {name: value for name, value in summary.items() if name in gates and value < gates[name]}
        if failed:
            raise SystemExit(f"RAGAS gate failed: {failed}")

        return output
    finally:
        await close_milvus()
        await close_redis()
        await close_mysql()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against the local RAG backend.")
    parser.add_argument("--eval-set", default="scripts/eval_set.json", help="Path to eval_set.json")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--user-id", default=None, help="Eval user id")
    parser.add_argument("--role-id", default="lawyer_01", help="Default role id")
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval top_k override")
    parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature")
    parser.add_argument("--judge-base-url", default=None, help="OpenAI-compatible base URL for RAGAS judge")
    parser.add_argument("--judge-api-key", default=None, help="Bearer token for the judge model")
    parser.add_argument("--judge-model", default=None, help="Judge model name")
    parser.add_argument("--embedding-model-path", default=None, help="Embedding model path/name for RAGAS")
    parser.add_argument("--embedding-device", default=None, help="Embedding device (cpu/cuda)")
    parser.add_argument("--min-faithfulness", type=float, default=0.7)
    parser.add_argument("--min-answer-relevancy", type=float, default=0.7)
    parser.add_argument("--min-context-recall", type=float, default=0.7)
    parser.add_argument("--min-context-precision", type=float, default=0.7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    args.user_id = args.user_id or f"eval_{datetime.now().strftime('%Y%m%d')}"
    args.output = args.output or f"data/eval/ragas_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    args.judge_base_url = args.judge_base_url or settings.vllm_base_url
    args.judge_api_key = args.judge_api_key or settings.vllm_api_key
    args.judge_model = args.judge_model or settings.vllm_model
    args.embedding_model_path = args.embedding_model_path or settings.embedding_model_path or settings.embedding_model_name
    args.embedding_device = args.embedding_device or settings.embedding_device
    if args.top_k is None:
        args.top_k = settings.retrieval_top_k

    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
