from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from backend.config import settings
from backend.schemas import QueryResponse

if TYPE_CHECKING:
    from backend.retrieval.unified_query_service import UnifiedDefaultQueryService
    from backend.services.rag_pipeline import RAGPipeline


DEFAULT_RAGAS_METRICS = ["context_precision", "context_recall", "faithfulness"]
SUPPORTED_RAGAS_METRICS = {
    "context_precision",
    "context_recall",
    "faithfulness",
    "context_utilization",
    "answer_relevancy",
    "response_relevancy",
    "factual_correctness",
    "exact_match",
}


class RagasSetupError(RuntimeError):
    """Raised when ragas evaluation dependencies or config are unavailable."""


@dataclass
class RagasCsvPaths:
    input_csv: Path
    output_csv: Path
    dataset_jsonl: Path
    summary_json: Path


def normalize_openai_base_url(raw_url: str) -> str:
    normalized = str(raw_url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/embeddings", "/completions"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode CSV: {path}")


def build_page_string(response: QueryResponse, limit: int = 5) -> str:
    pages: list[str] = []
    for item in response.citations[:limit]:
        page_number = str(item.page_number or "").strip()
        if page_number and page_number not in pages:
            pages.append(page_number)
    return "|".join(pages)


def build_citation_id_string(response: QueryResponse, limit: int = 5) -> str:
    ids: list[str] = []
    for item in response.citations[:limit]:
        metadata = dict(item.metadata or {})
        candidate = str(
            metadata.get("table_id")
            or metadata.get("visual_id")
            or metadata.get("chunk_id")
            or item.chunk_id
            or ""
        ).strip()
        if candidate and candidate not in ids:
            ids.append(candidate)
    return "|".join(ids)


def split_reference_contexts(raw_text: str) -> list[str]:
    normalized = str(raw_text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return []
    parts = [
        item.strip()
        for item in normalized.replace("\uFF1B", "\n").replace(";", "\n").split("\n")
        if item.strip()
    ]
    return parts


def resolve_reference_answer(row: dict[str, str]) -> str:
    for key in ("gold_answer", "answer", "reference", "ground_truth"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def build_dataset_record(row: dict[str, str], response: QueryResponse) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    reference_answer = resolve_reference_answer(row)
    return {
        "id": str(row.get("id") or "").strip(),
        "user_input": question,
        "response": response.answer,
        "reference": reference_answer,
        "retrieved_contexts": [citation.text for citation in response.citations if str(citation.text or "").strip()],
        "reference_contexts": split_reference_contexts(str(row.get("gold_evidence") or row.get("notes") or "")),
    }


def ensure_supported_metrics(metrics: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in metrics or DEFAULT_RAGAS_METRICS:
        metric_name = str(item or "").strip().lower()
        if not metric_name:
            continue
        if metric_name not in SUPPORTED_RAGAS_METRICS:
            supported = ", ".join(sorted(SUPPORTED_RAGAS_METRICS))
            raise ValueError(f"Unsupported ragas metric: {metric_name}. Supported metrics: {supported}")
        if metric_name not in normalized:
            normalized.append(metric_name)
    return normalized or list(DEFAULT_RAGAS_METRICS)


class RagasEvaluator:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self._default_query_service: UnifiedDefaultQueryService | None = None
        self._pipeline: RAGPipeline | None = None

    @property
    def default_query_service(self) -> UnifiedDefaultQueryService:
        if self._default_query_service is None:
            from backend.retrieval.unified_query_service import UnifiedDefaultQueryService

            self._default_query_service = UnifiedDefaultQueryService(project_root=self.project_root)
        return self._default_query_service

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            from backend.services.rag_pipeline import RAGPipeline

            self._pipeline = RAGPipeline()
        return self._pipeline

    def _build_paths(
        self,
        input_csv: str,
        output_csv: str = "",
        dataset_jsonl: str = "",
        summary_json: str = "",
    ) -> RagasCsvPaths:
        input_path = Path(input_csv).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        stem = input_path.stem
        output_path = Path(output_csv).expanduser().resolve() if output_csv else input_path.with_name(f"{stem}.ragas.csv")
        dataset_path = (
            Path(dataset_jsonl).expanduser().resolve()
            if dataset_jsonl
            else input_path.with_name(f"{stem}.ragas.dataset.jsonl")
        )
        summary_path = (
            Path(summary_json).expanduser().resolve()
            if summary_json
            else input_path.with_name(f"{stem}.ragas.summary.json")
        )
        return RagasCsvPaths(
            input_csv=input_path,
            output_csv=output_path,
            dataset_jsonl=dataset_path,
            summary_json=summary_path,
        )

    def _query(
        self,
        question: str,
        *,
        top_k: int,
        corpus: str,
        upload_id: str,
        use_llm: bool,
    ) -> QueryResponse:
        if corpus == "uploaded":
            return self.pipeline.ask(
                question,
                top_k=top_k,
                use_llm=use_llm,
                corpus="uploaded",
                upload_id=upload_id or None,
            )
        return self.default_query_service.ask(question, top_k=top_k, use_llm=use_llm)

    def _load_ragas_runtime(self, timeout_seconds: int) -> tuple[Any, Any, dict[str, Callable[[], Any]]]:
        try:
            from ragas import evaluate
            from ragas.dataset_schema import EvaluationDataset
            from ragas.llms import llm_factory
            # from ragas.embeddings import embedding_factory
            from langchain_openai import OpenAIEmbeddings  # 添加
            from ragas.embeddings import LangchainEmbeddingsWrapper  # 添加
            from ragas.metrics import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                ContextUtilization,
                ExactMatch,
                FactualCorrectness,
                Faithfulness,
                ResponseRelevancy,
            )
        except Exception as exc:  # pragma: no cover - depends on optional install
            raise RagasSetupError(
                "Failed to import ragas runtime dependencies. "
                "This is often caused by an incomplete or incompatible ragas/langchain install. "
                f"Original error: {exc}"
            ) from exc

        base_url = normalize_openai_base_url(settings.ragas_api_url or settings.llm_api_url or settings.llm_fallback_api_url)
        api_key = (
            settings.ragas_api_key
            or settings.llm_api_key
            or settings.llm_fallback_api_key
            or "dummy"
        )
        model_name = settings.ragas_model_name or settings.llm_model_name or settings.llm_fallback_model_name
        if not base_url:
            raise RagasSetupError("RAGAS_API_URL is not configured, and no reusable LLM API URL was found.")
        if not model_name:
            raise RagasSetupError("RAGAS_MODEL_NAME is not configured, and no reusable LLM model name was found.")

        os.environ["OPENAI_API_KEY"] = api_key
        llm = llm_factory(model=model_name, base_url=base_url)
        # embeddings = embedding_factory(model="BAAI/bge-m3", base_url=base_url)
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="BAAI/bge-m3",
                api_key=api_key,
                base_url=base_url,
            )
        )
        print(f"[DEBUG] base_url={base_url}, model={model_name}, api_key={api_key[:15]}...")
        metric_factories: dict[str, Callable[[], Any]] = {
            "context_precision": lambda: ContextPrecision(llm=llm),
            "context_recall": lambda: ContextRecall(llm=llm),
            "faithfulness": lambda: Faithfulness(llm=llm),
            "context_utilization": lambda: ContextUtilization(llm=llm),
            "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
            "response_relevancy": lambda: ResponseRelevancy(llm=llm),
            "factual_correctness": lambda: FactualCorrectness(llm=llm),
            "exact_match": lambda: ExactMatch(),
        }
        return (EvaluationDataset, evaluate, metric_factories)

    def evaluate_csv(
        self,
        *,
        input_csv: str,
        output_csv: str = "",
        dataset_jsonl: str = "",
        summary_json: str = "",
        top_k: int = 5,
        corpus: str = "default",
        upload_id: str = "",
        use_llm: bool = True,
        metrics: list[str] | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        selected_metrics = ensure_supported_metrics(metrics or list(DEFAULT_RAGAS_METRICS))
        paths = self._build_paths(input_csv, output_csv, dataset_jsonl, summary_json)
        rows = read_csv_rows(paths.input_csv)
        if not rows:
            raise ValueError(f"Input CSV is empty: {paths.input_csv}")

        dataset_records: list[dict[str, Any]] = []
        export_rows: list[dict[str, Any]] = []
        evaluated_export_positions: list[int] = []
        skipped_rows = 0
        for row in rows:
            export_row = dict(row)
            question = str(row.get("question") or "").strip()
            reference_answer = resolve_reference_answer(row)
            if not question or not reference_answer:
                skipped_rows += 1
                export_row["rag_answer"] = ""
                export_row["rag_pages"] = ""
                export_row["rag_citation_ids"] = ""
                export_row["rag_grounded"] = ""
                export_row["rag_retrieval_mode"] = ""
                export_row["rag_latency_ms"] = ""
                export_rows.append(export_row)
                continue
            response = self._query(
                question,
                top_k=top_k,
                corpus=corpus,
                upload_id=upload_id,
                use_llm=use_llm,
            )
            record = build_dataset_record(row, response)
            dataset_records.append(record)
            export_row["rag_answer"] = response.answer
            export_row["rag_pages"] = build_page_string(response)
            export_row["rag_citation_ids"] = build_citation_id_string(response)
            export_row["rag_grounded"] = str(bool(response.grounded)).lower()
            export_row["rag_retrieval_mode"] = str(response.retrieval_mode or "")
            export_row["rag_latency_ms"] = str(response.latency_ms or "")
            export_rows.append(export_row)
            evaluated_export_positions.append(len(export_rows) - 1)

        if not dataset_records:
            raise ValueError("No evaluable rows found. Ensure the CSV contains `question` and `gold_answer` columns.")

        paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
        paths.dataset_jsonl.parent.mkdir(parents=True, exist_ok=True)
        paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with paths.dataset_jsonl.open("w", encoding="utf-8") as handle:
            for item in dataset_records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        EvaluationDataset, evaluate, metric_factories = self._load_ragas_runtime(timeout_seconds)
        metric_objects = [metric_factories[name]() for name in selected_metrics]
        dataset = EvaluationDataset.from_list(dataset_records)
        result = evaluate(
            dataset=dataset,
            metrics=metric_objects,
            show_progress=False,

        )
        result_df = result.to_pandas()
        if "user_input" not in result_df.columns:
            result_df["user_input"] = [item["user_input"] for item in dataset_records]

        metrics_summary: dict[str, float | None] = {}
        for metric_name in selected_metrics:
            if metric_name not in result_df.columns:
                metrics_summary[metric_name] = None
                continue
            series = pd.to_numeric(result_df[metric_name], errors="coerce")
            metrics_summary[metric_name] = None if series.dropna().empty else round(float(series.mean()), 6)

        scored_rows = result_df.to_dict(orient="records")
        scored_row_by_export_index = {
            export_index: scored_rows[result_index]
            for result_index, export_index in enumerate(evaluated_export_positions)
            if result_index < len(scored_rows)
        }
        for export_index, row in enumerate(export_rows):
            score_row = scored_row_by_export_index.get(export_index, {})
            for metric_name in selected_metrics:
                score_value = score_row.get(metric_name)
                row[f"ragas_{metric_name}"] = "" if score_value is None or pd.isna(score_value) else f"{float(score_value):.6f}"

        fieldnames = list(export_rows[0].keys())
        with paths.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_rows)

        summary = {
            "status": "ok",
            "input_csv": str(paths.input_csv),
            "output_csv": str(paths.output_csv),
            "dataset_jsonl": str(paths.dataset_jsonl),
            "summary_json": str(paths.summary_json),
            "total_rows": len(rows),
            "evaluated_rows": len(dataset_records),
            "skipped_rows": skipped_rows,
            "corpus": corpus,
            "upload_id": upload_id,
            "use_llm": use_llm,
            "metrics": metrics_summary,
        }
        paths.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
