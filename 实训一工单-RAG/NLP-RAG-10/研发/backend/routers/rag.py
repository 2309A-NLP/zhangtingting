from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.conversation.service import ConversationQueryService, UploadedConversationQueryService
from backend.conversation.store import build_conversation_store
from backend.schemas import (
    AudioQueryResponse,
    ConversationStateResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RagasEvaluationRequest,
    RagasEvaluationResponse,
    ResetIndexResponse,
    TranscriptionResponse,
    UploadPdfResponse,
)
from backend.retrieval.unified_query_service import UnifiedDefaultQueryService
from backend.services.rag_pipeline import RAGPipeline
from backend.services.ragas_eval import RagasEvaluator, RagasSetupError
from backend.services.speech_to_text import SpeechToTextService
from backend.utils.logging import get_logger


router = APIRouter(prefix="/api", tags=["rag"])
logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@lru_cache(maxsize=1)
def get_stt_service() -> SpeechToTextService:
    return SpeechToTextService()


@lru_cache(maxsize=1)
def get_unified_query_service() -> UnifiedDefaultQueryService:
    return UnifiedDefaultQueryService(project_root=Path(__file__).resolve().parents[2])


@lru_cache(maxsize=1)
def get_conversation_store():
    return build_conversation_store()


@lru_cache(maxsize=1)
def get_conversation_query_service() -> ConversationQueryService:
    return ConversationQueryService(query_service=get_unified_query_service(), store=get_conversation_store())


@lru_cache(maxsize=1)
def get_uploaded_conversation_query_service() -> UploadedConversationQueryService:
    return UploadedConversationQueryService(pipeline=get_pipeline(), store=get_conversation_store())


@lru_cache(maxsize=1)
def get_ragas_evaluator() -> RagasEvaluator:
    return RagasEvaluator(project_root=Path(__file__).resolve().parents[2])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**get_pipeline().health())


@router.post("/reset-index", response_model=ResetIndexResponse)
def reset_index() -> ResetIndexResponse:
    try:
        pipeline = get_pipeline()
        pipeline.reset_default_collection()
        return ResetIndexResponse(
            status="ok",
            collection_name=pipeline.vector_store.collection_name,
            cleared_artifacts=True,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest", response_model=IngestResponse)
def ingest(force: bool = False) -> IngestResponse:
    try:
        pipeline = get_pipeline()
        chunks = pipeline.ingest_main(force=force)
        return IngestResponse(
            status="ok",
            chunks=chunks,
            collection_name=pipeline.vector_store.collection_name,
            mode="heavy_unified",
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest-enhance", response_model=IngestResponse)
def ingest_enhance(force: bool = False) -> IngestResponse:
    try:
        pipeline = get_pipeline()
        chunks = pipeline.ingest_enhancement(force=force)
        return IngestResponse(
            status="ok",
            chunks=chunks,
            collection_name=pipeline.vector_store.collection_name,
            mode="heavy_unified",
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/upload-pdf", response_model=UploadPdfResponse)
def upload_pdf(pdf: UploadFile = File(...)) -> UploadPdfResponse:
    if not (pdf.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    started = time.perf_counter()
    filename = pdf.filename or "uploaded.pdf"
    logger.info("[upload-pdf] start filename=%s", filename)
    try:
        pipeline = get_pipeline()
        saved_path = pipeline.save_uploaded_pdf_stream(filename, pdf.file)
        logger.info("[upload-pdf] saved filename=%s path=%s", filename, saved_path)
        upload_result = pipeline.ingest_uploaded_pdf(saved_path, filename)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "[upload-pdf] ok filename=%s upload_id=%s chunks=%s collection=%s elapsed_ms=%s",
            filename,
            upload_result["upload_id"],
            upload_result["chunks"],
            upload_result["collection_name"],
            elapsed_ms,
        )
        return UploadPdfResponse(
            status="ok",
            filename=filename,
            upload_id=str(upload_result["upload_id"]),
            chunks=int(upload_result["chunks"]),
            collection_name=str(upload_result["collection_name"]),
            visual_collection_name=str(upload_result.get("visual_collection_name") or ""),
        )
    except Exception as exc:  # pragma: no cover
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.exception("[upload-pdf] failed filename=%s elapsed_ms=%s", filename, elapsed_ms)
        raise HTTPException(status_code=500, detail=f"upload-pdf failed: {exc}") from exc


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    try:
        if request.corpus == "default":
            if request.enable_conversation:
                result = get_conversation_query_service().ask(
                    request.query,
                    session_id=request.session_id,
                    top_k=request.top_k,
                    use_llm=request.use_llm,
                )
            else:
                result = get_unified_query_service().ask(
                    request.query,
                    top_k=request.top_k,
                    use_llm=request.use_llm,
                )
        else:
            if request.enable_conversation:
                result = get_uploaded_conversation_query_service().ask(
                    request.query,
                    session_id=request.session_id,
                    upload_id=request.upload_id or None,
                    top_k=request.top_k,
                    use_llm=request.use_llm,
                )
            else:
                result = get_pipeline().ask(
                    request.query,
                    top_k=request.top_k,
                    use_llm=request.use_llm,
                    corpus=request.corpus,
                    upload_id=request.upload_id or None,
                )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(
            f"[query] ok elapsed_ms={elapsed_ms} corpus={request.corpus} use_llm={request.use_llm} "
            f"question={request.query[:80]}"
        )
        return result
    except Exception as exc:  # pragma: no cover
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(
            f"[query] error elapsed_ms={elapsed_ms} corpus={request.corpus} use_llm={request.use_llm} "
            f"question={request.query[:80]} error={exc}"
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ragas/evaluate", response_model=RagasEvaluationResponse)
def evaluate_ragas(request: RagasEvaluationRequest) -> RagasEvaluationResponse:
    try:
        summary = get_ragas_evaluator().evaluate_csv(
            input_csv=request.input_csv,
            output_csv=request.output_csv,
            dataset_jsonl=request.dataset_jsonl,
            summary_json=request.summary_json,
            top_k=request.top_k,
            corpus=request.corpus,
            upload_id=request.upload_id,
            use_llm=request.use_llm,
            metrics=list(request.metrics),
            timeout_seconds=request.timeout_seconds,
        )
        return RagasEvaluationResponse(**summary)
    except (FileNotFoundError, ValueError, RagasSetupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/conversation/{session_id}", response_model=ConversationStateResponse)
def get_conversation_state(session_id: str) -> ConversationStateResponse:
    state = get_conversation_query_service().get_state(session_id)
    if state is None:
        return ConversationStateResponse(session_id=session_id)
    return ConversationStateResponse(
        session_id=state.session_id,
        current_corpus=state.current_corpus,
        current_upload_id=state.current_upload_id,
        current_company=state.current_company,
        current_profile_id=state.current_profile_id,
        current_topic=state.current_topic,
        current_question_type=state.current_question_type,
        current_subject=state.current_subject,
        last_rewritten_query=state.last_rewritten_query,
        last_answer_summary=state.last_answer_summary,
        history_turns=state.history_turns,
    )


@router.delete("/conversation/{session_id}")
def clear_conversation_state(session_id: str) -> dict[str, str]:
    get_conversation_query_service().clear(session_id)
    return {"status": "ok", "session_id": session_id}


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name


@router.post("/transcribe", response_model=TranscriptionResponse)
def transcribe_audio(audio: UploadFile = File(...)) -> TranscriptionResponse:
    temp_path = ""
    try:
        temp_path = _save_upload_to_temp(audio)
        transcript = get_stt_service().transcribe(temp_path)
        return TranscriptionResponse(
            text=str(transcript["text"]),
            language=str(transcript["language"]),
            duration=float(transcript["duration"]),
            backend=str(transcript["backend"]),
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@router.post("/query-audio", response_model=AudioQueryResponse)
def query_audio(
    audio: UploadFile = File(...),
    top_k: int = Form(5),
    use_llm: bool = Form(True),
    corpus: str = Form("default"),
    upload_id: str = Form(""),
    session_id: str = Form(""),
    enable_conversation: bool = Form(True),
) -> AudioQueryResponse:
    temp_path = ""
    try:
        temp_path = _save_upload_to_temp(audio)
        transcript = get_stt_service().transcribe(temp_path)
        if corpus == "uploaded":
            if enable_conversation:
                query_response = get_uploaded_conversation_query_service().ask(
                    str(transcript["text"]),
                    session_id=session_id or None,
                    upload_id=upload_id or None,
                    top_k=top_k,
                    use_llm=use_llm,
                )
            else:
                query_response = get_pipeline().ask(
                    str(transcript["text"]),
                    top_k=top_k,
                    use_llm=use_llm,
                    corpus="uploaded",
                    upload_id=upload_id or None,
                )
        else:
            if enable_conversation:
                query_response = get_conversation_query_service().ask(
                    str(transcript["text"]),
                    session_id=session_id or None,
                    top_k=top_k,
                    use_llm=use_llm,
                )
            else:
                query_response = get_unified_query_service().ask(
                    str(transcript["text"]),
                    top_k=top_k,
                    use_llm=use_llm,
                )
        return AudioQueryResponse(
            transcript=TranscriptionResponse(
                text=str(transcript["text"]),
                language=str(transcript["language"]),
                duration=float(transcript["duration"]),
                backend=str(transcript["backend"]),
            ),
            result=query_response,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
