from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
from functools import lru_cache
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import (
    AudioQueryResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ResetIndexResponse,
    TranscriptionResponse,
    UploadPdfResponse,
)
from app.services.rag_pipeline import RAGPipeline
from app.services.speech_to_text import SpeechToTextService


router = APIRouter(prefix="/api", tags=["rag"])


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@lru_cache(maxsize=1)
def get_stt_service() -> SpeechToTextService:
    return SpeechToTextService()


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
    try:
        pipeline = get_pipeline()
        saved_path = pipeline.save_uploaded_pdf_stream(pdf.filename or "uploaded.pdf", pdf.file)
        chunks = pipeline.ingest_uploaded_pdf(saved_path, pdf.filename or saved_path.name)
        return UploadPdfResponse(
            status="ok",
            filename=pdf.filename or saved_path.name,
            chunks=chunks,
            collection_name=pipeline.uploaded_vector_store.collection_name,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    try:
        result = get_pipeline().ask(
            request.query,
            top_k=request.top_k,
            use_llm=request.use_llm,
            corpus=request.corpus,
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
) -> AudioQueryResponse:
    temp_path = ""
    try:
        temp_path = _save_upload_to_temp(audio)
        transcript = get_stt_service().transcribe(temp_path)
        query_response = get_pipeline().ask(
            str(transcript["text"]),
            top_k=top_k,
            use_llm=use_llm,
            corpus="uploaded" if corpus == "uploaded" else "default",
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
