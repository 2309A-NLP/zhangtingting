from __future__ import annotations

# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.services.embedding import EmbeddingService


DEFAULT_STAGE3_DIRNAME = "stage3_text_chunking"
DEFAULT_OUTPUT_DIRNAME = "stage4_vectorized_chunks"
DEFAULT_BATCH_SIZE = 64


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[stage4 {timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize stage3 text chunks without inserting into Milvus.")
    parser.add_argument("--artifact-dir", type=str, default="", help="Parent artifact dir that contains stage3_text_chunking")
    parser.add_argument("--chunk-dir", type=str, default="", help="Direct path to stage3_text_chunking")
    parser.add_argument("--chunk-file", type=str, default="", help="Direct path to text_chunks.jsonl or text_chunks.json")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for vector outputs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    return parser.parse_args()


def resolve_chunk_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.chunk_file:
        chunk_file = Path(args.chunk_file).resolve()
        chunk_dir = chunk_file.parent
        return chunk_dir, chunk_file

    if args.chunk_dir:
        chunk_dir = Path(args.chunk_dir).resolve()
    elif args.artifact_dir:
        chunk_dir = Path(args.artifact_dir).resolve() / DEFAULT_STAGE3_DIRNAME
    else:
        chunk_dir = settings.artifact_dir / DEFAULT_STAGE3_DIRNAME

    jsonl_path = chunk_dir / "text_chunks.jsonl"
    json_path = chunk_dir / "text_chunks.json"
    if jsonl_path.exists():
        return chunk_dir, jsonl_path
    if json_path.exists():
        return chunk_dir, json_path
    raise FileNotFoundError(f"No chunk file found under {chunk_dir}")


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
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

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Chunk file must contain a list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def chunk_batch(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    chunk_dir, chunk_file = resolve_chunk_paths(args)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (chunk_dir.parent / DEFAULT_OUTPUT_DIRNAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunk_file)
    if not chunks:
        raise ValueError(f"No chunks found in {chunk_file}")

    embedder = EmbeddingService(settings.model_dir / "embedding", configured_path=settings.embedding_model_path)
    batch_size = max(1, int(args.batch_size))

    log(f"chunk_file={chunk_file}")
    log(f"output_dir={output_dir}")
    log(f"chunk_count={len(chunks)} batch_size={batch_size}")
    log(f"embedding_backend={embedder.backend} dimension={embedder.dimension}")

    vectors: list[list[float]] = []
    vector_index_rows: list[dict[str, Any]] = []
    batches = chunk_batch(chunks, batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        texts = [str(item.get("text") or "") for item in batch]
        embeddings = embedder.embed_texts(texts)
        vectors.extend(embeddings)
        for chunk_item, embedding in zip(batch, embeddings):
            vector_index_rows.append(
                {
                    "chunk_id": str(chunk_item.get("chunk_id") or ""),
                    "chunk_index": int(chunk_item.get("chunk_index") or 0),
                    "source_pages": list(chunk_item.get("source_pages") or []),
                    "source_page_count": int(chunk_item.get("source_page_count") or 0),
                    "char_count": int(chunk_item.get("char_count") or 0),
                    "marker_count": int(chunk_item.get("marker_count") or 0),
                    "markers": list(chunk_item.get("markers") or []),
                    "embedding_offset": len(vector_index_rows),
                    "embedding_dim": len(embedding),
                    "text_preview": str(chunk_item.get("text") or "")[:240],
                }
            )
        log(f"embedded_batch {batch_index}/{len(batches)} size={len(batch)} total_vectors={len(vectors)}")

    matrix = np.asarray(vectors, dtype=np.float32)
    np.save(output_dir / "chunk_embeddings.npy", matrix)
    write_jsonl(output_dir / "chunk_vector_index.jsonl", vector_index_rows)
    write_json(
        output_dir / "vector_manifest.json",
        {
            "source_chunk_dir": str(chunk_dir),
            "source_chunk_file": str(chunk_file),
            "output_dir": str(output_dir),
            "chunk_count": len(chunks),
            "vector_count": int(matrix.shape[0]),
            "embedding_dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "embedding_backend": embedder.backend,
            "embedding_model_path": settings.embedding_model_path,
            "batch_size": batch_size,
            "generated_files": [
                "chunk_embeddings.npy",
                "chunk_vector_index.jsonl",
                "vector_manifest.json",
            ],
        },
    )

    log(f"vector_count={matrix.shape[0]}")
    log(f"embedding_dimension={matrix.shape[1] if matrix.ndim == 2 else 0}")
    log(f"vector_manifest={output_dir / 'vector_manifest.json'}")


if __name__ == "__main__":
    main()
