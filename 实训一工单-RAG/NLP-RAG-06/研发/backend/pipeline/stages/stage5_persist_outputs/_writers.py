# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化

"""Stage5 数据写入器：Milvus、MongoDB、MinIO。"""

from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Any

import numpy as np


class MilvusCollectionWriter:
    def __init__(self, uri: str) -> None:
        from pymilvus import connections

        self.alias = f"stage5_{int(time.time() * 1000)}"
        connections.connect(alias=self.alias, uri=uri)
        self.connections = connections

    def close(self) -> None:
        try:
            self.connections.disconnect(alias=self.alias)
        except Exception:
            pass

    def ensure_text_collection(self, name: str, dim: int, recreate: bool) -> Any:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if recreate and utility.has_collection(name, using=self.alias):
            Collection(name=name, using=self.alias).drop()

        if utility.has_collection(name, using=self.alias):
            collection = Collection(name=name, using=self.alias)
            collection.load()
            return collection

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="parse_version", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="ingest_run_id", dtype=DataType.VARCHAR, max_length=96),
                FieldSchema(name="page_start", dtype=DataType.INT64),
                FieldSchema(name="page_end", dtype=DataType.INT64),
                FieldSchema(name="source_page_count", dtype=DataType.INT64),
                FieldSchema(name="marker_count", dtype=DataType.INT64),
                FieldSchema(name="source_pages_json", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="markers_json", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=32768),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ],
            description="Stage5 text chunk vectors",
        )
        collection = Collection(name=name, schema=schema, using=self.alias)
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
        )
        collection.load()
        return collection

    def ensure_visual_collection(self, name: str, dim: int, recreate: bool) -> Any:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if recreate and utility.has_collection(name, using=self.alias):
            Collection(name=name, using=self.alias).drop()

        if utility.has_collection(name, using=self.alias):
            collection = Collection(name=name, using=self.alias)
            collection.load()
            return collection

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="visual_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="parse_version", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="ingest_run_id", dtype=DataType.VARCHAR, max_length=96),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="visual_type", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="marker_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="source_pages_json", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="source_region_ids_json", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="summary_text", dtype=DataType.VARCHAR, max_length=16384),
                FieldSchema(name="search_text", dtype=DataType.VARCHAR, max_length=32768),
                FieldSchema(name="minio_path", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=32768),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ],
            description="Stage5 visual explanation vectors",
        )
        collection = Collection(name=name, schema=schema, using=self.alias)
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
        )
        collection.load()
        return collection

    def delete_ids(self, collection: Any, field_name: str, ids: list[str]) -> None:
        if not ids:
            return
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            batch = [item.replace("\\", "\\\\").replace('"', '\\"') for item in ids[start : start + batch_size]]
            expr = f'{field_name} in ["' + '","'.join(batch) + '"]'
            collection.delete(expr)

    def insert_text_rows(self, collection: Any, rows: list[dict[str, Any]], vectors: np.ndarray) -> int:
        if not rows:
            return 0
        self.delete_ids(collection, "chunk_id", [str(row["chunk_id"]) for row in rows])
        collection.insert([
            [str(row["chunk_id"]) for row in rows],
            [str(row["doc_name"]) for row in rows],
            [str(row["parse_version"]) for row in rows],
            [str(row["ingest_run_id"]) for row in rows],
            [int(row["page_start"]) for row in rows],
            [int(row["page_end"]) for row in rows],
            [int(row["source_page_count"]) for row in rows],
            [int(row["marker_count"]) for row in rows],
            [str(row["source_pages_json"]) for row in rows],
            [str(row["markers_json"]) for row in rows],
            [str(row["text"]) for row in rows],
            [str(row["metadata_json"]) for row in rows],
            vectors.tolist(),
        ])
        collection.flush()
        collection.load()
        return len(rows)

    def insert_visual_rows(self, collection: Any, rows: list[dict[str, Any]], vectors: np.ndarray) -> int:
        if not rows:
            return 0
        self.delete_ids(collection, "visual_id", [str(row["visual_id"]) for row in rows])
        collection.insert([
            [str(row["visual_id"]) for row in rows],
            [str(row["task_id"]) for row in rows],
            [str(row["doc_name"]) for row in rows],
            [str(row["parse_version"]) for row in rows],
            [str(row["ingest_run_id"]) for row in rows],
            [int(row["page_number"]) for row in rows],
            [str(row["visual_type"]) for row in rows],
            [str(row["marker_id"]) for row in rows],
            [str(row["source_pages_json"]) for row in rows],
            [str(row["source_region_ids_json"]) for row in rows],
            [str(row["summary_text"]) for row in rows],
            [str(row["search_text"]) for row in rows],
            [str(row["minio_path"]) for row in rows],
            [str(row["metadata_json"]) for row in rows],
            vectors.tolist(),
        ])
        collection.flush()
        collection.load()
        return len(rows)


class MongoTableWriter:
    def __init__(self, uri: str, db_name: str, collection_name: str) -> None:
        from pymongo import MongoClient

        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def replace_documents(self, docs: list[dict[str, Any]]) -> int:
        count = 0
        for doc in docs:
            self.collection.replace_one({"table_id": doc["table_id"]}, doc, upsert=True)
            count += 1
        return count


class MinioUploader:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool) -> None:
        import boto3
        from botocore.client import Config

        endpoint = endpoint.strip()
        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            endpoint = ("https://" if secure else "http://") + endpoint
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            region_name="us-east-1",
        )

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self.client.head_bucket(Bucket=bucket)
        except Exception:
            self.client.create_bucket(Bucket=bucket)

    def upload_file(self, bucket: str, key: str, path: Path) -> str:
        extra_args: dict[str, Any] = {}
        guessed_type, _ = mimetypes.guess_type(path.name)
        if guessed_type:
            extra_args["ContentType"] = guessed_type
        if extra_args:
            self.client.upload_file(str(path), bucket, key, ExtraArgs=extra_args)
        else:
            self.client.upload_file(str(path), bucket, key)
        return f"s3://{bucket}/{key}"
