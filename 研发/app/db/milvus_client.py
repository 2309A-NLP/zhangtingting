import asyncio
from typing import Any

from pymilvus import DataType, MilvusClient

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_milvus_client: MilvusClient | None = None


def _build_schema() -> Any:
    settings = get_settings()
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=64,
    )
    schema.add_field(
        field_name="tenant_key",
        datatype=DataType.VARCHAR,
        max_length=128,
        # 把 tenant_key 字段设置为分区键，用于数据分片和隔离。
        is_partition_key=True,
    )
    schema.add_field(
        field_name="role_category",
        datatype=DataType.VARCHAR,
        max_length=32,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=settings.embedding_dimension,
    )
    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=512,
    )
    schema.add_field(
        field_name="doc_id",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="created_at",
        datatype=DataType.INT64,
    )
    schema.add_field(
        field_name="updated_at",
        datatype=DataType.INT64,
    )
    return schema


def _build_index_params() -> Any:
    settings = get_settings()
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_name="idx_embedding",
        index_type=settings.milvus_index_type,
        metric_type=settings.milvus_metric_type,
        params={"nlist": settings.milvus_nlist},
    )
    return index_params


def _create_client() -> MilvusClient:
    settings = get_settings()
    return MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        db_name=settings.milvus_db_name,
    )


def _ensure_collection(client: MilvusClient) -> None:
    settings = get_settings()
    collection_name = settings.milvus_collection_name

    if client.has_collection(collection_name=collection_name):
        client.load_collection(collection_name=collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        schema=_build_schema(),
        index_params=_build_index_params(),
        consistency_level=settings.milvus_consistency_level,
    )
    client.load_collection(collection_name=collection_name)

    logger.info(
        "milvus_collection_created",
        collection_name=collection_name,
        metric_type=settings.milvus_metric_type,
        index_type=settings.milvus_index_type,
    )


async def init_milvus() -> None:
    global _milvus_client

    if _milvus_client is not None:
        return

    _milvus_client = await asyncio.to_thread(_create_client)
    await asyncio.to_thread(_ensure_collection, _milvus_client)

    settings = get_settings()
    logger.info(
        "milvus_initialized",
        uri=settings.milvus_uri,
        db_name=settings.milvus_db_name,
        collection_name=settings.milvus_collection_name,
    )


async def close_milvus() -> None:
    global _milvus_client

    if _milvus_client is not None:
        close_method = getattr(_milvus_client, "close", None)
        if callable(close_method):
            await asyncio.to_thread(close_method)
        logger.info("milvus_closed")

    _milvus_client = None


def get_milvus_client() -> MilvusClient:
    if _milvus_client is None:
        raise RuntimeError("Milvus client is not initialized. Call init_milvus() first.")
    return _milvus_client
