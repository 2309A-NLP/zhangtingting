from functools import lru_cache
from typing import Annotated, List
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

'''
是因为 alias 吗？	✅ 是，alias 告诉 Pydantic 去环境变量里找哪个名字
为什么不直接用环境变量名？	代码规范（小写）和环境变量规范（大写）冲突
设置两次的意义？	不是两次设置，是一次定义 + 一次映射
核心目的	让代码优雅，同时遵循两边的最佳实践
一句话：alias 是翻译官，让代码里的 app_secret_key 和环境变量里的 APP_SECRET_KEY 成为同一个东西。
'''

class Settings(BaseSettings):
    """
    应用配置类，继承自Pydantic的BaseSettings
    用于从环境变量和.env文件中加载配置
    支持类型验证和自动转换
    """
    # Pydantic v2的配置字典
    model_config = SettingsConfigDict(
        env_file=".env",              # 从.env文件加载环境变量
        env_file_encoding="utf-8",    # .env文件使用UTF-8编码
        case_sensitive=False,         # 环境变量不区分大小写
        extra="ignore",               # 忽略未定义的额外字段
    )

    app_name: str = Field(default="multi-role-rag-backend", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    app_runtime_trace: bool = Field(default=True, alias="APP_RUNTIME_TRACE")
    app_runtime_print: bool = Field(default=True, alias="APP_RUNTIME_PRINT")
    app_runtime_content_preview_chars: int = Field(default=160, alias="APP_RUNTIME_CONTENT_PREVIEW_CHARS")
    app_cors_origins: Annotated[List[str], NoDecode] = Field(default_factory=list, alias="APP_CORS_ORIGINS")
    app_cors_origin_regex: str = Field(default="", alias="APP_CORS_ORIGIN_REGEX")
    app_api_prefix: str = Field(default="/api/v1", alias="APP_API_PREFIX")
    app_timezone: str = Field(default="Asia/Shanghai", alias="APP_TIMEZONE")
    app_secret_key: str = Field(default="replace-with-a-long-random-secret", alias="APP_SECRET_KEY")
    app_access_token_expire_minutes: int = Field(default=1440, alias="APP_ACCESS_TOKEN_EXPIRE_MINUTES")
    auth_jwt_algorithm: str = Field(default="HS256", alias="AUTH_JWT_ALGORITHM")
    auth_enable_dev_header: bool = Field(default=True, alias="AUTH_ENABLE_DEV_HEADER")
    shared_preset_user_id: str = Field(default="__preset__", alias="SHARED_PRESET_USER_ID")

    # ==================== MySQL数据库配置 ====================
    # MySQL服务器地址，Docker环境下使用服务名
    mysql_host: str = Field(default="127.0.0.1", alias="MYSQL_HOST")
    # MySQL服务端口
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    # 数据库名称
    mysql_database: str = Field(default="rag_app", alias="MYSQL_DATABASE")
    # 数据库用户名
    mysql_user: str = Field(default="rag_user", alias="MYSQL_USER")
    # 数据库密码
    mysql_password: str = Field(default="rag_password", alias="MYSQL_PASSWORD")
    # MySQL root用户密码，用于初始化
    mysql_root_password: str = Field(default="root_password", alias="MYSQL_ROOT_PASSWORD")
    # 连接池大小，控制同时打开的最大连接数
    mysql_pool_size: int = Field(default=20, alias="MYSQL_POOL_SIZE")
    # 连接池最大溢出数，当连接池满时可额外创建的连接数
    mysql_max_overflow: int = Field(default=20, alias="MYSQL_MAX_OVERFLOW")
    # 连接回收时间（秒），超过此时间的连接会被回收重建，防止连接过期
    mysql_pool_recycle: int = Field(default=1800, alias="MYSQL_POOL_RECYCLE")
    # 是否在每次使用连接前ping检查连接有效性
    mysql_pool_pre_ping: bool = Field(default=True, alias="MYSQL_POOL_PRE_PING")

    # ==================== Redis缓存配置 ====================
    # Redis服务器地址
    redis_host: str = Field(default="127.0.0.1", alias="REDIS_HOST")
    # Redis服务端口
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    # Redis数据库编号
    redis_db: int = Field(default=0, alias="REDIS_DB")
    # Redis密码，空字符串表示无密码
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    # Redis连接池最大连接数
    redis_max_connections: int = Field(default=200, alias="REDIS_MAX_CONNECTIONS")
    # Redis键前缀，用于区分不同应用的键
    redis_key_prefix: str = Field(default="rag", alias="REDIS_KEY_PREFIX")
    # 聊天最近消息缓存过期时间（秒），默认24小时
    redis_chat_recent_ttl_seconds: int = Field(default=86400, alias="REDIS_CHAT_RECENT_TTL_SECONDS")
    # 聊天会话过期时间（秒），默认30分钟
    redis_chat_session_ttl_seconds: int = Field(default=1800, alias="REDIS_CHAT_SESSION_TTL_SECONDS")
    # 查询缓存过期时间（秒），默认1小时
    redis_query_cache_ttl_seconds: int = Field(default=3600, alias="REDIS_QUERY_CACHE_TTL_SECONDS")
    # 限流计数器过期时间（秒），默认1分钟
    redis_rate_limit_ttl_seconds: int = Field(default=60, alias="REDIS_RATE_LIMIT_TTL_SECONDS")
    # 记忆摘要过期时间（秒），默认7天
    redis_memory_summary_ttl_seconds: int = Field(default=604800, alias="REDIS_MEMORY_SUMMARY_TTL_SECONDS")
    # 知识库摄取状态过期时间（秒），默认7天
    redis_ingest_status_ttl_seconds: int = Field(default=604800, alias="REDIS_INGEST_STATUS_TTL_SECONDS")
    # 分布式锁过期时间（秒），默认5分钟
    redis_lock_ttl_seconds: int = Field(default=300, alias="REDIS_LOCK_TTL_SECONDS")

    # ==================== Milvus向量数据库配置 ====================
    # Milvus服务器地址，用于存储和检索向量嵌入
    milvus_uri: str = Field(default="http://127.0.0.1:19530", alias="MILVUS_URI")
    # Milvus认证令牌，格式为"用户名:密码"
    milvus_token: str = Field(default="root:Milvus", alias="MILVUS_TOKEN")
    # Milvus数据库名称
    milvus_db_name: str = Field(default="default", alias="MILVUS_DB_NAME")
    # Milvus集合名称，用于存储文档向量
    milvus_collection_name: str = Field(default="rag_chunks", alias="MILVUS_COLLECTION_NAME")
    # 向量索引类型：IVF_FLAT适合中等规模数据，IVF_PQ适合大规模数据
    milvus_index_type: str = Field(default="IVF_FLAT", alias="MILVUS_INDEX_TYPE")
    # 向量距离度量类型：COSINE(余弦相似度)/L2(欧氏距离)/IP(内积)
    milvus_metric_type: str = Field(default="COSINE", alias="MILVUS_METRIC_TYPE")
    # IVF索引的聚类中心数量，影响索引构建和查询性能
    milvus_nlist: int = Field(default=2048, alias="MILVUS_NLIST")
    # 搜索时查询的聚类中心数量，值越大越精确但越慢
    milvus_search_nprobe: int = Field(default=32, alias="MILVUS_SEARCH_NPROBE")
    # 一致性级别：Strong(强一致)/Bounded(有界一致)/Session(会话一致)/Eventually(最终一致)
    milvus_consistency_level: str = Field(default="Bounded", alias="MILVUS_CONSISTENCY_LEVEL")

    # ==================== MinIO对象存储配置 ====================
    # MinIO服务器地址，格式为"主机:端口"
    minio_endpoint: str = Field(default="127.0.0.1:9000", alias="MINIO_ENDPOINT")
    # MinIO访问密钥ID
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    # MinIO访问密钥密码
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    # 是否使用HTTPS协议连接MinIO
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")
    # 存储原始文档的桶名称
    minio_bucket_raw: str = Field(default="rag-raw", alias="MINIO_BUCKET_RAW")
    # 存储解析后文档的桶名称
    minio_bucket_parsed: str = Field(default="rag-parsed", alias="MINIO_BUCKET_PARSED")

    # ==================== vLLM本地推理服务配置 ====================
    # vLLM服务器基础URL，用于本地模型推理
    vllm_base_url: str = Field(default="http://127.0.0.1:8001/v1", alias="VLLM_BASE_URL")
    # vLLM API密钥，本地部署通常为"EMPTY"
    vllm_api_key: str = Field(default="EMPTY", alias="VLLM_API_KEY")
    # vLLM使用的模型名称
    vllm_model: str = Field(default="Qwen2.5-0.5B-Instruct", alias="VLLM_MODEL")
    # vLLM请求超时时间（秒）
    vllm_timeout_seconds: int = Field(default=120, alias="VLLM_TIMEOUT_SECONDS")
    # vLLM请求失败时的最大重试次数
    vllm_max_retries: int = Field(default=2, alias="VLLM_MAX_RETRIES")
    # vLLM健康检查超时时间（秒）
    vllm_health_timeout_seconds: int = Field(default=5, alias="VLLM_HEALTH_TIMEOUT_SECONDS")
    # 应用启动时等待 vLLM 就绪的最长时间（秒）
    vllm_startup_max_wait_seconds: int = Field(default=180, alias="VLLM_STARTUP_MAX_WAIT_SECONDS")
    # 应用启动时轮询 vLLM 就绪状态的时间间隔（秒）
    vllm_startup_poll_interval_seconds: float = Field(default=2.0, alias="VLLM_STARTUP_POLL_INTERVAL_SECONDS")
    # vLLM实际加载的 HuggingFace 模型名
    vllm_hf_model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct", alias="VLLM_HF_MODEL")
    # vLLM张量并行数
    vllm_tensor_parallel_size: int = Field(default=1, alias="VLLM_TENSOR_PARALLEL_SIZE")
    # vLLM显存利用率
    vllm_gpu_memory_utilization: float = Field(default=0.9, alias="VLLM_GPU_MEMORY_UTILIZATION")
    # vLLM最大上下文长度
    vllm_max_model_len: int = Field(default=8192, alias="VLLM_MAX_MODEL_LEN")
    # vLLM权重精度
    vllm_dtype: str = Field(default="auto", alias="VLLM_DTYPE")
    local_llm_enabled: bool = Field(default=True, alias="LOCAL_LLM_ENABLED")
    local_llm_model_name: str = Field(default="Qwen2.5-0.5B-Instruct", alias="LOCAL_LLM_MODEL_NAME")
    local_llm_model_path: str = Field(default="./data/models/Qwen2.5-0.5B-Instruct", alias="LOCAL_LLM_MODEL_PATH")
    local_llm_device: str = Field(default="cpu", alias="LOCAL_LLM_DEVICE")
    local_llm_dtype: str = Field(default="float32", alias="LOCAL_LLM_DTYPE")
    local_llm_max_new_tokens: int = Field(default=256, alias="LOCAL_LLM_MAX_NEW_TOKENS")
    local_llm_top_p: float = Field(default=0.9, alias="LOCAL_LLM_TOP_P")
    local_llm_trust_remote_code: bool = Field(default=True, alias="LOCAL_LLM_TRUST_REMOTE_CODE")

    # ==================== SiliconFlow在线API配置 ====================
    # SiliconFlow API基础URL，作为vLLM的降级方案
    siliconflow_base_url: str = Field(default="https://api.siliconflow.cn/v1", alias="SILICONFLOW_BASE_URL")
    # SiliconFlow API密钥，需要注册获取
    siliconflow_api_key: str = Field(default="", alias="SILICONFLOW_API_KEY")
    # SiliconFlow使用的模型名称
    siliconflow_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", alias="SILICONFLOW_MODEL")
    # SiliconFlow请求超时时间（秒）
    siliconflow_timeout_seconds: int = Field(default=120, alias="SILICONFLOW_TIMEOUT_SECONDS")
    # SiliconFlow请求失败时的最大重试次数
    siliconflow_max_retries: int = Field(default=2, alias="SILICONFLOW_MAX_RETRIES")
    # 是否启用 LLM 预热
    llm_warmup_enabled: bool = Field(default=True, alias="LLM_WARMUP_ENABLED")
    # 是否在应用启动时预热 LLM
    llm_warmup_on_startup: bool = Field(default=True, alias="LLM_WARMUP_ON_STARTUP")
    # LLM 预热时请求的最大 token 数
    llm_warmup_max_tokens: int = Field(default=8, alias="LLM_WARMUP_MAX_TOKENS")

    # ==================== 嵌入模型配置 ====================
    # 嵌入模型名称，用于将文本转换为向量
    embedding_model_name: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL_NAME")
    # 嵌入模型本地路径，为空则从HuggingFace下载
    embedding_model_path: str = Field(default="", alias="EMBEDDING_MODEL_PATH")
    # 向量维度，必须与模型输出的维度一致
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    # 模型运行设备：cpu/cuda/mps(苹果芯片)
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    # 最小批处理大小，用于动态批处理
    embedding_batch_size_min: int = Field(default=8, alias="EMBEDDING_BATCH_SIZE_MIN")
    # 最大批处理大小，用于动态批处理
    embedding_batch_size_max: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE_MAX")

    # ==================== 重排序模型配置 ====================
    # 重排序模型名称，用于对检索结果进行精细化排序
    rerank_model_name: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANK_MODEL_NAME")
    # 重排序模型本地路径，为空则从HuggingFace下载
    rerank_model_path: str = Field(default="", alias="RERANK_MODEL_PATH")
    # 重排序模型运行设备：cpu/cuda/mps(苹果芯片)
    rerank_device: str = Field(default="cpu", alias="RERANK_DEVICE")
    # 重排序保留的候选文档数量
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")

    # ==================== 知识库处理配置 ====================
    # 文件上传目录
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    # 解析后文件存储目录
    parsed_dir: str = Field(default="./data/parsed", alias="PARSED_DIR")
    # 最大上传文件大小（MB）
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")
    # 知识库任务队列工作线程数
    knowledge_task_queue_workers: int = Field(default=2, alias="KNOWLEDGE_TASK_QUEUE_WORKERS")
    # 知识库摄取完成后是否清理本地文件
    knowledge_cleanup_local_after_ingest: bool = Field(default=True, alias="KNOWLEDGE_CLEANUP_LOCAL_AFTER_INGEST")
    # 允许上传的文件类型列表
    allowed_upload_types: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["pdf", "txt", "json", "html"],
        alias="ALLOWED_UPLOAD_TYPES",
    )
    # 是否启用OCR功能，用于从图片中提取文字
    ocr_enabled: bool = Field(default=True, alias="OCR_ENABLED")
    # OCR识别语言：ch(中文)/en(英文)
    ocr_language: str = Field(default="ch", alias="OCR_LANGUAGE")
    pdf_use_pdfplumber: bool = Field(default=True, alias="PDF_USE_PDFPLUMBER")
    pdf_use_pdfminer: bool = Field(default=True, alias="PDF_USE_PDFMINER")
    pdf_use_camelot: bool = Field(default=False, alias="PDF_USE_CAMELOT")
    pdf_use_tabula: bool = Field(default=False, alias="PDF_USE_TABULA")
    pdf_extract_tables: bool = Field(default=True, alias="PDF_EXTRACT_TABLES")
    pdf_text_min_chars_per_page: int = Field(default=80, alias="PDF_TEXT_MIN_CHARS_PER_PAGE")
    pdf_ocr_force_all_pages: bool = Field(default=False, alias="PDF_OCR_FORCE_ALL_PAGES")
    pdf_ocr_render_dpi: int = Field(default=200, alias="PDF_OCR_RENDER_DPI")
    mineru_api_enabled: bool = Field(default=False, alias="MINERU_API_ENABLED")
    mineru_api_base_url: str = Field(default="", alias="MINERU_API_BASE_URL")
    mineru_api_parse_path: str = Field(default="/parse", alias="MINERU_API_PARSE_PATH")
    mineru_api_key: str = Field(default="", alias="MINERU_API_KEY")
    mineru_api_timeout_seconds: int = Field(default=180, alias="MINERU_API_TIMEOUT_SECONDS")
    mineru_fallback_quality_threshold: float = Field(default=45.0, alias="MINERU_FALLBACK_QUALITY_THRESHOLD")
    pdf_parse_cache_dir: str = Field(default="./data/pdf_parse_cache", alias="PDF_PARSE_CACHE_DIR")
    # 文本分块大小（字符数）
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    # 文本分块重叠大小（字符数）
    chunk_overlap: int = Field(default=51, alias="CHUNK_OVERLAP")
    # 最小块文本长度，小于此值的块会被过滤
    min_chunk_text_length: int = Field(default=20, alias="MIN_CHUNK_TEXT_LENGTH")
    # 敏感词文件路径
    sensitive_words_path: str = Field(default="./app/resources/sensitive_words.txt", alias="SENSITIVE_WORDS_PATH")

    # ==================== 检索配置 ====================
    # 最终返回的检索结果数量
    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")
    # BM25关键词检索返回的候选数量
    retrieval_bm25_top_k: int = Field(default=12, alias="RETRIEVAL_BM25_TOP_K")
    # 向量检索返回的候选数量
    retrieval_vector_top_k: int = Field(default=12, alias="RETRIEVAL_VECTOR_TOP_K")
    # RRF(Reciprocal Rank Fusion)融合参数，影响多路检索结果的融合方式
    retrieval_rrf_k: int = Field(default=60, alias="RETRIEVAL_RRF_K")
    # 是否启用检索功能
    retrieval_enabled: bool = Field(default=True, alias="RETRIEVAL_ENABLED")
    # 是否启用查询重写功能
    retrieval_enable_query_rewrite: bool = Field(default=True, alias="RETRIEVAL_ENABLE_QUERY_REWRITE")
    # 是否启用重排序功能
    retrieval_enable_rerank: bool = Field(default=True, alias="RETRIEVAL_ENABLE_RERANK")

    # ==================== 聊天配置 ====================
    # 聊天历史中保留的最近对话轮数
    chat_recent_rounds: int = Field(default=10, alias="CHAT_RECENT_ROUNDS")
    # 从数据库加载的历史记录条数限制
    chat_history_load_limit: int = Field(default=20, alias="CHAT_HISTORY_LOAD_LIMIT")
    # 记忆摘要的最大行数
    chat_memory_summary_max_lines: int = Field(default=12, alias="CHAT_MEMORY_SUMMARY_MAX_LINES")
    # 记忆摘要的最大字符数
    chat_memory_summary_max_chars: int = Field(default=2000, alias="CHAT_MEMORY_SUMMARY_MAX_CHARS")
    # 限流：每分钟允许的最大请求数
    rate_limit_requests_per_minute: int = Field(default=60, alias="RATE_LIMIT_REQUESTS_PER_MINUTE")
    # 限流：突发流量允许的请求数
    rate_limit_burst: int = Field(default=20, alias="RATE_LIMIT_BURST")
    # SSE心跳间隔时间（秒），用于保持长连接
    sse_heartbeat_seconds: int = Field(default=15, alias="SSE_HEARTBEAT_SECONDS")

    # ==================== 追踪和监控配置 ====================
    # 是否启用分布式追踪
    trace_enabled: bool = Field(default=False, alias="TRACE_ENABLED")
    # OpenTelemetry OTLP导出器端点
    otel_exporter_otlp_endpoint: str = Field(default="", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    # 是否启用指标收集
    metrics_enabled: bool = Field(default=False, alias="METRICS_ENABLED")

    # ==================== 测试配置 ====================
    # 测试用户ID
    test_user_id: str = Field(default="test-user-001", alias="TEST_USER_ID")
    # 测试角色ID
    test_role_id: str = Field(default="lawyer_01", alias="TEST_ROLE_ID")

    @field_validator("app_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | List[str]) -> List[str]:
        """
        验证和转换CORS源地址列表
        支持逗号分隔的字符串或列表格式
        """
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("allowed_upload_types", mode="before")
    @classmethod
    def parse_allowed_upload_types(cls, value: str | List[str]) -> List[str]:
        """
        验证和转换允许上传的文件类型列表
        支持逗号分隔的字符串或列表格式，统一转换为小写
        """
        if isinstance(value, list):
            return [item.lower() for item in value]
        return [item.strip().lower() for item in value.split(",") if item.strip()]

    @property
    def mysql_async_dsn(self) -> str:
        """
        构建MySQL异步数据源名称(DSN)
        使用asyncmy驱动，支持异步操作
        """
        user = quote_plus(self.mysql_user)
        password = quote_plus(self.mysql_password)
        return (
            f"mysql+asyncmy://{user}:{password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )

    @property
    def redis_url(self) -> str:
        """
        构建Redis连接URL
        自动处理密码的URL编码
        """
        auth = f":{quote_plus(self.redis_password)}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def minio_endpoint_url(self) -> str:
        """
        构建MinIO端点URL
        根据secure配置自动选择http或https协议
        """
        scheme = "https" if self.minio_secure else "http"
        return f"{scheme}://{self.minio_endpoint}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    获取配置单例
    使用LRU缓存确保配置只加载一次
    避免重复读取环境变量和.env文件
    """
    return Settings()
