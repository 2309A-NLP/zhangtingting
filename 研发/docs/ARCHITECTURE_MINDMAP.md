# RAG 项目架构图

原先使用的是 Mermaid `mindmap`，部分编辑器或 Mermaid 低版本会报错。

这里改成兼容性更高的 `flowchart` 版本，信息仍然是项目架构总览，通常在 GitHub、Obsidian、Typora、VS Code Mermaid 插件里都更稳。

```mermaid
flowchart TB

    U[用户]
    FE[前端应用<br/>Vite + React + Router + Zustand]
    API[FastAPI 应用<br/>app/main.py]

    U --> FE
    FE -->|HTTP / SSE| API

    subgraph FE_LAYER[前端层]
        FE1[AuthPage]
        FE2[RolesPage]
        FE3[ChatPage]
        FE4[KnowledgePage]
        FE5[api.ts<br/>fetchWithFallback]
        FE --> FE1
        FE --> FE2
        FE --> FE3
        FE --> FE4
        FE --> FE5
    end

    subgraph API_LAYER[API 接入层]
        R1[/auth]
        R2[/roles]
        R3[/chat]
        R4[/knowledge]
        R5[/health]
        D1[dependencies.py<br/>DB Session / Auth / RoleService]
        S1[schemas.py<br/>Envelope + Request/Response]
        API --> R1
        API --> R2
        API --> R3
        API --> R4
        API --> R5
        API --> D1
        API --> S1
    end

    subgraph CORE_LAYER[应用核心层]
        C1[RequestContextMiddleware]
        C2[exception_handlers]
        C3[config.py]
        C4[response.py]
        C5[logging.py]
        API --> C1
        API --> C2
        API --> C3
        API --> C4
        API --> C5
    end

    subgraph AUTH_ROLE_LAYER[认证与角色层]
        A1[AuthService]
        A2[RoleService]
        A3[预置角色<br/>lawyer / doctor / stock / history]
        A4[自定义角色 / 自动角色]
        R1 --> A1
        R2 --> A2
        A2 --> A3
        A2 --> A4
    end

    subgraph CHAT_LAYER[对话与 RAG 主链路]
        CH1[chat router]
        CH2[RateLimiter]
        CH3[RoleGuard]
        CH4[ChatCacheService]
        CH5[ContextBuilder]
        CH6[MemoryService]
        CH7[LLMClient]
        CH8[SSE EventSourceResponse]
        R3 --> CH1
        CH1 --> CH2
        CH1 --> CH3
        CH1 --> CH4
        CH1 --> CH5
        CH1 --> CH6
        CH1 --> CH7
        CH1 --> CH8
    end

    subgraph RETRIEVAL_LAYER[检索层]
        RT1[HybridRetriever]
        RT2[QueryRewriter]
        RT3[Dense Search]
        RT4[BM25 Candidate Rerank]
        RT5[RRF Fusion]
        RT6[BGE Reranker]
        RT7[BgeM3Embedder]
        CH5 --> RT1
        RT1 --> RT2
        RT1 --> RT3
        RT1 --> RT4
        RT1 --> RT5
        RT1 --> RT6
        RT3 --> RT7
    end

    subgraph KNOWLEDGE_LAYER[知识库导入链路]
        K1[knowledge router]
        K2[MinioStorageService]
        K3[KnowledgeTaskQueue]
        K4[KnowledgeIngestService]
        K5[loader]
        K6[cleaner]
        K7[chunker]
        K8[embedder]
        K9[KnowledgeFileService]
        K10[AuditService]
        R4 --> K1
        K1 --> K2
        K1 --> K3
        K1 --> K9
        K1 --> K10
        K3 --> K4
        K4 --> K5
        K4 --> K6
        K4 --> K7
        K4 --> K8
    end

    subgraph STORAGE_LAYER[数据存储层]
        DB1[(MySQL)]
        DB2[(Redis)]
        DB3[(Milvus)]
        DB4[(MinIO)]
    end

    subgraph MYSQL_DATA[MySQL 主要表]
        M1[users]
        M2[preset_roles]
        M3[preset_role_keywords]
        M4[custom_roles]
        M5[conversations]
        M6[user_role_mapping]
        M7[knowledge_files]
        M8[audit_logs]
    end

    subgraph REDIS_DATA[Redis 主要职责]
        RD1[recent chat]
        RD2[session marker]
        RD3[memory summary]
        RD4[query cache]
        RD5[rate limit]
        RD6[ingest status]
    end

    subgraph MILVUS_DATA[Milvus 主要内容]
        V1[rag_chunks]
        V2[tenant_key]
        V3[role_category]
        V4[doc_id / chunk_id]
        V5[text + embedding + source]
    end

    subgraph MINIO_DATA[MinIO 主要内容]
        O1[rag-raw]
        O2[rag-parsed]
    end

    A1 --> DB1
    A2 --> DB1

    CH4 --> DB2
    CH5 --> DB2
    CH5 --> DB1
    CH5 --> RT1
    CH6 --> DB2
    CH7 --> LLM1[本地 vLLM]
    CH7 --> LLM2[在线 SiliconFlow]
    CH1 --> DB1
    CH1 --> DB2

    RT3 --> DB3
    RT4 --> DB3

    K2 --> DB4
    K3 --> DB2
    K4 --> DB3
    K4 --> DB4
    K9 --> DB1
    K10 --> DB1

    DB1 --- M1
    DB1 --- M2
    DB1 --- M3
    DB1 --- M4
    DB1 --- M5
    DB1 --- M6
    DB1 --- M7
    DB1 --- M8

    DB2 --- RD1
    DB2 --- RD2
    DB2 --- RD3
    DB2 --- RD4
    DB2 --- RD5
    DB2 --- RD6

    DB3 --- V1
    DB3 --- V2
    DB3 --- V3
    DB3 --- V4
    DB3 --- V5

    DB4 --- O1
    DB4 --- O2

    ISO[隔离设计<br/>user_id + role_id + session_id]
    ISO --> DB2
    ISO --> DB1
    ISO --> DB3

    DEP[运行环境<br/>Docker Compose]
    DEP --> API
    DEP --> DB1
    DEP --> DB2
    DEP --> DB3
    DEP --> DB4
```

## 说明

- 如果你的 Mermaid 环境版本较旧，`flowchart` 基本比 `mindmap` 稳定。
- 这张图更偏“系统架构总览”，不是时序图。
- 如果你要更接近示例那种“脑图分叉风格”，我建议下一步改成 `xmind`/`draw.io`/`ProcessOn` 可直接导入的格式，而不是继续赌 Mermaid `mindmap` 兼容性。
