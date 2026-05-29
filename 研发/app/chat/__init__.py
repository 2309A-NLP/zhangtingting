"""Chat services."""


"""
app/chat/ 目录完整结构
text
app/chat/
├── __init__.py          # 模块入口
├── models.py            # ✅ 数据模型定义
├── llm_client.py        # ✅ LLM 调用客户端
├── memory_service.py    # ✅ 记忆服务（摘要存储）
├── context_builder.py   # ✅ 上下文构建器
├── cache_service.py     # ✅ 缓存服务
├── role_guard.py        # ✅ 角色守卫
└── rate_limiter.py      # ✅ 限流器
各文件核心职责速记
文件	一句话职责	核心数据/概念
models.py	定义数据结构和传输对象	ChatMessage, ContextSource, BuiltContext, ChatCompletionResult
llm_client.py	调用 LLM API，支持自动降级	vLLM（本地） → SiliconFlow（在线）
memory_service.py	将对话压缩成摘要存到 Redis	update_summary() 写入，_merge_summary() 压缩算法
context_builder.py	组装最终发给 LLM 的 messages	系统提示 + 长期记忆 + 检索结果 + 历史对话 + 当前问题
cache_service.py	缓存 LLM 响应，避免重复调用	Redis 存 JSON，key 含 query 哈希
role_guard.py	生成角色专属 system prompt，后处理回复	法律/医疗/金融角色有特殊约束和免责声明
rate_limiter.py	限流（漏桶算法）	Redis + Lua 脚本，限制用户/角色请求频率
数据流概览
text
用户请求
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  api/routers/chat.py                                        │
│  ↓                                                          │
│  1. RoleGuard.build_system_prompt()  → 系统提示词            │
│  2. RateLimiter.check()              → 限流检查             │
│  3. CacheService.get_cached_response() → 查缓存            │
│  4. ContextBuilder.build()           → 构建上下文           │
│  5. LLMClient.complete()/stream()    → 调用 LLM            │
│  6. MemoryService.update_summary()   → 更新记忆            │
│  7. CacheService.set_cached_response() → 存缓存            │
│  8. RoleGuard.validate_and_postprocess() → 后处理          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
返回响应
文件间的依赖关系
text
                    ┌─────────────────┐
                    │   models.py     │（被所有文件依赖）
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ memory_service│    │context_builder│    │  llm_client   │
└───────────────┘    └───────┬───────┘    └───────────────┘
        │                    │                    │
        │              ┌─────┴─────┐              │
        │              │role_guard │              │
        │              └───────────┘              │
        │                                         │
        └─────────────────┬───────────────────────┘
                          ▼
              ┌───────────────────────┐
              │   cache_service       │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   rate_limiter        │
              └───────────────────────┘
              
              
              
关键配置参数速记
参数	默认值	作用
chat_recent_rounds	10	发给 LLM 的最近对话轮数
chat_history_load_limit	20	从 MySQL 加载的历史轮数
chat_memory_summary_max_lines	20	记忆摘要最大行数
chat_memory_summary_max_chars	2000	记忆摘要最大字符数
MIN_CONTEXT_SOURCE_SCORE	0.5	检索结果最低分数
MAX_CONTEXT_SOURCE_COUNT	5	最多引用几条检索结果
rate_limit_burst	20	漏桶容量（突发请求数）
rate_limit_requests_per_minute	60	每分钟最多请求数
redis_query_cache_ttl_seconds	3600	缓存过期时间（秒）
"""