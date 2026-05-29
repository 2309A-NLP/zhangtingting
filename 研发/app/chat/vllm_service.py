from __future__ import annotations
'''
时间轴 (秒)
↓
0s  ┌─────────────────────────────────────────────────────┐
    │ FastAPI 应用启动                                      │
    │ 创建 VLLMService 实例                                │
    └─────────────────────────────────────────────────────┘
    │
    ▼
0s  ┌─────────────────────────────────────────────────────┐
    │ wait_until_ready() 开始执行                          │
    │ deadline = now + 60秒                                │
    └─────────────────────────────────────────────────────┘
    │
    ▼
0s  ┌─────────────────────────────────────────────────────┐
    │ 第1次健康检查                                         │
    │ GET /models                                          │
    │ vLLM 可能还在启动中 → 返回错误                        │
    └─────────────────────────────────────────────────────┘
    │
    ▼ sleep 2秒
    │
2s  ┌─────────────────────────────────────────────────────┐
    │ 第2次健康检查                                         │
    │ GET /models                                          │
    │ vLLM 仍在加载模型 → 返回错误                          │
    └─────────────────────────────────────────────────────┘
    │
    ▼ sleep 2秒
    │
4s  ┌─────────────────────────────────────────────────────┐
    │ 第3次健康检查                                         │
    │ GET /models → 200 OK ✅                              │
    │ 状态变为 "ok"                                        │
    └─────────────────────────────────────────────────────┘
    │
    ▼
4s  ┌─────────────────────────────────────────────────────┐
    │ wait_until_ready() 返回 True                         │
    │ 日志: "vllm_ready"                                  │
    └─────────────────────────────────────────────────────┘
    │
    ▼
4s  ┌─────────────────────────────────────────────────────┐
    │ warmup() 开始执行                                    │
    │ 发送 POST /chat/completions                          │
    │ 消息: "hello"                                        │
    └─────────────────────────────────────────────────────┘
    │
    ▼ 等待模型推理 (通常 0.5-2秒)
    │
6s  ┌─────────────────────────────────────────────────────┐
    │ warmup() 完成                                        │
    │ 日志: "llm_warmup_succeeded"                        │
    │ CUDA kernels 已编译，模型已预热                      │
    └─────────────────────────────────────────────────────┘
    │
    ▼
6s  ┌─────────────────────────────────────────────────────┐
    │ 应用开始接收用户请求                                  │
    │ 第一个用户请求 → 直接使用已预热模型 → 快速响应         │
    └─────────────────────────────────────────────────────┘
'''
import asyncio
import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class VLLMService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vllm_api_key}"
        return headers

    async def healthcheck(self) -> tuple[str, str | None]:
        base_url = self.settings.vllm_base_url.rstrip("/")
        url = f"{base_url}/models"
        '''
        📌 核心 OpenAI 兼容端点（固定格式）
        端点	                    方法	          描述
        /v1/models	            GET	         列出当前加载的可用模型
        /v1/completions	        POST	     文本补全（适用于 base 模型）
        /v1/chat/completions	POST	   多轮对话补全（适用于 instruction 模型）
        /v1/embeddings	        POST	   生成文本向量（适用于 embedding 模型）
        🎯 其他支持的端点
        音频相关
        端点	                         方法	描述
        /v1/audio/transcriptions	 POST	语音转文字 
        /v1/audio/translations	     POST	语音翻译 
        Rerank/排序相关
        端点	                             方法	描述
        /rerank、/v1/rerank、/v2/rerank	 POST	语义重排序（适用于 cross-encoder 模型）
        其他功能端点
        端点	            方法	    描述
        /v1/responses	POST	OpenAI Responses API（实验性）
        /tokenize	    POST	将文本转换为 token ID 
        /detokenize	    POST	将 token ID 转回文本 
        /classify	    POST	文本分类（适用于分类模型）
        /pooling	    POST	向量池化（如 mean pooling）
        /score	        POST	评分/打分接口 
        系统/管理端点
        端点	           方法	描述
        /health	       GET	健康检查 
        /metrics	   GET	Prometheus 监控指标 
        /docs	       GET	Swagger UI 交互式 API 文档 
        /openapi.json  GET	OpenAPI 规范描述 
        /version	   GET	vLLM 版本号 
        📝 记笔记建议
        可以按类别整理：
        生成类：/completions、/chat/completions
        向量类：/embeddings、/pooling
        处理类：/tokenize、/detokenize、/classify、/rerank
        系统类：/health、/metrics、/models、/docs
        💡 小提示：启动 vLLM 服务后，访问 http://localhost:8000/docs 可以直接看到当前服务支持的所有端点，这是最准确的参考。
        '''
        headers = self._build_headers()

        try:
            # 这段代码中的 httpx.AsyncClient 是一个 HTTP 客户端，用于向 vLLM 服务的 HTTP API 发送请求。vLLM 本身是一个 HTTP 服务器，所以我们需要通过 HTTP 协议与它通信。
            async with httpx.AsyncClient(timeout=self.settings.vllm_health_timeout_seconds) as client:
                # 可优化：复用 HTTP 客户端
                '''
                class VLLMService:
                def __init__(self) -> None:
                    # 在初始化时创建一个长期复用的客户端
                    self._client = httpx.AsyncClient(
                        timeout=self.settings.vllm_timeout_seconds,
                        limits=httpx.Limits(max_keepalive_connections=10)  # 连接池大小
                    )
                
                async def healthcheck(self) -> tuple[str, str | None]:
                    # 复用同一个客户端，可以：
                    # 1. 复用 TCP 连接（减少握手开销）
                    # 2. 复用连接池
                    response = await self._client.get(url, headers=headers)
                    
                # ❌ 每次都创建：每次请求 ~50-100ms 额外开销
                async with httpx.AsyncClient() as client:
                    await client.get(url)  # 建立 TCP 连接
                # 关闭连接
                
                # ✅ 复用客户端：后续请求 ~1-5ms
                await self._client.get(url)  # 直接复用已有连接
                '''
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    return "ok", None
                if response.status_code in (401, 403):
                    return "unauthorized", f"vLLM returned {response.status_code}"
                return "down", f"vLLM returned {response.status_code}"
        except httpx.ConnectError as exc:
            return "unreachable", str(exc)
        except httpx.TimeoutException as exc:
            return "timeout", str(exc)
        except Exception as exc:
            logger.warning("vllm_healthcheck_failed", error=str(exc))
            return "down", str(exc)

    # 带超时的轮询等待机制，用于等待 vLLM 服务启动就绪
    async def wait_until_ready(self) -> bool:
        deadline = (
            asyncio.get_running_loop().time() + self.settings.vllm_startup_max_wait_seconds
        )
        last_status = "unknown"
        last_error: str | None = None

        while asyncio.get_running_loop().time() < deadline:
            status, error = await self.healthcheck()
            if status == "ok":
                logger.info("vllm_ready", provider="vllm")
                return True

            last_status = status
            last_error = error
            logger.debug(
                "vllm_not_ready_yet",
                provider="vllm",
                status=status,
                error=error,
            )
            # 等待配置的间隔时间，避免频繁请求
            await asyncio.sleep(self.settings.vllm_startup_poll_interval_seconds)

        logger.warning(
            "vllm_ready_timeout",
            provider="vllm",
            status=last_status,
            error=last_error,
            wait_seconds=self.settings.vllm_startup_max_wait_seconds,
        )
        return False

    # warmup（预热）功能，用于在服务启动后向 vLLM 发送一个测试请求，确保模型真正加载完成并可用。
    '''
    预热的主要作用：
    触发模型加载 - 确保模型完全加载到 GPU 内存
    CUDA Kernel 初始化 - 触发 GPU 内核的 JIT 编译
    缓存预热 - 让 vLLM 的 PagedAttention 等机制初始化
    减少首次请求延迟 - 避免第一个真实用户请求等待时间过长
    
    ⚠️ 注意事项
    不是所有模型都需要预热 - 某些小模型加载很快
    预热会增加启动时间 - 需要权衡
    预热失败不应该阻止服务启动 - 只记录警告，继续运行
    可以预热多个请求 - 触发不同的计算路径
    '''
    async def warmup(self) -> bool:
        if not self.settings.llm_warmup_enabled:
            logger.info("llm_warmup_skipped", reason="disabled")
            return False

        base_url = self.settings.vllm_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = self._build_headers()

        payload = {
            "model": self.settings.vllm_model,
            "messages": [
                {"role": "system", "content": "You are a warmup request."},
                {"role": "user", "content": "hello"},
            ],
            "temperature": 0.0,
            "max_tokens": self.settings.llm_warmup_max_tokens,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.settings.vllm_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            logger.info("llm_warmup_succeeded", provider="vllm", model=self.settings.vllm_model)
            return True
        except Exception as exc:
            logger.warning("llm_warmup_failed", provider="vllm", error=str(exc))
            return False
