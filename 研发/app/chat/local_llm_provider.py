from __future__ import annotations

import asyncio
import json
# 导入异步生成器类型   用途：类型注解，表示流式方法返回异步生成器
from collections.abc import AsyncGenerator
# 导入数据类装饰器  简化 _LoadedLocalModel 类的定义  自动生成 __init__、__repr__ 等方法
from dataclasses import dataclass
# 缓存装饰器  确保模型只加载一次，后续调用直接返回缓存的实例
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.chat.models import ChatCompletionResult
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LocalLLMNotAvailableError(RuntimeError):
    pass


@dataclass(slots=True)
class _LoadedLocalModel:
    tokenizer: Any
    model: Any
    device: str

# 将配置字符串转换为 PyTorch 数据类型
def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        # 匹配半精度浮点数（节省显存，训练时常用）
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        # 匹配 BF16 格式（Google提出的格式，动态范围更大）
        return torch.bfloat16
    # 默认返回单精度（精度最高但最耗显存）
    return torch.float32


@lru_cache(maxsize=1)  # 最多缓存1个结果，相同的参数直接返回缓存
def _load_local_model(
    model_path: str,
    device: str,
    dtype_name: str,
    trust_remote_code: bool,
) -> _LoadedLocalModel:
    torch_dtype = _resolve_torch_dtype(dtype_name)
    logger.info(
        "local_llm_loading_started",
        model_path=model_path,
        device=device,
        dtype=str(torch_dtype),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,  # 只从本地加载，不联网下载
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        torch_dtype=torch_dtype,
    )
    # 作用：设置填充token（padding token）  某些分词器没有 pad_token，但生成时需要  将结束符（EOS）复用为填充符
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # 作用：将模型移动到指定设备（CPU或GPU）
    model.to(device)
    model.eval()

    logger.info(
        "local_llm_loading_finished",
        model_path=model_path,
        device=device,
    )
    return _LoadedLocalModel(tokenizer=tokenizer, model=model, device=device)


class LocalLLMProvider:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def model_name(self) -> str:
        configured = self.settings.local_llm_model_name.strip()
        if configured:
            return configured
        return Path(self.settings.local_llm_model_path).name

    def is_enabled(self) -> bool:
        return bool(self.settings.local_llm_enabled and self.settings.local_llm_model_path.strip())

    def healthcheck(self) -> tuple[str, str | None]:
        if not self.settings.local_llm_enabled:
            return "disabled", "LOCAL_LLM_ENABLED=false"

        model_path = Path(self.settings.local_llm_model_path)
        if not model_path.exists():
            return "missing_model", f"Model path not found: {model_path}"
        if not model_path.is_dir():
            return "invalid_model_path", f"Model path is not a directory: {model_path}"
        return "ok", None

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResult:
        loaded = self._get_loaded_model()
        content, prompt_tokens, completion_tokens = await asyncio.to_thread(
            self._generate_text,
            loaded,
            messages,
            temperature,
            max_tokens,
        )
        return ChatCompletionResult(
            content=content,
            model=self.model_name,
            tokens_used=prompt_tokens + completion_tokens,
            degraded_to_online_api=False,
            raw_response={
                "provider": "local_transformers",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    async def stream(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[dict[str, str], None]:
        # 伪流式
        result = await self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        '''
        关键点：
        yield 只执行一次（因为结果只有一个）
        发送的是完整内容，不是增量
        event: "delta" 表示这是一个数据块事件
        ensure_ascii=False 确保中文显示正常
        '''
        if result.content:  # 检查是否有内容（非空字符串）
            yield {
                "event": "delta",  # 事件类型：增量数据块
                "data": json.dumps(
                    {
                        "content": result.content,  # 完整的回复内容
                        "model": result.model,  # 模型名称
                        "degraded": False,  # 是否降级（这里总是 False）
                    },
                    ensure_ascii=False,  # 保留中文等 Unicode 字符，不转义成 \uXXXX
                ),
            }
        yield {
            "event": "end",
            "data": json.dumps(
                {"model": result.model, "degraded": False},
                ensure_ascii=False,
            ),
        }

    def _get_loaded_model(self) -> _LoadedLocalModel:
        if not self.is_enabled():
            raise LocalLLMNotAvailableError("Local CPU LLM fallback is disabled.")

        status, error = self.healthcheck()
        if status != "ok":
            raise LocalLLMNotAvailableError(error or "Local CPU LLM fallback is not available.")

        device = self._resolve_device()
        return _load_local_model(
            self.settings.local_llm_model_path,
            device,
            self.settings.local_llm_dtype,
            self.settings.local_llm_trust_remote_code,
        )

    def _resolve_device(self) -> str:
        requested = self.settings.local_llm_device.strip().lower() or "cpu"
        if requested == "cuda":
            if not torch.cuda.is_available():
                logger.warning("local_llm_cuda_unavailable_falling_back_to_cpu")
                return "cpu"
            return "cuda"
        return "cpu"

    def _generate_text(
        self,
        loaded: _LoadedLocalModel,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        # 将消息列表转换为模型需要的提示词格式
        '''
        为什么需要这一步？
        不同模型使用不同的对话格式
        apply_chat_template 会根据模型的 tokenizer_config.json 自动应用正确的格式
        add_generation_prompt=True 会在末尾添加助手回复的起始标记，告诉模型"该你回答了"
        '''
        prompt = loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 不进行tokenization，只返回字符串
            add_generation_prompt=True,  # 添加助手回复的提示符
        )
        # 将文本转换为模型能理解的数字 ID
        inputs = loaded.tokenizer([prompt], return_tensors="pt")
        # 将输入张量移动到模型所在的设备（CPU 或 CUDA）
        # 模型在 GPU 上，输入数据也必须在 GPU 上
        inputs = {key: value.to(loaded.device) for key, value in inputs.items()}

        generation_kwargs = {
            **inputs,
            "max_new_tokens": min(
                max(1, max_tokens),  # 至少生成1个token
                self.settings.local_llm_max_new_tokens,  # 不超过配置上限
            ),
            # 优先使用 pad_token_id
            # 如果没有，使用 eos_token_id 作为替代
            # 这是为了处理变长序列的批处理
            "pad_token_id": loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True  # 启用随机采样
            generation_kwargs["temperature"] = temperature  # 控制随机性程度
            generation_kwargs["top_p"] = self.settings.local_llm_top_p  # 核采样
        else:
            generation_kwargs["do_sample"] = False  # 贪婪解码（总是选概率最高的）
        '''
        temperature 的作用：
        # temperature = 0: 贪婪解码
        # 总是选概率最高的 token，结果确定
        # temperature = 0.7: 适度随机
        # 概率分布被"软化"，高概率的 token 仍大概率被选中
        # temperature = 1.0: 原始概率分布
        # 直接按模型输出的概率采样
        # temperature = 2.0: 高随机性
        # 概率分布被"平滑"，低概率 token 也有机会被选中
        
        top_p (核采样) 的作用：
        # top_p = 0.9 表示从累计概率达到 90% 的最小 token 集合中采样
        # 例如 token 概率: [0.5, 0.3, 0.1, 0.05, 0.05]
        # 累计到 0.9 的 tokens: [0.5, 0.3, 0.1] (累计 0.9)
        # 只从这三个 token 中采样，忽略后面的低概率 token
        '''

        with torch.inference_mode():
            '''
            # inference_mode()等价于老的 torch.no_grad()，但更快
            # 作用：禁用梯度计算，大幅减少内存占用
            # 训练时需要梯度，推理时不需要
            
            # 这是 transformers 库的核心方法
            # 输入: input_ids = [1, 2, 3, 4]  # "你好世界"
            # 输出: output_ids = [1, 2, 3, 4, 5, 6, 7]  # "你好世界，我是AI"
            # 自回归生成过程（简化）:
            # Step 1: 输入 [1,2,3,4] → 输出 token 5
            # Step 2: 输入 [1,2,3,4,5] → 输出 token 6
            # Step 3: 输入 [1,2,3,4,5,6] → 输出 token 7
            # 直到达到 max_new_tokens 或遇到 EOS token
            '''
            output_ids = loaded.model.generate(**generation_kwargs)

        '''
        # inputs["input_ids"] 的形状: (batch_size, sequence_length)
        # 例子: tensor([[101, 234, 567, 890]]) 
        # shape = (1, 4)
        # prompt_tokens = 4
        
        # int() 转换是因为 torch 的 shape 返回 torch.Size，需要转成 Python int
        '''
        # 计算输入 token 数
        prompt_tokens = int(inputs["input_ids"].shape[1])
        # 提取生成的 token
        '''
        # output_ids 形状: (batch_size, total_sequence_length)
        # 假设：
        # input_ids = [101, 234, 567, 890]     # 4个token
        # output_ids = [101, 234, 567, 890, 111, 222, 333]  # 总共7个token
        
        # output_ids[0] 获取第一个样本 → [101, 234, 567, 890, 111, 222, 333]
        # [prompt_tokens:] 从第4个token后开始切片 → [111, 222, 333]
        # completion_ids = [111, 222, 333]  # 生成的3个token
        '''
        completion_ids = output_ids[0][prompt_tokens:]
        # 计算生成 token 数
        '''
        # completion_ids 是一维张量: tensor([111, 222, 333])
        # shape = (3,)
        # completion_tokens = 3
        '''
        completion_tokens = int(completion_ids.shape[0])
        # 解码为文本
        '''
        # completion_ids = [111, 222, 333]
        # 解码：
        # 111 → "我是"
        # 222 → "AI"
        # 333 → "助手"
        # 结果: "我是AI助手"
        
        # skip_special_tokens=True 的作用：
        # 移除特殊 token，如 <|im_end|>, </s>, <pad> 等
        # 例如 token 列表中有 1234 → "<|im_end|>"，会被跳过
        '''
        text = loaded.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("Local CPU LLM returned an empty response.")
        return text, prompt_tokens, completion_tokens
