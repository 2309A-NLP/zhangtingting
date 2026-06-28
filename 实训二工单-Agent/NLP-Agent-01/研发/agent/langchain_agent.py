from __future__ import annotations

import json
from typing import Any

from agent.llm_agent import LLMAgent, TOOLS, TOOL_MAP
from agent.prompts import load_system_prompt

# 假langchain 遗留代码
class LangChainCompatibleAgent(LLMAgent):
    """A lightweight compatibility layer that keeps the current runtime
    while exposing a structure close to the LangChain-style agent workflow.
    轻量级兼容层，保持当前运行时，同时暴露接近 LangChain 风格 Agent 工作流的结构。
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_template = load_system_prompt()
        self.tools = TOOLS
        self.tool_map = TOOL_MAP

    def build_agent(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt_template,
            "tools": self.tools,
            "executor": "openai-compatible-function-calling",
        }

    def inspect_runtime(self) -> dict[str, Any]:
        return {
            "tool_names": [tool["function"]["name"] for tool in self.tools],
            "message_count": len(self.memory.messages),
            "executor": "openai-compatible-function-calling",
        }

    def invoke(self, user_input: str) -> dict[str, Any]:
        output = self.reply(user_input)  # 调用父类方法 
        return {
            "input": user_input,
            "output": output,
            "runtime": self.inspect_runtime(),
        }
