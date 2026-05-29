from __future__ import annotations

# dataclass：Python 3.7+ 引入的装饰器。
# 你给它一个类，它会自动生成 __init__、__repr__、__eq__ 等方法。
# field：用于给单个字段提供额外配置，比如默认值、是否在 __repr__ 中显示、是否参与比较等。下面会看到用法。
from dataclasses import dataclass, field
# Any：表示"可以是任何类型"。这是放弃类型检查时的逃生舱口。
# 在这里用于 retrieval_debug 和 raw_response，因为这些字典的内部结构是不固定的，取决于具体的检索器或 LLM API 返回的原始数据。
from typing import Any

'''
slots=True 是一个重要的性能优化。加了它之后，Python 不会为每个实例创建 __dict__ 字典。
普通类的实例在内存中是一个字典（obj.__dict__），字典开销很大——每个键值对都要存哈希表条目。
使用 slots 后，实例的属性变成固定位置存储（类似于 C 结构体）。

效果对比：
无 slots：每个 ChatMessage 实例约 56+ 字节（对象头）+ 字典开销（72 字节起步）
有 slots：每个 ChatMessage 实例约 48 字节固定大小
副作用：不能动态添加新属性（但这对数据类通常是好事，防止拼写错误）。

省内存，因为相对于普通对象，会少一个存属性的字典内存，
速度快，因为声明属性的固定位置，省掉字典，用数组直接存,直接用下标去数据，
要是字典的话还需要去找key对应的值加偏移量，(字典需要计算hash → 定位槽位 → 处理冲突 → 读取value)
'''
@dataclass(slots=True)
class ChatMessage:
    '''
    role: str 和 content: str：
    这是类型注解（type hint），不是强制约束。运行时 Python 不会检查你传的是不是字符串。
    但 IDE（PyCharm、VS Code）和类型检查器（mypy、pyright）会用这些信息来提示和验证。
    role 的期望值：一般是 "system"、"user"、"assistant" 三者之一，对应 OpenAI API 的消息角色规范。
    content：消息的实际文本内容。
    没有默认值的原因：这两个字段没有在定义时赋默认值，意味着在构造 ChatMessage 时必须提供它们
    '''
    role: str
    content: str

# 这个类的用途：表示从向量数据库或检索系统中找回的一个"知识片段"。这是 RAG 流程的核心数据结构。
@dataclass(slots=True)
class ContextSource:
    """
    逐字段解释：
    doc_id: str：原始文档的唯一标识符。比如入库时的 document_12345。如果一次检索返回 5 个片段，它们可能来自同一个文档（相同的 doc_id），也可能来自不同文档。
    chunk_id: str：文档被切分后的块的 ID。一个文档通常被切成多个 chunk（比如每 500 字符一块），每个 chunk 有独立的 ID。chunk_id 用于精确定位是哪一块被检索到了。
    source: str：来源的文本描述，供前端或日志显示。可能是文件名、URL 或数据库表名。例如："用户手册_第四章.pdf" 或 "https://docs.example.com/page"。
    score: float：相似度分数，通常在 0 到 1 之间（也可能是其他范围，取决于向量检索的实现）。分数越高表示这个片段与用户查询越相关。context_builder.py 可能会用这个分数来决定是否要包含该片段（比如过滤掉分数低于 0.7 的），或者按分数排序后只取前 K 个。
    text: str：实际检索到的文本内容。这是要交给 LLM 作为上下文参考的原始文字。
    为什么 doc_id 和 chunk_id 分开：因为同一文档的不同 chunk 有着相同的 doc_id 但不同的 chunk_id。如果用户问的问题跨越了多个 chunk，前端可以知道这些回答都来自同一份文档（doc_id 相同），但具体引用了文档的不同位置（chunk_id 不同）。
    """
    doc_id: str
    chunk_id: str
    source: str
    score: float
    text: str

# 这个类的用途：context_builder.py 的最终产出。它包含了"发给 LLM 之前的所有准备内容"。
@dataclass(slots=True)
class BuiltContext:
    # 注意这个类型：list[dict[str, str]] —— 它是一个字典列表，每个字典的键和值都是字符串。
    '''
    为什么用 dict[str, str] 而不是 list[ChatMessage]？
    因为 llm_client.py 最终要调用 LLM API（比如 OpenAI），而 API 原生接受 JSON 字典格式。
    用字典避免了在最后一步再做一次转换。设计上选择在 context_builder 这一层就输出 API 所需的最终格式。
    '''
    messages: list[dict[str, str]]  # OpenAI API 的 messages 格式
    '''
    包含所有被纳入上下文的检索片段（可能经过筛选、排序、去重）。这个字段不送给 LLM（LLM 只需要看到拼接好的文本），而是用于：
        返回给前端显示"引用来源"
        调试和日志记录
        分析检索质量
    '''
    context_sources: list[ContextSource]
    '''
    这是一个重要的 RAG 优化机制。用户原始的查询可能很模糊或口语化（比如"那个上次说的东西"），在检索之前，系统会调用 LLM 把原始查询改写成更适合向量检索的形式。例如：
        原始："它多少钱？"（没有上下文，无法检索）
        改写后："根据对话历史，用户在询问之前讨论过的产品的价格"
        改写后的查询才是真正去向量数据库里搜索的字符串。保留这个字段便于：
        调试检索效果（看原始查询和改写后查询的差异）
        在 UI 上显示"系统将您的问题理解为：xxx"（用户体验优化）
    '''
    rewritten_query: str
    '''
    类型是 dict[str, Any]，默认值是一个空字典
    （default_factory=dict 让每个实例获得一个独立的新字典，而不是所有实例共享同一个字典
    ——如果用 = {} 作为默认值，所有实例会共享同一个字典对象）。
    field(default_factory=dict) 的语法：告诉 @dataclass 对这个字段特殊处理。
        普通字段 x: dict = {} 会导致所有实例共享同一个字典（一个大坑）。
        用 default_factory=dict，每次创建实例时都会调用 dict() 生成一个全新的空字典。
    '''
    # 这个字段的内容完全由context_builder.py决定，用于记录检索过程的调试信息。
    retrieval_debug: dict[str, Any] = field(default_factory=dict)

# llm_client.py 的返回结果，封装了从 LLM API 得到的响应。
@dataclass(slots=True)
class ChatCompletionResult:
    # LLM生成的最终回复内容。就是拿到后直接返回给用户的文本。
    content: str
    # 实际使用的模型名称。为什么需要？因为可能有降级策略：优先用本地模型（比如Llama.cpp或vLLM），
    # 如果本地不可用则降级到OpenAIGPT - 4。这个字段用于记录最终是哪个模型生成了回复。例如"llama3-8b-local"或"gpt-4-turbo"。
    model: str
    # 本次请求消耗的token总数。
    tokens_used: int
    # 这是一个标志位，表示是否触发了降级机制（fallback）。值是True说明首选模型失败了，切换到了在线API；False说明正常使用了首选模型。
    degraded_to_online_api: bool
    # 保留LLMAPI返回的原始JSON响应。比如OpenAI API返回的内容可能包含：
    '''
    存储原始响应的目的：
    如果需要 content 之外的字段（比如 finish_reason 是 "length" 还是 "stop"）
    调试
    审计和合规记录
    用 default_factory=dict 的原因同上（避免共享默认字典）
    '''
    raw_response: dict[str, Any] = field(default_factory=dict)
