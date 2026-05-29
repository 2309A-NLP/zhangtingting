from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
# get_redis()：获取 Redis 客户端连接
# memory_summary_key：一个函数，用于生成 Redis 中存储摘要的键名
from app.db.redis_client import get_redis, memory_summary_key

logger = get_logger(__name__)


class MemoryService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.redis = get_redis()  # 返回Redis客户端实例

    # 整个类的核心方法，接收新的一轮对话，更新 Redis 中的摘要。
    async def update_summary(
        self,
        *,
        user_id: str,
        role_id: str,
        session_id: str | None = None,
        query: str,
        response: str,
    ) -> str:
        '''
        参数	             类型	            说明
        user_id	         str	      用户 ID（哪个用户在问）
        role_id	         str	      角色 ID（用户在和哪个 AI 角色对话）
        session_id	  str | None	  会话 ID（可选，不提供则所有会话共享记忆）
        query	         str	      用户刚才问的问题
        response	     str	      AI 刚才的回答
        返回值：更新后的摘要字符串。
        '''
        # 不同用户、不同角色、不同会话之间相互隔离
        key = memory_summary_key(user_id, role_id, session_id)
        # 如果 key 存在，redis.get() 返回字符串（如 "User: 你好\nAssistant: 你好！"）
        # 如果 key 不存在，redis.get() 返回 None
        # None or "" 的结果是 ""（空字符串）:or取值逻辑：返回第一个为真的值，如果所有值都是假值，就返回最后一个假值。None 属于假值
        # 为什么要转成空字符串？
        # 因为后面的 _merge_summary 期望参数是字符串，空字符串代表"还没有任何对话历史"。
        current_summary = await self.redis.get(key) or ""
        next_summary = self._merge_summary(
            current_summary=current_summary,
            query=query,
            response=response,
        )
        # 写回 Redis
        # key：Redis 键名
        # next_summary：新生成的摘要字符串
        # ex：过期时间（Expiration），单位是秒
        '''
        为什么要设置过期时间？
            防止 Redis 无限增长（用户长期不对话，记忆慢慢失效）
            自动清理不活跃用户的记忆
            符合"短期记忆"的设计理念（太久远的对话可能不重要）
        '''
        '''
        优势	                 说明
        Token 可控	  摘要大小固定，LLM 调用成本可预测
        内存可控	      Redis 存储空间有限，不会无限增长
        响应快速	      一次读取整个摘要，无需多次查询
        实现简单	      纯字符串操作，无需复杂数据结构
        自动清理	      TTL 自动删除过期数据，无需维护
        牺牲	       说明	                   严重程度
        细节丢失	摘要截断会丢失部分信息	    ⚠️ 中等（可接受）
        不可追溯	无法找回原始完整对话	    ⚠️ 中等（可接受）
        更新成本	每次对话都要重新生成摘要	✅ 低（CPU 成本小）
        为什么可以接受？
            对于对话场景，用户只关心最近几轮（短期记忆）
            很早之前的对话即使被截断，影响也不大
            如果需要永久保存，应该有"长期记忆"系统（向量数据库）配合
        用可控的代价（压缩摘要），换来了可预测的成本（Token/内存）和简单的实现，完美适配短期记忆场景。
        '''
        # SET 命令是覆盖的，不是追加。 存储类型是字符串
        await self.redis.set(key, next_summary, ex=self.settings.redis_memory_summary_ttl_seconds)
        logger.info("memory_summary_updated", user_id=user_id, role_id=role_id, summary_length=len(next_summary))
        return next_summary

    # 把旧的对话摘要 + 新的一轮问答 → 合并成新的摘要
    def _merge_summary(
        self,
        *,
        current_summary: str,
        query: str,
        response: str,
    ) -> str:
        # *,：后面的参数必须用关键字传递，不能按位置传。
        # 清理数据，去掉空行和多余空格。
        entries = [line.strip() for line in current_summary.splitlines() if line.strip()]
        # self._normalize_text()：调用静态方法，规范化文本（多个空格变一个）并截断到指定长度。
        # 用户问题：最多 160 个字符
        # AI 回答：最多 240 个字符（AI 通常说话更多）
        entries.append(f"User: {self._normalize_text(query, max_length=160)}")
        entries.append(f"Assistant: {self._normalize_text(response, max_length=240)}")
        # 限制总行数（滑动窗口） 取最后 N 个元素。
        # chat_memory_summary_max_lines:记忆摘要的最大行数
        if len(entries) > self.settings.chat_memory_summary_max_lines:
            entries = entries[-self.settings.chat_memory_summary_max_lines :]
        # 拼成字符串
        summary = "\n".join(entries)
        # chat_memory_summary_max_chars:记忆摘要的最大字符数
        if len(summary) > self.settings.chat_memory_summary_max_chars:
            summary = summary[-self.settings.chat_memory_summary_max_chars :]
            # find("\n")：找到第一个换行符的位置（索引）。
            # summary[first_newline + 1:]：从第一个换行符后面开始截取。
            # 效果：丢弃第一行（可能不完整），从第二行开头开始。
            # 边缘情况：
            # first_newline == 0：第一个字符就是换行符？不太可能，但如果是，summary[1:] 从第二个字符开始
            # first_newline == -1：没找到换行符，说明整个摘要只有一行，那就保留这一行（不断）
            first_newline = summary.find("\n")
            if first_newline > 0:
                summary = summary[first_newline + 1 :]
        return summary.strip()

    # 这是文本规范化 + 截断函数。
    # @staticmethod 静态方法装饰器，不需要self实例，直接能用，直接用类名调用
    @staticmethod
    def _normalize_text(value: str, *, max_length: int) -> str:
        '''
        为什么把它做成静态方法？
            因为这个函数：
            不需要访问任何实例属性（不需要 self.settings、self.redis）
            不需要访问类属性
            输入明确（value + max_length）→ 输出明确（处理后的文本）
            纯函数：同样的输入永远得到同样的输出
            明确意图：告诉读代码的人"这个方法不依赖实例状态"
        '''
        # value.split()：按任意空白分割，返回单词列表（连续多个空格当作一个分隔符）
        # " ".join(...)：用单个空格重新拼接
        # 效果：多个空格/换行/制表符 → 单个空格。
        collapsed = " ".join(value.split())
        # 如果长度没超，直接返回
        # 如果超了，保留前 max_length - 3 个字符，去掉末尾空格，再加 ...
        if len(collapsed) <= max_length:
            return collapsed
        return f"{collapsed[: max_length - 3].rstrip()}..."
