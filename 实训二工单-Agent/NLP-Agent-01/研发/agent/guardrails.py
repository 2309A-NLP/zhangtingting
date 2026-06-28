# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
# 来源：北京八维信息集团 · 八维文化与产业研究院

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from config import DEFAULT_WELCOME, MAX_DB_RETRIES
from tools.db_tools import delete_record, insert_record, query_records, verify_record

MEMBER_ALIASES = {
    "爸爸": "爸爸",
    "爸": "爸爸",
    "父亲": "爸爸",
    "老爸": "爸爸",
    "爹": "爸爸",
    "老公": "爸爸",
    "他": None,
    "妈妈": "妈妈",
    "妈": "妈妈",
    "母亲": "妈妈",
    "老妈": "妈妈",
    "娘": "妈妈",
    "老婆": "妈妈",
    "女儿": "女儿",
    "闺女": "女儿",
    "孩子": "女儿",
    "丫头": "女儿",
    "她": "女儿",
    "我": "爸爸",
}

ADD_EXPENSE_WORDS = ["买", "花了", "花钱", "支出", "消费", "买了", "订购", "付了", "用了", "下单", "吃了", "报"]
ADD_INCOME_WORDS = ["收到", "到账", "收入", "工资", "报销", "奖金", "发了", "进账"]
DELETE_WORDS = ["删除", "删掉", "不要了", "撤销", "取消记录", "不要那条了", "不要那笔了"]
QUERY_WORDS = ["明细", "多少", "查询", "看看", "看下", "哪天", "什么时候", "统计", "花了多少钱"]
ACCOUNTING_WORDS = ADD_EXPENSE_WORDS + ADD_INCOME_WORDS + DELETE_WORDS + QUERY_WORDS
CONFIRM_WORDS = {"确认", "是", "是的", "对", "对的", "好", "好的", "嗯", "行", "可以", "删除"}
REJECT_WORDS = {"不是", "不对", "不用", "取消", "先不要", "不确认"}
FAMILY_WORDS = ["家里", "全家", "家庭"]
INCOME_QUERY_WORDS = ["收入", "工资", "报销", "奖金", "到账", "收到"]
EXPENSE_QUERY_WORDS = ["花了多少钱", "支出", "消费", "付", "花钱"]
DETAIL_QUERY_WORDS = ["花钱明细", "支出明细", "所有支出", "所有花费"]
DATE_TOKENS = ["今天", "昨天", "前天", "大前天", "上周五", "这个月", "本月", "上个月"]
CATEGORY_SPECIAL_CASES = {
    "旅游团": "旅游团费",
    "买书": "买书",
    "三体": "买书",
    "书": "买书",
}

# 作用：记住对话历史和未完成的操作，实现多轮对话。
@dataclass
class ConversationState:
    # 这种设计模式通常称为对话状态机，用于处理多轮对话中的复杂业务操作。
    pending_action: str | None = None          # 当前状态：fill_slots收集信息阶段/confirm_insert确认插入数据/confirm_delete确认删除数据/confirm_duplicate_insert确认重复插入数据/confirm_delete确认删除数据/recent_records_query查询最近记录/last_record_query查询最后一条记录
    pending_payload: dict[str, Any] = field(default_factory=dict)
    pending_records: list[dict] = field(default_factory=list)
    current_index: int = 0
    last_member: str | None = None
    last_intent: str | None = None
    last_record: dict | None = None
    recent_records: list[dict] = field(default_factory=list)
    duplicate_candidates: list[dict] = field(default_factory=list)


class GuardrailAgent:
    def __init__(self) -> None:
        self.state = ConversationState()

    def reply(self, user_input: str) -> str:
        text = user_input.strip()
        # 1. 空输入处理
        if not text:
            return DEFAULT_WELCOME
        # 2. 状态机处理（优先级最高）
        if self.state.pending_action == "fill_slots":
            return self._handle_fill_slots(text)
        if self.state.pending_action == "confirm_insert":
            return self._handle_confirm_insert(text)
        if self.state.pending_action == "confirm_duplicate_insert":
            return self._handle_confirm_duplicate_insert(text)
        if self.state.pending_action == "confirm_delete":
            return self._handle_confirm_delete(text)
        # 3. 引用查询（如"刚才那笔"）
        # 让用户可以用自然语言回溯查询刚刚记录的账目，而不需要重新输入具体信息。
        if self._is_recent_records_reference(text) and self._is_reference_query(text) and self.state.recent_records:
            return self._handle_recent_records_query(text)
        if self._is_last_record_reference(text) and self._is_reference_query(text) and self.state.last_record:
            return self._handle_last_record_query(text)
        # 4. 记账词汇检查（过滤无关输入）
        if not any(word in text for word in ACCOUNTING_WORDS):
            return DEFAULT_WELCOME
        # 5. 意图识别并分发
        intent = self._detect_intent(text)
        self.state.last_intent = intent
        if intent in {"ADD_EXPENSE", "ADD_INCOME"}:
            return self._handle_add(text, "支出" if intent == "ADD_EXPENSE" else "收入")
        if intent == "DELETE":
            return self._handle_delete(text)
        return self._handle_query(text)

    # 一、处理收集信息阶段
    def _handle_fill_slots(self, text: str) -> str:
        # 1. 拒绝处理
        if self._is_reject(text):
            # 重置状态 重置状态后，会清空pending_action、pending_payload、pending_records、current_index、last_member、last_intent、last_record、recent_records、duplicate_candidates
            self._reset_pending()
            return "好的，那我先不写入。这笔账如果要继续记录，您可以重新告诉我详细信息。"
        # 2. 处理payload：提取成员、日期、金额、事项、备注   <保护 state 中的数据不被意外修改，等确认完整后再覆盖。>
        # copy() 是 Python 字典的内置方法，用于创建字典的浅拷贝（shallow copy）。
        # 浅拷贝：只复制第一层，如果嵌套了其他对象（如列表、字典），则只复制引用，不会创建新的对象。
        # 深拷贝：会递归复制所有层级，创建新的对象，不会引用原来的对象。
        # 通常情况下，使用浅拷贝就足够了，除非需要完全独立的数据副本。
        payload = self.state.pending_payload.copy()
        # 3.提取成员、日期、金额、事项、备注
        # 为什么要先 payload.get() 再调用方法？  支持多轮对话，避免覆盖已有信息
        # 为什么要单独处理 note？  备注是累积的，而不是替换的
        payload["member"] = payload.get("member") or self._extract_member(text)
        payload["date"] = payload.get("date") or self._extract_date(text)
        payload["amount"] = payload.get("amount") if payload.get("amount") is not None else self._extract_amount(text)
        payload["category"] = payload.get("category") or self._extract_category(text, payload.get("type", "支出"))
        payload["note"] = self._merge_note(payload.get("note", ""), text)

        # 4.检查缺失字段
        missing = self._missing_fields(payload)
        if missing:
            self.state.pending_payload = payload
            # 返回询问提示
            return self._build_missing_prompt(missing)
        # 5. 设置状态
        self.state.pending_records = [payload]
        self.state.current_index = 0
        self.state.pending_action = "confirm_insert"
        self.state.pending_payload = payload
        self.state.last_member = payload.get("member")
        # 返回确认消息
        return self._format_insert_confirmation(payload, 1, 1)

    # 二、处理确认插入
    def _handle_confirm_insert(self, text: str) -> str:
        # 1. 拒绝处理
        if self._is_reject(text):
            self._reset_pending()
            return "好的，那我先不写入。这笔账如果要继续记录，您可以重新告诉我详细信息。"
        # 2. 处理估算值
        if self._contains_estimate_word(text) and not self._is_confirm(text):
            return "这笔金额听起来像估算值，我先按您说的记下来。您确认无误的话，直接回复“确认”就行。"
        # 3. 处理确认
        if not self._is_confirm(text):
            return "您回复“确认”后，我就立即写入；如果有哪里不对，也可以直接告诉我修改。"
        records = self.state.pending_records or [self.state.pending_payload]
        # 4. 处理重复记录
        duplicate = self._find_duplicate_record(records[0]) if len(records) == 1 else None
        # 5. 处理重复记录
        if duplicate and self.state.pending_action != "confirm_duplicate_insert":
            # 设置状态
            self.state.pending_action = "confirm_duplicate_insert"
            self.state.duplicate_candidates = [duplicate]
            return (
                "这笔记录好像已经存过了：\n"
                f"{duplicate['date']} | {duplicate['member']} | {duplicate['type']} | {duplicate['category']} | {duplicate['amount']}元"
                f" | {duplicate.get('note', '')}\n"
                "如果您确认要再存一条，请回复“确认”。"
            )
        # 没有重复记录 直接调用保存方法
        return self._save_records(records)

    # 三、处理确认重复插入
    def _handle_confirm_duplicate_insert(self, text: str) -> str:
        # 1. 拒绝处理
        if self._is_reject(text):
            # 重置状态
            self._reset_pending()
            return "好的，那这笔重复记录我先不写入。"
        # 2. 处理确认
        if not self._is_confirm(text):
            return "如果您确认这次也要写入，请直接回复“确认”；如果不需要，回复“取消”就行。"
        records = self.state.pending_records or [self.state.pending_payload]
        return self._save_records(records)

    # 保存记录
    def _save_records(self, records: list[dict[str, Any]]) -> str:
        saved: list[str] = []
        for record in records:
            result = self._with_retry(lambda record=record: insert_record(**record))
            """
            错误的写法（常见陷阱）：
            lambdas = []
            for i in range(3):
                lambdas.append(lambda: print(i))  # 没有捕获当前值
            for f in lambdas:
                f()
            # 输出：
            # 2
            # 2
            # 2
            # 期望输出 0,1,2，但实际全是 2！
            为什么？
            lambda 内部的 i 是外部变量，不是lambda创建时的快照
            lambda 在调用时才去查找 i 的当前值
            循环结束后 i=2，所以所有lambda打印的都是2

            正确的写法（使用默认参数捕获）：
            lambdas = []
            for i in range(3):
                lambdas.append(lambda i=i: print(i))  # i=i 把当前值作为默认参数
            for f in lambdas:
                f()
            # 输出：0 1 2 ✓

            应用到代码中：
            # 错误写法（会出bug）
            for record in records:
                result = self._with_retry(lambda: insert_record(**record))
                # 所有lambda都使用最后一轮的record值！
            # 正确写法（捕获当前值）
            for record in records:
                result = self._with_retry(lambda record=record: insert_record(**record))
                # 每个lambda都保存了自己这轮的record值
            """
            if not result.get("success"):
                # 原子性思维    要么全部成功，要么全部失败（通过返回错误让上层处理）
                self._reset_pending()
                return result.get("message", "数据库写入失败，请稍后重试。")
            record_id = result.get("id")
            verified = self._with_retry(lambda record_id=record_id: verify_record(record_id))
            if not verified.get("exists"):
                self._reset_pending()
                return "抱歉，记录写入后校验失败，请稍后重试。"
            record["id"] = record_id
            saved.append(f"{record['date']}，{record['member']}{record['type']}{record['amount']}元（{record['category']}）")
        # 保存最后一条记录（用于快速撤销或重复记账）
        self.state.last_record = records[-1].copy() if records else None
        # 保存所有记录的快照（用于批量操作回顾）
        self.state.recent_records = [record.copy() for record in records]
        # 重置状态
        self._reset_pending()
        if len(saved) == 1:
            return f"已记录！{saved[0]}。"
        return "已全部记录！\n" + "\n".join(f"- {item}" for item in saved)

    # 四、处理确认删除
    def _handle_confirm_delete(self, text: str) -> str:
        # 1. 拒绝处理
        if self._is_reject(text):
            self._reset_pending()
            return "好的，那这条记录我先不删。"
        if not self._is_confirm(text):
            return "如果确认删除，请直接回复“确认”或“删除”；如果不是这条，也可以告诉我更多信息。"

        payload = self.state.pending_payload.copy()
        # 2. 处理删除
        result = self._with_retry(lambda: delete_record(int(payload["id"])))
        # 3. 重置状态
        self._reset_pending()
        # 4. 处理成功
        if result.get("success"):
            return f"已删除，数据库已更新：{payload['date']} {payload['member']} {payload['category']} {payload['amount']}元。"
        return result.get("message", "删除失败，请稍后再试。")

    # 处理添加记录
    def _handle_add(self, text: str, record_type: str) -> str:
        # 1. 分割记录
        records = self._split_records(text, record_type)
        # 2. 规范化记录
        normalized = self._normalize_record_payloads(records, record_type)
        # 3. 检查缺失字段
        incomplete = [payload for payload in normalized if self._missing_fields(payload)]
        if incomplete:
            first = incomplete[0]
            self.state.pending_action = "fill_slots"
            self.state.pending_payload = first
            self.state.pending_records = []
            return self._build_missing_prompt(self._missing_fields(first))
        # 4. 设置状态
        self.state.pending_action = "confirm_insert"
        self.state.pending_records = normalized
        self.state.pending_payload = normalized[0]
        self.state.current_index = 0
        self.state.last_member = normalized[-1]["member"]
        # 5. 处理重复记录
        if len(normalized) == 1:
            duplicate = self._find_duplicate_record(normalized[0])
            if duplicate:
                self.state.pending_action = "confirm_duplicate_insert"
                self.state.duplicate_candidates = [duplicate]
                return (
                    "这笔记录好像已经存过了：\n"
                    f"{duplicate['date']} | {duplicate['member']} | {duplicate['type']} | {duplicate['category']} | {duplicate['amount']}元"
                    f" | {duplicate.get('note', '')}\n"
                    "如果您确认还要再存一条，请回复“确认”。"
                )
            return self._format_insert_confirmation(normalized[0], 1, 1)

        lines = ["好的，我来帮您逐条确认，这次一共识别到以下账目："]
        for idx, payload in enumerate(normalized, start=1):
            lines.append(self._format_insert_confirmation(payload, idx, len(normalized)))
        lines.append("如果都没问题，您回复“确认”后我就一次写入数据库~")
        return "\n".join(lines)

    # 处理查询
    def _handle_query(self, text: str) -> str:
        # 1. 提取成员    有"全家"/"家庭"等词 → None（查所有人）
        member = None if any(word in text for word in FAMILY_WORDS) else self._extract_member(text)
        # 2. 检测记录类型
        record_type = self._detect_query_record_type(text)
        # 3. 提取日期范围
        date_from, date_to = self._extract_date_range(text)
        if not date_from and not date_to:
            date_from, date_to = self._default_month_range()
        # 4. 提取事项
        category = self._extract_query_category(text)
        # 5. 提取关键词
        keyword = self._extract_keyword(text)
        # 6. 查询记录
        result = self._with_retry(
            lambda: query_records(
                date_from=date_from,
                date_to=date_to,
                member=member,
                type=record_type,
                category=category,
                keyword=keyword if keyword and keyword != category else None,
            )
        )
        # 7. 处理查询失败
        if not result.get("success"):
            return result.get("message", "查询失败，请稍后再试。")
        records = result.get("records", [])
        if not records:   # 如果没有记录，返回提示
            return "该时间段内暂无记录。"
        # 8. 格式化查询结果
        return self._format_query_result(records, date_from, date_to, member=member, category=category, keyword=keyword)

    # 处理最后一条记录查询
    def _handle_last_record_query(self, text: str) -> str:
        record = self.state.last_record or {}
        if not record:
            return "我这边还没有可供参考的最近一条记录。"

        if any(token in text for token in ["多少钱", "多少元", "金额"]):
            return f"那笔是{record['amount']}元。"
        if any(token in text for token in ["谁买的", "谁花的", "谁记录的", "谁的"]):
            return f"那笔是{record['member']}记录的。"
        if any(token in text for token in ["是什么", "买了什么", "什么项目", "什么内容"]):
            return f"那笔是{record['member']}的{record['category']}，金额{record['amount']}元。"
        return (
            f"最近一条记录是：{record['date']}，{record['member']}{record['type']}{record['amount']}元"
            f"（{record['category']}）。"
        )

    # 处理最近记录查询
    def _handle_recent_records_query(self, text: str) -> str:
        record = self._resolve_recent_record_reference(text)
        if not record:
            return "我暂时没法定位您说的是哪一笔，您可以说第一笔、第二笔或最后一条。"

        if any(token in text for token in ["多少钱", "多少元", "金额"]):
            return f"这笔是{record['amount']}元。"
        if any(token in text for token in ["谁买的", "谁花的", "谁记录的", "谁的"]):
            return f"这笔是{record['member']}记录的。"
        if any(token in text for token in ["是什么", "买了什么", "什么项目", "什么内容"]):
            return f"这笔是{record['member']}的{record['category']}，金额{record['amount']}元。"
        return (
            f"这笔记录是：{record['date']}，{record['member']}{record['type']}{record['amount']}元"
            f"（{record['category']}）。"
        )

    # 处理删除记录
    def _handle_delete(self, text: str) -> str:
        # 1. 处理最近记录引用
        if self._is_recent_records_reference(text) and self.state.recent_records:
            target = self._resolve_recent_record_reference(text)
            if target:
                self.state.pending_action = "confirm_delete"
                self.state.pending_payload = target.copy()
                self.state.pending_records = []
                self.state.last_member = target.get("member")
                return (
                    "找到以下记录：\n"
                    f"{target['date']} | {target['member']} | {target['type']} | {target['category']} | {target['amount']}元"
                    f" | {target.get('note', '')}\n"
                    "确认删除这条记录吗？"
                )

        if self._is_last_record_reference(text) and self.state.last_record:
            target = self.state.last_record.copy()
            self.state.pending_action = "confirm_delete"
            self.state.pending_payload = target
            self.state.pending_records = []
            self.state.last_member = target.get("member")
            return (
                "找到以下记录：\n"
                f"{target['date']} | {target['member']} | {target['type']} | {target['category']} | {target['amount']}元"
                f" | {target.get('note', '')}\n"
                "确认删除这条记录吗？"
            )

        member = self._extract_member(text)
        keyword = self._extract_keyword(text) or self._extract_query_category(text)
        date_from, date_to = self._extract_date_range(text)
        category_terms = [keyword] if keyword else []
        note_terms = [keyword] if keyword else []
        if keyword == "旅游团":
            category_terms = ["旅游团费", "旅游团", "旅游"]
            note_terms = ["旅游团", "旅游团费", "报旅游团"]

        records: list[dict[str, Any]] = []
        result: dict[str, Any] = {"success": True, "records": []}

        # 1. 查询事项
        for term in category_terms:
            result = self._with_retry(
                lambda term=term: query_records(
                    member=member,
                    category=term,
                    date_from=date_from,
                    date_to=date_to,
                )
            )
            if result.get("success") and result.get("records"):
                records = result.get("records", [])
                break
        # 2. 查询备注
        if not records:
            for term in note_terms:
                result = self._with_retry(
                    lambda term=term: query_records(
                        member=member,
                        keyword=term,
                        date_from=date_from,
                        date_to=date_to,
                    )
                )
                if result.get("success") and result.get("records"):
                    records = result.get("records", [])
                    break
        # 3. 处理查询失败
        if not result.get("success"):
            return result.get("message", "查询待删除记录失败，请稍后再试。")
        if not records:
            return "没找到对应的记录，请确认信息是否正确，可以再描述一下吗？"
        # 4. 设置状态
        target = records[0]
        self.state.pending_action = "confirm_delete"
        self.state.pending_payload = target
        self.state.pending_records = []
        self.state.last_member = target.get("member")
        return (
            "找到以下记录：\n"
            f"{target['date']} | {target['member']} | {target['type']} | {target['category']} | {target['amount']}元"
            f" | {target.get('note', '')}\n"
            "确认删除这条记录吗？"
        )

    # 意图识别
    def _detect_intent(self, text: str) -> str:
        if any(word in text for word in DELETE_WORDS):
            return "DELETE"
        if any(word in text for word in QUERY_WORDS):
            return "QUERY"
        if any(word in text for word in ADD_INCOME_WORDS):
            return "ADD_INCOME"
        if any(word in text for word in ADD_EXPENSE_WORDS):
            return "ADD_EXPENSE"
        return "QUERY"

    # 构建记录的数据结构
    def _build_payload(self, text: str, record_type: str) -> dict[str, Any]:
        member = self._extract_member(text)
        parsed_date = self._extract_date(text)
        amount = self._extract_amount(text)
        category = self._extract_category(text, record_type)
        note = self._extract_note(text, category)
        return {
            "date": parsed_date,
            "member": member,
            "type": record_type,
            "category": category,
            "amount": amount,
            "note": note,
        }

    # 规范化记录的数据结构
    def _normalize_record_payloads(self, records: list[str], record_type: str) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        shared_date: str | None = None      # 存储上一个有效的日期
        shared_member: str | None = None    # 存储上一个有效的成员
        for item in records:
            payload = self._build_payload(item, record_type)

            # 日期传递逻辑
            if payload.get("date"):
                shared_date = payload["date"]      # 当前有日期，更新共享值
            elif shared_date:                       # 当前无日期但有共享值
                payload["date"] = shared_date       # 向下传递日期

            # 成员传递逻辑
            if payload.get("member"):
                shared_member = payload["member"]   # 当前有成员，更新共享值
            # 如果当前有共享成员且不是"我"，则向下传递成员  \b 表示单词边界
            elif shared_member and not re.search(r"\b我\b", item):   # 如果当前有共享成员且不是"我"，则向下传递成员
                payload["member"] = shared_member   # 向下传递成员

            normalized.append(payload)   # 将规范化后的记录添加到列表中
        return normalized   # 返回规范化后的记录列表

    # 缺失字段
    def _missing_fields(self, payload: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if not payload.get("date"):
            missing.append("日期")
        if not payload.get("member"):
            missing.append("成员")
        if payload.get("amount") is None:
            missing.append("金额")
        if not payload.get("category"):
            missing.append("事项")
        return missing

    # 构建缺失字段提示词
    def _build_missing_prompt(self, missing: list[str]) -> str:
        prompts = {
            "日期": "请问是哪天的账目呢？",
            "成员": "请问是谁的消费/收入呢？",
            "金额": "请问花了（或收到）多少钱呢？",
            "事项": "请问这笔是什么费用呢？",
        }
        lines = ["好的，帮您记录！有几笔信息需要确认一下："]
        for item in missing:
            lines.append(f"- {prompts[item]}")
        lines.append("请告诉我，我来补充完整~")
        return "\n".join(lines)

    # 分割记录
    def _split_records(self, text: str, record_type: str) -> list[str]:
        if not re.search(r"[，,、和及并且;；]", text):
            return [text]   # 没有分隔符，直接返回原文本
        # 1. 第一层分割：主要用分号
        parts = re.split(r"[；;]", text)
        records: list[str] = []
        for part in parts:
            # 2. 第二层分割：在"元"后面用逗号类分隔符分割
            sub_parts = re.split(r"(?<=元)[，,、和及并且]", part)
            for item in sub_parts:
                # 3. 清理空格和标点
                cleaned = item.strip(" ，,、。；;")
                if cleaned:
                    records.append(cleaned)
        return records or [text]   # 如果没有分割成功，返回原文本

    # 重试机制
    def _with_retry(self, operation):
        last_result = None
        for attempt in range(MAX_DB_RETRIES):           # 循环重试
            try:
                last_result = operation()                # 执行传入的操作
                if last_result:                          # 如果返回真值，直接返回
                    return last_result
            except Exception:                            # 捕获所有异常
                pass                                      # 什么都不做，继续重试
            if attempt < MAX_DB_RETRIES - 1:             # 不是最后一次
                last_result = {"success": False, "message": "抱歉，数据库连接异常，我正在重新尝试..."}
        return last_result or {"success": False, "message": "数据库连接异常，请稍后重试。"}

    # 检测记录类型  
    def _detect_query_record_type(self, text: str) -> str | None:
        if any(word in text for word in INCOME_QUERY_WORDS):
            return "收入"   # 收入查询
        if any(word in text for word in EXPENSE_QUERY_WORDS):
            return "支出"   # 支出查询
        if any(word in text for word in DETAIL_QUERY_WORDS):
            return None   # 详细查询
        return None   # 默认返回None

    # 判断是否包含最后一条记录引用
    def _is_last_record_reference(self, text: str) -> bool:
        return any(token in text for token in ["刚才那笔", "那笔", "那条", "刚才那条", "那是", "刚才那是"])

    # 判断是否包含最近记录引用
    def _is_recent_records_reference(self, text: str) -> bool:
        return any(token in text for token in ["第一笔", "第二笔", "最后一条", "最后一笔", "第一条", "第二条"])

    # 判断是否包含引用查询
    def _is_reference_query(self, text: str) -> bool:
        return any(token in text for token in ["多少", "多少钱", "谁", "什么", "是什么", "买了什么"])

    # 解析最近记录引用
    def _resolve_recent_record_reference(self, text: str) -> dict[str, Any] | None:
        records = self.state.recent_records
        if not records:
            return None
        if any(token in text for token in ["第一笔", "第一条"]):
            return records[0]
        if any(token in text for token in ["第二笔", "第二条"]):
            return records[1] if len(records) >= 2 else None
        if any(token in text for token in ["最后一条", "最后一笔"]):
            return records[-1]
        return None

    # 提取成员
    def _extract_member(self, text: str) -> str | None:
        # 遍历成员别名，如果提取成功，返回别名对应的成员
        for alias, normalized in MEMBER_ALIASES.items():
            if alias in text:
                if normalized is None:
                    return None
                return normalized
        # 如果提取失败，返回上次操作的成员
        return self.state.last_member

    # 提取金额
    def _extract_amount(self, text: str) -> float | None:
        # (\d+(?:\.\d+)?) - 捕获数字（整数或小数）  内部的非捕获组只匹配不捕获，如果捕获了那么匹配到的小数会单独出现
        # \s* - 可选的空白字符
        # (?:元|块) - 非捕获组，匹配"元"或"块"
        amount_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:元|块)", text)
        if amount_matches:
            return float(amount_matches[-1])
        # 提取独立金额
        # 前面的断言：(?<![年月日])(?<!\d)
        # 捕获组：(\d{2,}(?:\.\d+)?)
        # 后面的断言：(?![年月日])
        # 类型	    语法	       含义	        光标
        # 正向前瞻	(?=...)	       后面是...	不动
        # 负向前瞻	(?!...)	       后面不是...	不动
        # 正向后顾	(?<=...)	   前面是...	不动
        # 负向后顾	(?<!...)	   前面不是...	不动
        standalone = re.findall(r"(?<![年月日])(?<!\d)(\d{2,}(?:\.\d+)?)(?![年月日])", text)
        return float(standalone[-1]) if standalone else None

    # 提取日期
    def _extract_date(self, text: str) -> str | None:
        today = date.today()
        if "今天" in text:
            # 返回今天的日期
            return today.isoformat()
        if "昨天" in text:
            return (today - timedelta(days=1)).isoformat()
        if "前天" in text:
            return (today - timedelta(days=2)).isoformat()
        if "大前天" in text:
            return (today - timedelta(days=3)).isoformat()
        if "上周五" in text:
            return self._weekday_of_last_week(4).isoformat()
        # 提取完整日期
        full = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if full:
            return date(int(full.group(1)), int(full.group(2)), int(full.group(3))).isoformat()
        # 提取部分日期
        partial = re.search(r"(\d{1,2})月(\d{1,2})日", text)
        if partial:
            return date(today.year, int(partial.group(1)), int(partial.group(2))).isoformat()
        iso = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if iso:
            return iso.group(1)
        return None

    # 提取日期范围
    def _extract_date_range(self, text: str) -> tuple[str | None, str | None]:
        today = date.today()
        if "这个月" in text or "本月" in text or "当月" in text:
            return self._default_month_range()
        if "上个月" in text:
            first_this_month = today.replace(day=1)
            last_prev_month = first_this_month - timedelta(days=1)
            # 日期替换为当月第1天
            first_prev_month = last_prev_month.replace(day=1)
            return first_prev_month.isoformat(), last_prev_month.isoformat()
        if "今天" in text:
            value = today.isoformat()
            return value, value
        if "昨天" in text:
            value = (today - timedelta(days=1)).isoformat()
            return value, value
        return None, None

    # 提取默认月份范围
    def _default_month_range(self) -> tuple[str, str]:
        today = date.today()
        first = today.replace(day=1)   # 获取当月第一天
        last = today.replace(day=calendar.monthrange(today.year, today.month)[1])   # 获取当月最后一天
        return first.isoformat(), last.isoformat()   # 返回当月第一天和最后一天的日期

    # 提取事项
    def _extract_category(self, text: str, record_type: str) -> str | None:
        if record_type == "收入":
            if "报销" in text:
                return "报销"
            if "工资" in text:
                return "工资"
            if "奖金" in text:
                return "奖金"

        for token, category in CATEGORY_SPECIAL_CASES.items():
            if token in text:
                return category

        cleaned = text
        # 替换成员 时间 单位
        for token in list(MEMBER_ALIASES.keys()) + DATE_TOKENS + ["元", "块"]:
            cleaned = cleaned.replace(token, " ")
        # 替换日期
        cleaned = re.sub(r"\d{4}年\d{1,2}月\d{1,2}日", " ", cleaned)
        cleaned = re.sub(r"\d{1,2}月\d{1,2}日", " ", cleaned)
        # 替换金额
        cleaned = re.sub(r"\d+(?:\.\d+)?", " ", cleaned)
        # 替换动作词
        verbs = ADD_EXPENSE_WORDS if record_type == "支出" else ADD_INCOME_WORDS
        for verb in verbs + DELETE_WORDS + QUERY_WORDS:
            cleaned = cleaned.replace(verb, " ")
        # 替换空格
        # 多个空格变一个，去除首尾标点
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，。,.!！?？")
        # 如果清洗后的字符串为空，返回None
        if not cleaned:
            return None
        # 如果清洗后的字符串长度大于20，截取前20个字符
        if len(cleaned) > 20:
            cleaned = cleaned[:20]
        # 返回清洗后的字符串
        return cleaned

    # 提取备注
    def _extract_note(self, text: str, category: str | None) -> str:
        if not category:
            return text.strip()
        note = text.strip()
        return note if note != category else ""   # 如果备注和事项相同，返回空字符串

    # 提取查询类别
    def _extract_query_category(self, text: str) -> str | None:
        if "买书" in text or "书" in text or "三体" in text:
            return "买书"

        patterns = [
            r"买(.+?)花了多少钱",
            r"(.+?)支出多少",
            r"(.+?)消费多少",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).replace("这个月", "").replace("本月", "").replace("我", "").strip()
                if value and value not in MEMBER_ALIASES and value not in {"爸爸", "妈妈", "女儿"}:
                    return value
        return None

    # 提取关键词
    def _extract_keyword(self, text: str) -> str | None:
        if "旅游团" in text:
            return "旅游团"
        if "买书" in text:
            return None
        if "三体" in text:
            return "三体"
        if "书" in text:
            return None

        """
        买的      # 字面量 "买的"
        (         # 开始捕获组1
        .+?     # 匹配任意字符1次或多次，非贪婪（尽量少）
        )         # 结束捕获组1
        (?:       # 开始非捕获组（只匹配不提取）
        \?      # 匹配英文问号 "?"
        |       # 或
        ？      # 匹配中文问号 "？"
        )         # 结束非捕获组
        ?         # 整个非捕获组可选（出现0次或1次）
        $         # 字符串结尾
        """
        patterns = [
            r"买的(.+?)(?:\?|？)?$",
            r"删除女儿报(.+?)的费用",
            r"删除(.+?)的费用",
            r"删除(.+?)的",
            r"删除(.+?)(?:\?|？)?$",
            r"什么时候(.+?)(?:\?|？)?$",
            r"哪天买的(.+?)(?:\?|？)?$",
            r"哪天(.+?)(?:\?|？)?$",
            r"报(.+?)的费用",
            r"买(.+?)花了多少钱",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip(" ，。,.!！?？")
                value = value.replace("女儿", "").replace("爸爸", "").replace("妈妈", "").strip()
                return value or None
        return None

    # 格式化插入确认
    def _format_insert_confirmation(self, payload: dict[str, Any], idx: int, total: int) -> str:
        dt = datetime.strptime(payload["date"], "%Y-%m-%d")
        # 多条记录时：显示 "第1笔："、"第2笔：" 等
        # 单条记录时：显示友好的引导语 "好的，我来帮您记录："
        prefix = f"第{idx}笔：" if total > 1 else "好的，我来帮您记录："
        return "\n".join(
            [
                prefix,
                f"  日期：{dt.year}年{dt.month}月{dt.day}日",
                f"  成员：{payload['member']}",
                f"  类型：{payload['type']}",
                f"  事项：{payload['category']}",
                f"  金额：{payload['amount']}元",
                "  确认无误吗？确认后我立即写入数据库~",
            ]
        )

    # 格式化查询结果
    def _format_query_result(
        self,
        records: list[dict[str, Any]],
        date_from: str | None,
        date_to: str | None,
        member: str | None = None,
        category: str | None = None,
        keyword: str | None = None,
    ) -> str:
        # 模式1：个人支出汇总   member 有值，且 category 和 keyword 都为 None，且记录中有支出
        if member and not category and not keyword and any(item["type"] == "支出" for item in records):
            expenses = [item for item in records if item["type"] == "支出"]
            total = sum(float(item["amount"]) for item in expenses)
            lines = [f"根据记录，{member}本月共支出："]
            for item in expenses:
                dt = datetime.strptime(item["date"], "%Y-%m-%d")
                lines.append(f"  - {dt.month}月{dt.day}日 | {item['category']} | {item['amount']}元（{item.get('note', '')}）")
            lines.append(f"{member}本月总支出：{total:.0f}元，共{len(expenses)}笔记录。")
            return "\n".join(lines)

        # 模式2：相关记录列表    有 category 或 keyword（查特定类别/关键词）
        if category or keyword:
            total = sum(float(item["amount"]) for item in records if item["type"] == "支出")
            lines = ["为您查到这些相关记录："]
            for item in records:
                dt = datetime.strptime(item["date"], "%Y-%m-%d")
                lines.append(f"  - {dt.year}年{dt.month}月{dt.day}日 | {item['member']} | {item['category']} | {item['amount']}元 | {item.get('note', '')}")
            lines.append(f"您本月相关花费共计：{total:.0f}元，共{len(records)}笔。")
            return "\n".join(lines)

        # 模式3：家庭整体统计（按成员分组）   不满足以上条件（通常是查"全家"或没有指定成员/类别）
        lines = []
        if date_from and date_to:
            start = datetime.strptime(date_from, "%Y-%m-%d")
            end = datetime.strptime(date_to, "%Y-%m-%d")
            lines.append(f"您查看的时间段：{start.year}年{start.month}月{start.day}日 ~ {end.year}年{end.month}月{end.day}日")
            lines.append("")

        grouped: dict[str, list[dict[str, Any]]] = {"爸爸": [], "妈妈": [], "女儿": []}
        for record in records:   # 按成员分组
            # 遍历所有记录
            # 如果记录的 member 字段是"爸爸"、"妈妈"、"女儿"，会追加到对应的列表中
            # 如果记录的 member 是其他值（如"爷爷"、"奶奶"），setdefault 会创建新键并设置空列表，然后追加
            grouped.setdefault(record["member"], []).append(record)

        total_expense = 0.0   # 总支出
        total_income = 0.0   # 总收入
        for person in ["爸爸", "妈妈", "女儿"]:
            items = grouped.get(person, [])   # 获取对应成员的记录列表
            if not items:   # 如果对应成员的记录列表为空，则跳过
                continue
            lines.append(f"【{person}记录】")   # 添加成员记录标题
            for item in items:
                # 遍历对应成员的记录列表
                dt = datetime.strptime(item["date"], "%Y-%m-%d")
                sign = "+" if item["type"] == "收入" else ""
                lines.append(f"  {dt.month}月{dt.day}日 | {item['category']} | {sign}{item['amount']}元 | {item.get('note', '')}")
                if item["type"] == "支出":
                    total_expense += float(item["amount"])
                else:
                    total_income += float(item["amount"])
            lines.append("")   # 添加空行

        lines.append(f"本月家庭总支出：{total_expense:.0f}元  |  总收入：{total_income:.0f}元  |  结余：{(total_income - total_expense):.0f}元")
        return "\n".join(lines).strip()

    # 提取上周日期
    def _weekday_of_last_week(self, weekday: int) -> date:
        today = date.today()
        # 计算本周第一天
        start_of_this_week = today - timedelta(days=today.weekday())
        # 计算上周第一天
        start_of_last_week = start_of_this_week - timedelta(days=7)
        # 计算上周指定星期几
        return start_of_last_week + timedelta(days=weekday)

    # 判断是否包含估算词
    def _contains_estimate_word(self, text: str) -> bool:
        return any(token in text for token in ["大概", "好像", "估计"])

    # 查找重复记录
    def _find_duplicate_record(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        # 查询重复记录
        result = self._with_retry(
            lambda: query_records(
                date_from=payload.get("date"),
                date_to=payload.get("date"),
                member=payload.get("member"),
                type=payload.get("type"),
                category=payload.get("category"),
            )
        )
        if not result.get("success"):
            # 保守策略：宁可不判断重复，也要能正常录入
            return None
        for record in result.get("records", []):     # 遍历查询结果
            same_amount = float(record.get("amount", 0)) == float(payload.get("amount", 0))  # 判断金额是否相同
            same_note = (record.get("note") or "") == (payload.get("note") or "")  # 判断备注是否相同
            if same_amount and same_note:
                return record  # 返回重复记录
        return None

    # 合并备注
    def _merge_note(self, current: str, text: str) -> str:
        merged = f"{current} {text}".strip()
        return merged

    # 判断是否确认
    def _is_confirm(self, text: str) -> bool:
        return text.strip() in CONFIRM_WORDS

    # 判断是否拒绝
    def _is_reject(self, text: str) -> bool:
        return text.strip() in REJECT_WORDS

    # 重置状态
    def _reset_pending(self) -> None:
        self.state.pending_action = None
        self.state.pending_payload = {}
        self.state.pending_records = []
        self.state.current_index = 0
        self.state.duplicate_candidates = []


"""
                  [开始]
                     ↓
              用户输入新命令
                     ↓
        ┌────────────────────────┐
        │   意图识别 / 状态检查    │
        └────────────────────────┘
                     ↓
    ┌───────────────────────────────┐
    │                               │
    ↓                               ↓
[信息不完整]                    [信息完整]
    ↓                               ↓
fill_slots                      confirm_insert
    ↓                               ↓
收集缺失信息                      检查重复
    ↓                               ↓
[信息完整]                      [有重复] → confirm_duplicate_insert
    ↓                               ↓
confirm_insert                      ↓
    ↓                               ↓
用户确认                         [执行插入]
    ↓                               ↓
执行插入                         [保存记录]
    ↓                               ↓
[完成] ←───────────────────────────┘
"""