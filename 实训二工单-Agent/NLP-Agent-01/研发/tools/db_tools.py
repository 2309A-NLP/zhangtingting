# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
# 来源：北京八维信息集团 · 八维文化与产业研究院

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from db.operations import delete_by_id, insert, select, verify
from tools.logging_utils import log_tool_call

logger = logging.getLogger(__name__)
ALLOWED_MEMBERS = {"爸爸", "妈妈", "女儿"}  # 家庭成员白名单
ALLOWED_TYPES = {"支出", "收入"}           # 收支类型白名单


def _validate_fields(date: str, member: str, type: str, category: str, amount: float) -> tuple[bool, str]:
    """统一字段校验，返回 (是否通过, 错误信息)。"""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        return False, f"日期格式错误，请提供 YYYY-MM-DD 格式，当前值：{date}"
    try:
        # strptime(): 将字符串转换为日期时间对象  日期有效性校验
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return False, f"日期无效：{date}"
    if member not in ALLOWED_MEMBERS:
        return False, f"成员 '{member}' 不在允许范围内，仅支持：{', '.join(sorted(ALLOWED_MEMBERS))}"
    if type not in ALLOWED_TYPES:
        return False, f"收支类型 '{type}' 不合法，仅支持：{', '.join(sorted(ALLOWED_TYPES))}"
    if not category or not category.strip():
        return False, "类别不能为空"
    if not isinstance(amount, (int, float)) or amount <= 0:
        return False, f"金额必须为正数，当前值：{amount}"
    return True, ""


def insert_record(date: str, member: str, type: str, category: str, amount: float, note: str = "") -> dict:
    """往 money_notes 表写入一条收支记录。"""
    ok, err = _validate_fields(date, member, type, category, amount)
    if not ok:
        result = {"success": False, "message": err, "id": None}
        log_tool_call("insert_record", {"date": date, "member": member, "type": type, "category": category, "amount": amount, "note": note}, result)
        return result

    try:
        record_id = insert(date, member, type, category, float(amount), note)
        verify_result = verify_record(record_id)
        if not verify_result.get("exists"):
            result = {"success": False, "message": "写入后校验失败", "id": record_id}
            log_tool_call("insert_record", {"date": date, "member": member, "type": type, "category": category, "amount": amount, "note": note}, result)
            return result
        result = {"success": True, "message": f"记录已写入数据库，ID：{record_id}", "id": record_id}
        log_tool_call("insert_record", {"date": date, "member": member, "type": type, "category": category, "amount": amount, "note": note}, result)
        return result
    except Exception as exc:
        logger.exception("[insert_record] DB error")
        result = {"success": False, "message": f"数据库写入失败：{exc}", "id": None}
        log_tool_call("insert_record", {"date": date, "member": member, "type": type, "category": category, "amount": amount, "note": note}, result)
        return result


def query_records(
    date_from: str | None = None,
    date_to: str | None = None,
    member: str | None = None,
    type: str | None = None,
    category: str | None = None,
    keyword: str | None = None,
) -> dict:
    """从 money_notes 表中查询记录，支持多条件组合筛选。"""
    try:
        records = select(date_from, date_to, member, type, category, keyword)
        result = {"success": True, "records": records, "count": len(records)}
        log_tool_call("query_records", {"date_from": date_from, "date_to": date_to, "member": member, "type": type, "category": category, "keyword": keyword}, result)
        return result
    except Exception as exc:
        logger.exception("[query_records] DB error")
        result = {"success": False, "records": [], "count": 0, "message": f"查询失败：{exc}"}
        log_tool_call("query_records", {"date_from": date_from, "date_to": date_to, "member": member, "type": type, "category": category, "keyword": keyword}, result)
        return result


def delete_record(record_id: int) -> dict:
    """根据记录 ID 删除一条记录。"""
    try:
        ok = delete_by_id(record_id)
        if ok:
            result = {"success": True, "message": f"记录 ID {record_id} 已删除"}
            log_tool_call("delete_record", {"record_id": record_id}, result)
            return result
        result = {"success": False, "message": f"未找到 ID 为 {record_id} 的记录"}
        log_tool_call("delete_record", {"record_id": record_id}, result)
        return result
    except Exception as exc:
        logger.exception("[delete_record] DB error")
        result = {"success": False, "message": f"删除失败：{exc}"}
        log_tool_call("delete_record", {"record_id": record_id}, result)
        return result


def verify_record(record_id: int) -> dict:
    """验证指定 ID 的记录是否已成功写入数据库。"""
    try:
        record = verify(record_id)
        result = {"exists": record is not None, "record": record}
        log_tool_call("verify_record", {"record_id": record_id}, result)
        return result
    except Exception as exc:
        logger.exception("[verify_record] DB error")
        result = {"exists": False, "record": None, "message": f"验证失败：{exc}"}
        log_tool_call("verify_record", {"record_id": record_id}, result)
        return result
