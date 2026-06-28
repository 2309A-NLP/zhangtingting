from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional

from config import DB_PATH
from db.init import init_db

logger = logging.getLogger(__name__)


def get_conn() -> sqlite3.Connection:
    """获取数据库连接，自动建表（幂等操作）。"""
    # 初始化数据库  init_db() 是幂等的（重复执行不会出错）
    init_db()
    # check_same_thread=True（默认）：SQLite 会检查当前线程和创建连接的线程是否是同一个。如果是不同的线程操作这个连接，会报错。
    # check_same_thread=False：关闭这个检查，允许多个线程使用同一个连接。
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # sqlite3.Row：使返回的行像字典一样访问 row['column_name']
    conn.row_factory = sqlite3.Row
    return conn


def insert(date: str, member: str, type: str, category: str, amount: float, note: str = "") -> int:
    """写入一条记录，返回新记录的 id。"""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO money_notes (date, member, type, category, amount, note) VALUES (?, ?, ?, ?, ?, ?)",
        (date, member, type, category, amount, note),
    )
    conn.commit()
    # 获取自动生成的ID（lastrowid）
    record_id = cursor.lastrowid
    conn.close()
    logger.info(
        json.dumps(
            {
                # 记录时间戳
                # ISO 8601 格式：YYYY-MM-DD HH:MM:SS.mmmmmm
                # 1. 可排序（字符串排序等于时间排序）
                    # t1 = "2026-01-01T10:00:00"
                    # t2 = "2026-01-02T09:00:00"
                    # print(t1 < t2)  # True，可以直接字符串比较
                # 2. 无歧义（不像 01/02/2026 到底是1月2日还是2月1日）
                # 3. 解析方便
                    # dt = datetime.fromisoformat("2026-06-14T15:30:45")
                "timestamp": datetime.now().isoformat(),
                # 操作类型
                "action": "INSERT",
                # 记录ID
                "record_id": record_id,
                # 日期
                "date": date,
                # 成员
                "member": member,
                # 金额
                "amount": amount,
            },
            ensure_ascii=False,
        )
    )
    return int(record_id)


def select(
    # 开始日期
    date_from: str | None = None,
    # 结束日期
    date_to: str | None = None,
    # 成员
    member: str | None = None,
    # 类型
    type: str | None = None,
    # 类别
    category: str | None = None,
    # 备注模糊搜索关键词
    keyword: str | None = None,
) -> list[dict]:
    """多条件组合查询，返回记录列表。"""
    conn = get_conn()
    cursor = conn.cursor()
    # WHERE 1=1 是为了方便添加 AND 条件
    sql = "SELECT * FROM money_notes WHERE 1=1" # 1=1 表示永远为真
    params: list[str | int | float] = []

    if date_from:
        sql += " AND date >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND date <= ?"
        params.append(date_to)
    if member:
        sql += " AND member = ?"
        params.append(member)
    if type:
        sql += " AND type = ?"
        params.append(type)
    if category:
        sql += " AND category LIKE ?"
        params.append(f"%{category}%")
    if keyword:
        sql += " AND note LIKE ?"
        params.append(f"%{keyword}%")

    sql += " ORDER BY date DESC, id DESC"  # 按日期倒序，ID倒序
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_by_id(record_id: int) -> bool:
    """根据 ID 删除一条记录，返回是否成功。"""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM money_notes WHERE id = ?", (record_id,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    logger.info(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "action": "DELETE",
                "record_id": record_id,
                "affected": affected,
            },
            ensure_ascii=False,
        )
    )
    return affected > 0


def verify(record_id: int) -> Optional[dict]:
    """查询指定 id 是否存在，若存在返回记录 dict。"""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM money_notes WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
