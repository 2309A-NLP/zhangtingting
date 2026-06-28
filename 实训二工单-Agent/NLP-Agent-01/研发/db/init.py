# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
# 来源：北京八维信息集团 · 八维文化与产业研究院

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from config import DB_PATH, SCHEMA_PATH


def init_db() -> None:
    """执行建表 SQL，确保数据库和表结构就绪。幂等操作，可重复执行。"""
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(db_path.parent):
        # os.makedirs(db_path.parent, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        # execute(): 执行单条SQL
        # executescript(): 执行多条SQL（用分号分隔）
        cursor.executescript(f.read())
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"数据库初始化完成：{DB_PATH}")
