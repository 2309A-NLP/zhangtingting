#!/usr/bin/env python3
"""
数据库初始化脚本 — 提取元数据、验证完整性、生成初始 few-shot 案例
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from config import settings


def main():
    db_path = settings.DB_PATH_RESOLVED
    if not db_path.exists():
        print(f"[ERROR] 数据库文件不存在: {db_path}")
        print("请先运行 scripts/download_data.sh 下载数据")
        return

    print(f"[INFO] 正在分析数据库: {db_path}")
    print(f"   文件大小: {db_path.stat().st_size / 1024 / 1024:.2f} MB")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. 列出所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n[TABLE] 共发现 {len(tables)} 张数据表:\n")

    metadata = []

    for table_name in tables:
        print(f"--- {table_name} ---")

        # 表信息
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        columns = cursor.fetchall()
        print(f"  字段数: {len(columns)}")

        # 行数
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
        row_count = cursor.fetchone()[0]
        print(f"  行数: {row_count:,}")

        # 外键
        cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
        fks = cursor.fetchall()

        table_meta = {
            "table_name": table_name,
            "row_count": row_count,
            "columns": [],
            "foreign_keys": [],
        }

        for col in columns:
            col_id, col_name, col_type, not_null, default_val, is_pk = col
            print(f"    |- {col_name:30s} {col_type or '':15s} {'PK' if is_pk else '  '}")
            table_meta["columns"].append({
                "name": col_name,
                "type": col_type or "TEXT",
                "not_null": bool(not_null),
                "is_primary_key": bool(is_pk),
            })

        for fk in fks:
            _, _, ref_table, fk_from, fk_to, *_ = fk
            print(f"    |- FK: {fk_from} -> {ref_table}({fk_to})")
            table_meta["foreign_keys"].append({
                "from": fk_from,
                "to_table": ref_table,
                "to_column": fk_to,
            })

        metadata.append(table_meta)

    conn.close()

    # 保存元数据
    meta_path = Path("data/processed/schema_metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 元数据已保存: {meta_path}")

    # 生成 few-shot 示例模板
    few_shot_path = Path(settings.FEW_SHOT_PATH)
    if not few_shot_path.exists():
        template = [
            {
                "question": "景顺长城中短债债券C基金在20210331的季报里，前三大持仓占比的债券名称是什么？",
                "sql": "SELECT b.bond_name, b.proportion "
                       "FROM fund_bond_holdings b "
                       "JOIN fund_basic_info f ON b.fund_code = f.fund_code "
                       "WHERE f.fund_name LIKE '%景顺长城中短债债券C%' "
                       "AND b.report_date = '2021-03-31' "
                       "ORDER BY b.proportion DESC LIMIT 3;",
                "table_names": ["fund_bond_holdings", "fund_basic_info"],
                "category": "data_query",
            },
            {
                "question": "在20210105，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？",
                "sql": "SELECT a.stock_code, "
                       "ROUND((a.close_price - a.pre_close) / a.pre_close * 100, 2) AS change_pct "
                       "FROM a_share_daily_market a "
                       "JOIN a_share_industry i ON a.stock_code = i.stock_code "
                       "WHERE a.trade_date = '2021-01-05' "
                       "AND i.industry_level1 = '综合金融' "
                       "ORDER BY change_pct DESC LIMIT 1;",
                "table_names": ["a_share_daily_market", "a_share_industry"],
                "category": "data_query",
            },
            {
                "question": "请帮我查询出20210415日，建筑材料一级行业涨幅超过5%（不包含）的股票数量。",
                "sql": "SELECT COUNT(*) AS stock_count "
                       "FROM a_share_daily_market a "
                       "JOIN a_share_industry i ON a.stock_code = i.stock_code "
                       "WHERE a.trade_date = '2021-04-15' "
                       "AND i.industry_level1 = '建筑材料' "
                       "AND (a.close_price - a.pre_close) / a.pre_close > 0.05;",
                "table_names": ["a_share_daily_market", "a_share_industry"],
                "category": "data_query",
            },
        ]
        with open(few_shot_path, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        print(f"[OK] 初始 few-shot 示例已生成: {few_shot_path}")
        print(f"   (包含 {len(template)} 条示例，建议人工补充至 100-200 条)")
    else:
        print(f"[INFO] few-shot 文件已存在: {few_shot_path}")


if __name__ == "__main__":
    main()
