#!/usr/bin/env python3
"""
few-shot 案例标注辅助工具

从 question.jsonl 中抽样题目，辅助人工标注 SQL
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def main():
    question_paths = [
        "data/raw/bs_challenge_financial_14b_dataset/question.json",
        "data/raw/bs_challenge_financial_14b_dataset/question.jsonl",
        "data/raw/question.json",
        "data/raw/question.jsonl",
        "question.json",
        "question.jsonl",
    ]
    question_path = None
    for p in question_paths:
        if Path(p).exists():
            question_path = p
            break
    if not question_path:

    # 加载问题
    questions = []
    with open(question_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"📖 共 {len(questions)} 道题")
    print()

    # 按类别分层抽样
    print("=" * 60)
    print("分层抽样建议标注的题目：")
    print("=" * 60)

    # 使用关键词简单分类
    categories = {
        "单表简单查询": [],
        "聚合计算": [],
        "排序Top-N": [],
        "跨表关联": [],
        "复杂条件": [],
    }

    for q in questions[:500]:  # 只看前 500 题
        text = q.get("question", "")
        qid = q.get("id", "")
        if any(kw in text for kw in ["多少只", "几个", "数量", "共有", "COUNT"]):
            categories["聚合计算"].append((qid, text))
        elif any(kw in text for kw in ["前", "最大", "最小", "最高", "最低", "TOP"]):
            categories["排序Top-N"].append((qid, text))
        elif any(kw in text for kw in ["和", "与", "以及", "同时"]) or \
             any(kw in text for kw in ["基金.*股票", "股票.*行业"]):
            categories["跨表关联"].append((qid, text))
        elif any(kw in text for kw in ["超过", "大于", "小于", "介于", "之间", "不低于"]):
            categories["复杂条件"].append((qid, text))
        else:
            categories["单表简单查询"].append((qid, text))

    # 从每类抽 3-5 条
    sample_targets = {
        "单表简单查询": 3,
        "聚合计算": 5,
        "排序Top-N": 5,
        "跨表关联": 5,
        "复杂条件": 4,
    }

    total_sample = 0
    for cat, count in sample_targets.items():
        pool = categories[cat]
        selected = random.sample(pool, min(count, len(pool)))
        print(f"\n📂 {cat}（共 {len(pool)} 题，建议标注 {count} 条）:")
        for qid, text in selected:
            total_sample += 1
            text_short = text[:80] + "..." if len(text) > 80 else text
            print(f"  [{qid}] {text_short}")

    print(f"\n{'='*60}")
    print(f"建议总标注量: {total_sample} 条（可扩展至 100-200 条）")
    print(f"{'='*60}")
    print()
    print("标注格式（保存到 data/processed/few_shot_examples.json）：")
    print("""
[
  {
    "question": "...",
    "sql": "SELECT ... FROM ... WHERE ...;",
    "table_names": ["table1", "table2"],
    "category": "data_query"
  }
]
    """)


if __name__ == "__main__":
    main()
