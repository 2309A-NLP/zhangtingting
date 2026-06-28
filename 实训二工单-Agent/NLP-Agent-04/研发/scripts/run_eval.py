#!/usr/bin/env python3
"""
评测脚本 — 在 question.jsonl 上批量跑分
支持完整评测 + 错误分析 + 报告生成
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from config import settings


def load_questions(path: str) -> list[dict[str, Any]]:
    """加载 question.jsonl """
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="基金问答系统评测")
    parser.add_argument("--sample", "-n", type=int, default=100, help="评测样本数量，默认100")
    parser.add_argument("--full", "-f", action="store_true", help="运行完整1000题评测")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")
    parser.add_argument("--save-sql", action="store_true", help="保存SQL语句")
    args = parser.parse_args()

    # 查找 question 文件（支持 .json 和 .jsonl）
    question_files = [
        "data/raw/bs_challenge_financial_14b_dataset/question.json",
        "data/raw/bs_challenge_financial_14b_dataset/question.jsonl",
        "data/raw/question.json",
        "data/raw/question.jsonl",
        "question.json",
        "question.jsonl",
    ]
    question_path = None
    for p in question_files:
        if Path(p).exists():
            question_path = p
            break

    if not question_path:
        print("[ERROR] 未找到 question 文件，请先下载数据")
        return

    print(f"[INFO] 加载测试题: {question_path}")
    questions = load_questions(question_path)
    print(f"   共 {len(questions)} 道题目\n")

    sample_size = len(questions) if args.full else min(len(questions), args.sample)
    print(f"[INFO] 将评测前 {sample_size} 道题目...\n")

    from src.core.engine.pipeline import NL2SQLPipeline
    from src.core.models import ChatRequest
    from src.services.llm_service import LLMService
    from src.services.db_service import DatabaseService
    from src.services.cache_service import CacheService
    from src.core.retriever.few_shot import FewShotRetriever

    print("初始化 NL2SQL Pipeline...")
    pipeline = NL2SQLPipeline(
        llm_service=LLMService(),
        db_service=DatabaseService(),
        cache_service=CacheService(),
        few_shot_retriever=FewShotRetriever(),
    )

    print(f"\n{'='*60}")
    print(f"开始评测 - 模型: {settings.LLM_MODEL_NAME}")
    print(f"{'='*60}\n")

    results = []
    success_count = 0
    total_time = 0.0
    error_types: Counter = Counter()
    category_stats: dict[str, dict] = {}

    for i in range(sample_size):
        q = questions[i]
        question_text = q.get("question", q.get("query", ""))
        qid = q.get("id", i)

        start = time.perf_counter()
        try:
            result = pipeline.run(ChatRequest(question=question_text))
            elapsed = (time.perf_counter() - start) * 1000
            total_time += elapsed

            if result.success:
                success_count += 1
                status = "[OK]"
            else:
                status = "[FAIL]"
                # 统计错误类型
                error_msg = result.error_message or "Unknown"
                if "no such table" in error_msg.lower():
                    error_types["表名错误"] += 1
                elif "no such column" in error_msg.lower():
                    error_types["字段名错误"] += 1
                elif "syntax error" in error_msg.lower():
                    error_types["SQL语法错误"] += 1
                elif "超时" in error_msg:
                    error_types["执行超时"] += 1
                elif "空" in error_msg:
                    error_types["查询结果为空"] += 1
                else:
                    error_types["其他错误"] += 1

            print(f"  [{i+1}/{sample_size}] {status} Q{qid}: {question_text[:40]}...")
            if not result.success:
                print(f"      SQL: {result.sql[:60]}...")
                print(f"      Error: {result.error_message[:60] if result.error_message else 'None'}...")
            print(f"      耗时: {elapsed:.0f}ms")

            res_entry = {
                "id": qid,
                "question": question_text,
                "answer": result.answer,
                "success": result.success,
                "latency_ms": round(elapsed, 2),
                "category": result.category.value if result.category else "unknown",
                "tables_used": result.tables_used,
                "error_message": result.error_message,
            }
            if args.save_sql:
                res_entry["sql"] = result.sql
            results.append(res_entry)

        except Exception as e:
            print(f"  [{i+1}/{sample_size}] [ERROR] Q{qid}: 执行异常 - {e}")
            total_time += (time.perf_counter() - start) * 1000
            error_types["执行异常"] += 1
            results.append({
                "id": qid,
                "question": question_text,
                "success": False,
                "error": str(e),
                "latency_ms": 0,
            })

    # 统计
    avg_time = total_time / sample_size if sample_size else 0
    success_rate = success_count / sample_size * 100 if sample_size else 0

    print(f"\n{'='*60}")
    print(f"评测结果汇总")
    print(f"{'='*60}")
    print(f"  总题数:        {sample_size}")
    print(f"  成功:          {success_count}")
    print(f"  失败:          {sample_size - success_count}")
    print(f"  成功率:        {success_rate:.2f}%")
    print(f"  平均耗时:      {avg_time:.0f}ms")
    print(f"  模型:          {settings.LLM_MODEL_NAME}")
    print(f"{'='*60}")

    # 错误类型分析
    if error_types:
        print(f"\n[错误类型分析]")
        for error_type, count in error_types.most_common():
            pct = count / (sample_size - success_count) * 100 if sample_size > success_count else 0
            print(f"  {error_type}: {count} ({pct:.1f}%)")

    # 保存结果
    output_path = args.output or f"eval_results_{int(time.time())}.json"
    output_data = {
        "summary": {
            "total": sample_size,
            "success": success_count,
            "failed": sample_size - success_count,
            "success_rate": round(success_rate, 2),
            "avg_latency_ms": round(avg_time, 2),
            "model": settings.LLM_MODEL_NAME,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_types": dict(error_types),
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 详细结果已保存: {output_path}")

    # 生成CSV格式的简单报告
    csv_path = output_path.replace(".json", "_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("问题ID,问题摘要,成功,耗时(ms),使用表,错误信息\n")
        for r in results:
            q_preview = r["question"][:30].replace(",", ";").replace("\n", " ")
            success_str = "是" if r["success"] else "否"
            latency = r.get("latency_ms", 0)
            tables = ";".join(r.get("tables_used", []))
            error = r.get("error_message", r.get("error", "")).replace(",", ";").replace("\n", " ")[:100]
            f.write(f'{r["id"]},"{q_preview}",{success_str},{latency},"{tables}","{error}"\n')
    print(f"[OK] CSV摘要已保存: {csv_path}")


if __name__ == "__main__":
    main()
