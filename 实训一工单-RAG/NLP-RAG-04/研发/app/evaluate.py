# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

from app.services.rag_pipeline import RAGPipeline


TEST_SET = [
    ("军用领域收入？", "军用领域收入"),
    ("参与制定的技术标准？", "技术标准"),
    ("军用收入占比？", "军用收入占比"),
    ("电子信息行业上游企业？", "上游企业"),
    ("重要供应商领域？", "重要供应商"),
    ("电子信息行业下游行业？", "下游行业"),
    ("国家科技进步一等奖工程？", "一等奖工程"),
    ("注册资本？", "注册资本"),
    ("法定代表人？", "法定代表人"),
    ("补充流动资金募集金额？", "补充流动资金"),
]


def contains_keyword(answer: str, keyword: str) -> int:
    return int(keyword in answer)


def main() -> None:
    pipeline = RAGPipeline()
    pipeline.ingest(force=False)
    report_path = Path("reports") / "evaluation.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    faithfulness_scores = []
    recall_scores = []
    correctness_scores = []
    for query, keyword in TEST_SET:
        rag_response = pipeline.ask(query, use_llm=True)
        baseline = pipeline.ask(query, use_llm=False)
        faithfulness = 1 if rag_response.grounded else 0
        correctness = contains_keyword(rag_response.answer, keyword)
        recall = 1 if rag_response.citations else 0
        faithfulness_scores.append(faithfulness)
        correctness_scores.append(correctness)
        recall_scores.append(recall)
        rows.append(
            {
                "question": query,
                "rag_answer": rag_response.answer,
                "llm_baseline": baseline.answer,
                "citations": "; ".join(str(item.page_number) for item in rag_response.citations),
                "faithfulness": faithfulness,
                "correctness": correctness,
                "recall": recall,
                "latency_ms": rag_response.latency_ms,
            }
        )
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = Path("reports") / "metrics.md"
    metrics_path.write_text(
        "\n".join(
            [
                "# 指标结果",
                "",
                f"- 忠实度: {mean(faithfulness_scores):.2f}",
                f"- 正确率: {mean(correctness_scores):.2f}",
                f"- 召回率: {mean(recall_scores):.2f}",
                f"- 平均响应时间(ms): {mean([row['latency_ms'] for row in rows]):.2f}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
