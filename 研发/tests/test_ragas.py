from __future__ import annotations

import json
import os

import pytest
from datasets import Dataset

ragas = pytest.importorskip("ragas")
metrics_mod = pytest.importorskip("ragas.metrics")


def _build_eval_dataset(raw_items: list[dict]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": item["question"],
                "ground_truth": item["expected_answer"],
                "contexts": item["expected_context"],
                "answer": item["answer"],
            }
            for item in raw_items
        ]
    )


@pytest.mark.skipif(os.getenv("RUN_RAGAS_TESTS") != "true", reason="Set RUN_RAGAS_TESTS=true to run integration evaluation.")
def test_ragas_evaluation_framework(tmp_path):
    sample_items = [
        {
            "question": "我的劳动合同到期后公司不续签，能要求赔偿吗？",
            "expected_answer": "在符合法定条件时，可主张经济补偿或赔偿。",
            "expected_context": [
                "劳动合同到期后，用人单位不续签的，一般需要根据劳动者工作年限支付经济补偿。",
                "若单位违法解除劳动关系，还可能构成赔偿责任。",
            ],
            "answer": "若符合劳动合同法规定，劳动者可以主张经济补偿；若违法解除，可能还可主张赔偿。",
        },
        {
            "question": "发热超过39度应该怎么处理？",
            "expected_answer": "持续高热并伴有严重症状时应及时就医。",
            "expected_context": [
                "若体温持续超过39摄氏度、伴有呼吸困难、意识改变或严重脱水，应及时就医。",
            ],
            "answer": "若持续高热达到39度并伴有呼吸困难、意识改变等情况，应尽快就医。",
        },
    ]

    dataset = _build_eval_dataset(sample_items)
    result = ragas.evaluate(
        dataset,
        metrics=[
            metrics_mod.faithfulness,
            metrics_mod.answer_relevancy,
            metrics_mod.context_recall,
            metrics_mod.context_precision,
        ],
    )

    output_file = tmp_path / "ragas_result.json"
    output_file.write_text(json.dumps(result.to_pandas().to_dict(), ensure_ascii=False), encoding="utf-8")

    summary = result.to_pandas().mean(numeric_only=True).to_dict()
    assert set(summary.keys()) >= {"faithfulness", "answer_relevancy", "context_recall", "context_precision"}
