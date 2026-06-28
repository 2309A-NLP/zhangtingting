"""评测主程序 — 参考 FinQwen evaluate.py"""

from __future__ import annotations


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, float]:
    """评测预测结果"""
    from . import compute_recall, compute_f1_score

    total = len(predictions)
    if total == 0:
        return {"recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    recalls = []
    f1s = []
    correct = 0

    for pred, ref in zip(predictions, references):
        pred_answer = pred.get("answer", "")
        ref_answer = ref.get("answer", "")

        recall = compute_recall(ref_answer, pred_answer)
        f1 = compute_f1_score(ref_answer, pred_answer)

        recalls.append(recall)
        f1s.append(f1)

        if recall > 0.8:
            correct += 1

    return {
        "recall": sum(recalls) / total,
        "f1": sum(f1s) / total,
        "accuracy": correct / total,
        "total": total,
    }
