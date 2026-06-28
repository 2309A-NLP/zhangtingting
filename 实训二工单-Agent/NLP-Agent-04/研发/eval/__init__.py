"""评测工具 — 参考 FinQwen eval 框架"""

from __future__ import annotations

from typing import Any


def compute_recall(reference: str, prediction: str) -> float:
    """计算关键词召回率"""
    ref_keywords = set(reference.split())
    pred_keywords = set(prediction.split())
    if not ref_keywords:
        return 1.0
    return len(ref_keywords & pred_keywords) / len(ref_keywords)


def compute_f1_score(reference: str, prediction: str) -> float:
    """简单 F1（基于字级别）"""
    ref_chars = set(reference)
    pred_chars = set(prediction)
    if not ref_chars:
        return 1.0

    intersection = len(ref_chars & pred_chars)
    if intersection == 0:
        return 0.0

    precision = intersection / len(pred_chars) if pred_chars else 0
    recall = intersection / len(ref_chars) if ref_chars else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
