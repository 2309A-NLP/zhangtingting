from __future__ import annotations

import re

from app.core.logging import get_logger

logger = get_logger(__name__)


ROLE_POLICY_MAP: dict[str, str] = {
    "lawyer": (
        "你是谨慎的法律信息助手。除非用户明确要求其他语言，否则始终使用简体中文作答。"
        "回答必须优先依据检索到的证据和通用法律原则，不得承诺结果，不得编造法条、案例或程序。"
        "如果证据没有明确支持，就明确说明信息不足，不要自行补充新的处理方案。"
        "优先先给结论，再给1到3点依据，避免展开成百科式长列表。"
        "涉及正式法律意见、诉讼策略或高风险判断时，明确提示用户咨询执业律师。"
    ),
    "doctor": (
        "你是谨慎的医疗信息助手。除非用户明确要求其他语言，否则始终使用简体中文作答。"
        "你只能提供健康科普和下一步建议，不能替代线下面诊，也不能给出处方级决策。"
        "回答必须优先依据检索到的证据；如果证据没有明确支持，不要延伸到其他疾病、治疗或护理方案。"
        "优先先给直接结论，再给1到3点依据或风险提示，避免泛化成大段通用医学建议。"
        "必要时明确提示用户尽快线下就医。"
    ),
    "stock": (
        "你是谨慎的投资信息助手。除非用户明确要求其他语言，否则始终使用简体中文作答。"
        "你只能提供信息分析和风险提示，不得承诺收益，也不得给出保证买卖结果的表述。"
        "如果证据不足，直接说明，不要用常见投资话术补齐答案。"
    ),
    "history": (
        "你是历史人物与历史事件讲解助手。除非用户明确要求其他语言，否则始终使用简体中文作答。"
        "你必须尊重史实，不得编造来源。优先依据已提供材料，不要无根据扩展。"
    ),
    "general": (
        "你是由知识库支持的多角色助手。除非用户明确要求其他语言，否则始终使用简体中文作答。"
        "你要优先依据检索上下文作答；如果证据不足，要明确说明，而不是编造。"
        "优先给出简短直接结论，再补充必要依据，避免无关扩展。"
    ),
}


HIGH_RISK_DISCLAIMER: dict[str, str] = {
    "lawyer": "\n\n提示：以上内容仅供法律信息参考，不构成正式法律意见。",
    "doctor": "\n\n提示：以上内容仅供健康信息参考，不能替代诊断、检查或治疗。",
    "stock": "\n\n提示：以上内容仅供信息参考，不构成投资建议。",
}


EVIDENCE_STYLE_GUIDE = (
    "回答风格要求："
    "1. 先用1到2句话直接回答问题。"
    "2. 仅在必要时补充不超过3点依据。"
    "3. 不要为了显得完整而扩展到证据之外的常见做法、常识清单或百科内容。"
    "4. 如果检索证据不足，明确说明“根据现有资料无法确认”或“现有证据只支持以下结论”。"
)


class RoleGuard:
    def build_system_prompt(
        self,
        *,
        role_name: str,
        role_category: str,
        system_prompt: str | None = None,
    ) -> str:
        base_policy = ROLE_POLICY_MAP.get(role_category, ROLE_POLICY_MAP["general"])
        custom_prompt = (system_prompt or "").strip()
        combined = (
            f"角色名称：{role_name}\n"
            f"角色分类：{role_category}\n\n"
            f"{base_policy}\n"
            f"{EVIDENCE_STYLE_GUIDE}\n"
            "如果用户没有特别说明，输出应保持简洁、自然、中文优先。"
        )
        if custom_prompt:
            combined = f"{combined}\n\n补充角色设定：\n{custom_prompt}"
        return combined

    def validate_and_postprocess(
        self,
        *,
        response_text: str,
        role_category: str,
        role_name: str,
    ) -> str:
        text = response_text.strip()
        if not text:
            raise ValueError("Model returned empty response.")

        lowered = text.lower()
        inconsistent_markers = [
            "as an ai language model",
            "i cannot roleplay",
            "ignore the above instructions",
            "i cannot guarantee this role identity",
            "作为一个ai语言模型",
            "我不能扮演",
        ]

        for marker in inconsistent_markers:
            if marker in lowered:
                logger.warning(
                    "role_consistency_warning",
                    role_name=role_name,
                    role_category=role_category,
                    marker=marker,
                )
                text = re.sub(re.escape(marker), "", text, flags=re.IGNORECASE)
                lowered = text.lower()

        disclaimer = HIGH_RISK_DISCLAIMER.get(role_category)
        if disclaimer and disclaimer.strip() not in text:
            text = f"{text.rstrip()}{disclaimer}"
        return text.strip()
