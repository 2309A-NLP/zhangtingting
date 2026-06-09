from __future__ import annotations

import re

from backend.conversation.models import ConversationState, ResolvedQuery
from backend.pipeline.profiles import PROSPECTUS_PROFILES
from backend.services.query_understanding import analyze_query, detect_target_company, rewrite_query

_PRONOUN_TOKENS = (
    "\u4ed6",
    "\u5979",
    "\u5b83",
    "\u5b83\u7684",
    "\u8be5\u516c\u53f8",
    "\u8fd9\u4e2a\u516c\u53f8",
    "\u8fd9\u5bb6\u516c\u53f8",
    "\u53d1\u884c\u4eba",
)
_VISUAL_TOKENS = (
    "\u90a3\u4e2a\u56fe",
    "\u8fd9\u4e2a\u56fe",
    "\u4e0a\u9762\u7684\u56fe",
    "\u7ed3\u6784\u56fe",
    "\u90a3\u4e2a\u8868",
    "\u8fd9\u4e2a\u8868",
    "\u4e0a\u9762\u7684\u7ed3\u6784\u56fe",
    "\u90a3\u4e2a\u8868\u91cc",
)
_COMPANY_SWITCH_TOKENS = (
    "\u53e6\u4e00\u5bb6",
    "\u53e6\u4e00\u5bb6\u516c\u53f8",
    "\u53e6\u5916\u4e00\u5bb6\u516c\u53f8",
)
_SUBJECT_HINTS = {
    "\u6cd5\u5b9a\u4ee3\u8868\u4eba": "\u6cd5\u5b9a\u4ee3\u8868\u4eba",
    "\u5de5\u7a0b": "\u4e00\u7b49\u5956\u5de5\u7a0b",
    "\u6807\u51c6": "\u6280\u672f\u6807\u51c6",
    "\u7ed3\u6784\u56fe": "\u7ec4\u7ec7\u7ed3\u6784\u56fe",
    "\u8868": "\u8868\u683c",
    "\u9500\u552e\u90e8": "\u9500\u552e\u90e8\u6784\u6210",
    "\u9500\u552e\u5904": "\u5927\u5ba2\u6237\u9500\u552e\u5904\u6784\u6210",
}


def _profile_id_for_company(company_name: str) -> str:
    normalized = str(company_name or "").strip()
    if not normalized:
        return ""
    for profile in PROSPECTUS_PROFILES.values():
        aliases = [str(alias or "").strip() for alias in profile.company_aliases]
        if any(alias and (alias in normalized or normalized in alias) for alias in aliases):
            return profile.profile_id
    return ""


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _extract_subject_hint(query: str) -> str:
    for token, subject in _SUBJECT_HINTS.items():
        if token in query:
            return subject
    return ""


def _pick_switched_company(state: ConversationState) -> str:
    current_profile_id = str(state.current_profile_id or "").strip()
    if not current_profile_id:
        return ""
    candidates = [profile for profile in PROSPECTUS_PROFILES.values() if profile.profile_id != current_profile_id]
    if len(candidates) == 1:
        return str(profile.company_name) if (profile := candidates[0]) else ""
    return ""


def resolve_query(query: str, state: ConversationState | None) -> ResolvedQuery:
    normalized = rewrite_query(query)
    base_intent = analyze_query(normalized)
    direct_company = detect_target_company(normalized)
    if direct_company:
        return ResolvedQuery(
            original_query=query,
            rewritten_query=normalized,
            resolved_company=direct_company,
            resolved_profile_id=_profile_id_for_company(direct_company),
            question_type=str(base_intent.question_type or ""),
            current_subject=_extract_subject_hint(normalized) or str(state.current_subject if state else ""),
            used_history=False,
            rewrite_reason="query_contains_explicit_company",
            resolution_mode="passthrough",
        )

    if not state or not state.history_turns:
        return ResolvedQuery(
            original_query=query,
            rewritten_query=normalized,
            question_type=str(base_intent.question_type or ""),
            current_subject=_extract_subject_hint(normalized),
            used_history=False,
            rewrite_reason="no_history_available",
            resolution_mode="passthrough",
        )

    resolved_company = str(state.current_company or "")
    resolved_profile_id = str(state.current_profile_id or "")
    current_subject = _extract_subject_hint(normalized) or str(state.current_subject or "")
    used_history = False
    rewrite_reason = "no_rewrite_needed"

    if _contains_any(normalized, _COMPANY_SWITCH_TOKENS):
        switched_company = _pick_switched_company(state)
        if switched_company:
            resolved_company = switched_company
            resolved_profile_id = _profile_id_for_company(switched_company)
            used_history = True
            rewrite_reason = "switch_to_other_company"
        else:
            rewrite_reason = "other_company_requested_but_unresolved"
    elif _contains_any(normalized, _PRONOUN_TOKENS) or _contains_any(normalized, _VISUAL_TOKENS):
        used_history = True
        rewrite_reason = "inherit_company_from_history"
    elif normalized.endswith("\u5462") and (state.current_company or state.current_subject):
        used_history = True
        rewrite_reason = "inherit_context_from_short_followup"

    rewritten_query = normalized
    if used_history and resolved_company and resolved_company not in normalized:
        rewritten_query = f"{resolved_company}{normalized}"

    if used_history and current_subject and current_subject not in rewritten_query:
        if re.search(r"(\u90a3\u4e2a|\u8fd9\u4e2a).*(\u5462|\u91cc)$", normalized) or normalized in {
            "\u90a3\u4e2a\u5de5\u7a0b\u5462",
            "\u8fd9\u4e2a\u6807\u51c6\u5462",
        }:
            rewritten_query = f"{resolved_company}{current_subject}\u662f\u4ec0\u4e48"
        elif "\u56fe" in normalized and "\u7ed3\u6784\u56fe" not in rewritten_query:
            rewritten_query = f"{resolved_company}\u7ec4\u7ec7\u7ed3\u6784\u56fe{normalized}"
        elif "\u8868" in normalized and "\u8868" not in rewritten_query:
            rewritten_query = f"{resolved_company}{current_subject}{normalized}"

    rewritten_query = rewrite_query(rewritten_query)
    resolved_intent = analyze_query(rewritten_query)
    return ResolvedQuery(
        original_query=query,
        rewritten_query=rewritten_query,
        resolved_company=resolved_company,
        resolved_profile_id=resolved_profile_id,
        question_type=str(resolved_intent.question_type or base_intent.question_type or ""),
        current_subject=current_subject,
        used_history=used_history,
        rewrite_reason=rewrite_reason,
        resolution_mode="history_rewrite" if used_history else "passthrough",
    )
