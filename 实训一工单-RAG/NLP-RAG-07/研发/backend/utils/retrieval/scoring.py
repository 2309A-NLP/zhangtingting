# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化

from __future__ import annotations

import math
import re
from collections import Counter

from backend.utils.retrieval.constants import COMPANY_RE, PHRASE_HINTS


def build_query_tokens(query: str) -> list[str]:
    import backend.utils.retrieval.text_normalize as tn
    normalized = tn.normalize_text(query)
    tokens: list[str] = []
    CJK = re.compile(r"[\u4e00-\u9fff]{2,}")
    LATIN = re.compile(r"[A-Za-z0-9_.%-]{2,}")
    for token in CJK.findall(normalized):
        tokens.append(token)
        if len(token) >= 4:
            for size in (4, 3, 2):
                if len(token) >= size:
                    for index in range(0, len(token) - size + 1):
                        tokens.append(token[index : index + size])
    tokens.extend(LATIN.findall(normalized))
    for token in re.split(r"[\s，。；：,.!??、""]\'()（）/]+", normalized):
        token = token.strip()
        if len(token) >= 2:
            tokens.append(token)
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def keyword_score(query: str, text: str) -> float:
    import backend.utils.retrieval.text_normalize as tn
    query_tokens = build_query_tokens(query)
    haystack = tn.normalize_text(text)
    if not haystack:
        return 0.0
    hits = 0.0
    for token in query_tokens:
        if token and token in haystack:
            hits += max(1.0, len(token) / 2.0)
    return hits


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = {t for t in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{1,}", query) if t.strip()}
    text_terms = {t for t in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{1,}", text) if t.strip()}
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / max(1, len(query_terms))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = norm_a = norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


def normalize_score_map(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    mn, mx = min(values), max(values)
    if math.isclose(mn, mx):
        return {k: 0.0 if mx <= 0 else 1.0 for k in score_map}
    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


class SimpleBM25Index:
    def __init__(self, corpus_tokens: list[list[str]], k1: float, b: float) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(t) for t in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.term_frequencies = [Counter(t) for t in corpus_tokens]
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        doc_freq: Counter[str] = Counter()
        for tokens in self.corpus_tokens:
            doc_freq.update(set(tokens))
        n = len(self.corpus_tokens)
        return {
            token: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def score(self, query_tokens: list[str]) -> list[float]:
        if not query_tokens or not self.corpus_tokens:
            return [0.0] * len(self.corpus_tokens)
        qtf = Counter(query_tokens)
        scores: list[float] = [0.0] * len(self.corpus_tokens)
        for idx, doc_tf in enumerate(self.term_frequencies):
            dl = self.doc_lengths[idx] or 1
            denom = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
            total = 0.0
            for token, q in qtf.items():
                tf = doc_tf.get(token, 0)
                if tf <= 0:
                    continue
                idf = self.idf.get(token, 0.0)
                total += idf * (tf * (self.k1 + 1)) / max(tf + denom, 1e-6) * q
            scores[idx] = total
        return scores


def extract_company_aliases(query: str) -> list[str]:
    import backend.utils.retrieval.text_normalize as tn
    sanitized = tn.strip_company_query_prefixes(query)
    match = COMPANY_RE.search(sanitized)
    if not match:
        return []
    company = match.group(1).strip()
    aliases = [company]
    for suffix in ("股份有限公司", "有限责任公司", "集团股份有限公司", "集团有限公司"):
        if company.endswith(suffix):
            short = company[:-len(suffix)].strip()
            if short:
                aliases.append(short)
            break
    return aliases


def extract_focus_terms(query: str) -> list[str]:
    import backend.utils.retrieval.text_normalize as tn
    from backend.utils.retrieval.constants import GENERIC_STOPWORDS
    normalized = tn.normalize_text(query)
    for alias in extract_company_aliases(query):
        normalized = normalized.replace(alias, " ")
    candidates: list[str] = []
    for phrase in PHRASE_HINTS:
        if phrase in normalized:
            candidates.append(phrase)
    for token in re.findall(r"[\u4e00-\u9fff]{2,10}|[A-Za-z0-9]{2,}", normalized):
        token = token.strip()
        if not token or token in GENERIC_STOPWORDS:
            continue
        if token.endswith("有限公司") or token.endswith("股份有限公司"):
            continue
        candidates.append(token)
    years = re.findall(r"\d{4}年|\d{4}", normalized)
    candidates.extend(years)
    seen: set[str] = set()
    ordered: list[str] = []
    for token in candidates:
        if len(token) > 1 and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def compute_focus_signal(query: str, text: str) -> tuple[float, dict[str, object]]:
    import backend.utils.retrieval.text_normalize as tn
    haystack = tn.normalize_text(text)
    if not haystack:
        return -0.12, {"focus_terms": [], "hits": [], "penalty": True}
    focus_terms = extract_focus_terms(query)
    if not focus_terms:
        return 0.0, {"focus_terms": [], "hits": [], "penalty": False}
    hits = [t for t in focus_terms if t in haystack]
    ratio = len(hits) / max(1, len(focus_terms))
    score = min(0.16, 0.05 + 0.035 * len(hits)) if hits else -0.14
    numeric_ctx = re.search(r"\d[\d,]*(?:\.\d+)?(?:%|万元|亿元|元|万股|股)?", haystack)
    query_normalized = tn.normalize_text(query)
    if numeric_ctx and any(t in query_normalized for t in ["多少", "金额", "占比", "比例", "收入", "股数", "注册资本"]):
        score += 0.03
    return score, {"focus_terms": focus_terms, "hits": hits, "hit_ratio": ratio, "penalty": not bool(hits)}


def compute_company_signal(query: str, text: str, doc_name: str = "") -> tuple[float, dict[str, object]]:
    aliases = extract_company_aliases(query)
    if not aliases:
        return 0.0, {"aliases": [], "matched": False}
    import backend.utils.retrieval.text_normalize as tn
    haystack = tn.normalize_text(text)
    doc_haystack = tn.normalize_text(doc_name)
    matched = any(a and (a in haystack or a in doc_haystack) for a in aliases)
    if matched:
        return 0.08, {"aliases": aliases, "matched": True}
    others = COMPANY_RE.findall(haystack)
    if others:
        return -0.08, {"aliases": aliases, "matched": False, "other_companies": others[:3]}
    return -0.03, {"aliases": aliases, "matched": False}


def infer_question_type(query: str) -> str:
    import backend.utils.retrieval.text_normalize as tn
    normalized = tn.normalize_text(query)
    if any(t in normalized for t in ["组织结构图", "流程图", "结构图", "销售部", "销售处"]):
        return "org_structure"
    if any(t in normalized for t in ["增长率", "负增长", "最快", "图中可以看出", "折线图", "柱形图"]):
        return "chart_trend"
    if any(t in normalized for t in ["占比", "比例", "分别是多少", "金额", "收入", "股数"]):
        return "table_numeric"
    if any(t in normalized for t in ["哪些", "包括", "分别", "项目", "关联方"]):
        return "table_list"
    if any(t in normalized for t in ["注册资本", "法定代表人", "技术标准", "一等奖", "供应商"]):
        return "field_lookup"
    return "fact_text"


def infer_query_tags(query: str) -> list[str]:
    import backend.utils.retrieval.text_normalize as tn
    normalized = tn.normalize_text(query)
    pairs = [
        ("募集资金", "fundraising"), ("募投", "fundraising"),
        ("关联方", "related_party"), ("军用领域", "military_revenue"),
        ("技术标准", "technical_standard"), ("组织结构图", "org_chart"),
        ("增长率", "chart_trend"), ("上游", "industry_chain"), ("下游", "industry_chain"),
    ]
    tags: list[str] = []
    for keyword, tag in pairs:
        if keyword in normalized and tag not in tags:
            tags.append(tag)
    return tags


def compute_page_position_penalty(
    source_pages: list[int],
    total_pages: int,
    *,
    is_visual: bool = False,
) -> tuple[float, dict[str, object]]:
    pages = sorted({int(p) for p in source_pages if int(p) > 0})
    if not pages or total_pages <= 0:
        return 0.0, {"pages": pages, "total_pages": total_pages, "zone": "", "reason": ""}
    first_page = min(pages)
    last_page = max(pages)
    head_window = min(5, max(1, total_pages // 100 or 1))
    tail_window = max(12, min(24, total_pages // 20 if total_pages >= 20 else total_pages))
    if last_page <= head_window:
        penalty = -0.14 if not is_visual else -0.10
        zone, reason = "head", f"within_first_{head_window}_pages"
    elif first_page >= max(1, total_pages - tail_window + 1):
        penalty = -0.24 if not is_visual else -0.18
        zone, reason = "tail", f"within_last_{tail_window}_pages"
    else:
        penalty = 0.0
        zone, reason = "body", "in_body_pages"
    return penalty, {
        "pages": pages, "total_pages": total_pages,
        "zone": zone, "reason": reason,
        "head_window": head_window, "tail_window": tail_window,
    }


def compute_answer_boost(query: str, text: str) -> float:
    import backend.utils.retrieval.text_normalize as tn
    nq = tn.normalize_text(query)
    nt = tn.normalize_text(text)
    if not nt:
        return 0.0
    bonus = 0.0
    if any(t in nq for t in ["多少", "几", "比例", "占比", "金额", "收入", "股数", "注册资本"]):
        if re.search(r"\d[\d,]*(?:\.\d+)?(?:%|万元|亿元|元|万股|股)?", nt):
            bonus += 0.05
    if any(t in nq for t in ["项目", "哪些", "包括", "分别", "关联方", "供应商"]):
        if any(t in nt for t in ["包括", "如下", "分别", "项目", "合计", "指", "为"]):
            bonus += 0.05
    if any(t in nq for t in ["图", "结构图", "流程图", "增长率", "负增长", "最快"]):
        if any(t in nt for t in ["增长率", "最快", "负增长", "流程", "结构", "部门", "销售处"]):
            bonus += 0.04
    return min(0.12, bonus)
