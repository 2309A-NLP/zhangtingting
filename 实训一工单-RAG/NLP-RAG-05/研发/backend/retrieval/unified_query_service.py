from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from backend.config import settings
from backend.pipeline.profiles import PROSPECTUS_PROFILES, ProspectusProfile
from backend.schemas import QueryResponse, SourceChunk
from backend.services.query_understanding import analyze_query
from backend.retrieval import quality_check as qc


@dataclass
class ProfileQueryResult:
    profile: ProspectusProfile
    answer: str
    diagnostics: dict[str, Any]
    text_rows: list[dict[str, Any]]
    visual_rows: list[dict[str, Any]]
    table_rows: list[dict[str, Any]]
    top_score: float


_COMPANY_SUFFIX_RE = re.compile(
    r"(股份有限公司|有限责任公司|集团股份有限公司|集团有限公司)$"
)
_ROUTING_CLEAN_RE = re.compile(r"[\s,，。；;：:（）()、/\\\-]+")


@contextmanager
def apply_profile_settings(profile: ProspectusProfile) -> Iterator[None]:
    original = {
        "collection_name": settings.collection_name,
        "text_vector_collection_name": settings.text_vector_collection_name,
        "visual_vector_collection_name": settings.visual_vector_collection_name,
        "mongodb_table_collection": settings.mongodb_table_collection,
    }
    settings.collection_name = profile.collection_name
    settings.text_vector_collection_name = profile.text_collection_name
    settings.visual_vector_collection_name = profile.visual_collection_name
    settings.mongodb_table_collection = profile.mongo_collection_name
    try:
        yield
    finally:
        settings.collection_name = original["collection_name"]
        settings.text_vector_collection_name = original["text_vector_collection_name"]
        settings.visual_vector_collection_name = original["visual_vector_collection_name"]
        settings.mongodb_table_collection = original["mongodb_table_collection"]


class UnifiedDefaultQueryService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.profiles = list(PROSPECTUS_PROFILES.values())

    def _normalize_for_routing(self, text: str) -> str:
        normalized = _ROUTING_CLEAN_RE.sub("", str(text or "").strip())
        return _COMPANY_SUFFIX_RE.sub("", normalized)

    def _profile_route_terms(self, profile: ProspectusProfile) -> list[str]:
        ordered_terms: list[str] = []
        seen: set[str] = set()
        for alias in profile.company_aliases:
            candidates = [str(alias or "").strip(), self._normalize_for_routing(alias)]
            for candidate in candidates:
                candidate = candidate.strip()
                if len(candidate) < 2 or candidate in seen:
                    continue
                seen.add(candidate)
                ordered_terms.append(candidate)
        ordered_terms.sort(key=len, reverse=True)
        return ordered_terms

    def _score_profile_route_match(self, query: str, profile: ProspectusProfile) -> tuple[int, int]:
        raw_query = str(query or "").strip()
        normalized_query = self._normalize_for_routing(raw_query)
        best_score = 0
        best_length = 0
        for term in self._profile_route_terms(profile):
            if term in raw_query:
                score = 300 + len(term)
            elif term in normalized_query:
                score = 260 + len(term)
            elif len(term) >= 4 and normalized_query.startswith(term):
                score = 220 + len(term)
            elif len(term) >= 5 and term.startswith(normalized_query):
                score = 180 + len(normalized_query)
            else:
                continue
            if score > best_score:
                best_score = score
                best_length = len(term)
        return best_score, best_length

    def _detect_candidate_profiles(self, query: str) -> list[ProspectusProfile]:
        normalized = (query or "").strip()
        route_scores: list[tuple[int, int, ProspectusProfile]] = []
        for profile in self.profiles:
            score, term_length = self._score_profile_route_match(normalized, profile)
            if score > 0:
                route_scores.append((score, term_length, profile))
        if route_scores:
            route_scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
            top_score, top_length, top_profile = route_scores[0]
            if len(route_scores) == 1:
                return [top_profile]
            second_score, second_length, _ = route_scores[1]
            if top_score > second_score or top_length > second_length:
                return [top_profile]
        intent = analyze_query(normalized)
        target_company = str(intent.target_company or "").strip()
        if target_company:
            normalized_target = self._normalize_for_routing(target_company)
            for profile in self.profiles:
                for alias in self._profile_route_terms(profile):
                    if alias in target_company or alias in normalized_target:
                        return [profile]
                    if len(normalized_target) >= 4 and alias.startswith(normalized_target):
                        return [profile]
                    if len(alias) >= 4 and normalized_target.startswith(alias):
                        return [profile]
                if any(alias and (alias in target_company or target_company in alias) for alias in profile.company_aliases):
                    return [profile]
        return list(self.profiles)

    def _run_profile_query(
        self,
        profile: ProspectusProfile,
        query: str,
        top_k: int,
        use_llm: bool,
    ) -> ProfileQueryResult:
        artifact_dir = profile.default_artifact_dir(self.project_root)
        with apply_profile_settings(profile):
            table_lookup = qc.build_table_lookup(artifact_dir)
            visual_lookup = qc.build_visual_lookup(artifact_dir)
            total_pages = qc.infer_total_pages_from_chunks(qc.load_text_chunk_lookup(artifact_dir))
            text_rows = qc.search_text_vectors(
                query,
                artifact_dir,
                table_lookup,
                visual_lookup,
                text_top_k=max(top_k, 5),
                text_candidate_k=max(top_k * 4, 20),
                disable_rerank=False,
            )
            visual_rows = qc.rerank_visual_rows_by_page_position(
                qc.search_visual_vectors(query, visual_top_k=max(3, top_k)),
                total_pages,
            )
            linked_marker_candidates = qc.extract_linked_marker_candidates_v2(text_rows)
            linked_table_rows = qc.build_linked_table_results_v2(linked_marker_candidates, table_lookup, query)
            keyword_table_rows = qc.search_tables_keyword(query, artifact_dir, table_top_k=max(5, top_k))

            merged_table_rows: list[dict[str, Any]] = []
            seen_table_ids: set[str] = set()
            for row in [*linked_table_rows, *keyword_table_rows]:
                table_id = str(row.get("table_id") or "")
                if table_id in seen_table_ids:
                    continue
                seen_table_ids.add(table_id)
                merged_table_rows.append(row)
            merged_table_rows.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            table_rows = merged_table_rows[: max(1, top_k)]

            if use_llm:
                llm_answer, llm_error, llm_diagnostics = qc.generate_llm_answer_with_diagnostics(
                    query=query,
                    text_rows=text_rows,
                    table_rows=table_rows,
                    visual_rows=visual_rows,
                    table_lookup=table_lookup,
                    visual_lookup=visual_lookup,
                    limit=max(top_k, 6),
                )
                answer = llm_answer or (f"Unavailable: {llm_error}" if llm_error else "Unavailable: LLM returned no response.")
                diagnostics = llm_diagnostics or {}
            else:
                diagnostics = {
                    "contexts": [],
                    "answer_source": "no_llm",
                }
                answer = "\n".join(str(item.get("preview") or "") for item in text_rows[:top_k]).strip()

        top_score = 0.0
        if text_rows:
            top_score = max(top_score, float(text_rows[0].get("score") or 0.0))
        if visual_rows:
            top_score = max(top_score, float(visual_rows[0].get("score") or 0.0))
        if table_rows:
            top_score = max(top_score, float(table_rows[0].get("score") or 0.0))
        return ProfileQueryResult(
            profile=profile,
            answer=answer,
            diagnostics=diagnostics,
            text_rows=text_rows,
            visual_rows=visual_rows,
            table_rows=table_rows,
            top_score=top_score,
        )

    def _choose_best_result(self, results: list[ProfileQueryResult]) -> ProfileQueryResult:
        ranked = sorted(
            results,
            key=lambda item: (
                item.top_score,
                len(list(item.diagnostics.get("contexts") or [])),
                0 if "未检索到足够证据" in item.answer else 1,
            ),
            reverse=True,
        )
        return ranked[0]

    def ask(self, query: str, top_k: int | None = None, use_llm: bool = True) -> QueryResponse:
        started = time.perf_counter()
        effective_top_k = top_k or settings.top_k
        intent = analyze_query(query)
        profiles = self._detect_candidate_profiles(query)
        results = [
            self._run_profile_query(profile=profile, query=query, top_k=effective_top_k, use_llm=use_llm)
            for profile in profiles
        ]
        best = self._choose_best_result(results)
        contexts = list(best.diagnostics.get("contexts") or [])
        citations = [
            SourceChunk(
                chunk_id=str(
                    item.get("chunk_id")
                    or item.get("table_id")
                    or item.get("visual_id")
                    or item.get("metadata", {}).get("chunk_id")
                    or item.get("metadata", {}).get("table_id")
                    or item.get("metadata", {}).get("visual_id")
                    or f"{best.profile.profile_id}_ctx_{index + 1}"
                ),
                page_number=int(item.get("page_number") or 0),
                logical_page=str(item.get("logical_page") or "") or None,
                score=float(item.get("score") or best.top_score or 0.0),
                text=str(item.get("text") or ""),
                metadata={
                    "profile": best.profile.profile_id,
                    "doc_name": best.profile.doc_name,
                    **dict(item.get("metadata") or {}),
                },
            )
            for index, item in enumerate(contexts[:effective_top_k])
        ]
        latency_ms = int((time.perf_counter() - started) * 1000)
        retrieval_mode = f"unified_quality_check[{','.join(profile.profile_id for profile in profiles)}]"
        grounded = "未检索到足够证据" not in best.answer and "Unavailable:" not in best.answer
        return QueryResponse(
            answer=best.answer,
            citations=citations,
            intent=intent,
            latency_ms=latency_ms,
            retrieval_mode=retrieval_mode,
            grounded=grounded,
            rewritten_query=query,
            resolved_company=str(intent.target_company or ""),
            resolved_profile=best.profile.profile_id,
            used_history=False,
        )
