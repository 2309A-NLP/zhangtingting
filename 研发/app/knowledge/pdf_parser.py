from __future__ import annotations
'''
这是一个多策略、可降级、带缓存的解析器。核心思想：
本地多引擎并行：同时使用 PyMuPDF、PyPDF、pdfplumber、pdfminer 提取文本，选择最佳结果。
OCR后备：当文本提取质量差时，调用 PaddleOCR 进行图像识别。
表格专项提取：使用 pdfplumber、Camelot、Tabula 多工具提取表格。
质量评估与远程降级：评估本地解析质量，若低于阈值则调用 MinerU API 重新解析。
结果缓存：基于文件内容和元数据的 SHA256 缓存解析结果，避免重复计算。
'''
import asyncio
import json
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import httpx
import numpy as np
from paddleocr import PaddleOCR
from pypdf import PdfReader

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.knowledge.models import DocumentSection, ParsedDocument, RawDocument

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
except ImportError:  # pragma: no cover - optional dependency
    extract_pages = None
    LTTextContainer = None

try:
    import camelot
except ImportError:  # pragma: no cover - optional dependency
    camelot = None

try:
    import tabula
except ImportError:  # pragma: no cover - optional dependency
    tabula = None

logger = get_logger(__name__)

HEADING_PATTERN = re.compile(
    r"^\s*("
    r"(chapter|section|appendix)\s+\S+"
    r"|"
    r"[0-9]+(\.[0-9]+)*"
    r"|"
    r"[一二三四五六七八九十百千万亿]+[、.)）]"
    r"|"
    r"第[一二三四五六七八九十百千万亿\d]+[章节条款]"
    r")",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(slots=True)
class PageParseArtifact:
    page_number: int
    text: str
    extractor: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    text_candidates: dict[str, str] = field(default_factory=dict)
    used_ocr: bool = False


class ComplexPdfParser:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logger
        self._ocr: PaddleOCR | None = None
        self._cache_dir = Path(self.settings.pdf_parse_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @log_timed("complex_pdf_parse")
    async def parse(self, raw_document: RawDocument) -> ParsedDocument:
        emit_runtime_trace(
            self.logger,
            "complex_pdf_parse_entered",
            doc_id=raw_document.file_id,
            file_name=raw_document.file_name,
            local_path=raw_document.local_path,
        )
        return await asyncio.to_thread(self._parse_sync, raw_document)

    # 同步核心逻辑 : 检查缓存 → 本地解析 → 质量评估 → (低质量则MinerU降级) → 保存缓存
    def _parse_sync(self, raw_document: RawDocument) -> ParsedDocument:
        # 缓存键生成 内容 = 文件名 + 路径 + 文件大小 + 修改时间 + 元数据
        # 确保文件内容变化时缓存失效
        cache_key = self._build_cache_key(raw_document)

        # 从磁盘读取并恢复之前解析过的文档对象
        cached_document = self._load_cache(cache_key)
        # 从磁盘读取到了
        if cached_document is not None:
            emit_runtime_trace(
                self.logger,
                "complex_pdf_parse_cache_hit",
                doc_id=raw_document.file_id,
                cache_key=cache_key,
                parser_name=cached_document.parser_name,
                parse_strategy=cached_document.metadata.get("parse_strategy", "cached"),
            )
            return cached_document

        # 没读到
        # 本地多策略解析 返回最佳文本列表
        page_artifacts = self._parse_locally(raw_document)
        # 将文本和表格进行格式输出 ， 返回str
        plain_text = "\n\n".join(
            self._build_page_text(page_artifact)
            for page_artifact in page_artifacts
            if self._build_page_text(page_artifact)
        ).strip()
        # 单独收集表格
        all_tables = [table for page in page_artifacts for table in page.tables]

        # 用camelot和tabula再次提取表格
        advanced_tables = self._extract_advanced_tables(raw_document.local_path)
        # 合并表格
        all_tables = self._merge_tables(all_tables, advanced_tables)
        # 将高级表格追加到文本末尾
        if advanced_tables:
            plain_text = self._append_advanced_tables_text(plain_text, advanced_tables)
        # 提取标题
        title = self._resolve_title(raw_document.file_name, plain_text)
        # 将解析后的页面转换为结构化的章节对象
        sections = self._build_sections(page_artifacts, plain_text)
        # 记录和标识文档解析过程中使用了哪些提取工具
        parser_name = self._build_parser_name(page_artifacts, advanced_tables)

        local_document = ParsedDocument(
            doc_id=raw_document.file_id,
            user_id=raw_document.user_id,
            role_id=raw_document.role_id,
            title=title,
            plain_text=plain_text,
            source_uri=raw_document.source_uri,
            file_name=raw_document.file_name,
            content_type=raw_document.content_type,
            parser_name=parser_name,
            sections=sections,
            tables=all_tables,
            metadata={
                "source_type": raw_document.source_type,
                **raw_document.metadata,
                "page_count": len(page_artifacts),
                "page_artifacts": [
                    {
                        "page_number": artifact.page_number,
                        "extractor": artifact.extractor,
                        "used_ocr": artifact.used_ocr,
                        "table_count": len(artifact.tables),
                        "text_preview": preview_text(artifact.text, 120),
                    }
                    for artifact in page_artifacts
                ],
                "table_extractors": sorted(
                    {
                        str(table.get("extractor", "unknown"))
                        for table in all_tables
                    }
                ),
                "parse_strategy": "local",
            },
        )
        # 评估本地提取质量
        quality_report = self._evaluate_local_quality(local_document)
        # 提取得分 记录日志
        local_document.metadata["parse_quality_score"] = quality_report["quality_score"]
        local_document.metadata["parse_quality_reasons"] = quality_report["reasons"]
        emit_runtime_trace(
            self.logger,
            "complex_pdf_parse_local_quality_scored",
            doc_id=raw_document.file_id,
            quality_score=quality_report["quality_score"],
            should_fallback=quality_report["should_fallback"],
            reasons=quality_report["reasons"],
            avg_chars_per_page=quality_report["avg_chars_per_page"],
            ocr_ratio=quality_report["ocr_ratio"],
        )
        # 如果需要回退
        if quality_report["should_fallback"]:
            mineru_result = self._try_parse_with_mineru_api(
                raw_document,
                fallback_reason=", ".join(quality_report["reasons"]) or "low_quality",
                local_quality_report=quality_report,
                cache_key=cache_key,
            )
            if mineru_result is not None:
                self._save_cache(cache_key, mineru_result)
                return mineru_result

        # 不需要回退
        emit_runtime_trace(
            self.logger,
            "complex_pdf_parse_completed",
            doc_id=raw_document.file_id,
            parser_name=parser_name,
            page_count=len(page_artifacts),
            table_count=len(all_tables),
            text_preview=preview_text(plain_text, 160),
        )
        # 缓存
        self._save_cache(cache_key, local_document)
        return local_document

    def _try_parse_with_mineru_api(
            self,
            raw_document: RawDocument,  # 原始文档信息
            *,  # 强制关键字参数
            fallback_reason: str,  # 回退原因（如"low_quality_score"）
            local_quality_report: dict[str, Any],  # 本地质量评估报告
            cache_key: str,  # 缓存键
    ) -> ParsedDocument | None:  # 返回解析结果或None
        if not self.settings.mineru_api_enabled or not self.settings.mineru_api_base_url.strip():
            return None

        url = f"{self.settings.mineru_api_base_url.rstrip('/')}{self.settings.mineru_api_parse_path}"
        headers: dict[str, str] = {}
        if self.settings.mineru_api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.mineru_api_key.strip()}"

        emit_runtime_trace(
            self.logger,
            "complex_pdf_parse_mineru_started",
            url=url,
            file_name=raw_document.file_name,
            fallback_reason=fallback_reason,
            local_quality_score=local_quality_report["quality_score"],
            cache_key=cache_key,
        )
        try:
            with open(raw_document.local_path, "rb") as handle:
                response = httpx.post(
                    url,
                    headers=headers,
                    files={
                        "file": (
                            raw_document.file_name,
                            handle,
                            raw_document.content_type or "application/pdf",
                        )
                    },
                    data={"return_format": "json"},
                    timeout=self.settings.mineru_api_timeout_seconds,
                )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - network/integration path
            emit_runtime_trace(
                self.logger,
                "complex_pdf_parse_mineru_failed",
                error=str(exc),
                fallback_reason=fallback_reason,
                cache_key=cache_key,
            )
            return None

        text = self._extract_first_text(
            payload,
            [
                ("data", "markdown"),
                ("data", "text"),
                ("data", "content"),
                ("result", "markdown"),
                ("result", "text"),
                ("markdown",),
                ("text",),
                ("content",),
            ],
        )
        if not text:
            emit_runtime_trace(
                self.logger,
                "complex_pdf_parse_mineru_empty",
                payload_keys=list(payload.keys())[:10],
                fallback_reason=fallback_reason,
                cache_key=cache_key,
            )
            return None

        pages = self._normalize_mineru_pages(payload)
        sections = self._build_mineru_sections(pages, text)
        tables = self._normalize_mineru_tables(payload)
        title = self._resolve_title(raw_document.file_name, text)
        emit_runtime_trace(
            self.logger,
            "complex_pdf_parse_mineru_finished",
            section_count=len(sections),
            table_count=len(tables),
            text_preview=preview_text(text, 160),
            fallback_reason=fallback_reason,
            cache_key=cache_key,
        )
        return ParsedDocument(
            doc_id=raw_document.file_id,
            user_id=raw_document.user_id,
            role_id=raw_document.role_id,
            title=title,
            plain_text=text,
            source_uri=raw_document.source_uri,
            file_name=raw_document.file_name,
            content_type=raw_document.content_type,
            parser_name="mineru_api",
            sections=sections,
            tables=tables,
            metadata={
                "source_type": raw_document.source_type,
                **raw_document.metadata,
                "mineru_api_used": True,
                "parse_strategy": "mineru_api_fallback",
                "fallback_reason": fallback_reason,
                "local_quality_report": local_quality_report,
                "parse_quality_score": local_quality_report["quality_score"],
                "parse_quality_reasons": local_quality_report["reasons"],
                "cache_key": cache_key,
                "mineru_payload_keys": list(payload.keys())[:20],
            },
        )

    # 文档质量评估函数
    def _evaluate_local_quality(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        # 输入：本地解析的文档
        # 输出：质量评估结果（包含评分、回退建议等）

        # 1. 获取基础信息
        page_count = int(parsed_document.metadata.get("page_count") or max(len(parsed_document.sections), 1))
        text_chars = len(parsed_document.plain_text.strip())
        avg_chars_per_page = round(text_chars / max(page_count, 1), 2)
        # 2. 统计OCR使用情况
        page_artifacts = parsed_document.metadata.get("page_artifacts") or []
        if isinstance(page_artifacts, list):
            ocr_pages = sum(1 for item in page_artifacts if isinstance(item, dict) and item.get("used_ocr"))
        else:
            ocr_pages = 0
        ocr_ratio = round(ocr_pages / max(page_count, 1), 2)
        # 3. 统计表格和提取器
        table_count = len(parsed_document.tables)
        extractors = {
            str(item.get("extractor", ""))
            for item in page_artifacts
            if isinstance(item, dict) and item.get("extractor")
        }
        reasons: list[str] = []
        # 4. 质量评分计算（满分100分）
        # 基础分：文本密度（最高50分）
        # 每页160字符为满分（50分）
        # avg_chars_per_page = 0    → 0分
        # avg_chars_per_page = 80   → 25分
        # avg_chars_per_page = 160  → 50分
        # avg_chars_per_page = 320  → 50分（封顶）
        quality_score = 0.0
        if text_chars == 0:
            reasons.append("empty_text")
        quality_score += min(avg_chars_per_page / 160.0, 1.0) * 50.0
        # 加分：表格数量（最高15分）
        # table_count = 0  → 0分
        # table_count = 1  → 5分
        # table_count = 2  → 10分
        # table_count = 3+ → 15分（封顶）
        quality_score += min(table_count * 5.0, 15.0)
        # 加分：使用高质量提取器（10分）
        if extractors & {"pymupdf_layout", "pdfplumber", "pdfminer"}:
            quality_score += 10.0
        # 减分：OCR比例过高（-20分）
        if ocr_ratio > 0.5:
            quality_score -= 20.0
            reasons.append("high_ocr_ratio")
        # 减分：文本密度过低（-15分）  少于80
        if avg_chars_per_page < self.settings.pdf_text_min_chars_per_page:
            quality_score -= 15.0
            reasons.append("low_text_density")
        # 减分：总文本不足（-10分）  80
        if page_count >= 3 and text_chars < page_count * self.settings.pdf_text_min_chars_per_page:
            quality_score -= 10.0
            reasons.append("insufficient_total_text")
        # 有解析器但无表格（-5分）
        if table_count == 0 and page_count >= 2 and any(token in extractors for token in {"pymupdf_layout", "pdfplumber", "pdfminer"}):
            quality_score -= 5.0
        # 最终评分处理  限制在 0-100 分之间
        quality_score = round(max(0.0, min(100.0, quality_score)), 2)
        # 回退条件（三个条件都满足）：
        # MinerU API 已启用
        # API 地址已配置
        # 文本为空 或 质量分数低于阈值 45
        should_fallback = (
            self.settings.mineru_api_enabled
            and bool(self.settings.mineru_api_base_url.strip())
            and (
                text_chars == 0
                or quality_score < self.settings.mineru_fallback_quality_threshold
            )
        )

        if should_fallback and "mineru_fallback" not in reasons:
            reasons.append("mineru_fallback")

        return {
            "page_count": page_count,
            "text_chars": text_chars,
            "avg_chars_per_page": avg_chars_per_page,
            "ocr_pages": ocr_pages,
            "ocr_ratio": ocr_ratio,
            "table_count": table_count,
            "quality_score": quality_score,
            "should_fallback": should_fallback,
            "reasons": reasons,
        }

    # 缓存键生成器
    def _build_cache_key(self, raw_document: RawDocument) -> str:
        # 创建一个SHA-256哈希算法对象 类型是哈希 _hashlib.HASH（CPython 实现中）
        # SHA-256会产生64位十六进制字符串（32字节）
        # 特点：确定性（相同输入→相同输出）、不可逆
        hasher = hashlib.sha256()

        '''这些属性定义了文档的身份标识，任何不同都会导致不同缓存键。'''
        # ---添加文件名---
        # raw_document.file_name：获取文件名，如 "report.pdf"
        # .encode("utf-8")：将字符串转为字节串（哈希算法需要字节输入）
        # hasher.update()：将字节数据喂给哈希计算器
        # 作用：不同文件名产生不同缓存键
        hasher.update(raw_document.file_name.encode("utf-8"))
        # ---添加内容类型---
        # content_type：MIME类型，如 "application/pdf"
        # 区分不同类型文档（即使文件名相同）
        # 例：同名"data"可能是json、csv、pdf，需要区分
        hasher.update(raw_document.content_type.encode("utf-8"))
        # ---添加源URI---
        # source_uri：文档来源，如 "s3://bucket/file.pdf" 或 "https://example.com/file.pdf"
        # 标识文档从哪里来
        # 同一文件不同来源应区分缓存
        hasher.update(raw_document.source_uri.encode("utf-8"))
        # ---添加本地路径---
        # local_path：本地文件系统路径，如 "/data/cache/report.pdf"
        # 记录文件物理位置
        hasher.update(raw_document.local_path.encode("utf-8"))
        try:
            '''
            核心思想：
            文件存在 → 用文件大小+修改时间区分不同版本
            文件不存在 → 只用元信息（文件名、路径等）作为缓存键
            设计选择：
            FileNotFoundError：常见、可恢复（文件可能延迟生成）
            PermissionError：不常见、通常表示配置错误，应该暴露出来
            
            # ⚠️ 这些操作可能失败（系统调用）
            stat = Path(raw_document.local_path).stat()  # ← 文件可能不存在、权限不足
            hasher.update(str(stat.st_size).encode("utf-8"))   # 依赖上面的stat成功
            hasher.update(str(int(stat.st_mtime)).encode("utf-8")) # 依赖上面的stat成功
            '''
            # ---获取文件统计信息---
            # Path(...)：将字符串路径转为pathlib.Path对象
            # .stat()：调用系统调用获取文件元数据
            # 返回包含：大小、权限、时间戳等
            # 可能抛异常：FileNotFoundError、PermissionError
            stat = Path(raw_document.local_path).stat()
            # ---添加文件大小---
            # stat.st_size：文件字节数，整数类型
            # str(...)：转为字符串，如 "1024000"
            # .encode("utf-8")：转为字节
            # 作用：检测文件内容长度是否变化
            # 注意：大小相同内容可能不同（碰撞可能性）
            hasher.update(str(stat.st_size).encode("utf-8"))
            # ---添加修改时间---
            # stat.st_mtime：最后修改时间，浮点数（Unix时间戳，带小数）
            # int(...)：截断小数部分，转为整数
            # 原因：浮点数精度在不同系统/语言可能不一致
            # str(...)：转为字符串
            # 作用：时间戳变化表示文件被修改过
            # 缺陷：1秒内的修改检测不到（但大多数场景够用）
            hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
        except FileNotFoundError:
            pass
        # ---序列化元数据---
        # raw_document.metadata：字典对象，存储额外信息
        # json.dumps()：将Python对象转为JSON字符串
        # ensure_ascii=False：保留中文等非ASCII字符（不转义为\uXXXX）
        # sort_keys=True：对字典键排序（关键！保证顺序一致）
        # default=str：遇到不可序列化对象时调用str()转换
        # 例：datetime对象会变成字符串 "2024-01-01 12:00:00"
        metadata_blob = json.dumps(raw_document.metadata, ensure_ascii=False, sort_keys=True, default=str)
        # ---添加元数据到哈希---
        hasher.update(metadata_blob.encode("utf-8"))
        # 返回64字符的十六进制--字符串--
            # hasher.digest()      # 返回32个字节
            # SHA-256 产生 256 位（bit）的哈希值：
            # 256 bits = 32 bytes    (因为 1 byte = 8 bits)
            # 32 bytes = 64 个十六进制字符  (因为 1 byte 用 2 个十六进制字符表示)
        return hasher.hexdigest()


    def _cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.json"

    # 缓存加载逻辑 ：根据缓存键从磁盘读取并恢复之前解析过的文档对象
    def _load_cache(self, cache_key: str) -> ParsedDocument | None:
        # 返回 None 表示缓存未命中或加载失败

        # ---构建缓存文件路径---
        path = self._cache_path(cache_key)

        # 注意：这里没有检查文件是否可读，假设存在即可读
        # 文件可能存在，但是：
        # - 权限不足（PermissionError）
        # - 被其他进程锁定（OSError）
        # - 是目录而非文件（IsADirectoryError）
        # - 损坏的符号链接
        if not path.exists():
            # 主要危险：TOCTOU (Time-Of-Check to Time-Of-Use) 竞态条件  检查和操作之间有间隙
            # 优点：避免每次都尝试读取不存在的文件（但 try-except 也很快）
            return None
        # 改进写法
        '''
        def _load_cache(self, cache_key: str) -> ParsedDocument | None:
        path = self._cache_path(cache_key)
        # 同时解决 TOCTOU 和可读性问题
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except FileNotFoundError:
            return None  # 文件不存在
        except PermissionError:
            return None  # 文件存在但不可读
        except IsADirectoryError:
            return None  # 路径是目录
        except Exception as e:
            # 其他读取错误
            emit_runtime_trace(...)
            return None
        # 反序列化...
        '''

        # 加载和解析缓存
        try:
            # path.read_text(encoding="utf-8")：读取整个文件为字符串
            # 注意：大文件可能有性能问题
            # 改进写法
            """
            # 改进1：直接传文件对象（推荐）
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)  # 更高效，内存更好
            # 改进2：读字节（避免双重解码）
            with open(path, 'rb') as f:
                payload = json.load(f)  # json.load 可以直接读 bytes
            # 改进3：流式解析（超大文件）
            import ijson
            with open(path, 'rb') as f:
                parser = ijson.parse(f)
                payload = ijson.util.parse_as_dict(parser)
            # 改进4：使用 orjson（更快的JSON库）
            import orjson
            payload = orjson.loads(path.read_bytes())
            """
            # json.loads(...)：将JSON字符串解析为Python对象（dict/list）
            # 等价于：
            # with open(path, 'r', encoding='utf-8') as f:
            #     content = f.read()  # 一次性读取整个文件到内存
            # payload = json.loads(content)
            payload = json.loads(path.read_text(encoding="utf-8"))
            # 将JSON对象转回 ParsedDocument 对象
            # 用 JSON 是因为安全、可读、通用，而 pickle 虽然方便但有严重安全风险和版本兼容问题。
            # 在缓存这种需要长期存储、可能被不同服务读取的场景，JSON 是更稳妥的选择。
            return self._deserialize_parsed_document(payload)
        # 捕获了 所有异常 (Exception)
        # 可能的异常：
        # json.JSONDecodeError：JSON格式损坏
        # FileNotFoundError：文件在检查和读取之间被删除（竞争条件）
        # PermissionError：权限问题
        # KeyError：反序列化时缺少必要字段
        # UnicodeDecodeError：编码问题
        except Exception as exc:
            emit_runtime_trace(
                self.logger,
                "complex_pdf_parse_cache_load_failed",
                cache_key=cache_key,
                error=str(exc),
            )
            return None

    # 缓存保存函数
    def _save_cache(self, cache_key: str, parsed_document: ParsedDocument) -> None:
        path = self._cache_path(cache_key)
        payload = self._serialize_parsed_document(parsed_document)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        emit_runtime_trace(
            self.logger,
            "complex_pdf_parse_cache_saved",
            cache_key=cache_key,
            parser_name=parsed_document.parser_name,
            path=str(path),
            parse_strategy=parsed_document.metadata.get("parse_strategy", "unknown"),
        )

    def _serialize_parsed_document(self, document: ParsedDocument) -> dict[str, Any]:
        return {
            "doc_id": document.doc_id,
            "user_id": document.user_id,
            "role_id": document.role_id,
            "title": document.title,
            "plain_text": document.plain_text,
            "source_uri": document.source_uri,
            "file_name": document.file_name,
            "content_type": document.content_type,
            "parser_name": document.parser_name,
            "sections": [
                {
                    "heading": section.heading,
                    "level": section.level,
                    "content": section.content,
                }
                for section in document.sections
            ],
            "tables": document.tables,
            "metadata": document.metadata,
        }

    # 反序列化 ： 将 JSON 格式的字典数据转换回 ParsedDocument 对象
    def _deserialize_parsed_document(self, payload: dict[str, Any]) -> ParsedDocument:
        # 防御性编程：每个字段都有默认值
        # 类型转换：强制转为 str/int，避免类型错误
        # 过滤：if isinstance(item, dict) 跳过损坏的数据
        sections = [
            DocumentSection(
                heading=str(item.get("heading", "Body")),  # 默认值 "Body"
                level=int(item.get("level", 1)),  # 默认等级 1
                content=str(item.get("content", "")),  # 默认空字符串
            )
            for item in payload.get("sections", [])  # 默认空列表
            if isinstance(item, dict)  # 过滤非字典项
        ]
        return ParsedDocument(
            doc_id=str(payload.get("doc_id", "")),
            user_id=str(payload.get("user_id", "")),
            role_id=str(payload.get("role_id", "")),
            title=str(payload.get("title", "")),
            plain_text=str(payload.get("plain_text", "")),
            source_uri=str(payload.get("source_uri", "")),
            file_name=str(payload.get("file_name", "")),
            content_type=str(payload.get("content_type", "")),
            parser_name=str(payload.get("parser_name", "pdf_parser")),
            sections=sections,
            # 直接保留字典结构（未转换为对象） 节省内存和序列化成本
            tables=[table for table in payload.get("tables", []) if isinstance(table, dict)],
            # 确保返回字典类型  复制一份，避免外部修改
            metadata=dict(payload.get("metadata", {})),
        )

    # 多引擎PDF解析器
    # 核心思想：多引擎投票机制
    # 使用4个不同的PDF解析引擎，选择最好的结果
    # engines = {
    #     "pymupdf": fitz (PyMuPDF),      # 速度快，布局好
    #     "pypdf": pypdf (PyPDF2/4),      # 纯Python，兼容性好
    #     "pdfplumber": pdfplumber,       # 表格提取强
    #     "pdfminer": pdfminer.six        # 文本布局精确
    # }
    def _parse_locally(self, raw_document: RawDocument) -> list[PageParseArtifact]:

        # 库来源: PyMuPDF
        # 特点: 高性能，C语言绑定 擅长提取文本、图片、矢量图形 保留布局信息较好
        # 支持PDF转XML/HTML/JSON
        # 适用场景: 需要快速提取结构化内容、带坐标的文本、表格边界检测
        # 包默认导入  import pymupdf as fitz
        # 使用 PyMuPDF 库打开指定路径的 PDF 文件。
        # open() 是 PyMuPDF 库提供的一个类方法，作用类似于 Python 内置的 open() 函数，但专门用于打开 PDF 等文档格式。
        # 返回一个 Document 对象（文档对象，pymupdf包内自带）
        fitz_doc = fitz.open(raw_document.local_path)

        # 库来源: pypdf(社区维护的活跃分支)
        # 特点:纯Python实现，无外部依赖、功能基础但可靠：合并 / 分割 / 旋转 / 加密、文本提取能力较弱（基于操作符顺序）、对损坏PDF容错性好
        # 适用场景: 需要基础PDF操作、跨平台兼容性优先
        # 包默认导入  from pypdf import PdfReader
        # 使用 pypdf 库创建一个 PDF 读取器对象来读取指定路径的 PDF 文件。
        # () 调用这个类的构造函数，创建一个 PdfReader 的实例对象
        # PdfReader 返回的是一个读取器对象
        pypdf_reader = PdfReader(raw_document.local_path)

        # 库来源: pdfplumber（基于pdfminer.six（Python））
        # 特点: 强项：表格提取（基于垂直线 / 水平线检测） 保留字符级位置信息 能提取PDF中的曲线、矩形图形 文本提取质量优于PyPDF2
        # 条件加载原因: 重量级依赖（依赖cairo等图形库） 可能内存占用较大 通过配置项pdf_use_pdfplumber控制开关  pdfplumber is not None可能是懒加载检测
        # 返回对象类型	pdfplumber.pdf.PDF
        plumber_doc = (
            pdfplumber.open(raw_document.local_path)
            if pdfplumber is not None and self.settings.pdf_use_pdfplumber
            else None
        )

        # 自定义方法：使用 pdfminer.six 提取纯文本
        # 库来源：pdfminer.six（pdfminer的Python3分支）
        # 特点：
        #   - 基于布局分析（LAParams）提取文本
        #   - 使用 LTTextContainer 筛选文本容器，过滤图片、线条等非文本元素
        #   - 按阅读顺序逐页提取文本内容
        # 具体实现：
        #   - 检查配置开关 pdf_use_pdfminer 和依赖可用性，不可用时返回空列表
        #   - 遍历 extract_pages 生成的每一页布局（LTPage）
        #   - 遍历页面中的每个元素，通过 isinstance(element, LTTextContainer) 筛选文本容器
        #   - 提取文本内容：element.get_text().strip()，过滤空字符串
        #   - 每页用换行符 "\n" 连接所有文本块，最终返回 list[str]（每页一个字符串）
        #   - 异常处理：捕获所有异常并记录日志，返回空列表实现优雅降级
        # 注意：不保留位置信息、字体样式等，仅提取纯文本内容
        pdfminer_pages = self._extract_pdfminer_pages(raw_document.local_path)

        emit_runtime_trace(
            self.logger,
            "complex_pdf_parse_local_started",
            page_count=fitz_doc.page_count,
            # 判断pdfplumber是否成功
            use_pdfplumber=plumber_doc is not None,
            # 判断pdfminer是否成功
            use_pdfminer=bool(pdfminer_pages),
        )

        # 返回值
        page_artifacts: list[PageParseArtifact] = []
        try:
            # 遍历每一页
            for page_index in range(fitz_doc.page_count):
                # 对每一页，用所有引擎提取文本

                # 1.fitz(PyMuPDF)  排序后返回str
                fitz_page = fitz_doc.load_page(page_index)
                layout_text = self._extract_layout_text(fitz_page)
                # 2.PyPDF 提取后返回str
                pypdf_text = self._safe_page_text(pypdf_reader, page_index)
                # 3.PDFPlumber
                '''
                pdfplumber vs PyMuPDF 表格提取对比
                特性	            pdfplumber	       PyMuPDF
                表格检测	      自动检测线条网格	      需要手动识别
                准确度	     较高（专为表格设计）	较低（需要额外处理）
                性能	               较慢	           较快
                依赖	             额外安装	      内置功能
                适用场景	         结构化表格	      简单表格或纯文本
                '''
                plumber_text = ""
                page_tables: list[dict[str, Any]] = []
                if plumber_doc is not None:
                    # 获取页面内容
                    plumber_page = plumber_doc.pages[page_index]
                    # 提取文本（安全处理）
                    plumber_text = (plumber_page.extract_text() or "").strip()
                    if self.settings.pdf_extract_tables:
                        # 返回值	list[dict]	表格信息列表，每个表格一个字典
                        page_tables = self._extract_pdfplumber_tables(plumber_page, page_index + 1)
                # 4.PDFMiner
                # 取一页的内容
                pdfminer_text = pdfminer_pages[page_index] if page_index < len(pdfminer_pages) else ""

                # 一页的文本候选
                text_candidates = {
                    "pymupdf_layout": layout_text,
                    "pypdf": pypdf_text,
                    "pdfplumber": plumber_text,
                    "pdfminer": pdfminer_text,
                }

                # 选择标准：提取到的文本最长的那一种
                best_extractor, best_text = self._choose_best_text(text_candidates)

                used_ocr = False
                if self._should_use_ocr(best_text):
                    # OCR需要的是原始图像，不是文本,所以传入fitz_page
                    # 返回str
                    ocr_text = self._ocr_page(fitz_page, page_index + 1)
                    # 更新最佳提取方式
                    if len(ocr_text.strip()) > len(best_text.strip()):
                        best_text = ocr_text
                        best_extractor = f"{best_extractor}+ocr" if best_extractor else "ocr"
                    used_ocr = bool(ocr_text.strip())
                    # 添加候选文本
                    text_candidates["ocr"] = ocr_text

                # 每一页存一次
                artifact = PageParseArtifact(
                    page_number=page_index + 1,
                    text=best_text.strip(),
                    extractor=best_extractor or "unknown",
                    tables=page_tables,
                    text_candidates=text_candidates,
                    used_ocr=used_ocr,
                )
                page_artifacts.append(artifact)

                emit_runtime_trace(
                    self.logger,
                    "complex_pdf_parse_page_finished",
                    page_number=artifact.page_number,
                    extractor=artifact.extractor,
                    used_ocr=artifact.used_ocr,
                    table_count=len(artifact.tables),
                    # 预览截断
                    text_preview=preview_text(artifact.text, 120),
                )
        finally:
            # 因为之前执行过fitz_doc = fitz.open(raw_document.local_path)
            # fitz_doc 持有：
            # - 文件句柄（操作系统资源）
            # - 内存映射（可能很大）
            # - 缓存的数据结构
            # 不关闭会导致：
            # - 文件锁未释放（Windows上无法删除/移动文件）
            # - 内存泄漏（大文档占用大量内存）
            # - 达到打开文件数限制（处理多个文档时）
            fitz_doc.close()
            # pdfplumber.open(raw_document.local_path)
            # plumber_doc 持有：
            # - 底层文件句柄
            # - 解析后的页面对象缓存
            # - 可能的内存映射
            if plumber_doc is not None:
                plumber_doc.close()
            # PyPDF2 的设计：一次性加载
            # 内部实现：
            # - 打开文件，读取全部内容到内存
            # - 立即关闭文件句柄
            # - 返回内存中的对象
            # PDFMiner直接返回字符串（无对象管理）
        return page_artifacts

    # 1.fitz(PyMuPDF)  从 PDF 页面中提取文本并按照布局（位置）进行排序
    def _extract_layout_text(self, page: fitz.Page) -> str:
        # 1. 获取页面的所有文本块       返回值是 列表套元组
        blocks = page.get_text("blocks")
        '''
        这是 PyMuPDF (fitz) 库的方法，它会：
        解析 PDF 页面：PDF 内部由"内容流"组成，包含绘制文本、线条、图像等操作
        提取文本块：算法将相邻、格式相似的文本合并成"块"（block）
        返回列表：每个块是一个元组
        
        每个块的格式详解 标准格式（8个元素）
        (x0, y0, x1, y1, "文本内容", block_no, block_type, flags)
        索引	名称	         含义	                示例
        0	x0	        块左上角 x 坐标（左边缘）	72.0
        1	y0	        块左上角 y 坐标（上边缘）	90.0
        2	x1	        块右下角 x 坐标（右边缘）	200.0
        3	y1	        块右下角 y 坐标（下边缘）	110.0
        4	text	    文本内容（字符串）	        "姓名：张三"
        5	block_no	块编号（从0开始）	        0
        6	block_type	块类型（0=文本，1=图像）	0
        7	flags	    格式标志（字体、颜色等）	0
        
        PDF 坐标系统：
            原点 (0,0)：在页面左下角
            x 轴：水平向右增加
            y 轴：垂直向上增加
        '''
        # 2. 排序：先按 y 坐标（垂直位置），再按 x 坐标（水平位置）  先判断这个文字块谁在上，如果同行，再判断左右看谁在前
        # item 是 blocks 列表中的一个块元组
        # round(value, 1) 四舍五入保留 1 位小数：
        # 问题：对于多栏文本，处理出来的顺序有误
        ordered = sorted(blocks, key=lambda item: (round(float(item[1]), 1), round(float(item[0]), 1)))
        #                                 item[1] = y0 (上边缘)  ↑     item[0] = x0 (左边缘) ↑
        # 3. 提取文本内容，去除空白
        fragments = [str(block[4]).strip() for block in ordered
                     if len(block) > 4 and str(block[4]).strip()]
        # 4. 用换行符连接后返回
        return "\n".join(fragments).strip()

    # 2.PyPDF  PDF页面文本提取方法
    def _safe_page_text(self, reader: PdfReader, page_index: int) -> str:
        '''
        reader: PdfReader：pypdf 的 PDF 读取器对象
        page_index: int：要提取的页码（从0开始）
        -> str：总是返回字符串，绝不会返回 None
        '''
        # 边界检查 #1：索引越界保护
        if page_index >= len(reader.pages):
            return ""
        # 三个层次的处理
        # 第1层：reader.pages[page_index]
        # 获取指定页面的PageObject,已有边界检查保证索引有效
        # 第2层：extract_text()
        # pypdf的文本提取方法,可能返回None（某些PDF可能没有文本内容）
        # 第3层：... or ""
        # Python 的短路求值
        # text = extract_text()  # 可能返回 ""、None 或其他字符串
        # result = text or ""  # 如果 text 是假值（None/空串），则使用 ""
        # 真假值判断：
        #     extract_text()返回        text or ""结果
        #     "Hello world"            "Hello world"
        #     ""（空字符串）             ""
        #     None                     ""
        #     " "（只含空格）            " "（先 or，后strip）
        # 第4层：.strip()
        # 去除首尾空白字符（空格、换行、制表符等）,确保返回的字符串是干净的
        return (reader.pages[page_index].extract_text() or "").strip()

    # 3.PDFPlumber  从PDF页面中提取表格，并将其转换为 Markdown 格式
    def _extract_pdfplumber_tables(self, page: Any, page_number: int) -> list[dict[str, Any]]:
        '''
        page	Any	pdfplumber 的页面对象（不是 fitz.Page）
        page_number	int	页码（从1开始，用于标记）
        返回值	list[dict]	表格信息列表，每个表格一个字典
        '''
        tables: list[dict[str, Any]] = []
        try:
            # pdfplumber 如何找到表格？
            # 检测线条：识别页面中的水平和垂直线条
            # 识别单元格：线条交叉形成网格
            # 提取文本：每个单元格内的文本内容
            # 返回结构：list[list[list[str]]] - 三维列表
            '''
            # extracted_tables 的结构
            [
                # 第1个表格
                [
                    ["姓名", "年龄", "城市"],      # 第1行
                    ["张三", "25", "北京"],        # 第2行
                    ["李四", "30", "上海"],        # 第3行
                ],
                # 第2个表格（如果存在）
                [
                    ["产品", "价格"],
                    ["苹果", "5元"],
                    ["香蕉", "3元"],
                ]
            ]
            '''
            extracted_tables = page.extract_tables()
        except Exception as exc:  # pragma: no cover - third-party parser path
            emit_runtime_trace(
                self.logger,
                "complex_pdf_parse_pdfplumber_tables_failed",
                page_number=page_number,
                error=str(exc),
            )
            return tables

        for table_index, rows in enumerate(extracted_tables or [], start=1):
            markdown = self._rows_to_markdown(rows) # 返回str
            if not markdown:
                continue
            tables.append(
                {
                    "page": page_number,
                    "caption": f"page_{page_number}_table_{table_index}",
                    "text": markdown,
                    "html": "",
                    "extractor": "pdfplumber",
                }
            )
        return tables

    def _extract_pdfminer_pages(self, local_path: str) -> list[str]:
        '''
        输入：PDF 文件的本地路径
        输出：字符串列表，每个元素是一页的文本内容
        '''
        if not self.settings.pdf_use_pdfminer or extract_pages is None or LTTextContainer is None:
            # 三个条件（任一为真就返回空列表）：
            # 配置开关关闭：not self.settings.pdf_use_pdfminer,用户主动禁用了pdfminer引擎
            # extract_pages不可用：extract_pages is Nonemm这个函数可能来自：from pdfminer.high_level import extract_pages,如果导入失败则为None
            # LTTextContainer不可用：LTTextContainer is None,这个类来自：from pdfminer.layout import LTTextContainer用于判断元素是否为文本容器
            return []

        # 返回值
        pages: list[str] = []
        try:
            # 遍历每一页
            for layout in extract_pages(local_path):
                # layout 代表 PDF 的**一页**
                # extract_pages 返回生成器，逐页产生 LTPage 对象
                lines: list[str] = []

                # 遍历页面中的每个元素
                for element in layout:
                    # layout 包含该页的所有元素：文本块、图片、线条、矩形等
                    # 元素类型包括：LTTextContainer, LTFigure, LTLine, LTRect 等

                    # 筛选文本容器
                    if isinstance(element, LTTextContainer):
                        # 提取并清理文本
                        text = element.get_text().strip()
                        if text:  # 过滤空字符串
                            lines.append(text)
                # 组装页面文本
                # "\n".join(lines) - 用换行符连接所有文本块
                # .strip() - 去掉首尾空白
                pages.append("\n".join(lines).strip())
        except Exception as exc:  # pragma: no cover - third-party parser path
            emit_runtime_trace(
                self.logger,
                "complex_pdf_parse_pdfminer_failed",
                error=str(exc),
            )
            return []
        return pages

    # 统一的表格提取接口
    def _extract_advanced_tables(self, local_path: str) -> list[dict[str, Any]]:
        # 输入：PDF文件的本地路径
        # 输出：从所有启用的引擎提取的表格列表（合并）
        tables: list[dict[str, Any]] = []
        if self.settings.pdf_use_camelot:
            # Camelot 特点
            # - 纯 Python 实现
            # - 专注于基于线条的表格
            # - 输出格式：pandas DataFrame
            # - 优点：准确度高，支持复杂表格
            # - 缺点：慢，依赖OpenCV和 Ghostscript
            # - 适用：有明确边框的表格
            # - 提供两种解析模式（lattice/stream）
            tables.extend(self._extract_camelot_tables(local_path))
        if self.settings.pdf_use_tabula:
            # Tabula 特点
            # - 基于 Apache PDFBox (Java)
            # - 通过子进程调用 Java JAR
            # - 速度快，适合大文件
            # - 不需要 OpenCV（Camelot 需要）
            # - 但需要安装 Java 环境
            tables.extend(self._extract_tabula_tables(local_path))
        return tables

    #  Camelot 库从 PDF 中提取表格，并尝试两种不同的解析策略（lattice 和 stream），选择能提取到表格的那个
    def _extract_camelot_tables(self, local_path: str) -> list[dict[str, Any]]:
        # 输入：PDF文件路径
        # 输出：统一格式的表格列表
        if camelot is None:
            emit_runtime_trace(self.logger, "complex_pdf_parse_camelot_missing")
            return []

        results: list[dict[str, Any]] = []
        for flavor in ("lattice", "stream"):
            '''
            1. Lattice（格子/线框模式）
            特点：
            识别页面中的线条（水平线和垂直线）
            通过线条交叉形成网格来定位表格
            适用于：有明确边框的表格
            2. Stream（流式模式）
            特点：
            通过文本对齐和空白区域来识别表格
            不依赖线条，基于文本的位置关系
            适用于：无边框或边框不完整的表格
            '''
            try:
                # camelot.read_pdf 参数：
                # flavor：解析策略（lattice/stream）
                # pages="all"：处理所有页面
                # 返回值：TableList 对象（可迭代的表格集合）
                found = camelot.read_pdf(local_path, flavor=flavor, pages="all")
            except Exception as exc:  # pragma: no cover - third-party parser path
                emit_runtime_trace(
                    self.logger,
                    "complex_pdf_parse_camelot_failed",
                    flavor=flavor,
                    error=str(exc),
                )
                continue

            for index, table in enumerate(found, start=1):
                # table.df 是 pandas DataFrame
                # 例如：
                #     姓名   年龄   城市
                # 0   张三   25    北京
                # 1   李四   30    上海
                # fillna("") 将 NaN 替换为空字符串
                # values.tolist() 转换为列表
                # [["张三", "25", "北京"], ["李四", "30", "上海"]]
                '''
                # 单个 Table 对象的结构
                table = tables[0]
                
                # 主要属性
                table.page      # 1 (页码)
                table.order     # 1 (该页的第几个表格)
                table.shape     # (3, 3) (行数, 列数)
                
                # pandas DataFrame - 核心数据
                table.df
                #     产品   销量   金额
                # 0   苹果   100   500
                # 1   香蕉   80    400
                
                # 转换为其他格式
                table.df.to_dict()           # 转为字典
                table.df.values.tolist()     # 转为列表 [['苹果', 100, 500], ['香蕉', 80, 400]]
                
                # 其他属性
                table.accuracy      # 准确度评分 (0-100)
                table.whitespace    # 空白区域分析
                table.cells         # 单元格详细信息
                '''
                rows = table.df.fillna("").values.tolist()
                markdown = self._rows_to_markdown(rows)
                if not markdown:
                    continue
                results.append(
                    {
                        # 取table的page属性，没有取到就用0
                        "page": int(getattr(table, "page", 0) or 0),
                        "caption": f"camelot_{flavor}_{index}",
                        "text": markdown,
                        "html": "",
                        "extractor": f"camelot_{flavor}",
                    }
                )
            # 如果 lattice 成功，就跳过 stream
            if results:
                break
        return results
    '''
    # 同一表格的提取结果对比
    # Camelot Lattice（有边框表格）
    {
        "extractor": "camelot_lattice",
        "text": "| 姓名 | 年龄 |\n|------|------|\n| 张三 | 25  |"
    }
    # Camelot Stream（无边框表格）
    {
        "extractor": "camelot_stream", 
        "text": "| 姓名 | 年龄 |\n| 张三 | 25  |"  # 可能识别为单行
    }
    # Tabula（基于位置）
    {
        "extractor": "tabula",
        "text": "| 姓名 | 年龄 |\n| 张三 | 25  |"
    }
    # pdfplumber（基于线条+文本）
    {
        "extractor": "pdfplumber",
        "text": "| 姓名 | 年龄 |\n| 张三 | 25  |"
    }
    '''

    # tabula提取表格
    def _extract_tabula_tables(self, local_path: str) -> list[dict[str, Any]]:
        if tabula is None:
            # 或者 Java 环境缺失
            # tabula-py 需要 Java 8+ 运行环境
            emit_runtime_trace(self.logger, "complex_pdf_parse_tabula_missing")
            return []

        try:
            # pages="all"	处理所有页面	也可用 "1-3,5"
            # multiple_tables=True	每页可提取多个表格	False 时只提取第一个
            # guess=True	自动检测表格区域	False 需手动指定 area
            frames = tabula.read_pdf(
                local_path,
                pages="all",  # 所有页面
                multiple_tables=True,  # 每页可能有多个表格
                guess=True  # 自动猜测表格区域
            )
            '''
            # Tabula 返回：List[pandas.DataFrame]
            frames = [
                df1,  # 第1个表格
                df2,  # 第2个表格
                df3,  # 第3个表格
            ]
            '''
        except Exception as exc:  # pragma: no cover - third-party parser path
            emit_runtime_trace(
                self.logger,
                "complex_pdf_parse_tabula_failed",
                error=str(exc),
            )
            return []

        results: list[dict[str, Any]] = []
        for index, frame in enumerate(frames or [], start=1):
            if frame is None or frame.empty:
                continue
            # 假设 DataFrame
            #     姓名   年龄   城市
            # 0   张三   25    北京
            # 1   李四   30    上海
            # 步骤1：获取列名
            # frame.columns.tolist()
            # ['姓名', '年龄', '城市']
            # 步骤2：获取数据部分（填充 NaN）
            # frame.fillna("").values.tolist()
            # [['张三', '25', '北京'], ['李四', '30', '上海']]
            # 步骤3：使用 * 解包 + 组合
            # rows = [['姓名', '年龄', '城市'], ['张三', '25', '北京'], ['李四', '30', '上海']]
            # Tabula 返回的 DataFrame 已经有列名
            # 需要把列名作为表格的第一行（表头）
            # *frame.values.tolist() 将数据行解包为多个元素
            rows = [frame.columns.tolist(), *frame.fillna("").values.tolist()]
            markdown = self._rows_to_markdown(rows)
            if not markdown:
                continue
            results.append(
                {
                    "page": 0,
                    "caption": f"tabula_{index}",
                    "text": markdown,
                    "html": "",
                    "extractor": "tabula",
                }
            )
        return results

    '''
    # Camelot 的方式
    rows = table.df.fillna("").values.tolist()
    # 注意：Camelot 的 df 已经包含列名作为第一行
    # [['姓名', '年龄', '城市'], ['张三', '25', '北京'], ...]
    
    # Tabula 的方式
    rows = [frame.columns.tolist(), *frame.fillna("").values.tolist()]
    # 需要手动添加列名
    
    Camelot 的 DataFrame
    # Camelot 返回的 Table 对象
    table = camelot_tables[0]
    df = table.df
    # DataFrame 的结构：
    #       0     1     2
    # 0    姓名   年龄   城市    ← 第0行是表头（数据）
    # 1    张三   25    北京    ← 第1行是数据
    # 2    李四   30    上海
    df.values.tolist()
    # [
    #     ['姓名', '年龄', '城市'],  ← 包含表头
    #     ['张三', '25', '北京'],
    #     ['李四', '30', '上海']
    # ]
    
    Tabula 的 DataFrame
    # Tabula 返回的 DataFrame
    df = tabula_frames[0]
    # DataFrame 的结构：
    #     姓名   年龄   城市    ← 列名（元数据，不是数据行）
    # 0   张三   25    北京    ← 第0行是数据
    # 1   李四   30    上海
    df.columns.tolist()
    # ['姓名', '年龄', '城市']  ← 列名单独存储
    df.values.tolist()
    # [
    #     ['张三', '25', '北京'],  ← 不包含列名
    #     ['李四', '30', '上海']
    # ]
    
    # Camelot: 面向表格识别
    # - 把识别到的"单元格内容"直接放入 DataFrame
    # - 第一行就是表格的第一行（可能是表头，也可能是数据）
    # - 不区分"列名"和"数据"
    
    # Tabula: 面向数据分析
    # - 遵循 pandas 的惯例
    # - 列名是 DataFrame 的元数据（.columns）
    # - 数据行不包含列名
    '''

    # 表格去重合并函数
    def _merge_tables(
        self,
        base_tables: list[dict[str, Any]],
        new_tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = list(base_tables)
        seen = {
            # 标准化文本
            self._normalize_table_text(str(table.get("text", "")))
            for table in base_tables
            if table.get("text")
        }
        for table in new_tables:
            normalized = self._normalize_table_text(str(table.get("text", "")))
            if not normalized or normalized in seen:
                continue
            merged.append(table)
            seen.add(normalized)
        return merged

    # 将高级表格追加到文本末尾
    def _append_advanced_tables_text(self, plain_text: str, tables: list[dict[str, Any]]) -> str:
        if not tables:
            return plain_text
        blocks = [plain_text] if plain_text else []
        for table in tables:
            caption = str(table.get("caption", "table"))
            extractor = str(table.get("extractor", "unknown"))
            table_text = str(table.get("text", "")).strip()
            if not table_text:
                continue
            blocks.append(f"[Advanced Table] {caption} ({extractor})\n{table_text}")
        return "\n\n".join(blocks).strip()

    # 将页面文本和表格组合成最终的输出格式
    def _build_page_text(self, page_artifact: PageParseArtifact) -> str:
        # 输入：单页的解析结果（包含文本和表格）
        # 输出：格式化的完整文本
        segments = [f"[Page {page_artifact.page_number}]"]
        if page_artifact.text:
            segments.append(page_artifact.text)
        for table in page_artifact.tables:
            segments.append(
                f"[Table {table.get('caption', '')} via {table.get('extractor', 'unknown')}]\n"
                f"{table.get('text', '')}"
            )
        return "\n\n".join(segment for segment in segments if segment).strip()

    # 构建文档结构化章节的函数
    def _build_sections(
            self,
            page_artifacts: list[PageParseArtifact],  # 页面解析结果列表
            plain_text: str,  # 纯文本（回退用）
    ) -> list[DocumentSection]:  # 章节列表
        sections: list[DocumentSection] = []
        for page_artifact in page_artifacts:
            content = self._build_page_text(page_artifact)
            if not content:
                continue
            heading = self._detect_page_heading(page_artifact.text) or f"Page {page_artifact.page_number}"
            sections.append(DocumentSection(heading=heading, level=1, content=content))

        if sections:
            return sections
        return self._extract_sections(plain_text)

    def _build_mineru_sections(self, pages: list[dict[str, Any]], fallback_text: str) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        for page in pages:
            text = str(page.get("text", "")).strip()
            if not text:
                continue
            heading = self._detect_page_heading(text) or f"Page {page.get('page_number', len(sections) + 1)}"
            sections.append(DocumentSection(heading=heading, level=1, content=text))
        if sections:
            return sections
        return self._extract_sections(fallback_text)

    def _extract_sections(self, text: str) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_heading = ""
        buffer: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if HEADING_PATTERN.match(line) and len(line) <= 120:
                if buffer:
                    sections.append(
                        DocumentSection(
                            heading=current_heading or "Body",
                            level=1,
                            content="\n".join(buffer).strip(),
                        )
                    )
                    buffer.clear()
                current_heading = line
            else:
                buffer.append(line)
        if buffer:
            sections.append(
                DocumentSection(
                    heading=current_heading or "Body",
                    level=1,
                    content="\n".join(buffer).strip(),
                )
            )
        if not sections and text.strip():
            sections.append(DocumentSection(heading="Body", level=1, content=text.strip()))
        return sections

    # 简洁的页面标题检测函数
    def _detect_page_heading(self, text: str) -> str | None:
        # 输入：页面文本内容
        # 输出：检测到的标题（或 None）
        for raw_line in text.splitlines()[:5]:
            line = raw_line.strip()
            if HEADING_PATTERN.match(line) and len(line) <= 120:
                return line
        return None

    # 智能解析文档标题
    def _resolve_title(self, file_name: str, text: str) -> str:
        # 输入：PDF文件路径 + 文档文本内容
        # 输出：文档标题（从内容提取 或 使用文件名）
        heading = self._detect_page_heading(text)
        return heading or Path(file_name).stem

    # 返回最长的文本提取
    def _choose_best_text(self, candidates: dict[str, str]) -> tuple[str, str]:
        # 输入：{"提取器名称": "文本内容", ...}
        # 输出：(选中的名称, 选中的文本)
        scored = sorted(
            ((name, text.strip()) for name, text in candidates.items() if text and text.strip()),
            key=lambda item: len(item[1]), # item[1] 是文本内容
            reverse=True,  # 降序：最长的在前
        )
        if not scored:
            return "", ""
        return scored[0]

    # 判断是否应该对PDF页面使用OCR（光学字符识别）
    def _should_use_ocr(self, text: str) -> bool:
        # 输入：从PDF提取的文本内容
        # 输出：True表示应该使用OCR，False表示不需要
        if self.settings.pdf_ocr_force_all_pages:
            return self.settings.ocr_enabled
        normalized = text.strip()
        # 提取到的文本长度小于80就用OCR再处理一次
        return self.settings.ocr_enabled and len(normalized) < self.settings.pdf_text_min_chars_per_page

    # 懒加载+单例模式
    def _get_ocr(self) -> PaddleOCR:
        if self._ocr is None:
            emit_runtime_trace(
                self.logger,
                "complex_pdf_parse_ocr_model_loading",
                language=self.settings.ocr_language,
            )
            # 这里是初始化的核心，主要做三件事：
            # 下载： 如果是首次使用，PaddleOCR 会自动下载预训练模型文件。
            # 加载： 将模型读入内存/显存。
            # 预热： 构建推理引擎。
            self._ocr = PaddleOCR(
                # 启用文本方向分类器。因为 PDF 里的文字可能是旋转的，开启后模型会自动检测并修正，提高识别率。
                use_angle_cls=True,
                # 指定识别语言，例如 ch（中文）、en（英文）。这会影响加载哪一个语言模型。
                lang=self.settings.ocr_language,
                # 关闭 PaddleOCR 内部的冗长日志输出，保持终端输出干净。
                show_log=False,
            )
        return self._ocr

    # 对PDF页面执行OCR（光学字符识别），将图片中的文字提取出来
    def _ocr_page(self, page: fitz.Page, page_number: int) -> str:
        # 输入：PyMuPDF页面对象，页码
        # 输出：OCR识别出的文本

        # 1.计算DPI缩放比例
        '''
        📖 原理：PDF 的物理尺寸与图像分辨率
        PDF 的内在单位是「点」：一个 PDF 页面或其中的图片，尺寸是用点 (point) 来衡量的，
        标准定义为 1 点 = 1/72 英寸。
        屏幕显示需要 DPI：当我们将 PDF 页面“渲染”成一张普通的图片时，需要指定一个 DPI (每英寸点数)。
        DPI 越高，生成的图片分辨率就越高，文字就越清晰，但 OCR 处理时间也更长，占用内存也越大。
        72 DPI 是基线：在 PDF 世界中，72 DPI 对应着 1:1 的原始大小。
        也就是说，如果你设置渲染 DPI 为 72，那么 PDF 中一个 12 点的文字，
        在生成的图片中就大约占据 12 个像素的高度。

        PDF 里的定义：1 英寸 = 72 点。这是铁律，就像 1 米 = 100 厘米一样。
        DPI 的含义：就是每英寸有多少个像素点。
        用一个等式帮你彻底记住
        PDF 渲染的核心等式：
        72 (PDF里的点/英寸) = DPI (渲染时的像素/英寸) 时，才是 1:1 原始大小
        如果 DPI = 72：PDF 里的 1 点 → 图片里占 1 像素。
        如果 DPI = 300：PDF 里的 1 点 → 图片里占 300 ÷ 72 ≈ 4.17 像素。
        打个比方
        PDF 的点：是地图上的比例尺。地图上 1 厘米代表地面 1 公里。
        DPI：是你把地图放大复印多少倍。复印 1 倍（72 DPI），地图上的 1 厘米在复印件上还是 1 厘米；复印 4 倍（300 DPI），地图上的 1 厘米在复印件上就变成了 4 厘米。
        '''
        # 确保最终的 dpi_scale 至少是 1.0,确保了 OCR 的下限质量。
        dpi_scale = max(self.settings.pdf_ocr_render_dpi / 72.0, 1.0)
        # 2.渲染PDF页面为图像   用矩阵把 PDF 里的 1 点（1/72英寸）放大到对应 dpi 的像素数，然后生成真实的像素图
        # page.get_pixmap()：这是 PyMuPDF 库的核心方法，它的职责是把 PDF的矢量指令
        # （如“这里画一条线，那里写一个字母A”）实时计算并转换成像素点阵，也就是我们常说的图片。
        # get_pixmap 参数：
        #   fitz.Page 对象内部不是一张图片，而是一个矢量指令的集合（或者说一个"绘图程序"）。
        # matrix：变换矩阵（缩放、旋转等）  Matrix 在这里是一个仿射变换矩阵，用来控制缩放和旋转。
        # alpha=False：不要透明度通道（RGB模式，不需要Alpha）
        # fitz.Matrix 的作用： 创建缩放变换矩阵
        '''
        仿射变换矩阵是一个 2×3 的数字表格，它用矩阵乘法一次性完成：缩放、旋转、平移、倾斜。
        为什么叫"仿射"？
        因为它保持两条性质：
        直线变换后还是直线（不会变曲线）
        平行线变换后依然平行（不会歪斜变形）
        矩阵的样子（3×3 齐次形式，实际常用2×3）
        text
        [ a  b  tx ]
        [ c  d  ty ]
        a, d：控制 X 和 Y 方向的缩放（a=2 → 放大2倍）
        b, c：控制倾斜/旋转（b=0.5 → 向右倾斜）
        tx, ty：控制平移（tx=10 → 向右移10像素）
        具体例子：你代码里的用法
        matrix = fitz.Matrix(dpi_scale, dpi_scale)
        这个矩阵是：
        text
        [ dpi_scale,    0,         0 ]
        [    0,      dpi_scale,    0 ]
        相当于 a = dpi_scale, b = 0, c = 0, tx = 0, ty = 0。
        效果：只在 X 和 Y 方向等比例缩放，不旋转、不平移。
        如果 dpi_scale = 300/72 ≈ 4.17
        原来 PDF 里 1 点（1/72英寸）→ 变成 4.17 像素
        12pt 的字 → 在图片中占 12 × 4.17 ≈ 50 像素高
        '''
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_scale, dpi_scale), alpha=False)
        # 3.转换为NumPy数组
        # pix.samples：原始字节数据
        # 例如：R,G,B,R,G,B,R,G,B,... (每个像素3个字节)
        # 步骤1：从字节缓冲区创建NumPy数组
        # np.frombuffer(pix.samples, dtype=np.uint8)
        # 结果：形状为 (总数, ) 的一维数组 [R,G,B,R,G,B,...]
        # 步骤2：重塑为3维数组
        # .reshape(pix.height, pix.width, pix.n)
        # pix.height = 1000 (高度像素)
        # pix.width = 800 (宽度像素)
        # pix.n = 3 (RGB通道数)
        # 结果形状：(1000, 800, 3) - 标准的图像数组
        '''
        🔧 代码逻辑拆解
        pix.samples：这是一个 bytes 对象，包含了图片所有像素的原始数据。它是扁平化的，像一串很长的一维数组：[R, G, B, R, G, B, ...]。
        np.frombuffer(..., dtype=np.uint8)：创建了一个 NumPy 数组，但没有复制数据（零拷贝），而是直接引用了 pix.samples 的内存块，声明每个元素都是 0-255 的 uint8 类型，表示像素值。
        .reshape(pix.height, pix.width, pix.n)：这是至关重要的一步，它将扁平的数组重组为高度、宽度、通道数 的三维结构。
        例如：一个 100 像素高，200 像素宽，3 通道 (RGB) 的图片，最终会变成一个 shape 为 (100, 200, 3) 的数组。这样，image[y, x] 就能直接获取到第 y 行、第 x 列像素的 [R, G, B] 值。
        '''
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # cls=True 参数： cls = Classification (文本方向分类)
        # 自动检测并纠正文本方向（0°, 90°, 180°, 270°）
        # 对旋转的扫描文档很重要
        result = self._get_ocr().ocr(image, cls=True)
        '''
        # PaddleOCR返回格式
        result = [
            [  # 第一组文字（通常整个页面为一组）
                [
                    [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # 文字边界框
                    ('识别出的文字', 0.95)  # (文本, 置信度)
                ],
                [
                    [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                    ('第二个文字', 0.98)
                ],
                ...
            ]
        ]
        # 或者格式2（简化版本）
        result = [
            [
                (box, ('文本', 置信度)),
                ...
            ]
        ]
        '''
        lines: list[str] = []
        # 提取结果
        for line_group in result or []:  # 提取每一页
            for line in line_group:  # 提取一页中的一块
                if len(line) > 1 and line[1]:  # 确保有文本
                    lines.append(str(line[1][0]).strip())
        ocr_text = "\n".join(line for line in lines if line).strip()
        emit_runtime_trace(
            self.logger,
            "complex_pdf_parse_page_ocr_finished",
            page_number=page_number,
            text_preview=preview_text(ocr_text, 120),
        )
        return ocr_text

    # 将二维列表（表格数据）转换为 Markdown 格式的表格
    def _rows_to_markdown(self, rows: list[list[Any]] | None) -> str:
        # 第一步：数据清洗和规范化
        normalized_rows: list[list[str]] = []
        for row in rows or []: # rows 为 None 时使用空列表
            # 潜在问题： 数字 0 会被转换成空字符串！
            cleaned_row = [str(cell or "").strip() for cell in row]
            if any(cleaned_row):  # 至少有一个非空字符串,去除空行
                normalized_rows.append(cleaned_row)
        if not normalized_rows:
            return ""

        # 第二步：列数对齐
        '''
        # 填充过程
        第1行: ["姓名","年龄","城市","电话"] + [""]*(5-4) = ["姓名","年龄","城市","电话",""]
        第2行: ["张三","25","北京"] + [""]*(5-3) = ["张三","25","北京","",""]
        第3行: ["李四","30","上海","123456","备注"] + [""]*(0) = ["李四","30","上海","123456","备注"]
        '''
        max_width = max(len(row) for row in normalized_rows)
        padded_rows = [row + [""] * (max_width - len(row)) for row in normalized_rows]

        # 第三步：分离表头和数据
        header = padded_rows[0]
        # 情况1：有表头 + 数据行
        # 情况2：只有表头（无数据）
        # 这个设计确保即使只有表头，也能生成一个完整的表格（表头行重复显示）。
        body = padded_rows[1:] or [padded_rows[0]]

        # 第四步：生成 Markdown
        separator = ["---"] * max_width
        markdown_rows = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in body:
            markdown_rows.append("| " + " | ".join(row) + " |")
        return "\n".join(markdown_rows)

    def _normalize_table_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    # 记录和标识文档解析过程中使用了哪些提取工具
    def _build_parser_name(
        self,
        page_artifacts: list[PageParseArtifact],
        advanced_tables: list[dict[str, Any]],
    ) -> str:
        extractors = {artifact.extractor for artifact in page_artifacts if artifact.extractor}
        extractors.update(str(table.get("extractor", "")) for table in advanced_tables if table.get("extractor"))
        return "+".join(sorted(extractors)) or "pdf_parser"

    # 从嵌套字典中按多个路径提取第一个非空文本的辅助方法
    def _extract_first_text(self, payload: dict[str, Any], paths: list[tuple[str, ...]]) -> str:
        for path in paths:
            value: Any = payload
            found = True
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    found = False
                    break
                value = value[key]
            if found and isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _normalize_mineru_pages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        pages = payload.get("pages") or payload.get("data", {}).get("pages") or payload.get("result", {}).get("pages")
        if not isinstance(pages, list):
            return []
        normalized: list[dict[str, Any]] = []
        for index, page in enumerate(pages, start=1):
            if isinstance(page, dict):
                normalized.append(
                    {
                        "page_number": int(page.get("page_number", page.get("page", index))),
                        "text": str(page.get("text") or page.get("markdown") or page.get("content") or "").strip(),
                    }
                )
        return normalized

    def _normalize_mineru_tables(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        tables = payload.get("tables") or payload.get("data", {}).get("tables") or payload.get("result", {}).get("tables")
        if not isinstance(tables, list):
            return []
        normalized: list[dict[str, Any]] = []
        for index, table in enumerate(tables, start=1):
            if not isinstance(table, dict):
                continue
            markdown = str(table.get("markdown") or table.get("text") or table.get("html") or "").strip()
            if not markdown:
                continue
            normalized.append(
                {
                    "page": int(table.get("page", table.get("page_number", 0)) or 0),
                    "caption": str(table.get("caption") or f"mineru_table_{index}"),
                    "text": markdown,
                    "html": str(table.get("html", "")),
                    "extractor": "mineru_api",
                }
            )
        return normalized
