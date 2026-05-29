from __future__ import annotations

import re
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.knowledge.models import CleanDocument, ParsedDocument

logger = get_logger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
SPECIAL_CHAR_PATTERN = re.compile(
    r"[^\w\s\u4e00-\u9fff，。！？；：、（）《》“”‘’【】A-Za-z0-9\-_.!?;:()/%]"
)
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")


class SensitiveContentError(ValueError):
    """Raised when document content contains blocked sensitive words."""


class DocumentCleaner:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logger
        self.sensitive_words = self._load_sensitive_words()

    @log_timed("document_clean")
    def clean(self, document: ParsedDocument) -> CleanDocument:
        removed_items: list[str] = []
        text = document.plain_text

        emit_runtime_trace(
            self.logger,
            "document_clean_entered",
            doc_id=document.doc_id,
            file_name=document.file_name,
            parser_name=document.parser_name,
            original_chars=len(text),
            original_preview=preview_text(text, 160),
        )

        filtered_lines = self._remove_repeated_header_footer_lines(text)
        if filtered_lines != text:
            removed_items.append("header_footer")
            emit_runtime_trace(
                self.logger,
                "document_clean_header_footer_removed",
                doc_id=document.doc_id,
                chars_before=len(text),
                chars_after=len(filtered_lines),
            )
        text = filtered_lines

        stripped_urls = URL_PATTERN.sub(" ", text)
        if stripped_urls != text:
            removed_items.append("url")
            emit_runtime_trace(
                self.logger,
                "document_clean_urls_removed",
                doc_id=document.doc_id,
                chars_before=len(text),
                chars_after=len(stripped_urls),
            )
        text = stripped_urls

        no_special = SPECIAL_CHAR_PATTERN.sub(" ", text)
        if no_special != text:
            removed_items.append("special_chars")
            emit_runtime_trace(
                self.logger,
                "document_clean_special_chars_removed",
                doc_id=document.doc_id,
                chars_before=len(text),
                chars_after=len(no_special),
            )
        text = no_special

        normalized_text = MULTI_SPACE_PATTERN.sub(" ", text)
        normalized_text = MULTI_NEWLINE_PATTERN.sub("\n\n", normalized_text).strip()
        if normalized_text != text:
            emit_runtime_trace(
                self.logger,
                "document_clean_whitespace_normalized",
                doc_id=document.doc_id,
                chars_before=len(text),
                chars_after=len(normalized_text),
            )
        text = normalized_text

        text = self._remove_advertisement_lines(text, removed_items)
        emit_runtime_trace(
            self.logger,
            "document_clean_after_advertisement_check",
            doc_id=document.doc_id,
            removed_items=removed_items,
            current_preview=preview_text(text, 160),
        )

        self._check_sensitive_words(text)
        emit_runtime_trace(
            self.logger,
            "document_clean_sensitive_check_passed",
            doc_id=document.doc_id,
            sensitive_words_loaded=len(self.sensitive_words),
        )

        self._check_min_length(text)
        effective_length = len(text.replace(" ", "").replace("\n", ""))
        emit_runtime_trace(
            self.logger,
            "document_clean_min_length_passed",
            doc_id=document.doc_id,
            effective_length=effective_length,
            min_length=self.settings.min_chunk_text_length,
        )

        cleaned = CleanDocument(
            doc_id=document.doc_id,
            user_id=document.user_id,
            role_id=document.role_id,
            title=document.title,
            clean_text=text,
            source_uri=document.source_uri,
            file_name=document.file_name,
            content_type=document.content_type,
            parser_name=document.parser_name,
            sections=document.sections,
            tables=document.tables,
            metadata=document.metadata,
            removed_items=removed_items,
        )
        logger.info("document_cleaned", doc_id=document.doc_id, removed_items=removed_items)
        emit_runtime_trace(
            self.logger,
            "document_clean_completed",
            doc_id=document.doc_id,
            removed_items=removed_items,
            cleaned_chars=len(text),
            cleaned_preview=preview_text(text, 160),
        )
        return cleaned

    def _load_sensitive_words(self) -> set[str]:
        path = Path(self.settings.sensitive_words_path)
        if not path.exists():
            emit_runtime_trace(
                self.logger,
                "document_clean_sensitive_words_missing",
                sensitive_words_path=str(path),
            )
            return set()

        words = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        emit_runtime_trace(
            self.logger,
            "document_clean_sensitive_words_loaded",
            sensitive_words_path=str(path),
            count=len(words),
        )
        return words

    def _remove_repeated_header_footer_lines(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 6:
            return text

        frequencies: dict[str, int] = {}
        for line in lines:
            frequencies[line] = frequencies.get(line, 0) + 1

        repeated = {line for line, count in frequencies.items() if count >= 3 and len(line) <= 60}
        if not repeated:
            return text

        emit_runtime_trace(
            self.logger,
            "document_clean_repeated_lines_detected",
            repeated_count=len(repeated),
            repeated_preview=list(repeated)[:5],
        )
        kept_lines = [line for line in text.splitlines() if line.strip() not in repeated]
        return "\n".join(kept_lines)

    def _remove_advertisement_lines(self, text: str, removed_items: list[str]) -> str:
        ad_markers = ("广告", "扫码", "关注公众号", "点击下载", "责任编辑", "免责声明")
        lines = text.splitlines()
        cleaned_lines = [line for line in lines if not any(marker in line for marker in ad_markers)]
        if len(cleaned_lines) != len(lines):
            removed_items.append("advertisement")
            emit_runtime_trace(
                self.logger,
                "document_clean_advertisement_removed",
                removed_line_count=len(lines) - len(cleaned_lines),
            )
        return "\n".join(cleaned_lines).strip()

    def _check_sensitive_words(self, text: str) -> None:
        for word in self.sensitive_words:
            if word in text:
                emit_runtime_trace(
                    self.logger,
                    "document_clean_sensitive_word_blocked",
                    blocked_word=word,
                    text_preview=preview_text(text, 120),
                )
                raise SensitiveContentError(f"Blocked by sensitive word: {word}")

    def _check_min_length(self, text: str) -> None:
        effective_length = len(text.replace(" ", "").replace("\n", ""))
        if effective_length < self.settings.min_chunk_text_length:
            emit_runtime_trace(
                self.logger,
                "document_clean_too_short",
                effective_length=effective_length,
                min_length=self.settings.min_chunk_text_length,
                text_preview=preview_text(text, 120),
            )
            raise ValueError(f"Document text too short after cleaning: {effective_length}")
