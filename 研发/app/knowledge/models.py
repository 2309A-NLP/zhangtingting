from __future__ import annotations
'''
定义了知识库处理流程中所有阶段的数据结构（DTO/数据类）

RawDocument → ParsedDocument → CleanDocument → ChunkedDocument → EmbeddedChunk
    (上传)         (解析)          (清洗)          (分块)           (向量化)
'''
from dataclasses import dataclass, field
from typing import Any

# 代表用户上传的原始文件信息
@dataclass(slots=True)
class RawDocument:
    file_id: str
    task_id: str
    user_id: str
    role_id: str
    file_name: str
    content_type: str
    source_uri: str      # MinIO 中的存储路径
    local_path: str      # 本地临时文件路径
    source_type: str = "upload"    # 区分上传、爬虫、API 等来源
    metadata: dict[str, Any] = field(default_factory=dict)

# 文档章节
# 解析器（loader）提取文档的结构化章节信息
@dataclass(slots=True)
class DocumentSection:
    heading: str   # 章节标题，如 "第三章 方法论"
    level: int     # 标题级别，如 H1=1, H2=2
    content: str   # 该章节的正文内容

# 解析后的文档   loader.py 的输出
@dataclass(slots=True)
class ParsedDocument:
    doc_id: str                      # 文档唯一 ID
    user_id: str
    role_id: str
    title: str                        # 文档标题
    plain_text: str                   # 提取的纯文本
    source_uri: str
    file_name: str
    content_type: str
    parser_name: str                  # 使用的解析器（pdf_parser, docx_parser 等）
    sections: list[DocumentSection] = field(default_factory=list)  # 章节列表
    tables: list[dict[str, str]] = field(default_factory=list)     # 表格数据
    metadata: dict[str, Any] = field(default_factory=dict)         # 扩展元数据

# 清洗后的文档
@dataclass(slots=True)
class CleanDocument:
    doc_id: str
    user_id: str
    role_id: str
    title: str
    clean_text: str   # 替代 plain_text
    source_uri: str
    file_name: str
    content_type: str
    parser_name: str
    sections: list[DocumentSection] = field(default_factory=list)
    tables: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    removed_items: list[str] = field(default_factory=list)

# 分块后的文档
@dataclass(slots=True)
class ChunkedDocument:
    id: str                           # chunk 唯一 ID（uuid）
    doc_id: str
    chunk_id: str                     # 如 "doc_123_0"
    user_id: str
    role_id: str
    role_category: str                # 角色分类（用于检索隔离）
    text: str                         # chunk 的文本内容
    token_count: int                  # 精确的 token 数（tiktoken 计算）
    chunk_index: int                  # 在文档中的位置顺序（0,1,2...）
    source: str                       # 来源 URI
    heading_path: str                 # 章节路径，如 "第1章 > 第2节"
    metadata: dict[str, Any] = field(default_factory=dict)

    # tenant_key 属性： 不需要额外存储  实时计算
    # 用于多租户隔离
    # 检索时只查 tenant_key 匹配的 chunks
    @property
    def tenant_key(self) -> str:
        return f"{self.user_id}:{self.role_id}"

# 向量化后的文档
@dataclass(slots=True)
class EmbeddedChunk:
    id: str
    doc_id: str
    chunk_id: str
    user_id: str
    role_id: str
    role_category: str
    text: str
    embedding: list[float]  # 浮点数列表
    source: str
    heading_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tenant_key(self) -> str:
        return f"{self.user_id}:{self.role_id}"
