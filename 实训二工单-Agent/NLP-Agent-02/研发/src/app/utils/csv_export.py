from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, time
from enum import Enum
from typing import Any


def build_csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    # 在内存中创建一个文本流缓冲区，可以像文件一样写入，但不会写入磁盘。
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    # 写入表头 将 fieldnames 作为第一行写入 CSV。
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _serialize_csv_value(row.get(field)) for field in fieldnames})
    # getvalue()获取缓冲区中的完整 CSV 字符串。     encode("utf-8-sig") 编码为字节流
    # 为什么用 utf-8-sig？ Excel 默认以 ANSI 编码打开 CSV，如果没有 BOM，中文会乱码。加上 BOM 后，Excel 会正确识别为 UTF-8，中文显示正常。
    # 这是生产环境导出 CSV 的最佳实践。
    return buffer.getvalue().encode("utf-8-sig")


def _serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
'''
设计点	                     优势
io.StringIO 内存操作	      无需创建临时文件，速度快，适合 Web 响应。
extrasaction="ignore"	  健壮性强，即使数据有额外字段也不会报错。
utf-8-sig 编码	          兼容 Excel，解决中文乱码问题。
递归序列化处理	          支持 None、枚举、时间、复杂结构，覆盖 99% 的场景。
JSON 序列化复杂结构	      保留数据完整性，即使 CSV 原生不支持列表/字典。
'''