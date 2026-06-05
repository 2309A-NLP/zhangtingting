from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request
import time

DEFAULT_API_URL = "http://127.0.0.1:8000/api/query"
DEFAULT_INPUT = Path(r"D:\Desktop\NLP-RAG-04\test\parse4\test4.csv")
DEFAULT_OUTPUT = Path(r"D:\Desktop\NLP-RAG-04\test\parse4\test4_result.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch query the local RAG API and fill rag_* columns in an evaluation CSV."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="RAG query endpoint.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k citations to request.")
    parser.add_argument(
        "--corpus",
        default="default",
        choices=["default", "uploaded"],
        help="Corpus used by the backend.",
    )
    parser.add_argument(
        "--use-llm",
        default="true",
        choices=["true", "false"],
        help="Whether to let the backend generate rag_answer with LLM.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-question timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite-input",
        action="store_true",
        help="Write results back to the input file instead of the output file.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode CSV: {path}")


def normalize_preview(text: str, limit: int = 120) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def post_query(
    api_url: str,
    question: str,
    top_k: int,
    corpus: str,
    use_llm: bool,
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "query": question,
        "top_k": top_k,
        "use_llm": use_llm,
        "corpus": corpus,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        content = response.read().decode("utf-8")
    return json.loads(content)


def build_page_string(citations: List[Dict[str, Any]]) -> str:
    pages: List[str] = []
    for item in citations[:5]:
        page_number = str(item.get("page_number", "")).strip()
        if page_number and page_number not in pages:
            pages.append(page_number)
    return "|".join(pages)


def build_preview_string(citations: List[Dict[str, Any]]) -> str:
    previews = [normalize_preview(str(item.get("text") or "")) for item in citations[:5]]
    previews = [preview for preview in previews if preview]
    return ",".join(previews)


def ensure_columns(rows: List[Dict[str, str]], fieldnames: List[str]) -> List[str]:
    required = ["rag_answer", "rag_pages", "rag_evidence_preview"]
    names = list(fieldnames)
    for column in required:
        if column not in names:
            names.append(column)
            for row in rows:
                row[column] = ""
    return names


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = input_path if args.overwrite_input else Path(args.output)
    use_llm = args.use_llm.lower() == "true"

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    rows = read_csv_rows(input_path)
    if not rows:
        raise ValueError(f"Input CSV is empty: {input_path}")

    fieldnames = ensure_columns(rows, list(rows[0].keys()))
    total = len(rows)

    for index, row in enumerate(rows, start=1):
        start=time.time()
        question = (row.get("question") or "").strip()
        if not question:
            print(f"[{index}/{total}] skip empty question")
            continue

        try:
            result = post_query(
                api_url=args.api_url,
                question=question,
                top_k=args.top_k,
                corpus=args.corpus,
                use_llm=use_llm,
                timeout=args.timeout,
            )
            citations = list(result.get("citations") or [])
            row["rag_answer"] = str(result.get("answer") or "")
            row["rag_pages"] = build_page_string(citations)
            row["rag_evidence_preview"] = build_preview_string(citations)
            end=time.time()
            print(f"[{index}/{total}] ok,timelong={end-start}")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            row["rag_answer"] = f"[HTTP {exc.code}] {detail or exc.reason}"
            row["rag_pages"] = ""
            row["rag_evidence_preview"] = ""
            print(f"[{index}/{total}] http error {exc.code}")
        except Exception as exc:  # pragma: no cover
            row["rag_answer"] = f"[ERROR] {exc}"
            row["rag_pages"] = ""
            row["rag_evidence_preview"] = ""
            print(f"[{index}/{total}] error: {exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
