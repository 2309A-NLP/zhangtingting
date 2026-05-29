from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.collectors.common import (  # noqa: E402
    RAW_SOURCES_DIR,
    ensure_dir,
    load_registry,
    seen_source_url,
    stable_doc_id,
    user_agent,
    write_collected_document,
    write_raw_payload,
)

UPDATED_AT_PATTERN = re.compile(r"最后更新\s*(\d{4}-\d{2}-\d{2})")


class ZgbkHistoryCollector:
    def __init__(self, *, contact_email: str, max_docs: int) -> None:
        registry = load_registry()["history_01"]
        self.role_id = "history_01"
        self.contact_email = contact_email
        self.max_docs = max_docs or int(registry["max_docs_default"])
        self.min_interval_seconds = float(registry["min_interval_seconds"])
        self.seed_names = list(registry["seed_names"])
        self.raw_dir = ensure_dir(RAW_SOURCES_DIR / "history_zhwiki")
        self._last_request_at = 0.0
        self._robots: RobotFileParser | None = None

    async def collect(self) -> list[dict[str, Any]]:
        headers = {"User-Agent": user_agent(self.contact_email)}
        async with httpx.AsyncClient(headers=headers, timeout=30, follow_redirects=True) as client:
            await self._load_robots(client)
            docs: list[dict[str, Any]] = []
            for name in self.seed_names:
                if len(docs) >= self.max_docs:
                    break
                query_url = f"https://www.zgbk.com/ecph/words?Name={quote(name)}&SiteID=1"
                if self._robots and not self._robots.can_fetch(headers["User-Agent"], query_url):
                    continue
                response = await self._get(client, query_url)
                source_url = str(response.url)
                if seen_source_url(self.role_id, source_url):
                    continue
                doc = self._extract_article(response.text, source_url, seed_name=name)
                if not doc:
                    continue
                write_raw_payload(self.raw_dir / f"{doc['doc_id']}.html", response.text)
                write_collected_document(self.role_id, doc)
                docs.append(doc)
            return docs

    async def _load_robots(self, client: httpx.AsyncClient) -> None:
        robots_url = "https://www.zgbk.com/robots.txt"
        try:
            response = await client.get(robots_url)
            if response.status_code >= 400:
                self._robots = None
                return
        except Exception:
            self._robots = None
            return

        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())
        self._robots = parser

    async def _get(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        now = asyncio.get_running_loop().time()
        sleep_for = self.min_interval_seconds - (now - self._last_request_at)
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        response = await client.get(url)
        response.raise_for_status()
        self._last_request_at = asyncio.get_running_loop().time()
        response.encoding = response.encoding or "utf-8"
        return response

    def _extract_article(self, html: str, source_url: str, *, seed_name: str) -> dict[str, Any] | None:
        soup = BeautifulSoup(html, "lxml")
        title = self._extract_title(soup)
        summary_lines = [
            node.get_text(" ", strip=True)
            for node in soup.select(".summary p")
            if node.get_text(" ", strip=True)
        ]
        infobox_lines = []
        for item in soup.select(".word-left dl"):
            key_node = item.select_one("dt")
            value_node = item.select_one("dd")
            if not key_node or not value_node:
                continue
            key = key_node.get_text(" ", strip=True)
            value = value_node.get_text(" ", strip=True)
            if key and value:
                infobox_lines.append(f"{key}: {value}")

        content_parts = []
        if summary_lines:
            content_parts.append("\n".join(summary_lines))
        if infobox_lines:
            content_parts.append("基本信息:\n" + "\n".join(infobox_lines))
        content = "\n\n".join(part for part in content_parts if part).strip()
        if not title or not content or len(content) < 80:
            return None

        published_at = None
        match = UPDATED_AT_PATTERN.search(soup.get_text(" ", strip=True))
        if match:
            published_at = match.group(1)

        doc_id = stable_doc_id("history", source_url)
        return {
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "source_name": "www.zgbk.com",
            "source_url": source_url,
            "source_domain": "www.zgbk.com",
            "published_at": published_at,
            "role_id": self.role_id,
            "source_tier": "reference",
            "tags": ["history_01", seed_name],
            "metadata": {
                "crawler": "history_zgbk",
                "content_type": "encyclopedia_entry",
                "seed_name": seed_name,
            },
        }

    def _extract_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            text = soup.title.string.strip()
            return text.replace(" - 中国百科网", "").strip()
        node = soup.select_one("h1")
        return node.get_text(" ", strip=True) if node else ""


async def _main_async(args: argparse.Namespace) -> None:
    collector = ZgbkHistoryCollector(contact_email=args.contact_email, max_docs=args.max_docs)
    docs = await collector.collect()
    print(f"history_01 collected {len(docs)} documents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-email", required=True)
    parser.add_argument("--max-docs", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
