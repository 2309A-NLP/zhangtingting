from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.collectors.common import (
    RAW_SOURCES_DIR,
    ensure_dir,
    load_registry,
    seen_source_url,
    source_domain,
    stable_doc_id,
    user_agent,
    write_collected_document,
    write_raw_payload,
)

ARTICLE_SELECTORS = (
    ".TRS_Editor",
    ".con",
    ".content",
    ".pages_content",
    ".detail",
    "article",
    "#xw_box",
    ".txt_con",
)
PUBLISHED_AT_PATTERN = re.compile(r"(20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}(?:日)?(?:\s+\d{2}:\d{2})?)")
ARTICLE_PATH_PATTERN = re.compile(r"/(?:\d{6}|\d{4}/\d{2}|c\d{6,})/")


@dataclass(slots=True)
class SeedPage:
    url: str
    label: str


class NhcDoctorCollector:
    def __init__(self, *, contact_email: str, max_docs: int) -> None:
        registry = load_registry()["doctor_01"]
        self.role_id = "doctor_01"
        self.contact_email = contact_email
        self.max_docs = max_docs or int(registry["max_docs_default"])
        self.min_interval_seconds = float(registry["min_interval_seconds"])
        self.seed_pages = [
            SeedPage(url=registry["seed_urls"][0], label="disease_control"),
            SeedPage(url=registry["seed_urls"][1], label="health_myth_busting"),
        ]
        self.raw_dir = ensure_dir(RAW_SOURCES_DIR / "doctor_nhc")
        self._last_request_at = 0.0
        self._robots: RobotFileParser | None = None

    async def collect(self) -> list[dict[str, Any]]:
        headers = {"User-Agent": user_agent(self.contact_email)}
        async with httpx.AsyncClient(headers=headers, timeout=20, follow_redirects=True) as client:
            await self._load_robots(client)
            results: list[dict[str, Any]] = []
            for seed in self.seed_pages:
                list_html = await self._get(client, seed.url)
                for article_url in self._discover_article_urls(list_html, base_url=seed.url):
                    if len(results) >= self.max_docs:
                        return results
                    if seen_source_url(self.role_id, article_url):
                        continue
                    if self._robots and not self._robots.can_fetch(headers["User-Agent"], article_url):
                        continue

                    detail_html = await self._get(client, article_url)
                    doc = self._extract_article(detail_html, article_url, seed.label)
                    if not doc:
                        continue

                    raw_path = self.raw_dir / f"{doc['doc_id']}.html"
                    write_raw_payload(raw_path, detail_html)
                    write_collected_document(self.role_id, doc)
                    results.append(doc)
            return results

    async def _load_robots(self, client: httpx.AsyncClient) -> None:
        robots_url = "https://www.nhc.gov.cn/robots.txt"
        try:
            response = await client.get(robots_url)
            response.raise_for_status()
        except Exception:
            self._robots = None
            return

        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())
        self._robots = parser

    async def _get(self, client: httpx.AsyncClient, url: str) -> str:
        now = asyncio.get_running_loop().time()
        sleep_for = self.min_interval_seconds - (now - self._last_request_at)
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        response = await client.get(url)
        response.raise_for_status()
        self._last_request_at = asyncio.get_running_loop().time()
        response.encoding = response.encoding or "utf-8"
        return response.text

    def _discover_article_urls(self, html: str, *, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []
        seen: set[str] = set()
        for anchor in soup.select("a[href]"):
            href = (anchor.get("href") or "").strip()
            text = anchor.get_text(" ", strip=True)
            if not href:
                continue
            absolute_url = urljoin(base_url, href)
            if source_domain(absolute_url) != "www.nhc.gov.cn":
                continue
            if not absolute_url.endswith(".shtml"):
                continue
            if not text or len(text) < 6:
                continue
            if "/wjw/" not in absolute_url and "/kppypt/" not in absolute_url and "/jkj/" not in absolute_url and "/yjb/" not in absolute_url:
                continue
            if not ARTICLE_PATH_PATTERN.search(absolute_url):
                continue
            if absolute_url in seen:
                continue
            seen.add(absolute_url)
            urls.append(absolute_url)
        return urls

    def _extract_article(self, html: str, source_url: str, seed_label: str) -> dict[str, Any] | None:
        soup = BeautifulSoup(html, "lxml")
        title = self._extract_title(soup)
        body = self._extract_body(soup)
        if not title or not body or len(body) < 200:
            return None

        doc_id = stable_doc_id("doctor", source_url)
        published_at = self._extract_published_at(soup, body)
        return {
            "doc_id": doc_id,
            "title": title,
            "content": body,
            "source_name": "nhc.gov.cn",
            "source_url": source_url,
            "source_domain": "www.nhc.gov.cn",
            "published_at": published_at,
            "role_id": self.role_id,
            "source_tier": "official",
            "tags": ["doctor_01", seed_label],
            "metadata": {
                "crawler": "doctor_nhc",
                "content_type": "article",
                "seed_label": seed_label,
            },
        }

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for selector in ("h1", ".tit", ".title", ".article-title"):
            node = soup.select_one(selector)
            if node:
                text = node.get_text(" ", strip=True)
                if text:
                    return text
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        return ""

    def _extract_body(self, soup: BeautifulSoup) -> str:
        candidates: list[str] = []
        for selector in ARTICLE_SELECTORS:
            node = soup.select_one(selector)
            if not node:
                continue
            for garbage in node.select("script, style, nav, footer, .share, .editor, .二维码, .ewm"):
                garbage.decompose()
            text = node.get_text("\n", strip=True)
            if text:
                candidates.append(text)

        if not candidates:
            fallback = soup.get_text("\n", strip=True)
            return fallback
        return max(candidates, key=len)

    def _extract_published_at(self, soup: BeautifulSoup, body: str) -> str | None:
        for selector in ("[class*=time]", "[class*=date]", ".info", ".source"):
            node = soup.select_one(selector)
            if not node:
                continue
            match = PUBLISHED_AT_PATTERN.search(node.get_text(" ", strip=True))
            if match:
                return match.group(1)
        match = PUBLISHED_AT_PATTERN.search(body[:500])
        return match.group(1) if match else None


async def _main_async(args: argparse.Namespace) -> None:
    collector = NhcDoctorCollector(contact_email=args.contact_email, max_docs=args.max_docs)
    docs = await collector.collect()
    print(f"doctor_01 collected {len(docs)} documents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-email", required=True)
    parser.add_argument("--max-docs", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
