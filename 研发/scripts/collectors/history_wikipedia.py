from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.robotparser import RobotFileParser

import httpx

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.collectors.common import (
    RAW_SOURCES_DIR,
    ensure_dir,
    load_registry,
    seen_source_url,
    stable_doc_id,
    user_agent,
    write_collected_document,
    write_raw_payload,
)


@dataclass(slots=True)
class CategorySeed:
    title: str


class WikipediaHistoryCollector:
    def __init__(self, *, contact_email: str, max_docs: int) -> None:
        registry = load_registry()["history_01"]
        self.role_id = "history_01"
        self.contact_email = contact_email
        self.max_docs = max_docs or int(registry["max_docs_default"])
        self.min_interval_seconds = float(registry["min_interval_seconds"])
        self.api_base = str(registry["api_base"])
        self.category_seeds = [CategorySeed(title=title) for title in registry["category_titles"]]
        self.raw_dir = ensure_dir(RAW_SOURCES_DIR / "history_zhwiki")
        self._last_request_at = 0.0
        self._robots: RobotFileParser | None = None

    async def collect(self) -> list[dict[str, Any]]:
        headers = {"User-Agent": user_agent(self.contact_email)}
        async with httpx.AsyncClient(headers=headers, timeout=20, follow_redirects=True) as client:
            await self._load_robots(client)
            docs: list[dict[str, Any]] = []
            for seed in self.category_seeds:
                async for page_title in self._iter_category_members(client, seed.title):
                    if len(docs) >= self.max_docs:
                        return docs
                    source_url = f"https://zh.wikipedia.org/wiki/{quote(page_title.replace(' ', '_'))}"
                    if seen_source_url(self.role_id, source_url):
                        continue
                    if self._robots and not self._robots.can_fetch(headers["User-Agent"], source_url):
                        continue

                    page_doc = await self._fetch_page_extract(client, page_title, seed.title)
                    if not page_doc:
                        continue
                    doc_id = page_doc["doc_id"]
                    raw_path = self.raw_dir / f"{doc_id}.json"
                    write_raw_payload(raw_path, page_doc.pop("_raw_payload"))
                    write_collected_document(self.role_id, page_doc)
                    docs.append(page_doc)
            return docs

    async def _load_robots(self, client: httpx.AsyncClient) -> None:
        robots_url = "https://zh.wikipedia.org/robots.txt"
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

    async def _iter_category_members(self, client: httpx.AsyncClient, category_title: str):
        cmcontinue: str | None = None
        yielded = 0
        while yielded < self.max_docs:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": category_title,
                "cmtype": "page",
                "cmlimit": "50",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            payload = await self._api_get(client, params)
            members = payload.get("query", {}).get("categorymembers", [])
            for member in members:
                title = str(member.get("title") or "").strip()
                if not title:
                    continue
                yielded += 1
                yield title
                if yielded >= self.max_docs:
                    return

            cmcontinue = payload.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                return

    async def _fetch_page_extract(
        self,
        client: httpx.AsyncClient,
        page_title: str,
        category_title: str,
    ) -> dict[str, Any] | None:
        params = {
            "action": "query",
            "format": "json",
            "redirects": "1",
            "prop": "extracts|info|pageprops",
            "inprop": "url",
            "explaintext": "1",
            "titles": page_title,
        }
        payload = await self._api_get(client, params)
        pages = payload.get("query", {}).get("pages", {})
        if not pages:
            return None

        page = next(iter(pages.values()))
        if "disambiguation" in page.get("pageprops", {}):
            return None

        title = str(page.get("title") or "").strip()
        extract = str(page.get("extract") or "").strip()
        fullurl = str(page.get("fullurl") or "").strip()
        if not title or not extract or len(extract) < 300 or not fullurl:
            return None

        doc_id = stable_doc_id("history", fullurl)
        return {
            "doc_id": doc_id,
            "title": title,
            "content": extract,
            "source_name": "zh.wikipedia.org",
            "source_url": fullurl,
            "source_domain": "zh.wikipedia.org",
            "published_at": None,
            "role_id": self.role_id,
            "source_tier": "reference",
            "tags": ["history_01", category_title],
            "metadata": {
                "crawler": "history_wikipedia",
                "content_type": "wiki_extract",
                "seed_category": category_title,
            },
            "_raw_payload": json.dumps(payload, ensure_ascii=False, indent=2),
        }

    async def _api_get(self, client: httpx.AsyncClient, params: dict[str, str]) -> dict[str, Any]:
        now = asyncio.get_running_loop().time()
        sleep_for = self.min_interval_seconds - (now - self._last_request_at)
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        response = await client.get(self.api_base, params=params)
        response.raise_for_status()
        self._last_request_at = asyncio.get_running_loop().time()
        return response.json()


async def _main_async(args: argparse.Namespace) -> None:
    collector = WikipediaHistoryCollector(contact_email=args.contact_email, max_docs=args.max_docs)
    docs = await collector.collect()
    print(f"history_01 collected {len(docs)} documents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-email", required=True)
    parser.add_argument("--max-docs", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
