"""Wikipedia source backed by the MediaWiki API (not HTML scraping).

``discover()`` lists article titles in a category (default: UN member states).
``fetch()`` pulls plain-text extracts plus each page's current revision id, batching
titles per request (MediaWiki caps ``exlimit`` at 20) and bounding concurrency with a
semaphore. The revision id is stored so the pipeline can skip unchanged pages on re-index.

Parsing is factored into ``parse_pages`` so it can be tested against a recorded API
response without touching the network.
"""
from __future__ import annotations

import asyncio

import httpx

from src.common.config import settings
from src.common.enums import SourceName
from src.common.logging import get_logger
from src.ingestion.base import SourceDocument

log = get_logger("wikipedia")


def _api_url(language: str) -> str:
    return f"https://{language}.wikipedia.org/w/api.php"


async def _get_json(
    client: httpx.AsyncClient, url: str, params: dict[str, str], *, retries: int = 3
) -> dict:
    """GET with bounded exponential backoff on transient HTTP/network errors."""
    delay = 0.5
    for attempt in range(retries):
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")  # pragma: no cover


def parse_pages(data: dict, *, language: str) -> list[SourceDocument]:
    """Turn a formatversion=2 query response into SourceDocuments (pure, testable)."""
    docs: list[SourceDocument] = []
    for page in data.get("query", {}).get("pages", []):
        if page.get("missing") or not page.get("extract"):
            continue
        revisions = page.get("revisions") or [{}]
        revision = str(revisions[0].get("revid", "0"))
        page_id = page["pageid"]
        docs.append(
            SourceDocument(
                doc_id=f"{SourceName.WIKIPEDIA.value}:{page_id}",
                title=page["title"],
                text=page["extract"],
                source_uri=f"https://{language}.wikipedia.org/?curid={page_id}",
                revision=revision,
                metadata={"source": SourceName.WIKIPEDIA.value, "title": page["title"]},
            )
        )
    return docs


class WikipediaSource:
    name = SourceName.WIKIPEDIA

    def __init__(
        self,
        *,
        language: str | None = None,
        category: str | None = None,
        concurrency: int | None = None,
        batch_titles: int | None = None,
    ) -> None:
        self._language = language or settings.wikipedia_language
        self._category = category or settings.wikipedia_category
        self._concurrency = concurrency or settings.ingestion_concurrency
        self._batch_titles = batch_titles or settings.wikipedia_batch_titles
        self._url = _api_url(self._language)
        self._headers = {"User-Agent": settings.user_agent}

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=settings.request_timeout_seconds, headers=self._headers
        )

    async def discover(self) -> list[str]:
        """List every article title in the configured category, following continuation."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{self._category}",
            "cmtype": "page",
            "cmlimit": "max",
            "format": "json",
            "formatversion": "2",
        }
        titles: list[str] = []
        async with self._client() as client:
            cont: dict[str, str] = {}
            while True:
                data = await _get_json(client, self._url, {**params, **cont})
                titles.extend(m["title"] for m in data["query"]["categorymembers"])
                if "continue" not in data:
                    break
                cont = data["continue"]
        log.info("discovered", source=self.name.value, count=len(titles))
        return titles

    async def fetch(self, keys: list[str]) -> list[SourceDocument]:
        """Fetch plain-text extracts + revision ids for all titles, batched and bounded."""
        batches = [
            keys[i : i + self._batch_titles]
            for i in range(0, len(keys), self._batch_titles)
        ]
        semaphore = asyncio.Semaphore(self._concurrency)
        async with self._client() as client:
            results = await asyncio.gather(
                *(self._fetch_batch(client, semaphore, batch) for batch in batches)
            )
        docs = [doc for batch in results for doc in batch]
        log.info("fetched", source=self.name.value, documents=len(docs))
        return docs

    async def _fetch_batch(
        self, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, titles: list[str]
    ) -> list[SourceDocument]:
        params = {
            "action": "query",
            "prop": "extracts|revisions",
            "explaintext": "1",
            "exlimit": "max",
            "rvprop": "ids",
            "titles": "|".join(titles),
            "format": "json",
            "formatversion": "2",
        }
        async with semaphore:
            data = await _get_json(client, self._url, params)
        return parse_pages(data, language=self._language)
