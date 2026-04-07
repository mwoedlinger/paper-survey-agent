"""Fetch and parse arXiv HTML pages for full paper text and figures."""

import asyncio
import json
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Sections worth extracting for deep dives
SECTION_KEYWORDS = {
    "introduction", "method", "approach", "model", "architecture",
    "experiment", "result", "evaluation", "ablation", "discussion",
    "conclusion", "limitation", "related work", "background",
    "training", "implementation", "setup", "framework", "design",
    "analysis", "findings", "our",
}


class ArxivHTMLFetcher:
    def __init__(self, config: dict):
        ah = config.get("arxiv_html", {})
        self._cache_dir = Path(ah.get("cache_dir", "./work/html_cache"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(ah.get("max_concurrency", 4))
        self._timeout = aiohttp.ClientTimeout(total=ah.get("timeout", 30))
        self._retry_count = ah.get("retry_count", 2)

    def _cache_path(self, arxiv_id: str) -> Path:
        return self._cache_dir / f"{arxiv_id.replace('/', '_')}.json"

    async def fetch_one(self, arxiv_id: str, session: aiohttp.ClientSession) -> dict:
        """Fetch and parse a single paper. Returns cached result if available."""
        cache = self._cache_path(arxiv_id)
        if cache.exists():
            return json.loads(cache.read_text())

        result = await self._fetch_and_parse(arxiv_id, session)
        cache.write_text(json.dumps(result, ensure_ascii=False))
        return result

    async def _fetch_and_parse(self, arxiv_id: str, session: aiohttp.ClientSession) -> dict:
        """Fetch HTML and extract sections."""
        async with self._semaphore:
            url = f"https://arxiv.org/html/{arxiv_id}"
            for attempt in range(self._retry_count + 1):
                try:
                    async with session.get(url, timeout=self._timeout) as resp:
                        if resp.status == 404:
                            return {"arxiv_id": arxiv_id, "html_available": False, "sections": {}}
                        resp.raise_for_status()
                        html = await resp.text()
                        return self._parse_html(arxiv_id, html)
                except Exception as e:
                    if attempt == self._retry_count:
                        logger.warning(f"Failed to fetch {arxiv_id}: {e}")
                        return {"arxiv_id": arxiv_id, "html_available": False, "sections": {}}
                    await asyncio.sleep(1.0)
        return {"arxiv_id": arxiv_id, "html_available": False, "sections": {}}

    def _parse_html(self, arxiv_id: str, html: str) -> dict:
        """Extract structured sections, abstract, and figures from arXiv HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Drop noise that confuses get_text: scripts, styles, and inline math
        # tooltips. Keep <math> visible labels via their plain alt text instead.
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for math in soup.find_all("math"):
            alt = math.get("alttext") or ""
            math.replace_with(alt)

        sections = {}
        for section in soup.find_all("section", class_="ltx_section"):
            heading = section.find(["h2", "h3", "h4"])
            if not heading:
                continue
            title = heading.get_text(" ", strip=True)
            # Strip leading section numbers like "1", "1.", "1.2", "1.2.3" + optional trailing punct
            title_clean = re.sub(r"^\d+(\.\d+)*\.?\s*", "", title).strip()

            title_lower = title_clean.lower()
            if not any(kw in title_lower for kw in SECTION_KEYWORDS):
                continue

            paragraphs = section.find_all("p", class_="ltx_p")
            # Use " " as the inline separator so words split across <a>/<span>/
            # math tags are not glued together. Then collapse runs of whitespace.
            chunks = []
            for p in paragraphs:
                txt = p.get_text(" ", strip=True)
                txt = re.sub(r"\s+", " ", txt)
                if txt:
                    chunks.append(txt)
            text = "\n\n".join(chunks)
            if text:
                sections[title_lower] = {"title": title_clean, "text": text[:12000]}

        # Extract abstract
        abstract = ""
        abstract_div = soup.find("div", class_="ltx_abstract")
        if abstract_div:
            abstract = abstract_div.get_text(" ", strip=True)
            abstract = re.sub(r"\s+", " ", abstract)
            if abstract.lower().startswith("abstract"):
                abstract = abstract[8:].lstrip(":. ").strip()

        figures = self._extract_figures(soup, arxiv_id)

        return {
            "arxiv_id": arxiv_id,
            "html_available": bool(sections),
            "abstract": abstract,
            "sections": sections,
            "figures": figures,
        }

    def _extract_figures(self, soup: BeautifulSoup, arxiv_id: str) -> list[dict]:
        """Extract figures with absolute image URLs and captions.

        Skips images that Obsidian can't render inline from a URL (SVG,
        data: URIs, non-raster extensions). We don't HEAD-check the URL at
        parse time — that would require a live network call inside a parser
        — but we do validate that the URL has a plausible raster extension.
        """
        base_url = f"https://arxiv.org/html/{arxiv_id}/"
        allowed_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")
        figures = []
        for fig in soup.find_all("figure", class_="ltx_figure"):
            img = fig.find("img")
            if not img or not img.get("src"):
                continue
            src = img["src"]
            if src.startswith("data:"):
                continue
            url = src if src.startswith("http") else urljoin(base_url, src)
            # Strip query params before checking extension.
            path = url.split("?", 1)[0].lower()
            if not path.endswith(allowed_exts):
                continue

            caption_el = fig.find(["figcaption", "div"], class_=re.compile(r"ltx_caption"))
            caption = ""
            if caption_el:
                caption = caption_el.get_text(" ", strip=True)
                caption = re.sub(r"\s+", " ", caption)

            # Try to pull a "Figure N" label from the caption
            label_match = re.match(r"(Figure\s+\d+[:.]?)\s*", caption, flags=re.IGNORECASE)
            label = label_match.group(1).rstrip(":. ") if label_match else f"Figure {len(figures) + 1}"
            caption_body = caption[label_match.end():] if label_match else caption

            figures.append({
                "label": label,
                "caption": caption_body[:600],
                "url": url,
            })
        return figures

    async def fetch_batch(self, arxiv_ids: list[str]) -> dict[str, dict]:
        """Fetch HTML for multiple papers concurrently."""
        logger.info(f"Fetching arXiv HTML for {len(arxiv_ids)} papers...")
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_one(aid, session) for aid in arxiv_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        out = {}
        html_count = 0
        for aid, result in zip(arxiv_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Error fetching {aid}: {result}")
                out[aid] = {"arxiv_id": aid, "html_available": False, "sections": {}}
            else:
                out[aid] = result
                if result.get("html_available"):
                    html_count += 1

        logger.info(f"HTML available for {html_count}/{len(arxiv_ids)} papers")
        return out
