"""Triage agent: batch-assess papers for novelty via DeepSeek JSON mode."""

import asyncio
import logging
from typing import Callable

from paper_discovery.llm_client import DeepSeekClient

logger = logging.getLogger(__name__)

TRIAGE_SYSTEM_PROMPT = """\
You are an expert ML research triage system. You evaluate papers for a researcher
with a PhD in computer vision and machine learning. Assess genuine novelty — not
popularity or production quality.

Assign each paper:
- novelty_score (1-5):
  5 = paradigm-shifting or foundational new idea
  4 = clearly novel technique/insight with broad implications
  3 = solid work, worth tracking but not deeply novel
  2 = incremental improvement on existing methods
  1 = routine application, scaling, or evaluation paper
- category: One of: {category_keys}
- one_line_insight: The single genuinely novel contribution in ONE sentence.
  Be specific about the technical mechanism.
  Bad: "Introduces a novel approach for image generation"
  Good: "Replaces diffusion denoising with single-step flow matching, achieving 10x faster inference at equivalent FID"
- tier: "deep_dive" if novelty_score >= 4, "brief_mention" if novelty_score == 3, "exclude" otherwise.

Category definitions:
{category_defs}

Respond in json with key "papers" containing an array of objects with keys:
arxiv_id, novelty_score, category, one_line_insight, tier."""


def _build_system_prompt(categories: dict) -> str:
    keys = list(categories.keys())
    defs = "\n".join(
        f"- {k}: {v.get('display_name', k)} — keywords: {', '.join(v.get('search_keywords', [])[:4])}"
        for k, v in categories.items()
        if k != "other"
    )
    defs += "\n- other: Papers that don't fit the above categories"
    return TRIAGE_SYSTEM_PROMPT.format(category_keys=", ".join(keys), category_defs=defs)


def _build_batch_prompt(papers: list[dict]) -> str:
    parts = [f"Evaluate these {len(papers)} papers. Be ruthless: most ML papers are incremental. Reserve score 5 for genuine breakthroughs.\n"]
    for p in papers:
        abstract = (p.get("abstract") or "")[:1200].replace("\n", " ")
        parts.append(
            f"---\n"
            f"arxiv_id: {p['arxiv_id']}\n"
            f"title: {p.get('title', '')}\n"
            f"citations: {p.get('citation_count', 0)} | influential: {p.get('influential_citation_count', 0)} | HF upvotes: {p.get('hf_upvotes', 0)}\n"
            f"abstract: {abstract}\n"
            f"---"
        )
    return "\n".join(parts)


class TriageAgent:
    def __init__(self, client: DeepSeekClient, config: dict):
        self.client = client
        self.config = config
        self.batch_size = config["deepseek"]["triage_batch_size"]
        self.max_tokens = config["deepseek"]["triage_max_tokens"]
        self.max_pool = config.get("triage", {}).get("max_triage_pool", 500)
        self.categories = config.get("categories", {})
        self._system_prompt = _build_system_prompt(self.categories)

    async def triage_all(
        self,
        scored_papers: list[dict],
        already_triaged: set[str] | None = None,
        on_batch_done: Callable[[list[str]], None] | None = None,
    ) -> list[dict]:
        """Triage top papers. Returns list of papers with novelty assessments.

        Args:
            scored_papers: Papers sorted by final_score descending.
            already_triaged: arxiv_ids to skip (checkpoint resume).
            on_batch_done: Callback with arxiv_ids after each batch completes.
        """
        pool = scored_papers[: self.max_pool]
        if already_triaged:
            pool = [p for p in pool if p["arxiv_id"] not in already_triaged]

        logger.info(f"Triaging {len(pool)} papers in batches of {self.batch_size}")
        batches = [pool[i : i + self.batch_size] for i in range(0, len(pool), self.batch_size)]

        results: list[dict] = []
        tasks = [self._triage_batch(batch) for batch in batches]

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                batch_results = await coro
                results.extend(batch_results)
                if on_batch_done:
                    on_batch_done([p["arxiv_id"] for p in batch_results])
                logger.info(f"Batch {i + 1}/{len(batches)} done: {len(batch_results)} papers kept")
            except Exception as e:
                logger.error(f"Batch {i + 1} failed: {e}")

        # Sort: deep_dive first (by novelty desc), then brief_mention
        results.sort(key=lambda p: (-p.get("novelty_score", 0), p.get("title", "")))
        return results

    async def _triage_batch(self, papers: list[dict]) -> list[dict]:
        """Triage a single batch via LLM."""
        user_prompt = _build_batch_prompt(papers)
        response = await self.client.complete_json(
            self._system_prompt, user_prompt, max_tokens=self.max_tokens
        )

        decisions = {d["arxiv_id"]: d for d in response.get("papers", [])}
        enriched = []
        for p in papers:
            d = decisions.get(p["arxiv_id"])
            if not d:
                continue
            p.update({
                "novelty_score": d.get("novelty_score", 2),
                "category": d.get("category", "other"),
                "one_line_insight": d.get("one_line_insight", ""),
                "tier": d.get("tier", "exclude"),
            })
            if p["tier"] != "exclude":
                enriched.append(p)
        return enriched
