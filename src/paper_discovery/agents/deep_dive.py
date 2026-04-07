"""Deep-dive agent: generate graduate-level paper summaries via DeepSeek."""

import asyncio
import logging
import re
from typing import Callable

from paper_discovery.llm_client import DeepSeekClient
from paper_discovery.arxiv_html import ArxivHTMLFetcher

logger = logging.getLogger(__name__)

DEEP_DIVE_SYSTEM_PROMPT = """\
You are writing a thorough graduate-level research note for a researcher with a
PhD in computer vision and machine learning. The reader has expert-level
familiarity with optimization, attention, loss functions, statistical
evaluation, and modern deep-learning architectures. Your goal is that after
reading your note they understand the paper deeply enough that they only need
to open the PDF for low-level implementation details or to inspect a specific
figure. Be substantive — aim for ~700-1100 words across the sections.

Write a deep dive with exactly these five markdown sections:

### Problem & Motivation
Why does this problem matter? What concrete gap in prior work does this paper
address? Spell out the technical limitation that motivated this work and the
context in the field. (4-6 sentences)

### Method
Detailed technical description at the level of a reading-group presentation.
Cover:
- The full pipeline: inputs, intermediate representations, outputs
- Key architectural / algorithmic choices and WHY each was made
- Mathematical formulation of the core contribution (use LaTeX: $...$ inline,
  $$...$$ display) — losses, update rules, key equations
- How this differs from the obvious baseline and from the closest prior work
- Training data, scale, optimisation specifics that matter for reproduction
(8-14 sentences, use sub-bullets where it aids clarity)

### Key Technical Details
The 3-5 specific insights that make this work novel. Things you would actually
flag at a reading group: clever design choices, non-obvious training tricks,
theoretical results, or empirical observations that drive the main claim.
(5-8 sentences)

### Results
Concrete numbers on the main benchmarks: which datasets, which metrics, the
strongest baselines, and the absolute / relative improvements. Call out the
most informative ablation. Mention any surprising findings, failure modes, or
results that contradict prior assumptions. (5-8 sentences)

### Limitations & Open Questions
Honest assessment. Assumptions the method bakes in, regimes where it would
likely fail, compute / data requirements that constrain adoption, and what
follow-up work this enables. Include [[wikilinks]] to related papers where
relevant, using the filenames listed in the prompt. (4-6 sentences)

Guidelines:
- Be technically precise. Use correct terminology and concrete numbers.
- Do NOT pad with filler phrases ("In summary," "It is important to note"). Cut.
- Do NOT just paraphrase the abstract — pull from the full paper text when
  available and add analytical depth.
- When full paper text is unavailable, clearly mark what you are inferring.
- If figures with image URLs are provided, embed the 1-2 most informative ones
  inline using markdown image syntax: ![Figure N: short caption](url). Place
  them right after the section where they help most (usually Method or
  Results). Do NOT invent figure URLs — only use the ones supplied.
- Wikilinks: use [[Filename]] format exactly matching the filenames provided."""

BRIEF_MENTION_SYSTEM_PROMPT = """\
You are writing a short research note for a PhD-level ML researcher about a
paper that is worth tracking but not deep-diving. Output 2-3 short paragraphs
of plain markdown (NO headings):

1. One sentence stating the genuinely novel contribution — be specific about
   the mechanism, not vague ("proposes a new method").
2. 3-5 sentences expanding on the method and the main result, in the style of
   a condensed abstract written for an expert. Mention the dataset / benchmark
   and the headline number if available.
3. (Optional) one sentence on the most important limitation or caveat.

Be concrete and technical. Do not invent numbers."""


def _sanitize_filename(title: str) -> str:
    """Mirror of ``scripts/06_generate_obsidian_notes.sanitize_filename``."""
    clean = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", title)
    clean = re.sub(r"\s+", " ", clean).strip().rstrip(". ")
    return clean[:120].rstrip(". ")


def _find_related(paper: dict, all_papers: list[dict], manifest_by_id: dict) -> list[str]:
    """Find filenames of related papers for wikilinks."""
    cat = paper.get("category", "")
    title_words = set(paper.get("title", "").lower().split())
    stopwords = {"a", "an", "the", "for", "of", "in", "to", "and", "with", "via", "from", "on", "by", "is", "are", "towards", "toward"}
    title_words -= stopwords

    related = []
    for other in all_papers:
        if other["arxiv_id"] == paper.get("arxiv_id"):
            continue
        other_words = set(other.get("title", "").lower().split()) - stopwords
        if len(title_words & other_words) >= 2 or other.get("category") == cat:
            entry = manifest_by_id.get(other["arxiv_id"])
            if entry:
                related.append((entry["filename"].replace(".md", ""), other.get("title", "")))
    return related[:8]


# Approximate token budget for the user-prompt body. DeepSeek's context is
# 64k tokens; ~4 chars/token gives us a generous budget while keeping room
# for the system prompt + a substantial completion.
_HTML_CHAR_BUDGET = 60_000
_PER_SECTION_CHAR_CAP = 12_000
_MAX_FIGURES = 4


def _build_deep_dive_prompt(
    paper: dict,
    html: dict,
    related: list[tuple[str, str]],
    char_budget: int = _HTML_CHAR_BUDGET,
) -> str:
    """Build user prompt for a deep-dive paper, respecting a global char budget."""
    parts = [
        "Write a deep dive for this paper.\n",
        "METADATA:",
        f"Title: {paper.get('title', '')}",
        f"Authors: {', '.join((paper.get('authors') or [])[:5])}",
        f"ArXiv: {paper.get('arxiv_id', '')}",
        f"Category: {paper.get('category', '')}",
        f"Key Insight: {paper.get('one_line_insight', '')}",
    ]
    abstract = paper.get("abstract") or html.get("abstract", "")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    if related:
        parts.append("\nRELATED PAPERS IN VAULT (use [[filename]] for wikilinks):")
        for fname, title in related:
            parts.append(f"- [[{fname}]] -- {title[:80]}")

    figures = (html.get("figures") or [])[:_MAX_FIGURES]
    if figures:
        parts.append(
            "\nAVAILABLE FIGURES (embed the 1-2 most informative inline using "
            "![label: short caption](url) — never invent URLs):"
        )
        for fig in figures:
            cap = (fig.get("caption") or "").strip()
            parts.append(f"- {fig.get('label', 'Figure')} | url: {fig['url']} | caption: {cap[:300]}")

    sections = html.get("sections", {})
    if html.get("html_available") and sections:
        parts.append("\nFULL PAPER TEXT:")

        section_order = [
            ("introduction",),
            ("method", "approach", "model", "architecture", "framework", "design"),
            ("experiment", "result", "evaluation", "findings"),
            ("ablation", "analysis"),
            ("discussion",),
            ("conclusion", "limitation"),
        ]
        used_keys: set[str] = set()
        remaining = char_budget
        for keywords in section_order:
            if remaining <= 0:
                break
            for key, sec in sections.items():
                if key in used_keys:
                    continue
                if any(kw in key for kw in keywords):
                    snippet = sec["text"][: min(_PER_SECTION_CHAR_CAP, remaining)]
                    parts.append(f"\n=== {sec['title']} ===")
                    parts.append(snippet)
                    remaining -= len(snippet)
                    used_keys.add(key)
                    break
    else:
        parts.append(
            f"\nABSTRACT ONLY (no HTML available — flag inferences explicitly):"
            f"\n{abstract[:char_budget]}"
        )

    return "\n".join(parts)


class DeepDiveAgent:
    def __init__(
        self,
        client: DeepSeekClient,
        fetcher: ArxivHTMLFetcher,
        config: dict,
        all_papers: list[dict],
        manifest: list[dict],
    ):
        self.client = client
        self.fetcher = fetcher
        self.config = config
        self.all_papers = all_papers
        self.manifest_by_id = {e["arxiv_id"]: e for e in manifest}
        self.max_tokens = config["deepseek"]["deep_dive_max_tokens"]

    async def generate_all(
        self,
        deep_dive_papers: list[dict],
        already_done: set[str] | None = None,
        on_done: Callable[[str], None] | None = None,
    ) -> dict[str, str]:
        """Generate deep dives. Returns {arxiv_id: markdown}."""
        todo = [p for p in deep_dive_papers if p["arxiv_id"] not in (already_done or set())]
        if not todo:
            logger.info("All deep dives already complete (checkpoint)")
            return {}

        logger.info(f"Generating deep dives for {len(todo)} papers...")

        # Phase 1: fetch HTML
        html_data = await self.fetcher.fetch_batch([p["arxiv_id"] for p in todo])

        # Phase 2: generate deep dives concurrently
        tasks = [self._generate_one(p, html_data.get(p["arxiv_id"], {})) for p in todo]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for paper, result in zip(todo, results_list):
            aid = paper["arxiv_id"]
            if isinstance(result, Exception):
                logger.error(f"Deep dive failed for {aid}: {result}")
                # Fallback: use one_line_insight
                results[aid] = f"*Deep dive generation failed. Key insight: {paper.get('one_line_insight', '')}*"
            else:
                results[aid] = result
            if on_done:
                on_done(aid)

        logger.info(f"Generated {len(results)} deep dives")
        return results

    async def _generate_one(self, paper: dict, html: dict) -> str:
        related = _find_related(paper, self.all_papers, self.manifest_by_id)
        user_prompt = _build_deep_dive_prompt(paper, html, related)
        return await self.client.complete_text(
            DEEP_DIVE_SYSTEM_PROMPT, user_prompt, max_tokens=self.max_tokens
        )

    async def generate_brief_mentions(self, papers: list[dict]) -> dict[str, str]:
        """Generate one-liner mentions for brief-mention papers."""
        logger.info(f"Generating brief mentions for {len(papers)} papers...")
        tasks = [self._brief_one(p) for p in papers]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for paper, result in zip(papers, results_list):
            aid = paper["arxiv_id"]
            if isinstance(result, Exception):
                results[aid] = paper.get("one_line_insight", "")
            else:
                results[aid] = result
        return results

    async def _brief_one(self, paper: dict) -> str:
        abstract = (paper.get("abstract") or "")[:2500]
        venue = paper.get("venue") or ""
        prompt = (
            f"Title: {paper.get('title', '')}\n"
            f"Venue: {venue}\n"
            f"Citations: {paper.get('citation_count', 0)} "
            f"(influential: {paper.get('influential_citation_count', 0)})\n"
            f"Abstract: {abstract}"
        )
        return await self.client.complete_text(
            BRIEF_MENTION_SYSTEM_PROMPT, prompt, max_tokens=600
        )
