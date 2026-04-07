#!/usr/bin/env python3
"""
Step 5: Generate deep-dive content for triaged papers via DeepSeek.

For each paper in ``triaged_papers.json``:
- ``tier == "deep_dive"`` (novelty >= 4): fetches the arXiv HTML, builds a
  prompt that includes selected sections, and asks DeepSeek for a structured
  graduate-level write-up using the five-section format defined in
  ``DeepDiveAgent``.
- ``tier == "brief_mention"`` (novelty == 3): generates a one-sentence note
  highlighting the genuinely novel contribution.

Generated markdown is cached to ``work/deep_dives.json`` so the script can be
re-run cheaply, and is then patched into the existing notes under
``staging/literature/papers/<year>/<category>/`` by replacing the placeholder
``## Deep Dive`` block.

Usage:
    DEEPSEEK_API_KEY=... python scripts/05_generate_deep_dives.py \
        --config config.yaml

    # Skip API calls and just patch notes from a previous run:
    python scripts/05_generate_deep_dives.py --config config.yaml --patch-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

# Make ``src/`` importable when running as a standalone script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_discovery.agents.deep_dive import DeepDiveAgent  # noqa: E402
from paper_discovery.arxiv_html import ArxivHTMLFetcher  # noqa: E402
from paper_discovery.checkpoint import PipelineCheckpoint  # noqa: E402
from paper_discovery.config import load_config  # noqa: E402
from paper_discovery.llm_client import DeepSeekClient  # noqa: E402

logger = logging.getLogger("deep_dives")

DEEP_DIVE_HEADER = "## Deep Dive"
NEXT_SECTION_RE = re.compile(r"^## (?!Deep Dive)", re.MULTILINE)

# Minimum plausible word count for a usable deep dive. DeepSeek occasionally
# returns a half-generated response that passes the API-level check but is
# obviously unusable — e.g. <200 words, no sub-headings. Treat those as
# failures so they're retried on the next run.
_MIN_DEEP_DIVE_WORDS = 400
_REQUIRED_SUBHEADINGS = ("### ", "Method")


def _deep_dive_looks_broken(md: str) -> bool:
    if not md:
        return True
    if md.startswith("*Deep dive generation failed"):
        return True
    if len(md.split()) < _MIN_DEEP_DIVE_WORDS:
        return True
    if "### " not in md:  # no sub-sections at all
        return True
    return False


def _split_papers(triaged: list[dict]) -> tuple[list[dict], list[dict]]:
    deep = [p for p in triaged if p.get("tier") == "deep_dive"]
    brief = [p for p in triaged if p.get("tier") == "brief_mention"]
    return deep, brief


def _render_deep_dive_block(markdown: str) -> str:
    """Wrap generated content in the standard ``## Deep Dive`` section."""
    return f"{DEEP_DIVE_HEADER}\n\n{markdown.strip()}\n\n"


def _render_brief_block(brief_text: str, abstract: str = "") -> str:
    """Render the Deep Dive section for a brief-mention paper.

    Always includes the paper's abstract verbatim so the note is useful even
    when the LLM-generated summary is short or the call failed.
    """
    body = f"{DEEP_DIVE_HEADER}\n\n*Brief mention.*\n\n{brief_text.strip()}\n\n"
    if abstract:
        body += f"### Abstract\n\n{abstract.strip()}\n\n"
    return body


def _patch_note(note_path: Path, new_block: str) -> bool:
    """Replace the ``## Deep Dive`` block in a note. Returns True on change."""
    text = note_path.read_text()
    start = text.find(DEEP_DIVE_HEADER)
    if start == -1:
        logger.warning(f"No '## Deep Dive' header in {note_path}")
        return False

    after = text[start + len(DEEP_DIVE_HEADER):]
    next_match = NEXT_SECTION_RE.search(after)
    end = start + len(DEEP_DIVE_HEADER) + (next_match.start() if next_match else len(after))

    new_text = text[:start] + new_block + text[end:]
    if new_text == text:
        return False
    note_path.write_text(new_text)
    return True


def _resolve_note_path(filepath: str, staging_dir: Path) -> Path | None:
    """Resolve a manifest filepath that may be absolute or project-relative."""
    p = Path(filepath)
    candidates = [p] if p.is_absolute() else [ROOT / p, staging_dir.parent / p, Path.cwd() / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _patch_all(
    triaged: list[dict],
    manifest: list[dict],
    deep_dives: dict[str, str],
    briefs: dict[str, str],
    staging_dir: Path,
) -> int:
    """Patch every staged note that has generated content. Returns count."""
    by_id = {p["arxiv_id"]: p for p in triaged}
    manifest_by_id = {m["arxiv_id"]: m for m in manifest}

    patched = 0
    missing = 0
    for arxiv_id, paper in by_id.items():
        entry = manifest_by_id.get(arxiv_id)
        if not entry:
            continue

        note_path = _resolve_note_path(entry["filepath"], staging_dir)
        if note_path is None:
            missing += 1
            continue

        if paper.get("tier") == "deep_dive" and arxiv_id in deep_dives:
            block = _render_deep_dive_block(deep_dives[arxiv_id])
        elif paper.get("tier") == "brief_mention" and arxiv_id in briefs:
            block = _render_brief_block(briefs[arxiv_id], paper.get("abstract", ""))
        else:
            continue

        if _patch_note(note_path, block):
            patched += 1

    if missing:
        logger.warning(f"{missing} notes referenced in manifest were not found on disk")
    return patched


class CostExceeded(RuntimeError):
    """Raised when projected DeepSeek spend exceeds the configured ceiling."""


async def _run_generation(
    config: dict,
    triaged: list[dict],
    manifest: list[dict],
    cache_path: Path,
    checkpoint_path: Path,
    failures_path: Path,
    max_cost_usd: float | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Run DeepDiveAgent and return ({arxiv_id: deep_dive_md}, {arxiv_id: brief_md})."""
    deep_papers, brief_papers = _split_papers(triaged)
    logger.info(f"Generating: {len(deep_papers)} deep dives, {len(brief_papers)} brief mentions")

    # Resume cache from previous runs.
    cached: dict[str, dict[str, str]] = {"deep_dives": {}, "briefs": {}}
    if cache_path.exists():
        cached.update(json.loads(cache_path.read_text()))
        # Drop previous failure fallbacks so we retry them this run. The
        # marker string is written by ``DeepDiveAgent.generate_all`` and by
        # the validator below.
        dropped = [
            aid for aid, md in list(cached["deep_dives"].items())
            if isinstance(md, str) and md.startswith("*Deep dive generation failed")
        ]
        for aid in dropped:
            del cached["deep_dives"][aid]
        if dropped:
            logger.info(f"Retrying {len(dropped)} previously-failed deep dives")
        logger.info(
            f"Resuming from cache: {len(cached['deep_dives'])} deep, "
            f"{len(cached['briefs'])} brief already done"
        )

    checkpoint = PipelineCheckpoint.load(checkpoint_path)

    client = DeepSeekClient(config)
    fetcher = ArxivHTMLFetcher(config)
    agent = DeepDiveAgent(
        client=client,
        fetcher=fetcher,
        config=config,
        all_papers=triaged,
        manifest=manifest,
    )

    # Failures file also doubles as an attempt counter so we stop hammering
    # a paper that consistently fails. Schema: {arxiv_id: {"attempts": int, "last_error": str}}.
    MAX_ATTEMPTS = 3
    failures: dict[str, dict] = {}
    if failures_path.exists():
        raw = json.loads(failures_path.read_text())
        for aid, v in raw.items():
            if isinstance(v, dict):
                failures[aid] = v
            else:  # legacy: bare error string
                failures[aid] = {"attempts": 1, "last_error": str(v)}

    # Don't retry papers that have already exhausted their attempts.
    exhausted = {aid for aid, v in failures.items() if v.get("attempts", 0) >= MAX_ATTEMPTS}
    if exhausted:
        logger.warning(f"{len(exhausted)} papers have exhausted retries; skipping")

    def _persist() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2))
        if failures:
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            failures_path.write_text(json.dumps(failures, indent=2))

    def _check_cost() -> None:
        if max_cost_usd is None:
            return
        spent = client.cost_estimate()["total_cost_usd"]
        if spent > max_cost_usd:
            raise CostExceeded(
                f"Projected spend ${spent:.2f} exceeds --max-cost-usd ${max_cost_usd:.2f}"
            )

    def _on_done(arxiv_id: str) -> None:
        checkpoint.mark_deep_dive(arxiv_id, checkpoint_path)
        _check_cost()

    already_deep = set(cached["deep_dives"].keys()) | exhausted
    new_deep = await agent.generate_all(
        deep_papers,
        already_done=already_deep,
        on_done=_on_done,
    )
    # Anything that came back as a failure-fallback or failed validation is
    # recorded with an incremented attempt counter.
    for aid, md in new_deep.items():
        if _deep_dive_looks_broken(md):
            prev = failures.get(aid, {"attempts": 0, "last_error": ""})
            failures[aid] = {
                "attempts": prev.get("attempts", 0) + 1,
                "last_error": md[:200],
            }
        else:
            # Success — clear any stale failure record.
            failures.pop(aid, None)
    cached["deep_dives"].update(new_deep)
    _persist()

    # Brief mentions: only run for those not yet cached.
    todo_brief = [p for p in brief_papers if p["arxiv_id"] not in cached["briefs"]]
    if todo_brief:
        new_brief = await agent.generate_brief_mentions(todo_brief)
        cached["briefs"].update(new_brief)
        _persist()
        _check_cost()
    else:
        logger.info("All brief mentions already cached")

    checkpoint.html_fetch_complete = True
    checkpoint.brief_mentions_done = True
    checkpoint.save(checkpoint_path)

    cost = client.cost_estimate()
    logger.info(
        f"DeepSeek usage: {cost['input_tokens']} in / {cost['output_tokens']} out "
        f"-> ${cost['total_cost_usd']:.4f}"
    )

    return cached["deep_dives"], cached["briefs"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--triaged",
        default="work/triaged_papers.json",
        help="Triaged papers JSON (output of step 04 + triage)",
    )
    parser.add_argument(
        "--manifest",
        default="staging/notes_manifest.json",
        help="Notes manifest from step 06",
    )
    parser.add_argument(
        "--cache",
        default="work/deep_dives.json",
        help="Where generated deep dives are cached for resume",
    )
    parser.add_argument(
        "--checkpoint",
        default="work/checkpoint.json",
        help="Pipeline checkpoint file",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="Skip API calls; only patch notes from cached deep dives",
    )
    parser.add_argument(
        "--failures",
        default="work/deep_dive_failures.json",
        help="Where per-paper failure messages are recorded",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Abort once estimated DeepSeek spend exceeds this many USD",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.patch_only:
        import yaml

        config = yaml.safe_load(Path(args.config).read_text())
    else:
        config = load_config(args.config)
    triaged = json.loads(Path(args.triaged).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    staging_dir = Path(config.get("staging_dir", "./staging")).resolve()
    cache_path = Path(args.cache)
    checkpoint_path = Path(args.checkpoint)

    if args.patch_only:
        if not cache_path.exists():
            sys.exit(f"--patch-only requires {cache_path} to exist")
        cached = json.loads(cache_path.read_text())
        deep_dives = cached.get("deep_dives", {})
        briefs = cached.get("briefs", {})
    else:
        try:
            deep_dives, briefs = asyncio.run(
                _run_generation(
                    config,
                    triaged,
                    manifest,
                    cache_path,
                    checkpoint_path,
                    Path(args.failures),
                    max_cost_usd=args.max_cost_usd,
                )
            )
        except CostExceeded as e:
            logger.error(str(e))
            sys.exit(2)

    patched = _patch_all(triaged, manifest, deep_dives, briefs, staging_dir)
    logger.info(f"Patched {patched} notes in {staging_dir}")
    print(
        f"\nDone. Deep dives: {len(deep_dives)}  "
        f"Brief mentions: {len(briefs)}  Notes patched: {patched}"
    )


if __name__ == "__main__":
    main()
