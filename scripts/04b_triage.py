#!/usr/bin/env python3
"""
Step 4b: LLM triage of scored papers via DeepSeek (JSON mode).

Reads ``work/scored_papers.json``, runs ``TriageAgent`` in batches, and writes
``work/triaged_papers.json``. Enforces the ``triage.min_score`` and
``triage.target_count`` thresholds from the config so the downstream
deep-dive stage operates on a bounded set.

Usage:
    DEEPSEEK_API_KEY=... python scripts/04b_triage.py --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_discovery.agents.triage import TriageAgent  # noqa: E402
from paper_discovery.checkpoint import PipelineCheckpoint  # noqa: E402
from paper_discovery.config import load_config  # noqa: E402
from paper_discovery.llm_client import DeepSeekClient  # noqa: E402

logger = logging.getLogger("triage")

CITATION_FIELDS = (
    "citation_count",
    "influential_citation_count",
    "venue",
    "s2_paper_id",
    "published_date",
)


def _merge_fresh_citations(triaged: list[dict], scored: list[dict]) -> int:
    """Refresh citation metadata on ``triaged`` from the latest ``scored`` list.

    Triage preserves the scored dict it receives, but if the user re-ran
    ``make score`` between triage runs those snapshots can drift. Merging
    here guarantees the emitted file reflects the most recent enrichment.
    """
    by_id = {p["arxiv_id"]: p for p in scored}
    refreshed = 0
    for p in triaged:
        fresh = by_id.get(p.get("arxiv_id"))
        if not fresh:
            continue
        changed = False
        for field in ("citation_count", "influential_citation_count"):
            v = fresh.get(field, 0) or 0
            if v > (p.get(field, 0) or 0):
                p[field] = v
                changed = True
        for field in ("venue", "s2_paper_id", "published_date"):
            if not p.get(field) and fresh.get(field):
                p[field] = fresh[field]
                changed = True
        if changed:
            refreshed += 1
    return refreshed


def _checkpoint_is_stale(checkpoint_path: Path, scored_path: Path) -> bool:
    """Return True if the checkpoint predates the scored file.

    We want to rerun triage on papers whose citation rank may have changed.
    Stale triage decisions are still correct (novelty is content-based), but
    the dict bodies need a citation refresh — handled separately by
    ``_merge_fresh_citations`` — so we only invalidate the checkpoint when
    there's a strong signal that the upstream data shifted.
    """
    if not checkpoint_path.exists() or not scored_path.exists():
        return False
    return scored_path.stat().st_mtime > checkpoint_path.stat().st_mtime


def apply_thresholds(papers: list[dict], config: dict) -> list[dict]:
    """Enforce ``min_score`` and ``target_count`` from config.

    Sorts deep-dive papers (>= min_score) ahead of brief mentions, then
    truncates to target_count. Brief mentions are kept only if there's
    headroom under target_count, since they cost an LLM call each.
    """
    triage_cfg = config.get("triage", {})
    min_score = triage_cfg.get("min_score", 4)
    target = triage_cfg.get("target_count", 100) or 0  # 0/None = unlimited

    deep = [p for p in papers if p.get("novelty_score", 0) >= min_score]
    brief = [
        p for p in papers
        if p.get("tier") == "brief_mention" and p.get("novelty_score", 0) < min_score
    ]

    deep.sort(key=lambda p: (-p.get("novelty_score", 0), -p.get("final_score", 0)))
    brief.sort(key=lambda p: -p.get("final_score", 0))

    # Re-tier deep-dive papers in case the LLM disagreed with itself.
    for p in deep:
        p["tier"] = "deep_dive"

    if target <= 0:
        # No cap — keep every deep dive and every brief mention.
        kept = deep + brief
    elif len(deep) >= target:
        kept = deep[:target]
    else:
        slack = target - len(deep)
        kept = deep + brief[:slack]

    logger.info(
        f"Threshold: kept {len(kept)} (deep={sum(1 for p in kept if p['tier']=='deep_dive')}, "
        f"brief={sum(1 for p in kept if p['tier']=='brief_mention')}) from {len(papers)}"
    )
    return kept


async def _run(config: dict, scored: list[dict], checkpoint_path: Path) -> list[dict]:
    checkpoint = PipelineCheckpoint.load(checkpoint_path)
    already = set(checkpoint.triaged_arxiv_ids)

    client = DeepSeekClient(config)
    agent = TriageAgent(client, config)

    def _on_batch(arxiv_ids: list[str]) -> None:
        checkpoint.mark_triage_batch(arxiv_ids, checkpoint_path)

    triaged = await agent.triage_all(
        scored, already_triaged=already, on_batch_done=_on_batch
    )

    checkpoint.triage_complete = True
    checkpoint.save(checkpoint_path)

    cost = client.cost_estimate()
    logger.info(
        f"DeepSeek usage: {cost['input_tokens']} in / {cost['output_tokens']} out "
        f"-> ${cost['total_cost_usd']:.4f}"
    )
    return triaged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default="work/scored_papers.json")
    parser.add_argument("--output", default="work/triaged_papers.json")
    parser.add_argument("--checkpoint", default="work/checkpoint.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    scored = json.loads(Path(args.input).read_text())
    logger.info(f"Loaded {len(scored)} scored papers from {args.input}")

    checkpoint_path = Path(args.checkpoint)
    if _checkpoint_is_stale(checkpoint_path, Path(args.input)):
        logger.warning(
            f"{args.input} is newer than {checkpoint_path}; resetting triage "
            "checkpoint so fresh citation data propagates."
        )
        checkpoint_path.unlink()

    new_triaged = asyncio.run(_run(config, scored, checkpoint_path))

    # If resuming with a warm checkpoint, the agent only re-triages newly
    # added papers. Merge the previous output so we don't drop everything.
    prior_path = Path(args.output)
    prior: dict[str, dict] = {}
    if prior_path.exists():
        for p in json.loads(prior_path.read_text()):
            prior[p["arxiv_id"]] = p
    for p in new_triaged:
        prior[p["arxiv_id"]] = p
    triaged = list(prior.values())

    # Even with a warm checkpoint, previously-triaged papers need their
    # citation bodies refreshed from the latest scored file.
    refreshed = _merge_fresh_citations(triaged, scored)
    if refreshed:
        logger.info(f"Refreshed citation data on {refreshed} triaged papers")

    final = apply_thresholds(triaged, config)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2))
    logger.info(f"Wrote {len(final)} triaged papers to {out}")


if __name__ == "__main__":
    main()
