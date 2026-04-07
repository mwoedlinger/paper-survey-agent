#!/usr/bin/env python3
"""
Step 2: Deduplicate candidates and compute composite scores.

Merges all candidates from different sources by arxiv ID, combines metadata,
and computes a recency-normalized citation score with source bonuses.

Scoring formula:
  base_score = influential_citation_count / max(1, months_since_publication)
  normalized_score = base_score / max_base_score  (0 to 1)
  source_bonus = 0.3 if conference best paper, 0.15 if blog mention
  final_score = normalized_score + source_bonus

This addresses the recency bias: a paper from December with 50 influential
citations in 2 months scores higher than a January paper with 200 over 12 months.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import requests
import yaml

logger = logging.getLogger(__name__)
_unparseable_dates: list[str] = []

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_FIELDS = "externalIds,citationCount,influentialCitationCount,venue,publicationDate,paperId"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_candidates(filepath: str) -> list[dict]:
    """Load all candidates from JSONL file."""
    candidates = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def merge_candidates(candidates: list[dict]) -> dict[str, dict]:
    """Merge candidates by arxiv ID, combining metadata from all sources."""
    merged = {}

    for c in candidates:
        arxiv_id = c.get("arxiv_id")
        if not arxiv_id:
            continue

        if arxiv_id not in merged:
            merged[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "title": c.get("title", ""),
                "abstract": c.get("abstract", ""),
                "authors": c.get("authors", []),
                "published_date": c.get("published_date", ""),
                "venue": c.get("venue", ""),
                "citation_count": c.get("citation_count", 0),
                "influential_citation_count": c.get("influential_citation_count", 0),
                "hf_upvotes": c.get("hf_upvotes", 0),
                "sources": set(),
                "s2_paper_id": c.get("s2_paper_id", ""),
            }

        paper = merged[arxiv_id]
        paper["sources"].add(c.get("source", "unknown"))

        # Take the best metadata from each source
        if c.get("abstract") and not paper["abstract"]:
            paper["abstract"] = c["abstract"]
        if c.get("title") and not paper["title"]:
            paper["title"] = c["title"]
        if c.get("venue") and not paper["venue"]:
            paper["venue"] = c["venue"]
        if c.get("authors") and not paper["authors"]:
            paper["authors"] = c["authors"]
        if c.get("published_date") and not paper["published_date"]:
            paper["published_date"] = c["published_date"]
        if c.get("s2_paper_id") and not paper["s2_paper_id"]:
            paper["s2_paper_id"] = c["s2_paper_id"]

        # Take highest values
        paper["citation_count"] = max(paper["citation_count"], c.get("citation_count", 0))
        paper["influential_citation_count"] = max(
            paper["influential_citation_count"], c.get("influential_citation_count", 0)
        )
        paper["hf_upvotes"] = max(paper["hf_upvotes"], c.get("hf_upvotes", 0))

    return merged


def enrich_citations_from_s2(
    merged: dict[str, dict],
    api_key: str | None,
    batch_size: int = 200,
) -> int:
    """Fetch fresh citation counts from Semantic Scholar for every merged paper.

    The S2 ``/paper/batch`` endpoint accepts up to 500 IDs per call. We use
    ``ArXiv:<id>`` lookups, then write the response back into ``merged`` so
    every paper has up-to-date citation_count and influential_citation_count
    regardless of which harvester originally surfaced it.

    Returns the number of papers that gained non-zero citation data.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    all_ids = list(merged.keys())
    arxiv_lookup = [f"ArXiv:{aid}" for aid in all_ids]
    enriched = 0

    for start in range(0, len(arxiv_lookup), batch_size):
        chunk = arxiv_lookup[start : start + batch_size]
        chunk_ids = all_ids[start : start + batch_size]
        try:
            resp = requests.post(
                S2_BATCH_URL,
                params={"fields": S2_BATCH_FIELDS},
                json={"ids": chunk},
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(
                f"  Warning: S2 batch enrichment failed for chunk {start}: {e}",
                file=sys.stderr,
            )
            time.sleep(2.0)
            continue

        # The endpoint returns a list aligned with the input order; missing
        # papers come back as ``null`` rather than being dropped.
        for arxiv_id, entry in zip(chunk_ids, data):
            if not entry:
                continue
            paper = merged.get(arxiv_id)
            if not paper:
                continue
            new_total = entry.get("citationCount") or 0
            new_inf = entry.get("influentialCitationCount") or 0
            paper["citation_count"] = max(paper.get("citation_count", 0), new_total)
            paper["influential_citation_count"] = max(
                paper.get("influential_citation_count", 0), new_inf
            )
            if not paper.get("venue") and entry.get("venue"):
                paper["venue"] = entry["venue"]
            if not paper.get("s2_paper_id") and entry.get("paperId"):
                paper["s2_paper_id"] = entry["paperId"]
            if not paper.get("published_date") and entry.get("publicationDate"):
                paper["published_date"] = entry["publicationDate"]
            if new_total > 0:
                enriched += 1

        # Be polite to S2: 1s with key, 3s without.
        time.sleep(1.0 if api_key else 3.0)

    return enriched


def get_s2_api_key(config: dict) -> str | None:
    key = config.get("semantic_scholar_api_key", "")
    if not key:
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    return key or None


def compute_months_since_publication(published_date: str, reference_date: datetime) -> float:
    """Compute months since publication for recency normalization.

    Unparseable dates fall back to 6 months and are collected in
    ``_unparseable_dates`` so the caller can surface a single summary
    warning rather than spamming stderr per paper.
    """
    if not published_date:
        _unparseable_dates.append("<empty>")
        return 6.0

    try:
        for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m"]:
            try:
                pub_dt = datetime.strptime(published_date[:len(fmt.replace("%", "X"))], fmt)
                break
            except (ValueError, IndexError):
                continue
        else:
            parts = published_date.split("-")
            if len(parts) >= 2:
                pub_dt = datetime(int(parts[0]), int(parts[1]), 15)
            else:
                _unparseable_dates.append(published_date)
                return 6.0

        delta = reference_date - pub_dt
        return max(0.5, delta.days / 30.44)
    except Exception:
        _unparseable_dates.append(published_date)
        return 6.0


def compute_scores(merged: dict[str, dict], config: dict) -> list[dict]:
    """Compute composite scores for all papers.

    The score combines three signals:
      - Citation signal: influential_citation_count / months (recency-normalized)
      - Community signal: HuggingFace upvotes (log-scaled)
      - Source bonus: conference best paper / top-cited at venue

    This ensures papers discovered via HF (community curation) and via S2
    (citation-backed) are ranked on a unified scale.
    """
    scoring = config.get("scoring", {})
    best_paper_bonus = scoring.get("conference_best_paper_bonus", 0.3)
    blog_bonus = scoring.get("blog_mention_bonus", 0.15)
    citation_weight = scoring.get("citation_weight", 0.5)
    upvote_weight = scoring.get("upvote_weight", 0.5)

    reference_date = datetime.now()
    papers = list(merged.values())

    # Step 1: Compute citation-based scores (influential citations per month)
    for p in papers:
        months = compute_months_since_publication(p["published_date"], reference_date)
        p["months_since_publication"] = round(months, 1)
        p["citation_rate"] = p["influential_citation_count"] / max(1, months)

    # Step 2: Normalize citation scores to [0, 1]
    max_citation_rate = max((p["citation_rate"] for p in papers), default=1)
    if max_citation_rate == 0:
        max_citation_rate = 1

    # Step 3: Normalize HF upvotes to [0, 1] using log scale
    import math
    max_upvotes = max((p.get("hf_upvotes", 0) for p in papers), default=1)
    if max_upvotes == 0:
        max_upvotes = 1
    log_max = math.log1p(max_upvotes)

    for p in papers:
        p["normalized_citation"] = p["citation_rate"] / max_citation_rate
        upvotes = p.get("hf_upvotes", 0)
        p["normalized_upvotes"] = math.log1p(upvotes) / log_max if log_max > 0 else 0

        # Step 4: Combined score = weighted sum of signals + source bonus
        source_bonus = 0.0
        if "conference_best_paper" in p["sources"]:
            source_bonus = best_paper_bonus
        elif "conference_top_cited" in p["sources"]:
            source_bonus = blog_bonus

        p["source_bonus"] = source_bonus
        p["final_score"] = (
            citation_weight * p["normalized_citation"]
            + upvote_weight * p["normalized_upvotes"]
            + source_bonus
        )

        # Convert sources set to list for JSON serialization
        p["sources"] = sorted(list(p["sources"]))

    # Sort by final score descending
    papers.sort(key=lambda p: p["final_score"], reverse=True)

    return papers


def main():
    parser = argparse.ArgumentParser(description="Deduplicate and score paper candidates")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input", default="candidates.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="scored_papers.json", help="Output JSON file")
    args = parser.parse_args()

    config = load_config(args.config)

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run harvesting scripts first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading candidates from {args.input}...")
    candidates = load_candidates(args.input)
    print(f"Loaded {len(candidates)} raw candidates")

    print("Merging by arxiv ID...")
    merged = merge_candidates(candidates)
    print(f"Merged to {len(merged)} unique papers")

    print("Enriching with Semantic Scholar citation counts...")
    api_key = get_s2_api_key(config)
    enriched = enrich_citations_from_s2(merged, api_key)
    print(f"  {enriched}/{len(merged)} papers now have non-zero citation counts")

    print("Computing scores...")
    scored = compute_scores(merged, config)

    if _unparseable_dates:
        sample = ", ".join(sorted(set(_unparseable_dates))[:3])
        print(
            f"Warning: {len(_unparseable_dates)} papers had unparseable "
            f"published_date (defaulted to 6 months). Samples: {sample}",
            file=sys.stderr,
        )

    # Print summary statistics
    source_counts = {}
    for p in scored:
        for s in p["sources"]:
            source_counts[s] = source_counts.get(s, 0) + 1

    print(f"\nSource distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    print(f"\nTop 20 papers by score:")
    for i, p in enumerate(scored[:20]):
        sources_str = ", ".join(p["sources"])
        print(
            f"  {i+1:3d}. [{p['final_score']:.3f}] {p['title'][:70]}"
            f" (IC:{p['influential_citation_count']}, {sources_str})"
        )

    # Save
    with open(args.output, "w") as f:
        json.dump(scored, f, indent=2)

    print(f"\nScored papers written to {args.output}")
    print(f"Total: {len(scored)} papers ready for LLM triage")


if __name__ == "__main__":
    main()
