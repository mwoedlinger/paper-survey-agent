#!/usr/bin/env python3
"""
Step 1c: Harvest conference best paper awards.

This script maintains a curated list of best paper award URLs and attempts
to extract paper titles from them. Since conference award pages have
inconsistent formats, this script takes a hybrid approach:

1. Try to fetch known award page URLs and extract titles
2. Fall back to Semantic Scholar search for "best paper award <venue> <year>"
3. Output candidates with source="conference_best_paper"

The conference best paper bonus gives these papers a significant scoring advantage,
ensuring they appear in the final selection regardless of citation count.
"""

import argparse
import json
import os
import sys
import time

import requests
import yaml

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,authors,citationCount,influentialCitationCount,publicationDate,venue,externalIds"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_api_key(config: dict) -> str | None:
    key = config.get("semantic_scholar_api_key", "")
    if not key:
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    return key if key else None


def search_s2_for_paper(title: str, year: str, api_key: str | None) -> dict | None:
    """Search Semantic Scholar for a specific paper by title."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": title,
        "year": year,
        "fields": S2_FIELDS,
        "limit": "3",
    }

    try:
        resp = requests.get(S2_SEARCH_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        papers = data.get("data", [])
        if papers:
            return papers[0]
    except requests.RequestException as e:
        print(f"  Warning: S2 search failed for '{title[:50]}': {e}", file=sys.stderr)

    return None


def search_best_papers_for_venue(
    venue: str, year: str, api_key: str | None
) -> list[dict]:
    """Search S2 for best paper award winners at a venue."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    # Search for papers at this venue, sorted by citation count
    # Best papers tend to be among the most cited
    params = {
        "query": f"best paper {venue}",
        "year": year,
        "venue": venue,
        "fields": S2_FIELDS,
        "limit": "10",
    }

    try:
        resp = requests.get(S2_SEARCH_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except requests.RequestException as e:
        print(f"  Warning: S2 venue search failed for {venue}: {e}", file=sys.stderr)
        return []


def s2_paper_to_candidate(paper: dict, venue: str) -> dict | None:
    """Convert S2 paper to candidate format with conference source."""
    ext_ids = paper.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv")
    if not arxiv_id:
        return None

    authors = [a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")]

    return {
        "arxiv_id": arxiv_id,
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "authors": authors[:20],
        "published_date": paper.get("publicationDate", ""),
        "hf_upvotes": 0,
        "source": "conference_best_paper",
        "hf_date": "",
        "venue": venue,
        "citation_count": paper.get("citationCount", 0) or 0,
        "influential_citation_count": paper.get("influentialCitationCount", 0) or 0,
        "s2_paper_id": paper.get("paperId", ""),
    }


# Known best paper titles by conference (to be populated by web search during execution)
# This dict is used as a fallback / supplement. The LLM triage step in SKILL.md
# instructs Claude Code to also web-search for best papers and add them manually.
KNOWN_BEST_PAPERS = {
    # Example format:
    # "NeurIPS": {
    #     "2024": [
    #         "Title of best paper 1",
    #         "Title of best paper 2",
    #     ]
    # }
}


def main():
    parser = argparse.ArgumentParser(description="Harvest conference best paper awards")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--output", default="candidates.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--best-papers-file",
        default=None,
        help="Optional JSON file with known best paper titles {venue: {year: [titles]}}",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    api_key = get_api_key(config)
    year = str(config["year"])
    venues = config.get("sources", {}).get("conferences", {}).get("venues", [])

    if not venues:
        print("No conference venues configured. Skipping.")
        return

    # Load known best papers if provided
    known_papers = dict(KNOWN_BEST_PAPERS)
    if args.best_papers_file and os.path.exists(args.best_papers_file):
        with open(args.best_papers_file) as f:
            known_papers.update(json.load(f))

    # Load existing candidates
    seen_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                data = json.loads(line)
                if data.get("arxiv_id"):
                    seen_ids.add(data["arxiv_id"])

    total_kept = 0

    with open(args.output, "a") as out_f:
        for venue in venues:
            print(f"\nSearching for {venue} {year} best papers...")

            # First: check known best papers
            venue_known = known_papers.get(venue, {}).get(year, [])
            for title in venue_known:
                print(f"  Looking up known paper: {title[:60]}...")
                paper = search_s2_for_paper(title, year, api_key)
                if paper:
                    candidate = s2_paper_to_candidate(paper, venue)
                    if candidate and candidate["arxiv_id"] not in seen_ids:
                        seen_ids.add(candidate["arxiv_id"])
                        out_f.write(json.dumps(candidate) + "\n")
                        total_kept += 1
                        print(f"    ✓ Found: {candidate['title'][:60]}")
                time.sleep(1.0 if api_key else 3.0)

            # Second: search S2 for top papers at this venue
            papers = search_best_papers_for_venue(venue, year, api_key)
            for p in papers:
                candidate = s2_paper_to_candidate(p, venue)
                if candidate and candidate["arxiv_id"] not in seen_ids:
                    seen_ids.add(candidate["arxiv_id"])
                    candidate["source"] = "conference_top_cited"  # Not confirmed best paper
                    out_f.write(json.dumps(candidate) + "\n")
                    total_kept += 1

            print(f"  Found {len(papers)} top papers at {venue}")
            time.sleep(1.0 if api_key else 3.0)

    print(f"\nDone! Added {total_kept} conference papers")
    print(f"Total candidates in {args.output}: {len(seen_ids)}")
    print(
        "\nNote: Conference best paper detection is approximate. For accurate results,"
    )
    print(
        "web-search for '<venue> <year> best paper award' during the LLM triage step"
    )
    print("and manually add any missing best papers to the best-papers JSON file.")


if __name__ == "__main__":
    main()
