#!/usr/bin/env python3
"""
Step 1a: Harvest paper candidates from HuggingFace Daily Papers API.

Iterates over each day of the target year (or date range) and collects
papers with their upvote counts. Appends results to candidates.jsonl.

HF Daily Papers API:
  GET https://huggingface.co/api/daily_papers?date=YYYY-MM-DD&limit=100

Each paper has: title, arxiv ID (paper.id), upvotes, publishedAt, summary.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import requests
import yaml

HF_API_BASE = "https://huggingface.co/api/daily_papers"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def date_range(start: str, end: str):
    """Yield dates from start to end inclusive as YYYY-MM-DD strings."""
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while current <= end_dt:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def fetch_daily_papers(date: str, limit: int = 100) -> list:
    """Fetch papers for a specific date from HF Daily Papers API."""
    params = {"date": date, "limit": limit}
    try:
        resp = requests.get(HF_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  Warning: Failed to fetch papers for {date}: {e}", file=sys.stderr)
        return []


def extract_arxiv_id(paper_data: dict) -> str | None:
    """Extract arxiv ID from HF paper data."""
    # The paper object has a nested 'paper' field with 'id' = arxiv ID
    paper = paper_data.get("paper", {})
    arxiv_id = paper.get("id")
    if arxiv_id:
        return arxiv_id
    return None


def paper_to_candidate(paper_data: dict, date: str) -> dict | None:
    """Convert HF paper data to our candidate format."""
    arxiv_id = extract_arxiv_id(paper_data)
    if not arxiv_id:
        return None

    paper = paper_data.get("paper", {})
    return {
        "arxiv_id": arxiv_id,
        "title": paper.get("title", ""),
        "abstract": paper.get("summary", ""),
        "authors": [a.get("name", "") for a in paper.get("authors", [])],
        "published_date": paper.get("publishedAt", ""),
        "hf_upvotes": paper_data.get("paper", {}).get("upvotes", 0),
        "source": "huggingface",
        "hf_date": date,
        "venue": "",
        "citation_count": 0,
        "influential_citation_count": 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Harvest papers from HuggingFace Daily Papers")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--output", default="candidates.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Determine date range
    if "start_date" in config and "end_date" in config:
        start = config["start_date"]
        end = config["end_date"]
    else:
        year = config["year"]
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        # Don't go past today
        today = datetime.now().strftime("%Y-%m-%d")
        if end > today:
            end = today

    min_upvotes = config.get("sources", {}).get("huggingface", {}).get("min_upvotes", 5)

    print(f"Harvesting HF Daily Papers from {start} to {end}")
    print(f"Minimum upvotes filter: {min_upvotes}")

    seen_ids = set()
    total_found = 0
    total_kept = 0

    # Load existing candidates to avoid duplicates
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                data = json.loads(line)
                if data.get("arxiv_id"):
                    seen_ids.add(data["arxiv_id"])
        print(f"Found {len(seen_ids)} existing candidates, will skip duplicates")

    with open(args.output, "a") as out_f:
        for date in date_range(start, end):
            papers = fetch_daily_papers(date)
            if not papers:
                continue

            day_count = 0
            for p in papers:
                total_found += 1
                candidate = paper_to_candidate(p, date)
                if candidate is None:
                    continue

                if candidate["arxiv_id"] in seen_ids:
                    continue

                if candidate["hf_upvotes"] < min_upvotes:
                    continue

                seen_ids.add(candidate["arxiv_id"])
                out_f.write(json.dumps(candidate) + "\n")
                total_kept += 1
                day_count += 1

            if day_count > 0:
                print(f"  {date}: {day_count} papers kept")

            # Be polite to the API
            time.sleep(0.2)

    print(f"\nDone! Found {total_found} total papers, kept {total_kept} after filtering")
    print(f"Candidates written to {args.output}")


if __name__ == "__main__":
    main()
