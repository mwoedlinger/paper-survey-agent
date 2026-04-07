#!/usr/bin/env python3
"""
Step 1b: Harvest paper candidates from Semantic Scholar bulk search API.

Uses the paper bulk search endpoint to find highly-cited CS papers from the
target year, filtered by fields of study. Fetches citation counts including
the influential citation count (which weights citations that actually build
on the paper's methods, not just background mentions).

S2 Bulk Search API:
  GET https://api.semanticscholar.org/graph/v1/paper/search/bulk
  - query: search terms
  - year: publication year filter
  - fieldsOfStudy: restrict to CS
  - fields: title, abstract, citationCount, influentialCitationCount, etc.
  - sort: citationCount:desc
  - Returns 1000 results per page, up to 10M total
"""

import argparse
import json
import os
import sys
import time

import requests
import yaml

S2_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "title,abstract,year,authors,citationCount,influentialCitationCount,publicationDate,venue,externalIds,publicationTypes,fieldsOfStudy"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_api_key(config: dict) -> str | None:
    key = config.get("semantic_scholar_api_key", "")
    if not key:
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    return key if key else None


def search_s2_bulk(
    query: str,
    year: str,
    fields_of_study: list[str],
    api_key: str | None = None,
    min_citations: int = 0,
    max_results: int = 1000,
) -> list[dict]:
    """Search Semantic Scholar bulk endpoint and return all results."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "year": year,
        "fieldsOfStudy": ",".join(fields_of_study),
        "fields": S2_FIELDS,
        "sort": "citationCount:desc",
    }
    if min_citations > 0:
        params["minCitationCount"] = str(min_citations)

    all_papers = []
    token = None
    page = 0

    while True:
        if token:
            params["token"] = token

        try:
            resp = requests.get(
                S2_BULK_SEARCH_URL, params=params, headers=headers, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Warning: S2 API error on page {page}: {e}", file=sys.stderr)
            if hasattr(e, "response") and e.response is not None:
                print(f"  Response: {e.response.text[:500]}", file=sys.stderr)
            break

        papers = data.get("data", [])
        if not papers:
            break

        all_papers.extend(papers)
        page += 1
        print(f"  Fetched page {page}: {len(papers)} papers (total: {len(all_papers)})")

        if len(all_papers) >= max_results:
            all_papers = all_papers[:max_results]
            break

        token = data.get("token")
        if not token:
            break

        # Rate limit: 1 req/sec with key, 3 sec without
        delay = 1.0 if api_key else 3.0
        time.sleep(delay)

    return all_papers


def extract_arxiv_id(paper: dict) -> str | None:
    """Extract arxiv ID from S2 paper's externalIds."""
    ext_ids = paper.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv")
    return arxiv_id


def s2_paper_to_candidate(paper: dict) -> dict | None:
    """Convert S2 paper data to our candidate format."""
    arxiv_id = extract_arxiv_id(paper)
    # We need an arxiv ID for deduplication; skip papers without one
    if not arxiv_id:
        return None

    authors = []
    for a in (paper.get("authors") or []):
        name = a.get("name", "")
        if name:
            authors.append(name)

    return {
        "arxiv_id": arxiv_id,
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "authors": authors[:20],  # Limit to first 20 authors
        "published_date": paper.get("publicationDate", ""),
        "hf_upvotes": 0,
        "source": "semantic_scholar",
        "hf_date": "",
        "venue": paper.get("venue", ""),
        "citation_count": paper.get("citationCount", 0) or 0,
        "influential_citation_count": paper.get("influentialCitationCount", 0) or 0,
        "s2_paper_id": paper.get("paperId", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Harvest papers from Semantic Scholar")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--output", default="candidates.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    config = load_config(args.config)
    api_key = get_api_key(config)

    if api_key:
        print("Using Semantic Scholar API key (1 req/sec rate limit)")
    else:
        print("No API key — using shared rate limit (slower, less reliable)")
        print("Get a free key at: https://www.semanticscholar.org/product/api")

    # Build year range for S2 API: supports "2025" or "2025-2026"
    if "start_date" in config and "end_date" in config:
        start_year = config["start_date"][:4]
        end_year = config["end_date"][:4]
        year = f"{start_year}-{end_year}" if start_year != end_year else start_year
    else:
        year = str(config["year"])
    print(f"S2 year filter: {year}")

    s2_config = config.get("sources", {}).get("semantic_scholar", {})
    fields_of_study = s2_config.get("fields_of_study", ["Computer Science"])
    min_influential = s2_config.get("min_influential_citations", 3)

    categories = config.get("categories", {})

    # Load existing candidates to track seen IDs
    seen_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                data = json.loads(line)
                if data.get("arxiv_id"):
                    seen_ids.add(data["arxiv_id"])
        print(f"Found {len(seen_ids)} existing candidates")

    total_kept = 0

    # Strategy: Run multiple targeted searches per category to get good coverage
    # Plus one broad search for highly-cited CS papers
    search_queries = []

    # Broad search for top CS papers
    search_queries.append(("TOP CS PAPERS (broad)", "machine learning deep learning", 1000))

    # Category-specific searches
    for cat_key, cat_data in categories.items():
        keywords = cat_data.get("search_keywords", [])
        if not keywords:
            continue
        # Combine 2-3 keywords per query for better coverage
        for i in range(0, len(keywords), 2):
            kw_group = keywords[i : i + 2]
            query = " ".join(kw_group)
            search_queries.append((f"{cat_key}: {query}", query, 500))

    with open(args.output, "a") as out_f:
        for label, query, max_results in search_queries:
            print(f"\nSearching: {label}")
            papers = search_s2_bulk(
                query=query,
                year=year,
                fields_of_study=fields_of_study,
                api_key=api_key,
                min_citations=0,
                max_results=max_results,
            )

            kept = 0
            for p in papers:
                candidate = s2_paper_to_candidate(p)
                if candidate is None:
                    continue

                if candidate["arxiv_id"] in seen_ids:
                    continue

                # Apply influential citation filter
                if candidate["influential_citation_count"] < min_influential:
                    continue

                seen_ids.add(candidate["arxiv_id"])
                out_f.write(json.dumps(candidate) + "\n")
                kept += 1
                total_kept += 1

            print(f"  Kept {kept} new papers from this search")

    print(f"\nDone! Added {total_kept} papers from Semantic Scholar")
    print(f"Total candidates in {args.output}: {len(seen_ids)}")


if __name__ == "__main__":
    main()
