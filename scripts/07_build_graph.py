#!/usr/bin/env python3
"""
Step 6: Build the Obsidian knowledge graph.

Generates:
1. Topic hub pages — one per category, listing all papers in that category
2. Map of Content (MOC) — yearly overview with papers organized by category
3. Cross-links — reads S2 citation data to add [[wikilinks]] between papers
   that cite each other in the vault
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import requests
import yaml

S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_manifest(staging_dir: str) -> list[dict]:
    manifest_path = os.path.join(staging_dir, "notes_manifest.json")
    with open(manifest_path) as f:
        return json.load(f)


def load_triaged_papers(filepath: str) -> dict:
    """Load triaged papers indexed by arxiv ID."""
    with open(filepath) as f:
        papers = json.load(f)
    return {p["arxiv_id"]: p for p in papers}


def refresh_citations_from_scored(
    papers_by_arxiv: dict, scored_path: str
) -> int:
    """Backfill citation counts from ``scored_papers.json`` (see note in 06)."""
    if not os.path.exists(scored_path):
        return 0
    with open(scored_path) as f:
        scored = {p["arxiv_id"]: p for p in json.load(f)}
    updated = 0
    for arxiv_id, p in papers_by_arxiv.items():
        s = scored.get(arxiv_id)
        if not s:
            continue
        changed = False
        for field in ("citation_count", "influential_citation_count"):
            fresh = s.get(field, 0) or 0
            if fresh > (p.get(field, 0) or 0):
                p[field] = fresh
                changed = True
        if not p.get("venue") and s.get("venue"):
            p["venue"] = s["venue"]
            changed = True
        if changed:
            updated += 1
    return updated


def get_api_key(config: dict) -> str | None:
    key = config.get("semantic_scholar_api_key", "")
    if not key:
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    return key if key else None


def fetch_citation_graph(
    arxiv_ids: list[str], api_key: str | None
) -> dict[str, list[str]]:
    """
    For each paper, fetch its references and citations from S2.
    Returns a dict mapping arxiv_id -> list of arxiv_ids it cites
    (filtered to only papers in our set).
    """
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    arxiv_id_set = set(arxiv_ids)
    cites_graph = {}  # paper -> [papers it cites that are in our set]

    print(f"Fetching citation relationships for {len(arxiv_ids)} papers...")

    for i, arxiv_id in enumerate(arxiv_ids):
        paper_id = f"ArXiv:{arxiv_id}"
        url = f"{S2_PAPER_URL}/{paper_id}/references"
        params = {"fields": "externalIds", "limit": "500"}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()

            refs = []
            for ref_entry in data.get("data", []):
                cited = ref_entry.get("citedPaper", {})
                ext_ids = cited.get("externalIds") or {}
                ref_arxiv = ext_ids.get("ArXiv")
                if ref_arxiv and ref_arxiv in arxiv_id_set and ref_arxiv != arxiv_id:
                    refs.append(ref_arxiv)

            if refs:
                cites_graph[arxiv_id] = refs

        except requests.RequestException as e:
            if i < 5:  # Only warn for first few failures
                print(f"  Warning: Failed to fetch refs for {arxiv_id}: {e}", file=sys.stderr)

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(arxiv_ids)}...")

        delay = 1.0 if api_key else 3.0
        time.sleep(delay)

    return cites_graph


def title_to_wikilink(title: str) -> str:
    """Match the filename format produced by ``06_generate_obsidian_notes``."""
    clean = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", title)
    clean = re.sub(r"\s+", " ", clean).strip().rstrip(". ")
    return clean[:120].rstrip(". ")


def generate_topic_hub(
    category_key: str,
    category_config: dict,
    papers: list[dict],
    year: str,
) -> str:
    """Generate a topic hub page for a category."""
    display_name = category_config.get("display_name", category_key)

    content = f"""---
title: "{display_name}"
type: "topic-hub"
year: {year}
paper_count: {len(papers)}
date_updated: "{datetime.now().strftime('%Y-%m-%d')}"
tags:
  - topic-hub
  - category/{category_key}
---

# {display_name}

*{len(papers)} papers from {year}*

"""

    # Sort by novelty score then by citation count
    sorted_papers = sorted(
        papers,
        key=lambda p: (p.get("novelty_score", 0), p.get("influential_citation_count", 0)),
        reverse=True,
    )

    for p in sorted_papers:
        link_name = title_to_wikilink(p.get("title", ""))
        score = p.get("novelty_score", 0)
        total_cites = p.get("citation_count", 0)
        infl_cites = p.get("influential_citation_count", 0)
        one_line = p.get("one_line_insight", "")
        star = " ⭐" if score == 5 else ""

        content += f"- [[{link_name}]]{star} — {one_line}\n"
        content += (
            f"  *Citations: {total_cites} (influential: {infl_cites}) "
            f"| Novelty: {score}/5*\n\n"
        )

    return content


def generate_moc(
    year: str,
    categories: dict,
    papers_by_category: dict[str, list[dict]],
) -> str:
    """Generate a Map of Content for the year."""
    total = sum(len(v) for v in papers_by_category.values())

    content = f"""---
title: "ML Papers {year} — Map of Content"
type: "moc"
year: {year}
total_papers: {total}
date_updated: "{datetime.now().strftime('%Y-%m-%d')}"
tags:
  - MOC
  - year-{year}
---

# ML Papers {year}

*{total} curated papers across {len(papers_by_category)} categories*

"""

    for cat_key, cat_config in categories.items():
        cat_papers = papers_by_category.get(cat_key, [])
        if not cat_papers:
            continue

        display_name = cat_config.get("display_name", cat_key)
        content += f"## [[{display_name}]] ({len(cat_papers)} papers)\n\n"

        # Show top 5 by novelty
        top = sorted(
            cat_papers,
            key=lambda p: (p.get("novelty_score", 0), p.get("influential_citation_count", 0)),
            reverse=True,
        )[:5]

        # Always list every paper — topic hubs already sort by score, the MOC
        # should give a complete flat index so nothing gets hidden behind a
        # truncation cue.
        all_sorted = sorted(
            cat_papers,
            key=lambda p: (p.get("novelty_score", 0), p.get("influential_citation_count", 0)),
            reverse=True,
        )
        for p in all_sorted:
            link_name = title_to_wikilink(p.get("title", ""))
            one_line = p.get("one_line_insight", "")
            content += f"- [[{link_name}]] — {one_line}\n"

        content += "\n"

    return content


def inject_crosslinks(
    manifest: list[dict],
    papers_by_arxiv: dict,
    cites_graph: dict[str, list[str]],
):
    """Inject [[wikilinks]] into paper notes based on citation relationships."""
    # Build arxiv_id -> filename mapping
    id_to_filename = {}
    id_to_filepath = {}
    for entry in manifest:
        arxiv_id = entry["arxiv_id"]
        # Strip .md extension for wikilinks
        id_to_filename[arxiv_id] = entry["filename"].replace(".md", "")
        id_to_filepath[arxiv_id] = entry["filepath"]

    crosslink_count = 0

    for arxiv_id, cited_ids in cites_graph.items():
        if arxiv_id not in id_to_filepath:
            continue

        filepath = id_to_filepath[arxiv_id]
        if not os.path.exists(filepath):
            continue

        # Build connections section
        links = []
        for cited_id in cited_ids:
            if cited_id in id_to_filename:
                filename = id_to_filename[cited_id]
                cited_title = papers_by_arxiv.get(cited_id, {}).get("title", filename)
                links.append(f"- **Cites:** [[{filename}]] — {cited_title[:60]}")

        # Also find papers that cite this one
        for other_id, other_cites in cites_graph.items():
            if arxiv_id in other_cites and other_id in id_to_filename:
                filename = id_to_filename[other_id]
                other_title = papers_by_arxiv.get(other_id, {}).get("title", filename)
                links.append(f"- **Cited by:** [[{filename}]] — {other_title[:60]}")

        if not links:
            continue

        # Replace the connections placeholder in the note
        with open(filepath, "r") as f:
            content = f.read()

        connections_section = "\n".join(links)
        old_placeholder = "_TODO: Add [[wikilinks]] to related papers during graph building step_"
        if old_placeholder in content:
            content = content.replace(old_placeholder, connections_section)
            with open(filepath, "w") as f:
                f.write(content)
            crosslink_count += 1

    return crosslink_count


def main():
    parser = argparse.ArgumentParser(description="Build Obsidian knowledge graph")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input", default="triaged_papers.json", help="Triaged papers JSON")
    parser.add_argument("--staging-dir", default=None, help="Override staging directory")
    parser.add_argument(
        "--skip-citations",
        action="store_true",
        help="Skip fetching citation relationships (faster, no cross-links)",
    )
    parser.add_argument(
        "--scored",
        default="work/scored_papers.json",
        help="Path to scored_papers.json for citation backfill",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    staging = args.staging_dir or config.get("staging_dir", "./staging")
    year = str(config.get("year", "unknown"))
    categories = config.get("categories", {})
    api_key = get_api_key(config)

    # Load data
    manifest = load_manifest(staging)
    papers_by_arxiv = load_triaged_papers(args.input)
    refreshed = refresh_citations_from_scored(papers_by_arxiv, args.scored)
    if refreshed:
        print(f"Refreshed citation data for {refreshed} papers from {args.scored}")
        # Persist the refreshed triaged file so the paper note frontmatter and
        # downstream consumers also benefit from the new numbers.
        with open(args.input, "w") as f:
            json.dump(list(papers_by_arxiv.values()), f, indent=2)

    # Organize papers by category
    papers_by_category = {}
    for entry in manifest:
        cat = entry.get("category", "other")
        if cat not in papers_by_category:
            papers_by_category[cat] = []
        paper_data = papers_by_arxiv.get(entry["arxiv_id"], {})
        paper_data["filepath"] = entry["filepath"]
        paper_data["filename"] = entry["filename"]
        papers_by_category[cat].append(paper_data)

    # 1. Generate topic hub pages
    topics_dir = os.path.join(staging, config.get("obsidian", {}).get("topics_dir", "topics"))
    os.makedirs(topics_dir, exist_ok=True)

    print("Generating topic hub pages...")
    for cat_key, cat_config in categories.items():
        cat_papers = papers_by_category.get(cat_key, [])
        if not cat_papers:
            continue

        hub_content = generate_topic_hub(cat_key, cat_config, cat_papers, year)
        display_name = cat_config.get("display_name", cat_key)
        hub_path = os.path.join(topics_dir, f"{display_name}.md")
        with open(hub_path, "w") as f:
            f.write(hub_content)
        print(f"  {display_name}: {len(cat_papers)} papers")

    # 2. Generate Map of Content
    mocs_dir = os.path.join(staging, config.get("obsidian", {}).get("mocs_dir", "MOCs"))
    os.makedirs(mocs_dir, exist_ok=True)

    print("\nGenerating Map of Content...")
    moc_content = generate_moc(year, categories, papers_by_category)
    moc_path = os.path.join(mocs_dir, f"ML-Papers-{year}.md")
    with open(moc_path, "w") as f:
        f.write(moc_content)
    print(f"  Written to {moc_path}")

    # 3. Cross-link papers using citation data
    if not args.skip_citations:
        arxiv_ids = [entry["arxiv_id"] for entry in manifest if entry.get("arxiv_id")]
        cites_graph = fetch_citation_graph(arxiv_ids, api_key)

        print(f"\nInjecting cross-links ({len(cites_graph)} papers have vault-internal citations)...")
        crosslinks = inject_crosslinks(manifest, papers_by_arxiv, cites_graph)
        print(f"  Updated {crosslinks} notes with cross-links")
    else:
        print("\nSkipping citation cross-links (--skip-citations)")

    print("\n✓ Knowledge graph built!")
    print(f"  Topic hubs: {topics_dir}/")
    print(f"  Map of Content: {moc_path}")
    print(f"\nTo copy to your vault:")
    print(f"  cp -r {staging}/papers/ <vault>/papers/")
    print(f"  cp -r {staging}/topics/ <vault>/topics/")
    print(f"  cp -r {staging}/MOCs/ <vault>/MOCs/")


if __name__ == "__main__":
    main()
