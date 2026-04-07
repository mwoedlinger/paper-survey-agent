#!/usr/bin/env python3
"""
Step 5: Generate Obsidian markdown notes from triaged papers.

Creates a structured markdown file for each paper using the template.
The deep-dive content section is left as a placeholder — Claude Code
fills it in interactively during step 5 of the SKILL.md procedure.

Also enriches papers with Semantic Scholar citation relationships
for cross-linking in the knowledge graph.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_triaged_papers(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return json.load(f)


def refresh_citations_from_scored(papers: list[dict], scored_path: str) -> int:
    """Backfill citation counts / venue from ``scored_papers.json``.

    Triage can race ahead of S2 enrichment — if the user reruns ``make score``
    after triage, the triaged JSON holds stale zeros. Rather than forcing a
    full re-triage, we look up each paper in the freshest scored file and
    merge the numeric fields back in. Returns the number of papers updated.
    """
    if not os.path.exists(scored_path):
        return 0
    with open(scored_path) as f:
        scored = {p["arxiv_id"]: p for p in json.load(f)}
    updated = 0
    for p in papers:
        s = scored.get(p.get("arxiv_id"))
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
        if not p.get("s2_paper_id") and s.get("s2_paper_id"):
            p["s2_paper_id"] = s["s2_paper_id"]
            changed = True
        if changed:
            updated += 1
    return updated


def sanitize_filename(title: str) -> str:
    """Create a valid Obsidian-friendly filename from a paper title.

    Preserves spaces — Obsidian handles them in wikilinks just fine — and only
    strips characters that are illegal on macOS/Linux/Windows filesystems
    (``/ \\ : * ? " < > |``) plus control chars. Collapses runs of whitespace.
    """
    # Replace path-illegal characters with a space, then collapse.
    clean = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", title)
    clean = re.sub(r"\s+", " ", clean).strip()
    # Trim trailing dots/spaces (Windows hates trailing dots, and they look bad).
    clean = clean.rstrip(". ")
    # Truncate then re-strip in case the cut landed inside whitespace.
    return clean[:120].rstrip(". ")


def format_authors(authors: list[str], max_display: int = 5) -> str:
    """Format author list for frontmatter."""
    if not authors:
        return "Unknown"
    if len(authors) <= max_display:
        return ", ".join(authors)
    return ", ".join(authors[:max_display]) + f" et al. (+{len(authors) - max_display})"


def format_authors_yaml(authors: list[str], max_display: int = 10) -> str:
    """Format authors as YAML list."""
    display = authors[:max_display]
    lines = []
    for a in display:
        # Escape quotes in names
        safe_name = a.replace('"', '\\"')
        lines.append(f'  - "{safe_name}"')
    if len(authors) > max_display:
        lines.append(f'  - "+{len(authors) - max_display} more"')
    return "\n".join(lines)


def _paper_year(paper: dict, default_year: str | int) -> str:
    """Resolve a paper's year from explicit field, published_date, or config."""
    year = paper.get("year")
    if year:
        return str(year)
    pub = paper.get("published_date") or ""
    if len(pub) >= 4 and pub[:4].isdigit():
        return pub[:4]
    return str(default_year)


def generate_tags(paper: dict, default_year: str | int = "unknown") -> list[str]:
    """Generate Obsidian tags for a paper."""
    tags = ["paper", f"year-{_paper_year(paper, default_year)}"]

    category = paper.get("category", "other")
    tags.append(f"category/{category}")

    venue = paper.get("venue", "")
    if venue:
        venue_tag = re.sub(r"[^\w]", "-", venue.lower()).strip("-")
        if venue_tag:
            tags.append(f"venue/{venue_tag}")

    for source in paper.get("sources", []):
        if source == "conference_best_paper":
            tags.append("best-paper")

    return tags


def generate_note(paper: dict, config: dict) -> str:
    """Generate a complete Obsidian markdown note for a paper."""
    year = _paper_year(paper, config.get("year", "unknown"))
    category = paper.get("category", "other")
    cat_display = config.get("categories", {}).get(category, {}).get(
        "display_name", category.replace("-", " ").title()
    )

    tags = generate_tags(paper, year)
    tags_yaml = "\n".join(f"  - {t}" for t in tags)
    authors_yaml = format_authors_yaml(paper.get("authors", []))

    # Build frontmatter
    frontmatter = f"""---
title: "{paper.get('title', '').replace('"', '\\"')}"
authors:
{authors_yaml}
year: {year}
arxiv_id: "{paper.get('arxiv_id', '')}"
category: "{category}"
tags:
{tags_yaml}
venue: "{paper.get('venue', '')}"
citation_count: {paper.get('citation_count', 0)}
influential_citation_count: {paper.get('influential_citation_count', 0)}
novelty_score: {paper.get('novelty_score', 0)}
date_added: "{datetime.now().strftime('%Y-%m-%d')}"
status: "unread"
---"""

    # Build body
    authors_str = format_authors(paper.get("authors", []))
    arxiv_id = paper.get("arxiv_id", "")
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    hf_url = f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else ""
    s2_id = paper.get("s2_paper_id", "")
    s2_url = f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else ""

    one_line = paper.get("one_line_insight", "")
    abstract = paper.get("abstract", "").strip()
    abstract_block = f"\n## Abstract\n\n{abstract}\n" if abstract else ""

    body = f"""
# {paper.get('title', 'Untitled')}

**Authors:** {authors_str}
**Category:** [[{cat_display}]]
**Links:** [arXiv]({arxiv_url}) | [HuggingFace]({hf_url}) | [Semantic Scholar]({s2_url})
**Citations:** {paper.get('citation_count', 0)} (influential: {paper.get('influential_citation_count', 0)})

## Key Insight

> {one_line}
{abstract_block}
## Deep Dive

<!-- Filled in by scripts/05_generate_deep_dives.py -->
_Pending — run `make deepdive` to generate._

## Connections

<!-- Links to related papers in the vault -->
_TODO: Add [[wikilinks]] to related papers during graph building step_

## Personal Notes

_Space for your own thoughts after reading_

"""
    return frontmatter + body


_PERSONAL_NOTES_RE = re.compile(
    r"## Personal Notes\s*\n(.*?)(?=\n## |\Z)", re.DOTALL
)
_DEEP_DIVE_RE = re.compile(
    r"## Deep Dive\s*\n(.*?)(?=\n## |\Z)", re.DOTALL
)
_PERSONAL_PLACEHOLDER = "_Space for your own thoughts after reading_"
_DEEP_DIVE_PLACEHOLDER = (
    "<!-- Filled in by scripts/05_generate_deep_dives.py -->\n"
    "_Pending — run `make deepdive` to generate._"
)


def _extract_section(existing_path: str, pattern: re.Pattern, placeholder: str) -> str | None:
    """Return the body of a section, or None if absent/unfilled."""
    try:
        with open(existing_path) as f:
            text = f.read()
    except FileNotFoundError:
        return None
    m = pattern.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    if not body or body == placeholder.strip():
        return None
    return body


def _splice_section(new_content: str, header: str, placeholder: str, preserved_body: str) -> str:
    """Replace a placeholder section body with the preserved text."""
    old = f"{header}\n\n{placeholder}\n"
    new = f"{header}\n\n{preserved_body}\n"
    return new_content.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description="Generate Obsidian notes from triaged papers")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input", default="triaged_papers.json", help="Input triaged papers JSON")
    parser.add_argument("--staging-dir", default=None, help="Override staging directory")
    parser.add_argument(
        "--scored",
        default="work/scored_papers.json",
        help="Path to scored_papers.json for citation backfill",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    papers = load_triaged_papers(args.input)
    refreshed = refresh_citations_from_scored(papers, args.scored)
    if refreshed:
        print(f"Refreshed citation data for {refreshed} papers from {args.scored}")
        # Persist the enrichment so downstream steps (graph, survey) see it too.
        with open(args.input, "w") as f:
            json.dump(papers, f, indent=2)

    staging = args.staging_dir or config.get("staging_dir", "./staging")
    papers_base = os.path.join(staging, config.get("obsidian", {}).get("papers_dir", "papers"))
    year = str(config.get("year", "unknown"))

    print(f"Generating Obsidian notes for {len(papers)} papers...")
    print(f"Output directory: {papers_base}/{year}/")

    # Track generated files for the graph builder
    generated = []

    for i, paper in enumerate(papers):
        category = paper.get("category", "other")
        cat_dir = os.path.join(papers_base, year, category)
        os.makedirs(cat_dir, exist_ok=True)

        filename = sanitize_filename(paper.get("title", f"paper-{i}")) + ".md"
        filepath = os.path.join(cat_dir, filename)

        note_content = generate_note(paper, config)

        # Preserve user edits + deep-dive content across re-runs.
        personal = _extract_section(filepath, _PERSONAL_NOTES_RE, _PERSONAL_PLACEHOLDER)
        if personal:
            note_content = _splice_section(
                note_content, "## Personal Notes", _PERSONAL_PLACEHOLDER, personal
            )
        deep_dive = _extract_section(filepath, _DEEP_DIVE_RE, _DEEP_DIVE_PLACEHOLDER)
        if deep_dive:
            note_content = _splice_section(
                note_content, "## Deep Dive", _DEEP_DIVE_PLACEHOLDER, deep_dive
            )

        with open(filepath, "w") as f:
            f.write(note_content)

        generated.append(
            {
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper.get("title", ""),
                "category": category,
                "filepath": os.path.abspath(filepath),
                "filename": filename,
            }
        )

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{len(papers)} notes")

    # Save manifest for the graph builder
    manifest_path = os.path.join(staging, "notes_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(generated, f, indent=2)

    # Print summary by category
    cat_counts = {}
    for g in generated:
        cat = g["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\nGenerated {len(generated)} notes:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    print(f"\nManifest saved to {manifest_path}")
    print("Next: Run the deep-dive generation interactively in Claude Code")


if __name__ == "__main__":
    main()
