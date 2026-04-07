"""Unit tests for the deep_dive agent helpers and the 05 script's patcher."""

from pathlib import Path

from paper_discovery.agents.deep_dive import (
    _build_deep_dive_prompt,
    _find_related,
    _sanitize_filename,
)

from .conftest import load_script

dd05 = load_script("05_generate_deep_dives")
notes06 = load_script("06_generate_obsidian_notes")


# ---------- _sanitize_filename ----------

def test_sanitize_preserves_spaces_and_strips_path_chars():
    # / and ? are filesystem-illegal; spaces, commas, ! are kept.
    assert _sanitize_filename("Hello, World! / Title?") == "Hello, World! Title"


def test_sanitize_collapses_whitespace_and_truncates():
    long = "a " * 200
    out = _sanitize_filename(long)
    assert len(out) <= 120
    assert "  " not in out
    assert not out.endswith(" ")


# ---------- _find_related ----------

def test_find_related_matches_same_category():
    paper = {"arxiv_id": "1", "title": "DINO contrastive vision", "category": "embeddings"}
    others = [
        {"arxiv_id": "2", "title": "CLIP contrastive vision encoder", "category": "embeddings"},
        {"arxiv_id": "3", "title": "Robot manipulation", "category": "robotics"},
    ]
    manifest = [
        {"arxiv_id": "2", "filename": "CLIP-paper.md"},
        {"arxiv_id": "3", "filename": "Robot.md"},
    ]
    related = _find_related(paper, others, {m["arxiv_id"]: m for m in manifest})
    fnames = [r[0] for r in related]
    # Both match: #2 by title overlap, #3 by category mismatch (so excluded)
    assert "CLIP-paper" in fnames


def test_find_related_excludes_self():
    paper = {"arxiv_id": "1", "title": "X Y Z", "category": "vision"}
    related = _find_related(paper, [paper], {"1": {"filename": "X.md"}})
    assert related == []


# ---------- _build_deep_dive_prompt token budget ----------

def test_prompt_respects_char_budget():
    paper = {"arxiv_id": "1234", "title": "T", "abstract": "abc"}
    huge = "x" * 100_000
    html = {
        "html_available": True,
        "sections": {
            "introduction": {"title": "Intro", "text": huge},
            "method": {"title": "Method", "text": huge},
        },
    }
    prompt = _build_deep_dive_prompt(paper, html, related=[], char_budget=2000)
    # Body should be small even though sections are huge.
    assert len(prompt) < 6000


def test_prompt_falls_back_to_abstract_when_no_html():
    paper = {"arxiv_id": "1", "title": "T", "abstract": "the abstract text"}
    prompt = _build_deep_dive_prompt(paper, {"html_available": False}, related=[])
    assert "ABSTRACT ONLY" in prompt
    assert "the abstract text" in prompt


# ---------- 05 _patch_note ----------

DEEP_DIVE_HEADER = "## Deep Dive"

NOTE_TEMPLATE = """---
title: "Test"
---
# Test

## Key Insight

> insight

## Deep Dive

OLD PLACEHOLDER

### Subsection

old subsection content

## Connections

links here

## Personal Notes
"""


def test_patch_note_replaces_only_deep_dive_section(tmp_path: Path):
    note = tmp_path / "n.md"
    note.write_text(NOTE_TEMPLATE)
    new_block = f"{DEEP_DIVE_HEADER}\n\nNEW CONTENT\n\n"
    assert dd05._patch_note(note, new_block) is True
    text = note.read_text()
    assert "NEW CONTENT" in text
    assert "OLD PLACEHOLDER" not in text
    assert "old subsection content" not in text  # subsections under Deep Dive are replaced
    assert "## Connections" in text  # next section preserved
    assert "links here" in text
    assert "## Personal Notes" in text


def test_patch_note_noop_when_no_header(tmp_path: Path):
    note = tmp_path / "n.md"
    note.write_text("# title\n\nno deep dive here\n")
    assert dd05._patch_note(note, "## Deep Dive\n\nx\n") is False


def test_split_papers():
    triaged = [
        {"arxiv_id": "1", "tier": "deep_dive"},
        {"arxiv_id": "2", "tier": "brief_mention"},
        {"arxiv_id": "3", "tier": "deep_dive"},
    ]
    deep, brief = dd05._split_papers(triaged)
    assert [p["arxiv_id"] for p in deep] == ["1", "3"]
    assert [p["arxiv_id"] for p in brief] == ["2"]


# ---------- 06 _paper_year ----------

def test_paper_year_explicit_field():
    assert notes06._paper_year({"year": 2024}, "fallback") == "2024"


def test_paper_year_from_published_date():
    assert notes06._paper_year({"published_date": "2025-03-15"}, "fallback") == "2025"


def test_paper_year_falls_back_to_default():
    assert notes06._paper_year({}, 2026) == "2026"
