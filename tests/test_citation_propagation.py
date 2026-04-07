"""End-to-end test that citation counts propagate from score -> notes -> graph.

This is the regression test for the bug where ``triaged_papers.json`` was
written before ``scored_papers.json`` had fresh S2 citation data, leaving
every note with ``citation_count: 0`` in the frontmatter.
"""

from __future__ import annotations

import json
from pathlib import Path

from .conftest import load_script

notes_mod = load_script("06_generate_obsidian_notes")
graph_mod = load_script("07_build_graph")
refresh_mod = load_script("refresh_metadata")


def _config(staging: Path) -> dict:
    return {
        "year": 2025,
        "staging_dir": str(staging),
        "categories": {"llms": {"display_name": "Large Language Models"}},
        "obsidian": {
            "papers_dir": "literature/papers",
            "topics_dir": "literature/topics",
            "mocs_dir": "literature/MOCs",
        },
    }


def _paper(arxiv_id: str, cc: int, ic: int) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "abstract": "An abstract.",
        "authors": ["A. Author"],
        "published_date": "2025-01-15",
        "venue": "",
        "citation_count": cc,
        "influential_citation_count": ic,
        "novelty_score": 4,
        "category": "llms",
        "one_line_insight": "Novel method.",
        "tier": "deep_dive",
    }


def test_citations_backfill_from_scored_into_notes(tmp_path: Path) -> None:
    """Stale triaged file should be healed by the notes step reading scored."""
    # Simulate: scored has fresh citations, triaged has zeros (stale snapshot).
    scored = [_paper("1111.0001", cc=42, ic=10), _paper("1111.0002", cc=7, ic=2)]
    triaged = [_paper("1111.0001", cc=0, ic=0), _paper("1111.0002", cc=0, ic=0)]

    scored_path = tmp_path / "scored.json"
    triaged_path = tmp_path / "triaged.json"
    scored_path.write_text(json.dumps(scored))
    triaged_path.write_text(json.dumps(triaged))

    # Step 06: refresh + regenerate notes
    papers = notes_mod.load_triaged_papers(str(triaged_path))
    refreshed = notes_mod.refresh_citations_from_scored(papers, str(scored_path))
    assert refreshed == 2
    assert papers[0]["citation_count"] == 42
    assert papers[1]["influential_citation_count"] == 2


def test_personal_notes_preserved_across_regeneration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paper = _paper("2222.0001", cc=5, ic=1)

    content = notes_mod.generate_note(paper, config)
    note_path = tmp_path / "note.md"
    note_path.write_text(
        content.replace(
            "_Space for your own thoughts after reading_",
            "My custom annotation: this matters because X.",
        )
    )

    preserved = notes_mod._extract_section(
        str(note_path),
        notes_mod._PERSONAL_NOTES_RE,
        notes_mod._PERSONAL_PLACEHOLDER,
    )
    assert preserved == "My custom annotation: this matters because X."

    # Splicing the preserved body into a fresh render should keep the edit.
    fresh = notes_mod.generate_note(paper, config)
    out = notes_mod._splice_section(
        fresh, "## Personal Notes", notes_mod._PERSONAL_PLACEHOLDER, preserved
    )
    assert "My custom annotation" in out
    assert "_Space for your own thoughts" not in out


def test_deep_dive_preserved_across_regeneration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paper = _paper("3333.0001", cc=5, ic=1)

    content = notes_mod.generate_note(paper, config)
    # Replace the placeholder with real deep-dive content.
    note_path = tmp_path / "note.md"
    note_path.write_text(
        content.replace(
            "<!-- Filled in by scripts/05_generate_deep_dives.py -->\n"
            "_Pending — run `make deepdive` to generate._",
            "### Problem & Motivation\nReal content.",
        )
    )

    preserved = notes_mod._extract_section(
        str(note_path),
        notes_mod._DEEP_DIVE_RE,
        notes_mod._DEEP_DIVE_PLACEHOLDER,
    )
    assert preserved is not None
    assert "Problem & Motivation" in preserved


def test_refresh_metadata_patches_frontmatter_in_place(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paper = _paper("4444.0001", cc=0, ic=0)
    fresh = _paper("4444.0001", cc=99, ic=33)

    # Write an old note with zero citations.
    cat_dir = tmp_path / "literature" / "papers" / "2025" / "llms"
    cat_dir.mkdir(parents=True)
    note_path = cat_dir / "note.md"
    note_path.write_text(notes_mod.generate_note(paper, config))

    # Run the in-place patcher.
    text = note_path.read_text()
    patched, fm_changed = refresh_mod._patch_frontmatter(text, fresh)
    patched, body_changed = refresh_mod._patch_body_citations(patched, fresh)
    assert fm_changed and body_changed
    assert "citation_count: 99" in patched
    assert "influential_citation_count: 33" in patched
    assert "**Citations:** 99 (influential: 33)" in patched


def test_moc_lists_all_papers_without_truncation(tmp_path: Path) -> None:
    """MOC must list every paper, not cut off with 'and N more'."""
    categories = {"llms": {"display_name": "Large Language Models"}}
    papers_by_category = {
        "llms": [_paper(f"5555.{i:04d}", cc=i, ic=i) for i in range(12)]
    }
    moc = graph_mod.generate_moc("2025", categories, papers_by_category)
    # Every paper must appear.
    for i in range(12):
        assert f"Paper 5555.{i:04d}" in moc
    assert "...and" not in moc
