#!/usr/bin/env python3
"""Patch frontmatter of existing staged notes in place.

Lightweight alternative to ``make notes``: walks the manifest, reads each
note, and overwrites only the citation / venue fields in the frontmatter
and the ``**Citations:**`` line in the body. Leaves the Deep Dive, Personal
Notes, and every other section untouched.

Use after rerunning ``make score`` when you've already spent money on deep
dives and don't want to regenerate notes from scratch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
CITATIONS_LINE_RE = re.compile(
    r"\*\*Citations:\*\* \d+ \(influential: \d+\)"
)


def _patch_frontmatter(text: str, paper: dict) -> tuple[str, bool]:
    m = FM_RE.match(text)
    if not m:
        return text, False
    fm_raw = m.group(1)
    changed = False

    def _set(field: str, value) -> None:
        nonlocal fm_raw, changed
        # Quoted (venue) vs bare int handling.
        if isinstance(value, str):
            repl = f'{field}: "{value}"'
            pat = re.compile(rf'^{field}: ".*"$', re.MULTILINE)
        else:
            repl = f"{field}: {value}"
            pat = re.compile(rf"^{field}: .*$", re.MULTILINE)
        if pat.search(fm_raw):
            new_fm = pat.sub(repl, fm_raw, count=1)
            if new_fm != fm_raw:
                fm_raw = new_fm
                changed = True

    _set("citation_count", int(paper.get("citation_count", 0) or 0))
    _set("influential_citation_count", int(paper.get("influential_citation_count", 0) or 0))
    if paper.get("venue"):
        _set("venue", str(paper["venue"]))

    if not changed:
        return text, False
    return text.replace(m.group(0), f"---\n{fm_raw}\n---\n", 1), True


def _patch_body_citations(text: str, paper: dict) -> tuple[str, bool]:
    new_line = (
        f"**Citations:** {int(paper.get('citation_count', 0) or 0)} "
        f"(influential: {int(paper.get('influential_citation_count', 0) or 0)})"
    )
    if CITATIONS_LINE_RE.search(text):
        new_text = CITATIONS_LINE_RE.sub(new_line, text, count=1)
        return new_text, new_text != text
    return text, False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--scored", default="work/scored_papers.json")
    p.add_argument("--manifest", default=None, help="Override notes manifest path")
    args = p.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    staging = Path(config.get("staging_dir", "./staging"))
    manifest_path = Path(args.manifest) if args.manifest else staging / "notes_manifest.json"

    manifest = json.loads(manifest_path.read_text())
    scored = {p["arxiv_id"]: p for p in json.loads(Path(args.scored).read_text())}

    updated = 0
    missing = 0
    for entry in manifest:
        path = Path(entry["filepath"])
        if not path.exists():
            missing += 1
            continue
        paper = scored.get(entry["arxiv_id"])
        if not paper:
            continue
        text = path.read_text()
        text, fm_changed = _patch_frontmatter(text, paper)
        text, body_changed = _patch_body_citations(text, paper)
        if fm_changed or body_changed:
            path.write_text(text)
            updated += 1

    print(f"Refreshed metadata on {updated} notes ({missing} missing on disk)")
    if missing:
        sys.exit(0)


if __name__ == "__main__":
    main()
