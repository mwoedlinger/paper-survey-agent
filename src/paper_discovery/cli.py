"""``paper-discovery`` CLI orchestrator.

Drives the eight pipeline stages with checkpoint awareness so that re-running
the command after an interruption skips already-completed stages. Each stage
is just a subprocess invocation of the corresponding ``scripts/NN_*.py`` so
the CLI stays a thin coordinator and the scripts remain independently usable.

Usage:
    paper-discovery run --config config.yaml
    paper-discovery run --config config.yaml --from triage
    paper-discovery run --config config.yaml --only deepdive
    paper-discovery status --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from paper_discovery.checkpoint import PipelineCheckpoint

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


@dataclass
class Stage:
    name: str
    script: str
    checkpoint_field: str
    extra_args: list[str]
    needs: Callable[[PipelineCheckpoint], bool]


def _stages() -> list[Stage]:
    """Define the pipeline. Order matters; ``--from`` indexes into this list."""
    return [
        Stage("harvest_hf",   "01_harvest_hf_papers.py",       "harvest_complete", [],
              lambda c: not c.harvest_complete),
        Stage("harvest_s2",   "02_harvest_semantic_scholar.py", "harvest_complete", [],
              lambda c: not c.harvest_complete),
        Stage("harvest_conf", "03_harvest_conferences.py",     "harvest_complete", [],
              lambda c: not c.harvest_complete),
        Stage("score",        "04_deduplicate_and_score.py",   "scoring_complete",
              ["--input", "work/candidates.jsonl", "--output", "work/scored_papers.json"],
              lambda c: not c.scoring_complete),
        Stage("triage",       "04b_triage.py",                 "triage_complete", [],
              lambda c: not c.triage_complete),
        Stage("notes",        "06_generate_obsidian_notes.py", "notes_generated",
              ["--input", "work/triaged_papers.json"],
              lambda c: not c.notes_generated),
        Stage("deepdive",     "05_generate_deep_dives.py",     "brief_mentions_done", [],
              lambda c: not c.brief_mentions_done),
        Stage("graph",        "07_build_graph.py",             "graph_built",
              ["--input", "work/triaged_papers.json"],
              lambda c: not c.graph_built),
        Stage("publish",      "08_copy_to_vault.py",           "vault_copied", [],
              lambda c: not c.vault_copied),
    ]


def _run_stage(stage: Stage, config: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPTS / stage.script), "--config", config, *stage.extra_args, *extra]
    print(f"\n=== [{stage.name}] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _select_stages(stages: list[Stage], from_: str | None, only: str | None) -> list[Stage]:
    if only:
        return [s for s in stages if s.name == only] or _die(f"unknown stage: {only}")
    if from_:
        idx = next((i for i, s in enumerate(stages) if s.name == from_), None)
        if idx is None:
            _die(f"unknown stage: {from_}")
        return stages[idx:]
    return stages


def _die(msg: str) -> "list[Stage]":
    print(msg, file=sys.stderr)
    sys.exit(1)


def cmd_run(args: argparse.Namespace) -> int:
    stages = _select_stages(_stages(), args.from_, args.only)
    cp_path = Path(args.checkpoint)
    extra = args.extra or []

    for stage in stages:
        cp = PipelineCheckpoint.load(cp_path)
        if not args.force and not stage.needs(cp):
            print(f"[{stage.name}] already complete (checkpoint), skipping")
            continue
        rc = _run_stage(stage, args.config, extra)
        if rc != 0:
            print(f"\n[{stage.name}] failed with exit code {rc}", file=sys.stderr)
            return rc
    print("\n✓ Pipeline complete")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cp = PipelineCheckpoint.load(Path(args.checkpoint))
    data = {
        "harvest_complete": cp.harvest_complete,
        "scoring_complete": cp.scoring_complete,
        "triage_complete": cp.triage_complete,
        "triaged_count": len(cp.triaged_arxiv_ids),
        "html_fetch_complete": cp.html_fetch_complete,
        "deep_dives_done": len(cp.deep_dives_done),
        "brief_mentions_done": cp.brief_mentions_done,
        "notes_generated": cp.notes_generated,
        "graph_built": cp.graph_built,
        "vault_copied": cp.vault_copied,
    }
    print(json.dumps(data, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paper-discovery", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run pipeline stages")
    run.add_argument("--config", required=True)
    run.add_argument("--checkpoint", default="work/checkpoint.json")
    run.add_argument("--from", dest="from_", help="Start at this stage", default=None)
    run.add_argument("--only", help="Run only this stage", default=None)
    run.add_argument("--force", action="store_true", help="Ignore checkpoint and re-run")
    run.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed through to scripts")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="Show pipeline checkpoint state")
    status.add_argument("--config", required=False, default=None)
    status.add_argument("--checkpoint", default="work/checkpoint.json")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
