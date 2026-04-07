"""Pipeline checkpoint for resume after interruption."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PipelineCheckpoint:
    harvest_complete: bool = False
    scoring_complete: bool = False
    triage_complete: bool = False
    triaged_arxiv_ids: list[str] = field(default_factory=list)
    html_fetch_complete: bool = False
    deep_dives_done: list[str] = field(default_factory=list)
    brief_mentions_done: bool = False
    notes_generated: bool = False
    graph_built: bool = False
    vault_copied: bool = False

    @classmethod
    def load(cls, path: Path) -> "PipelineCheckpoint":
        if path.exists():
            data = json.loads(path.read_text())
            valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**valid)
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    def mark_triage_batch(self, arxiv_ids: list[str], path: Path) -> None:
        self.triaged_arxiv_ids.extend(arxiv_ids)
        self.save(path)

    def mark_deep_dive(self, arxiv_id: str, path: Path) -> None:
        self.deep_dives_done.append(arxiv_id)
        self.save(path)
