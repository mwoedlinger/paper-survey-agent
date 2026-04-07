# paper-survey-agent

An agentic pipeline that finds important ML/AI research papers, triages them with an LLM, and turns the survivors into a structured Obsidian knowledge base — deep-dive notes, topic hubs, and a Map of Content with citation-based cross-links.

The pipeline is designed to run end-to-end with a single `make all`, resume cleanly via per-stage checkpoints, and degrade gracefully when an upstream API is flaky.

## What it does

1. **Harvest** candidates from HuggingFace Daily Papers, Semantic Scholar bulk search, and conference best-paper lists.
2. **Score** them by recency-normalized citations, community signal (HF upvotes), and source bonuses. Every paper is enriched via the S2 `/paper/batch` endpoint so citation counts reflect current reality, not whichever harvester surfaced the paper.
3. **Triage** with an LLM (DeepSeek by default, OpenAI-compatible API) in JSON-mode batches. Each paper gets a novelty score (1–5), a category, a one-line insight, and a tier (`deep_dive`, `brief_mention`, or `exclude`).
4. **Generate notes** — one Obsidian markdown file per paper with frontmatter, tags, category wikilinks, and an abstract block.
5. **Deep dives** — fetch the arXiv HTML version of the paper, extract Introduction/Method/Results/etc., harvest figures with absolute URLs, and ask the LLM for a 5-section graduate-level writeup that gets patched into each note's `## Deep Dive` block. Brief-mention papers get a shorter 2–3 paragraph summary.
6. **Build the graph** — topic hub pages per category, a yearly Map of Content listing every paper, and `[[wikilinks]]` cross-links derived from S2 citation relationships.
7. **Publish** — rsync the staged tree into your Obsidian vault.

Re-running `make notes` preserves any `## Personal Notes` and existing `## Deep Dive` content in place. Failed deep dives are retried automatically (up to 3 attempts per paper).

## Layout

```
config.yaml                # sources, categories, scoring, triage, obsidian paths
Makefile                   # one target per stage + `make all`
scripts/
  01_harvest_hf_papers.py         # HuggingFace Daily Papers
  02_harvest_semantic_scholar.py  # S2 bulk search
  03_harvest_conferences.py       # Conference best-paper lists
  04_deduplicate_and_score.py     # Merge + S2 batch enrichment + scoring
  04b_triage.py                   # LLM novelty triage
  05_generate_deep_dives.py       # Deep dives + brief mentions (supports retries)
  06_generate_obsidian_notes.py   # Markdown note generation
  07_build_graph.py               # Topic hubs + MOC + citation cross-links
  08_copy_to_vault.py             # rsync into the Obsidian vault
  refresh_metadata.py             # In-place citation frontmatter patcher
src/paper_discovery/
  agents/                  # Triage + DeepDive LLM agents
  arxiv_html.py            # Cached HTML fetcher + section & figure extractor
  checkpoint.py            # PipelineCheckpoint dataclass
  cli.py                   # `python -m paper_discovery.cli {run,status}`
  config.py                # Config loader with env var fallbacks
  llm_client.py            # Async rate-limited OpenAI-compatible client
tests/                     # pytest unit + integration tests
```

## Requirements

- Python ≥ 3.11
- `rsync` in `PATH` (for publishing to the vault)
- An OpenAI-compatible LLM API key (DeepSeek by default)
- Optional: a Semantic Scholar API key (raises rate limits substantially)
- Optional: an existing Obsidian vault to publish into — otherwise the staged tree under `staging/literature/` is self-contained markdown.

## Setup

```bash
git clone https://github.com/<you>/paper-survey-agent.git
cd paper-survey-agent

python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Configure secrets via environment variables (never commit them):

```bash
export DEEPSEEK_API_KEY=...           # required
export SEMANTIC_SCHOLAR_API_KEY=...    # optional, raises rate limits
export OBSIDIAN_VAULT_PATH=~/Obsidian/MyVault  # optional override for config.yaml
```

Edit `config.yaml` to:
- point `obsidian_vault_path` at your vault (or set the env var above),
- tweak the target year / date range,
- adjust categories, scoring weights, and triage thresholds.

`target_count: 0` in the `triage:` section means "keep every paper scoring ≥ `min_score`" — no cap.

## Running

Full pipeline:

```bash
make all
```

Or one stage at a time:

```bash
make harvest          # all three harvesters
make score            # dedupe + S2 batch enrichment + scoring
make triage           # LLM novelty triage
make notes            # generate Obsidian markdown notes
make deepdive         # deep dives + brief mentions via the LLM
make deepdive-patch   # only re-patch notes from cached deep_dives.json
make graph            # topic hubs + MOC + cross-links
make publish          # rsync into the vault
make publish-dry      # rsync --dry-run
```

Utility targets:

```bash
make refresh-metadata # patch citation frontmatter of existing notes in place,
                      # without touching deep-dive or personal-notes sections
make status           # print pipeline checkpoint state
make test             # pytest
make clean            # wipe pipeline state (keeps staged output)
```

Or via the CLI orchestrator:

```bash
python -m paper_discovery.cli run                # respects checkpoint
python -m paper_discovery.cli run --from triage  # resume from a stage
python -m paper_discovery.cli run --only notes   # single stage
python -m paper_discovery.cli run --force        # ignore checkpoint
python -m paper_discovery.cli status
```

## Cost control

LLM spend for a 100-paper year is typically a handful of USD with DeepSeek. Guardrails:

- `scripts/05_generate_deep_dives.py --max-cost-usd 5.0` aborts cleanly once projected spend exceeds the cap.
- All deep dives are cached in `work/deep_dives.json`; reruns re-use cached output and only retry failures.
- Per-paper failures are tracked with an attempt counter in `work/deep_dive_failures.json` and stop retrying after 3 attempts.
- The arXiv HTML fetcher caches every page under `work/html_cache/`, so rerunning the pipeline costs zero HTTP requests after the first pass.
- `make score`'s S2 enrichment is batched (200 IDs / call).

## Configuration reference

Key sections in `config.yaml`:

- **`sources`** — enable/disable individual harvesters and set per-source thresholds.
- **`categories`** — display name, arXiv subject codes, and search keywords for each topic. Papers are assigned to exactly one category by the triage agent.
- **`scoring`** — weights for the citation signal, HF upvote signal, and source bonuses.
- **`triage`** — `min_score` (minimum novelty, 1–5), `target_count` (cap on kept papers; 0 = unlimited), `batch_size`, `max_triage_pool`.
- **`deepseek`** — base URL, model name, concurrency, retry settings. Any OpenAI-compatible endpoint works.
- **`arxiv_html`** — cache directory, concurrency, and retry budget for the HTML fetcher.
- **`obsidian`** — subdirectory layout inside the vault (`papers_dir`, `topics_dir`, `mocs_dir`).

## Tests

```bash
make test
```

Covers the scoring formula, S2 enrichment, arXiv section + figure extraction, deep-dive prompt budgeting, the `## Deep Dive` patcher, the Personal Notes / Deep Dive preservation logic, filename sanitization, the in-place metadata refresher, and the end-to-end "MOC lists every paper" guarantee.

## License

MIT. See `LICENSE`.
