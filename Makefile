PYTHON ?= python
CONFIG ?= config.yaml
WORK   := work
STAGE  := staging

.PHONY: all harvest harvest-hf harvest-s2 harvest-conf score triage notes deepdive deepdive-patch graph publish status test clean refresh-metadata

all: harvest score triage notes deepdive graph publish

harvest: harvest-hf harvest-s2 harvest-conf

harvest-hf:
	$(PYTHON) scripts/01_harvest_hf_papers.py --config $(CONFIG)

harvest-s2:
	$(PYTHON) scripts/02_harvest_semantic_scholar.py --config $(CONFIG)

harvest-conf:
	$(PYTHON) scripts/03_harvest_conferences.py --config $(CONFIG)

score:
	$(PYTHON) scripts/04_deduplicate_and_score.py --config $(CONFIG) \
		--input $(WORK)/candidates.jsonl --output $(WORK)/scored_papers.json

triage:
	$(PYTHON) scripts/04b_triage.py --config $(CONFIG)

notes:
	$(PYTHON) scripts/06_generate_obsidian_notes.py --config $(CONFIG) \
		--input $(WORK)/triaged_papers.json

deepdive:
	$(PYTHON) scripts/05_generate_deep_dives.py --config $(CONFIG)

deepdive-patch:
	$(PYTHON) scripts/05_generate_deep_dives.py --config $(CONFIG) --patch-only

graph:
	$(PYTHON) scripts/07_build_graph.py --config $(CONFIG) \
		--input $(WORK)/triaged_papers.json

publish:
	$(PYTHON) scripts/08_copy_to_vault.py --config $(CONFIG)

publish-dry:
	$(PYTHON) scripts/08_copy_to_vault.py --config $(CONFIG) --dry-run

refresh-metadata:
	$(PYTHON) scripts/refresh_metadata.py --config $(CONFIG) \
		--scored $(WORK)/scored_papers.json

status:
	PYTHONPATH=src $(PYTHON) -m paper_discovery.cli status

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf $(WORK)/checkpoint.json $(WORK)/deep_dives.json $(WORK)/deep_dive_failures.json
