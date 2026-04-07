"""Tests for the scoring formula in 04_deduplicate_and_score.py."""

from datetime import datetime

from .conftest import load_script

score_mod = load_script("04_deduplicate_and_score")


def test_merge_combines_metadata_from_sources():
    candidates = [
        {"arxiv_id": "1", "title": "T", "abstract": "", "source": "hf", "hf_upvotes": 10},
        {"arxiv_id": "1", "title": "T", "abstract": "real abstract", "source": "s2",
         "influential_citation_count": 5, "citation_count": 50},
    ]
    merged = score_mod.merge_candidates(candidates)
    assert set(merged.keys()) == {"1"}
    p = merged["1"]
    assert p["abstract"] == "real abstract"
    assert p["hf_upvotes"] == 10
    assert p["influential_citation_count"] == 5
    assert p["sources"] == {"hf", "s2"}


def test_months_since_publication_handles_iso_date():
    months = score_mod.compute_months_since_publication(
        "2025-01-01", reference_date=datetime(2025, 7, 1)
    )
    assert 5 <= months <= 7


def test_months_since_publication_handles_missing_date():
    assert score_mod.compute_months_since_publication("", datetime(2025, 7, 1)) == 6.0


def test_compute_scores_normalizes_and_applies_bonus():
    merged = {
        "1": {
            "arxiv_id": "1", "title": "A", "abstract": "", "authors": [],
            "published_date": "2025-01-01", "venue": "", "citation_count": 100,
            "influential_citation_count": 20, "hf_upvotes": 50,
            "sources": {"conference_best_paper"}, "s2_paper_id": "",
        },
        "2": {
            "arxiv_id": "2", "title": "B", "abstract": "", "authors": [],
            "published_date": "2025-01-01", "venue": "", "citation_count": 1,
            "influential_citation_count": 0, "hf_upvotes": 1,
            "sources": {"hf"}, "s2_paper_id": "",
        },
    }
    config = {
        "scoring": {
            "conference_best_paper_bonus": 0.3,
            "blog_mention_bonus": 0.15,
            "citation_weight": 0.5,
            "upvote_weight": 0.5,
        }
    }
    scored = score_mod.compute_scores(merged, config)
    assert scored[0]["arxiv_id"] == "1"  # higher final_score
    assert scored[0]["source_bonus"] == 0.3
    assert scored[1]["source_bonus"] == 0.0
    # final_score in [0, ~1.3] range
    assert 0 <= scored[1]["final_score"] <= 1.3
