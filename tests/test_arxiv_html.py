"""Tests for the arxiv HTML section extractor."""

from paper_discovery.arxiv_html import ArxivHTMLFetcher


SAMPLE_HTML = """
<html><body>
<div class="ltx_abstract">Abstract: this is the abstract content.</div>
<section class="ltx_section">
  <h2>1. Introduction</h2>
  <p class="ltx_p">Intro paragraph one with <a>linked</a> word.</p>
  <p class="ltx_p">Intro paragraph two.</p>
</section>
<section class="ltx_section">
  <h2>2. Method</h2>
  <p class="ltx_p">Method details here.</p>
  <figure class="ltx_figure">
    <img src="x1.png" />
    <figcaption class="ltx_caption">Figure 1: The architecture overview.</figcaption>
  </figure>
</section>
<section class="ltx_section">
  <h2>3. Acknowledgements</h2>
  <p class="ltx_p">Thanks.</p>
</section>
</body></html>
"""


def test_parse_html_extracts_relevant_sections(tmp_path):
    fetcher = ArxivHTMLFetcher({"arxiv_html": {"cache_dir": str(tmp_path)}})
    parsed = fetcher._parse_html("1234.5678", SAMPLE_HTML)

    assert parsed["arxiv_id"] == "1234.5678"
    assert parsed["html_available"] is True
    assert "this is the abstract" in parsed["abstract"]
    titles = [s["title"] for s in parsed["sections"].values()]
    assert "Introduction" in titles
    assert "Method" in titles
    # Acknowledgements should be filtered out
    assert "Acknowledgements" not in titles
    # Inline tags must not glue words together
    intro_text = next(
        s["text"] for s in parsed["sections"].values() if s["title"] == "Introduction"
    )
    assert "linked word" in intro_text


def test_parse_html_extracts_figures_with_absolute_urls(tmp_path):
    fetcher = ArxivHTMLFetcher({"arxiv_html": {"cache_dir": str(tmp_path)}})
    parsed = fetcher._parse_html("1234.5678", SAMPLE_HTML)
    figs = parsed["figures"]
    assert len(figs) == 1
    assert figs[0]["url"] == "https://arxiv.org/html/1234.5678/x1.png"
    assert figs[0]["label"].lower().startswith("figure 1")
    assert "architecture" in figs[0]["caption"].lower()


def test_parse_html_returns_empty_when_no_sections(tmp_path):
    fetcher = ArxivHTMLFetcher({"arxiv_html": {"cache_dir": str(tmp_path)}})
    parsed = fetcher._parse_html("0", "<html><body></body></html>")
    assert parsed["html_available"] is False
    assert parsed["sections"] == {}
