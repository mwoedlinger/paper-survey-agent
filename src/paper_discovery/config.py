"""Configuration loading and validation."""

import os
from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load config.yaml and apply environment variable overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Environment variable overrides
    if not config.get("semantic_scholar_api_key"):
        config["semantic_scholar_api_key"] = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    ds = config.setdefault("deepseek", {})
    if not ds.get("api_key"):
        ds["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")

    # Defaults for deepseek section
    ds.setdefault("base_url", "https://api.deepseek.com")
    ds.setdefault("model", "deepseek-chat")
    ds.setdefault("max_concurrency", 8)
    ds.setdefault("max_retries", 3)
    ds.setdefault("retry_base_delay", 2.0)
    ds.setdefault("timeout", 120)
    ds.setdefault("temperature", 0.3)
    ds.setdefault("triage_batch_size", 10)
    ds.setdefault("deep_dive_max_tokens", 8192)
    ds.setdefault("triage_max_tokens", 2048)

    # Defaults for arxiv_html section
    ah = config.setdefault("arxiv_html", {})
    ah.setdefault("max_concurrency", 4)
    ah.setdefault("cache_dir", "./work/html_cache")
    ah.setdefault("timeout", 30)
    ah.setdefault("retry_count", 2)

    # Validate required fields
    if not ds["api_key"]:
        raise ValueError(
            "DeepSeek API key required. Set deepseek.api_key in config.yaml "
            "or DEEPSEEK_API_KEY environment variable."
        )

    return config
