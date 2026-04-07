"""Async DeepSeek API client with rate limiting, retries, and cost tracking."""

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self, config: dict):
        ds = config["deepseek"]
        self._client = AsyncOpenAI(
            api_key=ds["api_key"],
            base_url=ds["base_url"],
            timeout=ds["timeout"],
        )
        self._model = ds["model"]
        self._semaphore = asyncio.Semaphore(ds["max_concurrency"])
        self._max_retries = ds["max_retries"]
        self._retry_base_delay = ds["retry_base_delay"]
        self._temperature = ds["temperature"]
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        """Core API call with semaphore, retries, and cost tracking."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    response = await self._client.chat.completions.create(**kwargs)
                    if response.usage:
                        self.total_input_tokens += response.usage.prompt_tokens
                        self.total_output_tokens += response.usage.completion_tokens
                    return response.choices[0].message.content or ""
                except (RateLimitError, APITimeoutError) as e:
                    if attempt == self._max_retries:
                        raise
                    delay = self._retry_base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{self._max_retries} after {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                except APIError as e:
                    if e.status_code and e.status_code >= 500 and attempt < self._max_retries:
                        delay = self._retry_base_delay * (2 ** attempt)
                        logger.warning(f"Server error, retry {attempt + 1}: {e}")
                        await asyncio.sleep(delay)
                    else:
                        raise
        raise RuntimeError("Unreachable")

    async def complete_json(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> dict[str, Any]:
        """Completion expecting JSON output."""
        raw = await self._call(system_prompt, user_prompt, max_tokens, json_mode=True)
        return json.loads(raw)

    async def complete_text(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 4096
    ) -> str:
        """Completion returning free-form text (markdown)."""
        return await self._call(system_prompt, user_prompt, max_tokens, json_mode=False)

    def cost_estimate(self) -> dict[str, float]:
        """Estimated cost in USD (DeepSeek V3.2 pricing)."""
        input_cost = self.total_input_tokens / 1_000_000 * 0.28
        output_cost = self.total_output_tokens / 1_000_000 * 0.42
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(input_cost + output_cost, 4),
        }
