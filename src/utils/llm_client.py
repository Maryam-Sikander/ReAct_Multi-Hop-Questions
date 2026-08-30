from __future__ import annotations

import os
import time
import dataclasses
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclasses.dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict


class RateLimitError(Exception):
    pass


class LLMClient:
    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: int = 512,
                 timeout_s: int = 60, retry_attempts: int = 4, retry_backoff_s: int = 5):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retry_attempts = retry_attempts
        self.retry_backoff_s = retry_backoff_s

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def complete(self, prompt: str, stop: Optional[list[str]] = None) -> LLMResponse:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        last_err = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = requests.post(
                    OPENROUTER_URL, headers=self._headers, json=payload, timeout=self.timeout_s
                )
                if resp.status_code == 429:
                    raise RateLimitError(f"rate limited (attempt {attempt})")
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return LLMResponse(
                    text=choice,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    raw=data,
                )
            except (RateLimitError, requests.exceptions.RequestException) as e:
                last_err = e
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff_s * attempt)  # linear backoff, free tier doesn't need much more
                    continue
        raise RuntimeError(f"LLM call failed after {self.retry_attempts} attempts: {last_err}")
