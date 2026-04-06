from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request


class ModelClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass
class MockClient:
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def complete(self, prompt: str) -> str:
        # Minimal deterministic fallback for offline testing.
        lines = [x.strip() for x in prompt.splitlines() if x.strip()]
        if not lines:
            return ""
        low = prompt.lower()
        if "2+2" in low:
            return "4"
        if "capital of france" in low:
            return "Paris"
        if "how many minutes is the trip" in low:
            return "90"
        if "which number is larger: 19 or 91" in low:
            return "91"
        if "what day comes after monday" in low:
            return "Tuesday"
        choices = ["I need more context.", "Unknown", "N/A"]
        return choices[self._rng.randrange(len(choices))]


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout_sec: int = 120

    def complete(self, prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(self.temperature)},
        }
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except error.URLError as e:
            raise RuntimeError(
                f"Ollama request failed: {e}. Ensure ollama is running and model '{self.model}' is pulled."
            ) from e
        blob = json.loads(body)
        return str(blob.get("response", "")).strip()


@dataclass
class OpenAIChatClient:
    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    timeout_sec: int = 120

    def complete(self, prompt: str) -> str:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set and model.api_key was not provided")

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.temperature),
        }
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore") if e.fp else str(e)
            raise RuntimeError(f"OpenAI request failed: {detail}") from e
        except error.URLError as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

        blob = json.loads(body)
        try:
            return str(blob["choices"][0]["message"]["content"]).strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected OpenAI response schema: {body}") from e
