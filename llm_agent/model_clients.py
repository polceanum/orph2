from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


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
class OpenAIResponsesClient:
    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1/responses"
    timeout_sec: int = 120
    max_retries: int = 4
    retry_backoff_sec: float = 2.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

    def complete(self, prompt: str) -> str:
        payload = {"model": self.model, "input": prompt}
        attempt = 0
        while True:
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:  # nosec B310
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                status = int(getattr(exc, "code", 0))
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    sleep_s = self.retry_backoff_sec * (2**attempt)
                    time.sleep(sleep_s)
                    attempt += 1
                    continue
                raise
        # Responses API currently provides output_text as a convenience field.
        out = body.get("output_text")
        if isinstance(out, str):
            return out.strip()
        # Fallback parsing for future response shapes.
        if isinstance(body.get("output"), list):
            parts: list[str] = []
            for item in body["output"]:
                content = item.get("content", [])
                for c in content:
                    txt = c.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "\n".join(parts).strip()
        return ""


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://127.0.0.1:11434/api/generate"
    timeout_sec: int = 180

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:  # nosec B310
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Failed to reach local Ollama server. Start it with `ollama serve` "
                "and ensure the model is pulled (e.g., `ollama pull llama3.1:8b`)."
            ) from exc
        out = body.get("response")
        if isinstance(out, str):
            return out.strip()
        return ""
