from __future__ import annotations

import random
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
