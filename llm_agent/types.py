from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Task:
    task_id: str
    question: str
    answer: str
    split: str
    metadata: dict[str, Any]


@dataclass
class Prediction:
    task_id: str
    question: str
    gold_answer: str
    pred_answer: str
    correct: bool
    trace: dict[str, Any]

