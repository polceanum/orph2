from __future__ import annotations

import re


def normalize_text(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)

