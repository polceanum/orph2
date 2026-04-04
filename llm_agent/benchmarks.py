from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import Task


def load_jsonl_benchmark(path: str, split: str | None = None) -> list[Task]:
    rows: list[Task] = []
    p = Path(path)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        blob: dict[str, Any] = json.loads(line)
        task_split = str(blob.get("split", "test"))
        if split is not None and task_split != split:
            continue
        rows.append(
            Task(
                task_id=str(blob["id"]),
                question=str(blob["question"]),
                answer=str(blob["answer"]),
                split=task_split,
                metadata=dict(blob.get("metadata", {})),
            )
        )
    return rows

