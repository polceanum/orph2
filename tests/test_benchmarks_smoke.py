import json
from pathlib import Path

from llm_agent.benchmarks import load_jsonl_benchmark


def test_load_jsonl_benchmark_reads_all_rows(tmp_path: Path) -> None:
    bench = tmp_path / "mini.jsonl"
    rows = [
        {"id": "1", "question": "q1", "answer": "a1", "split": "iid", "metadata": {"type": "t1"}},
        {"id": "2", "question": "q2", "answer": "a2", "split": "ood", "metadata": {"type": "t2"}},
    ]
    bench.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    tasks = load_jsonl_benchmark(str(bench))
    assert len(tasks) == 2
    assert tasks[0].task_id == "1"
    assert tasks[1].split == "ood"


def test_load_jsonl_benchmark_split_filter(tmp_path: Path) -> None:
    bench = tmp_path / "mini.jsonl"
    rows = [
        {"id": "1", "question": "q1", "answer": "a1", "split": "iid"},
        {"id": "2", "question": "q2", "answer": "a2", "split": "ood"},
    ]
    bench.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    iid_tasks = load_jsonl_benchmark(str(bench), split="iid")
    ood_tasks = load_jsonl_benchmark(str(bench), split="ood")

    assert [t.task_id for t in iid_tasks] == ["1"]
    assert [t.task_id for t in ood_tasks] == ["2"]
