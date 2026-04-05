#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", x.strip().lower())


def _load_benchmark_strings(path: Path) -> tuple[list[str], list[str]]:
    questions: list[str] = []
    answers: list[str] = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        q = str(rec.get("question", "")).strip()
        a = str(rec.get("answer", "")).strip()
        if q:
            questions.append(_norm(q))
        if a:
            answers.append(_norm(a))
    return questions, answers


def _is_meaningful_answer(a: str) -> bool:
    # Avoid noisy matches on short/common answers.
    if len(a) < 12:
        return False
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", a):
        return False
    if " " not in a:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-path", required=True)
    ap.add_argument(
        "--scan-globs",
        nargs="+",
        default=["llm_agent/*.py", "configs/llm_agent/*.yaml", "scripts/*.py"],
    )
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    benchmark_path = Path(args.benchmark_path)
    questions, answers = _load_benchmark_strings(benchmark_path)
    answer_set = {a for a in answers if _is_meaningful_answer(a)}
    findings: list[dict[str, object]] = []
    scanned: list[str] = []

    for pattern in args.scan_globs:
        for p in sorted(Path(".").glob(pattern)):
            if not p.is_file():
                continue
            scanned.append(str(p))
            txt = _norm(p.read_text())
            for q in questions:
                if len(q) >= 40 and q in txt:
                    findings.append(
                        {
                            "severity": "high",
                            "kind": "question_literal_leak",
                            "file": str(p),
                            "message": "File contains full benchmark question text.",
                            "question_prefix": q[:120],
                        }
                    )
                    break
            for a in answer_set:
                if a in txt:
                    findings.append(
                        {
                            "severity": "medium",
                            "kind": "answer_literal_overlap",
                            "file": str(p),
                            "message": "File contains benchmark answer literal.",
                            "answer": a,
                        }
                    )
                    break

    out = {
        "ok": len(findings) == 0,
        "benchmark_path": str(benchmark_path),
        "n_questions": len(questions),
        "n_answers_meaningful": len(answer_set),
        "n_files_scanned": len(scanned),
        "n_findings": len(findings),
        "findings": findings,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if findings:
        sys.exit(1)


if __name__ == "__main__":
    main()
