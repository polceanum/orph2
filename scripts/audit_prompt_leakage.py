#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import importlib


def _contains_any(text: str, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for p in patterns:
        if re.search(p, text, flags=re.I):
            hits.append(p)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-file", default="llm_agent/agent.py")
    ap.add_argument("--runner-file", default="scripts/run_llm_agent_eval.py")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    agent_txt = Path(args.agent_file).read_text()
    runner_txt = Path(args.runner_file).read_text()

    findings: list[dict[str, object]] = []

    # Guard 1: no benchmark-answer literals in rewrite function source.
    rewrite_block = ""
    m = re.search(r"def _rewrite_question\(q: str\) -> str:(.*?)\n\ndef ", agent_txt, flags=re.S)
    if m:
        rewrite_block = m.group(1)
    suspicious_literals = [
        r"capital of france",
        r"2\+2",
        r"comes after monday",
        r"19 or 91",
        r"09:00",
    ]
    lit_hits = _contains_any(rewrite_block, suspicious_literals)
    if lit_hits:
        findings.append(
            {
                "severity": "high",
                "kind": "rewrite_literal_match",
                "message": "Rewrite function contains benchmark-specific literals.",
                "hits": lit_hits,
            }
        )

    # Guard 1b: runtime behavior of rewrite must be whitespace-only normalization.
    repo_root = str(Path(args.agent_file).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        mod = importlib.import_module("llm_agent.agent")
    except Exception:
        findings.append(
            {
                "severity": "high",
                "kind": "rewrite_import_error",
                "message": "Could not import agent module for runtime rewrite checks.",
            }
        )
    else:
        rewrite_fn = getattr(mod, "_rewrite_question", None)
        if rewrite_fn is None:
            findings.append(
                {
                    "severity": "high",
                    "kind": "rewrite_missing",
                    "message": "Agent module is missing _rewrite_question.",
                }
            )
        else:
            samples = [
                "  What is 2 + 2?  ",
                "Alice   has\t3 apples.\nHow many?",
                "If x=7, what is x*x?",
                "  09:00 to 17:00  ",
            ]
            for s in samples:
                expected = re.sub(r"\s+", " ", s.strip())
                got = rewrite_fn(s)
                if got != expected:
                    findings.append(
                        {
                            "severity": "high",
                            "kind": "rewrite_semantic_change",
                            "message": "Rewrite function changes content beyond whitespace normalization.",
                            "sample": s,
                            "expected": expected,
                            "got": got,
                        }
                    )
                    break

    # Guard 2: prompt templates must not include gold/answer/metadata/task_id fields.
    prompt_hits = _contains_any(
        agent_txt,
        [r"gold_answer", r"metadata", r"task_id", r"\banswer\s*:", r"\bgold\s*:"],
    )
    # allow "Answer with only the final answer" phrase
    prompt_hits = [h for h in prompt_hits if h not in {r"\banswer\s*:"}]
    if prompt_hits:
        findings.append(
            {
                "severity": "high",
                "kind": "prompt_field_leak",
                "message": "Agent prompt file references potential target/meta fields.",
                "hits": prompt_hits,
            }
        )

    # Guard 3: runner must pass only question into solve().
    if "agent.solve(t.question)" not in runner_txt:
        findings.append(
            {
                "severity": "high",
                "kind": "runner_call_shape",
                "message": "Runner does not use the expected question-only solve call.",
            }
        )
    if re.search(r"agent\.solve\([^)]*t\.answer", runner_txt):
        findings.append(
            {
                "severity": "high",
                "kind": "runner_target_flow",
                "message": "Runner appears to pass t.answer into agent.solve().",
            }
        )

    out = {
        "ok": len(findings) == 0,
        "n_findings": len(findings),
        "findings": findings,
        "checked_files": {
            "agent": args.agent_file,
            "runner": args.runner_file,
        },
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if findings:
        sys.exit(1)


if __name__ == "__main__":
    main()
