#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_agent.agent import AgentConfig, OrchestratedAgent
from llm_agent.benchmarks import load_jsonl_benchmark
from llm_agent.eval import exact_match
from llm_agent.model_clients import MockClient
from llm_agent.types import Prediction


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_client(cfg: dict[str, Any], seed: int):
    provider = str(_cfg_get(cfg, "model.provider", "mock")).lower()
    if provider == "mock":
        return MockClient(seed=seed)
    raise ValueError(
        f"Unsupported model.provider={provider!r}. "
        "This repository is configured for local-only runs; use model.provider=mock."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=0)
    ap.add_argument("--no-save-predictions", action="store_true")
    ap.add_argument("--emit-full-json", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(args.seed) if args.seed is not None else int(_cfg_get(cfg, "seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    benchmark_path = str(_cfg_get(cfg, "benchmark.path"))
    split = _cfg_get(cfg, "benchmark.split", None)
    max_tasks = int(_cfg_get(cfg, "run.max_tasks", 0))

    tasks = load_jsonl_benchmark(benchmark_path, split=split)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError("No tasks selected. Check benchmark.path and benchmark.split.")

    client = build_client(cfg, seed=seed)
    fallback_to_mock_on_api_error = False
    fallback_used = False
    agent_cfg = AgentConfig(
        mode=str(_cfg_get(cfg, "agent.mode", "plan_then_solve")),
        system_prompt=str(_cfg_get(cfg, "agent.system_prompt", AgentConfig.system_prompt)),
        self_consistency_k=int(_cfg_get(cfg, "agent.self_consistency_k", 5)),
        use_verifier=bool(_cfg_get(cfg, "agent.use_verifier", True)),
        use_query_rewrite=bool(_cfg_get(cfg, "agent.use_query_rewrite", True)),
        routing_conf_threshold=float(_cfg_get(cfg, "agent.routing_conf_threshold", 0.6)),
        routing_fast_k=int(_cfg_get(cfg, "agent.routing_fast_k", 3)),
        use_symbolic_solver=bool(_cfg_get(cfg, "agent.use_symbolic_solver", False)),
        learned_solver_path=_cfg_get(cfg, "agent.learned_solver_path", None),
    )
    agent = OrchestratedAgent(client=client, cfg=agent_cfg)

    preds: list[Prediction] = []
    n_done = 0
    n_correct = 0
    split_stats: dict[str, list[int]] = {}
    for t in tasks:
        pred_answer, trace = agent.solve(t.question)
        ok = exact_match(pred_answer, t.answer)
        n_done += 1
        n_correct += 1 if ok else 0
        split_stats.setdefault(t.split, [0, 0])
        split_stats[t.split][0] += 1
        split_stats[t.split][1] += 1 if ok else 0
        if args.progress_every > 0 and (n_done % args.progress_every == 0 or n_done == len(tasks)):
            parts = []
            for sk, (sn, sc) in sorted(split_stats.items()):
                sacc = (float(sc) / float(sn)) if sn > 0 else 0.0
                parts.append(f"{sk}={sacc:.3f} ({sc}/{sn})")
            print(
                f"[progress] {n_done}/{len(tasks)} "
                f"acc={float(n_correct)/float(n_done):.3f} "
                + " ".join(parts),
                flush=True,
            )
        preds.append(
            Prediction(
                task_id=t.task_id,
                question=t.question,
                gold_answer=t.answer,
                pred_answer=pred_answer,
                correct=ok,
                trace=trace,
            )
        )

    acc = float(np.mean([1.0 if p.correct else 0.0 for p in preds]))
    per_split: dict[str, list[float]] = {}
    per_type: dict[str, list[float]] = {}
    task_lookup = {t.task_id: t for t in tasks}
    for p in preds:
        t = task_lookup[p.task_id]
        per_split.setdefault(t.split, []).append(1.0 if p.correct else 0.0)
        ttype = str(t.metadata.get("type", "unknown"))
        per_type.setdefault(ttype, []).append(1.0 if p.correct else 0.0)
    out = {
        "config": args.config,
        "seed": seed,
        "summary": {
            "n_tasks": len(preds),
            "accuracy": acc,
            "per_split_accuracy": {k: float(np.mean(v)) for k, v in per_split.items()},
            "per_type_accuracy": {k: float(np.mean(v)) for k, v in per_type.items()},
        },
        "settings": {
            "benchmark_path": benchmark_path,
            "benchmark_split": split,
            "model_provider": _cfg_get(cfg, "model.provider", "mock"),
            "model_name": _cfg_get(cfg, "model.name", None),
            "agent_mode": agent_cfg.mode,
            "fallback_to_mock_on_api_error": fallback_to_mock_on_api_error,
            "fallback_used": fallback_used,
        },
    }
    if not args.no_save_predictions:
        out["predictions"] = [p.__dict__ for p in preds]
    route_counts: dict[str, int] = {}
    for p in preds:
        route = p.trace.get("route")
        if isinstance(route, str):
            route_counts[route] = route_counts.get(route, 0) + 1
    if route_counts:
        out["summary"]["route_counts"] = route_counts
        n = max(len(preds), 1)
        out["summary"]["route_fractions"] = {k: float(v / n) for k, v in route_counts.items()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    if args.emit_full_json:
        print(json.dumps(out, indent=2))
    else:
        summary_view = {
            "config": out["config"],
            "seed": out["seed"],
            "summary": out["summary"],
            "settings": out["settings"],
        }
        print(json.dumps(summary_view, indent=2))
    print(f"Wrote LLM-agent eval report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
