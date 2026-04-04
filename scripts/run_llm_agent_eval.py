#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
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
from llm_agent.model_clients import MockClient, OllamaClient, OpenAIResponsesClient
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
    if provider == "ollama":
        model = str(_cfg_get(cfg, "model.name", "llama3.1:8b"))
        base_url = str(_cfg_get(cfg, "model.base_url", "http://127.0.0.1:11434/api/generate"))
        return OllamaClient(model=model, base_url=base_url)
    if provider == "openai":
        model = str(_cfg_get(cfg, "model.name", "gpt-4.1-mini"))
        base_url = str(_cfg_get(cfg, "model.base_url", "https://api.openai.com/v1/responses"))
        return OpenAIResponsesClient(model=model, base_url=base_url)
    raise ValueError(f"Unsupported model.provider={provider!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
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

    try:
        client = build_client(cfg, seed=seed)
        fallback_to_mock_on_api_error = bool(_cfg_get(cfg, "model.fallback_to_mock_on_api_error", False))
        fallback_used = False
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc} Use model.provider=mock for offline smoke runs, "
            "or set OPENAI_API_KEY for provider=openai."
        ) from exc
    agent_cfg = AgentConfig(
        mode=str(_cfg_get(cfg, "agent.mode", "plan_then_solve")),
        system_prompt=str(_cfg_get(cfg, "agent.system_prompt", AgentConfig.system_prompt)),
        self_consistency_k=int(_cfg_get(cfg, "agent.self_consistency_k", 5)),
        use_verifier=bool(_cfg_get(cfg, "agent.use_verifier", True)),
        use_query_rewrite=bool(_cfg_get(cfg, "agent.use_query_rewrite", True)),
        routing_conf_threshold=float(_cfg_get(cfg, "agent.routing_conf_threshold", 0.6)),
        routing_fast_k=int(_cfg_get(cfg, "agent.routing_fast_k", 3)),
        use_symbolic_solver=bool(_cfg_get(cfg, "agent.use_symbolic_solver", False)),
    )
    agent = OrchestratedAgent(client=client, cfg=agent_cfg)

    preds: list[Prediction] = []
    for t in tasks:
        try:
            pred_answer, trace = agent.solve(t.question)
        except urllib.error.HTTPError as exc:
            if fallback_to_mock_on_api_error and int(getattr(exc, "code", 0)) == 429:
                fallback_used = True
                client = MockClient(seed=seed)
                agent = OrchestratedAgent(client=client, cfg=agent_cfg)
                pred_answer, trace = agent.solve(t.question)
                trace = {**trace, "fallback": "mock_after_429"}
            else:
                raise
        ok = exact_match(pred_answer, t.answer)
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
        "predictions": [p.__dict__ for p in preds],
    }
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
    print(json.dumps(out, indent=2))
    print(f"Wrote LLM-agent eval report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
