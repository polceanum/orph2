import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ood_solver.envs.hidden_mechanism_seq import HiddenMechanismSequenceEnv
from ood_solver.models.encoder import SimpleSequenceEncoder
from ood_solver.models.approach import ApproachProposer
from ood_solver.models.rules import RuleProposer
from ood_solver.models.probe_policy import ProbePolicy
from ood_solver.models.belief_updater import BeliefUpdater
from ood_solver.models.solver_head import SolverHead
from ood_solver.models.hypothesis_solver import HypothesisSolver
from ood_solver.baselines.no_rules_baseline import NoRulesSolver
from ood_solver.baselines.random_probe_baseline import RandomProbeSolver
from ood_solver.baselines.recurrent_baseline import RecurrentBaseline
from ood_solver.training.metrics import entropy_from_logits
from ood_solver.utils import move_episode_to_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cfg_get(cfg: dict, key: str, default=None):
    parts = key.split(".")
    cur = cfg
    ok = True
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            ok = False
            break
    if ok:
        return cur
    flat_key = parts[-1]
    return cfg.get(flat_key, default)


def build_structured_model(cfg: dict, device: str):
    d_model = int(cfg_get(cfg, "model.d_model", 128))
    vocab_size = int(cfg_get(cfg, "env.vocab_size", 16))
    model = HypothesisSolver(
        encoder=SimpleSequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=int(cfg_get(cfg, "model.nhead", 4)),
            num_layers=int(cfg_get(cfg, "model.num_layers", 2)),
            max_len=int(cfg_get(cfg, "env.seq_len", 12)) * 8,
        ),
        approach_proposer=ApproachProposer(
            d_model=d_model,
            num_slots=int(cfg_get(cfg, "model.num_approaches", 4)),
        ),
        rule_proposer=RuleProposer(
            d_model=d_model,
            num_rules=int(cfg_get(cfg, "model.num_rules", 8)),
        ),
        probe_policy=ProbePolicy(d_model=d_model),
        belief_updater=BeliefUpdater(
            d_model=d_model,
            archive_size=int(cfg_get(cfg, "model.archive_size", 16)),
        ),
        solver_head=SolverHead(
            d_model=d_model,
            vocab_size=vocab_size,
        ),
    ).to(device)
    return model


def build_random_probe_model(cfg: dict, device: str):
    d_model = int(cfg_get(cfg, "model.d_model", 128))
    vocab_size = int(cfg_get(cfg, "env.vocab_size", 16))
    model = RandomProbeSolver(
        encoder=SimpleSequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=int(cfg_get(cfg, "model.nhead", 4)),
            num_layers=int(cfg_get(cfg, "model.num_layers", 2)),
            max_len=int(cfg_get(cfg, "env.seq_len", 12)) * 8,
        ),
        approach_proposer=ApproachProposer(
            d_model=d_model,
            num_slots=int(cfg_get(cfg, "model.num_approaches", 4)),
        ),
        rule_proposer=RuleProposer(
            d_model=d_model,
            num_rules=int(cfg_get(cfg, "model.num_rules", 8)),
        ),
        probe_policy=ProbePolicy(d_model=d_model),
        belief_updater=BeliefUpdater(
            d_model=d_model,
            archive_size=int(cfg_get(cfg, "model.archive_size", 16)),
        ),
        solver_head=SolverHead(
            d_model=d_model,
            vocab_size=vocab_size,
        ),
    ).to(device)
    return model


def build_no_rules_model(cfg: dict, device: str):
    d_model = int(cfg_get(cfg, "model.d_model", 128))
    vocab_size = int(cfg_get(cfg, "env.vocab_size", 16))
    model = NoRulesSolver(
        encoder=SimpleSequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=int(cfg_get(cfg, "model.nhead", 4)),
            num_layers=int(cfg_get(cfg, "model.num_layers", 2)),
            max_len=int(cfg_get(cfg, "env.seq_len", 12)) * 8,
        ),
        approach_proposer=ApproachProposer(
            d_model=d_model,
            num_slots=int(cfg_get(cfg, "model.num_approaches", 4)),
        ),
        rule_proposer=RuleProposer(
            d_model=d_model,
            num_rules=int(cfg_get(cfg, "model.num_rules", 8)),
        ),
        probe_policy=ProbePolicy(d_model=d_model),
        belief_updater=BeliefUpdater(
            d_model=d_model,
            archive_size=int(cfg_get(cfg, "model.archive_size", 16)),
        ),
        solver_head=SolverHead(
            d_model=d_model,
            vocab_size=vocab_size,
        ),
    ).to(device)
    return model


def build_recurrent_baseline(cfg: dict, device: str):
    model = RecurrentBaseline(
        vocab_size=int(cfg_get(cfg, "env.vocab_size", 16)),
        d_model=int(cfg_get(cfg, "model.d_model", 128)),
        hidden_dim=int(cfg_get(cfg, "model.d_model", 128)),
        seq_len=int(cfg_get(cfg, "env.seq_len", 12)),
        num_probe_steps=int(cfg_get(cfg, "env.num_probe_steps", 4)),
    ).to(device)
    return model


@torch.no_grad()
def run_eval(model, env, episodes, device: str, structured: bool) -> dict:
    model.eval()
    ce = torch.nn.CrossEntropyLoss()

    overall = defaultdict(float)
    by_mech = defaultdict(lambda: defaultdict(float))
    total_batches = 0
    probe_steps = 0

    def probe_executor(batch, step, chosen_idx):
        return env.execute_probe_batch(batch, step, chosen_idx, device=device)

    for batch in episodes:
        batch = move_episode_to_device(batch, device)

        if structured:
            logits, belief, logs = model(batch, probe_executor)
            approach_entropy = entropy_from_logits(belief.approach_scores).item()
            rule_entropy = entropy_from_logits(belief.rule_scores).item()
            surprise_mean = belief.surprise.mean().item()
            probe_entropy = 0.0
            diagnostic_probe_acc = 0.0
            if len(logs) > 0:
                for step_idx, step_log in enumerate(logs):
                    probe_entropy += entropy_from_logits(step_log["probe_logits"]).item()
                    if batch.diagnostic_probe_targets is not None:
                        chosen = step_log["chosen_idx"]
                        target = batch.diagnostic_probe_targets[step_idx]
                        diagnostic_probe_acc += (chosen == target).float().mean().item()
                probe_entropy /= len(logs)
                diagnostic_probe_acc /= len(logs)
                probe_steps += 1
        else:
            logits, aux = model(batch, probe_executor)
            approach_entropy = float("nan")
            rule_entropy = float("nan")
            surprise_mean = float("nan")
            probe_entropy = float("nan")
            diagnostic_probe_acc = float("nan")

        b, l, v = logits.shape
        loss = ce(logits.reshape(b * l, v), batch.final_target.reshape(b * l)).item()
        pred = logits.argmax(dim=-1)
        seq_acc_per_item = (pred == batch.final_target).float().mean(dim=-1)

        overall["loss"] += loss
        overall["seq_acc"] += seq_acc_per_item.mean().item()
        if not np.isnan(approach_entropy):
            overall["approach_entropy"] += approach_entropy
        if not np.isnan(rule_entropy):
            overall["rule_entropy"] += rule_entropy
        if not np.isnan(surprise_mean):
            overall["surprise_mean"] += surprise_mean
        if not np.isnan(probe_entropy):
            overall["probe_entropy"] += probe_entropy
        if not np.isnan(diagnostic_probe_acc):
            overall["diagnostic_probe_acc"] += diagnostic_probe_acc
        total_batches += 1

        for i in range(batch.mechanism_id.size(0)):
            mech = int(batch.mechanism_id[i].item())
            by_mech[mech]["seq_acc"] += float(seq_acc_per_item[i].item())
            by_mech[mech]["count"] += 1

    result = {
        "overall": {
            "loss": overall["loss"] / max(total_batches, 1),
            "seq_acc": overall["seq_acc"] / max(total_batches, 1),
            "approach_entropy": overall["approach_entropy"] / max(total_batches, 1) if structured else None,
            "rule_entropy": overall["rule_entropy"] / max(total_batches, 1) if structured else None,
            "surprise_mean": overall["surprise_mean"] / max(total_batches, 1) if structured else None,
            "probe_entropy": overall["probe_entropy"] / max(probe_steps, 1) if structured else None,
            "diagnostic_probe_acc": overall["diagnostic_probe_acc"] / max(probe_steps, 1) if structured else None,
        },
        "by_mechanism": {},
    }

    for mech, vals in by_mech.items():
        cnt = int(vals["count"])
        result["by_mechanism"][str(mech)] = {
            "seq_acc": vals["seq_acc"] / max(cnt, 1),
            "count": cnt,
        }

    return result


def sample_eval_episodes(env, batch_size: int, num_batches: int):
    return [env.sample_episode(batch_size=batch_size) for _ in range(num_batches)]


def load_checkpoint(model, path: str):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--structured-ckpt", type=str, required=True)
    parser.add_argument("--random-probe-ckpt", type=str, default=None)
    parser.add_argument("--no-rules-ckpt", type=str, default=None)
    parser.add_argument("--recurrent-ckpt", type=str, default=None)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--out", type=str, default="artifacts/eval_baselines.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg_get(cfg, "seed", 0)))

    env = HiddenMechanismSequenceEnv(
        vocab_size=int(cfg_get(cfg, "env.vocab_size", 16)),
        seq_len=int(cfg_get(cfg, "env.seq_len", 12)),
        num_probe_steps=int(cfg_get(cfg, "env.num_probe_steps", 4)),
        num_candidate_probes=int(cfg_get(cfg, "env.num_candidate_probes", 8)),
        seed=int(cfg_get(cfg, "seed", 0) + 123),
    )

    episodes = sample_eval_episodes(
        env,
        batch_size=int(cfg_get(cfg, "train.batch_size", 32)),
        num_batches=args.num_batches,
    )

    structured_model = build_structured_model(cfg, device)
    load_checkpoint(structured_model, args.structured_ckpt)
    structured_metrics = run_eval(structured_model, env, episodes, device, structured=True)

    results = {
        "structured": structured_metrics,
    }

    if args.random_probe_ckpt is not None:
        random_probe_model = build_random_probe_model(cfg, device)
        load_checkpoint(random_probe_model, args.random_probe_ckpt)
        random_probe_metrics = run_eval(random_probe_model, env, episodes, device, structured=True)
        results["random_probe"] = random_probe_metrics

    if args.no_rules_ckpt is not None:
        no_rules_model = build_no_rules_model(cfg, device)
        load_checkpoint(no_rules_model, args.no_rules_ckpt)
        no_rules_metrics = run_eval(no_rules_model, env, episodes, device, structured=True)
        results["no_rules"] = no_rules_metrics

    if args.recurrent_ckpt is not None:
        recurrent_model = build_recurrent_baseline(cfg, device)
        load_checkpoint(recurrent_model, args.recurrent_ckpt)
        recurrent_metrics = run_eval(recurrent_model, env, episodes, device, structured=False)
        results["recurrent"] = recurrent_metrics

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
