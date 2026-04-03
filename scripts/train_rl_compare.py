import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ood_solver.baselines.recurrent_baseline import RecurrentBaseline
from ood_solver.envs.config import build_env_from_cfg
from ood_solver.models.approach import ApproachProposer
from ood_solver.models.belief_updater import BeliefUpdater
from ood_solver.models.encoder import SimpleSequenceEncoder
from ood_solver.models.hypothesis_solver import HypothesisSolver
from ood_solver.models.probe_policy import ProbePolicy
from ood_solver.models.rules import RuleProposer
from ood_solver.models.solver_head import SolverHead
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
    return cfg.get(parts[-1], default)


def parse_seeds(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError("No seeds parsed from --seeds")
    return vals


def append_jsonl(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def build_structured_model(cfg: dict, device: str) -> HypothesisSolver:
    d_model = int(cfg_get(cfg, "model.d_model", 64))
    vocab_size = int(cfg_get(cfg, "env.vocab_size", 12))
    seq_len = int(cfg_get(cfg, "env.seq_len", 6))
    model = HypothesisSolver(
        encoder=SimpleSequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=int(cfg_get(cfg, "model.nhead", 4)),
            num_layers=int(cfg_get(cfg, "model.num_layers", 2)),
            max_len=seq_len * 8,
            use_value_features=bool(cfg_get(cfg, "model.use_value_features", False)),
            value_features_only=bool(cfg_get(cfg, "model.value_features_only", False)),
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
            use_local_adapter=bool(cfg_get(cfg, "model.use_local_adapter", False)),
            use_value_features=bool(cfg_get(cfg, "model.use_value_features", False)),
            value_features_only=bool(cfg_get(cfg, "model.value_features_only", False)),
            use_modulo_shift_adapter=bool(cfg_get(cfg, "model.use_modulo_shift_adapter", False)),
            use_demo_shift_prior=bool(cfg_get(cfg, "model.use_demo_shift_prior", False)),
            use_mechanism_router=bool(cfg_get(cfg, "model.use_mechanism_router", False)),
            num_mechanism_experts=int(cfg_get(cfg, "model.num_mechanism_experts", 1)),
            mechanism_router_temperature=float(cfg_get(cfg, "model.mechanism_router_temperature", 1.0)),
        ),
        soft_probe_training=False,
        sample_probes_during_training=True,
        sample_probe_temp=float(cfg_get(cfg, "rl.sample_probe_temp", 1.0)),
    ).to(device)
    return model


def build_recurrent_model(cfg: dict, device: str) -> RecurrentBaseline:
    model = RecurrentBaseline(
        vocab_size=int(cfg_get(cfg, "env.vocab_size", 12)),
        d_model=int(cfg_get(cfg, "model.d_model", 64)),
        hidden_dim=int(cfg_get(cfg, "model.d_model", 64)),
        seq_len=int(cfg_get(cfg, "env.seq_len", 6)),
        num_probe_steps=int(cfg_get(cfg, "env.num_probe_steps", 2)),
        use_probe_policy=True,
        sample_probes_during_training=True,
        probe_temperature=float(cfg_get(cfg, "rl.sample_probe_temp", 1.0)),
    ).to(device)
    return model


def aggregate(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None}
    mean = float(sum(values) / len(values))
    var = float(sum((v - mean) ** 2 for v in values) / len(values))
    return {"mean": mean, "std": var**0.5}


@torch.no_grad()
def evaluate_model(model, env, device: str, num_batches: int, batch_size: int, structured: bool) -> dict:
    model.eval()
    seq_acc_vals = []

    def probe_executor(batch, step, chosen_idx):
        return env.execute_probe_batch(batch, step, chosen_idx, device=device)

    for _ in range(num_batches):
        batch = move_episode_to_device(env.sample_episode(batch_size=batch_size), device)
        if structured:
            logits, _, _ = model(batch, probe_executor)
        else:
            logits, _ = model(batch, probe_executor)
        pred = logits.argmax(dim=-1)
        seq_acc_vals.append(float((pred == batch.final_target).float().mean().item()))
    return {"seq_acc": float(sum(seq_acc_vals) / len(seq_acc_vals))}


def train_step_structured(
    model: HypothesisSolver,
    optimizer: torch.optim.Optimizer,
    batch,
    env,
    device: str,
    ce_weight: float,
    pg_weight: float,
    entropy_weight: float,
    baseline: float,
    baseline_momentum: float,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = move_episode_to_device(batch, device)

    def probe_executor(ep, step, chosen_idx):
        return env.execute_probe_batch(ep, step, chosen_idx, device=device)

    logits, _, logs = model(batch, probe_executor)
    b, l, v = logits.shape
    ce = F.cross_entropy(logits.reshape(b * l, v), batch.final_target.reshape(b * l))

    pred = logits.argmax(dim=-1)
    token_acc = (pred == batch.final_target).float().mean(dim=-1)
    reward = token_acc.detach()
    baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * float(reward.mean().item())
    advantage = reward - baseline

    if logs:
        logprob_sum = torch.stack([x["chosen_logprob"] for x in logs], dim=0).sum(dim=0)
        entropy = torch.stack([x["probe_entropy"] for x in logs], dim=0).mean()
        policy_loss = -(logprob_sum * advantage).mean()
    else:
        entropy = torch.tensor(0.0, device=device)
        policy_loss = torch.tensor(0.0, device=device)

    loss = ce_weight * ce + pg_weight * policy_loss - entropy_weight * entropy
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "ce": float(ce.item()),
        "pg": float(policy_loss.item()),
        "reward": float(reward.mean().item()),
        "probe_entropy": float(entropy.item()),
    }, baseline


def train_step_recurrent(
    model: RecurrentBaseline,
    optimizer: torch.optim.Optimizer,
    batch,
    env,
    device: str,
    ce_weight: float,
    pg_weight: float,
    entropy_weight: float,
    baseline: float,
    baseline_momentum: float,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = move_episode_to_device(batch, device)

    def probe_executor(ep, step, chosen_idx):
        return env.execute_probe_batch(ep, step, chosen_idx, device=device)

    logits, aux = model(batch, probe_executor)
    b, l, v = logits.shape
    ce = F.cross_entropy(logits.reshape(b * l, v), batch.final_target.reshape(b * l))

    pred = logits.argmax(dim=-1)
    token_acc = (pred == batch.final_target).float().mean(dim=-1)
    reward = token_acc.detach()
    baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * float(reward.mean().item())
    advantage = reward - baseline

    step_logs = aux.get("step_logs", [])
    logprob_terms = [x["chosen_logprob"] for x in step_logs if x.get("chosen_logprob") is not None]
    entropy_terms = [x["probe_entropy"] for x in step_logs if x.get("probe_entropy") is not None]
    if logprob_terms:
        logprob_sum = torch.stack(logprob_terms, dim=0).sum(dim=0)
        entropy = torch.stack(entropy_terms, dim=0).mean()
        policy_loss = -(logprob_sum * advantage).mean()
    else:
        entropy = torch.tensor(0.0, device=device)
        policy_loss = torch.tensor(0.0, device=device)

    loss = ce_weight * ce + pg_weight * policy_loss - entropy_weight * entropy
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "ce": float(ce.item()),
        "pg": float(policy_loss.item()),
        "reward": float(reward.mean().item()),
        "probe_entropy": float(entropy.item()),
    }, baseline


def run_seed(cfg: dict, seed: int, device: str, log_jsonl_path: Path | None = None) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = seed
    set_seed(seed)

    train_env = build_env_from_cfg(cfg, section="env", seed=seed, seed_offset=0)
    iid_env = build_env_from_cfg(cfg, section="eval.id", seed=seed, seed_offset=int(cfg_get(cfg, "eval.seed_offset", 123)))
    ood_env = build_env_from_cfg(
        cfg,
        section="eval.ood",
        seed=seed,
        seed_offset=int(cfg_get(cfg, "eval.ood.seed_offset", 999)),
    )

    structured = build_structured_model(cfg, device)
    recurrent = build_recurrent_model(cfg, device)

    lr = float(cfg_get(cfg, "rl.lr", 3e-4))
    opt_s = torch.optim.AdamW(structured.parameters(), lr=lr, weight_decay=float(cfg_get(cfg, "rl.weight_decay", 1e-4)))
    opt_r = torch.optim.AdamW(recurrent.parameters(), lr=lr, weight_decay=float(cfg_get(cfg, "rl.weight_decay", 1e-4)))

    epochs = int(cfg_get(cfg, "rl.epochs", 12))
    steps_per_epoch = int(cfg_get(cfg, "rl.steps_per_epoch", 120))
    batch_size = int(cfg_get(cfg, "rl.batch_size", cfg_get(cfg, "train.batch_size", 32)))
    log_every = int(cfg_get(cfg, "rl.log_every_steps", 40))
    ce_weight = float(cfg_get(cfg, "rl.ce_weight", 1.0))
    pg_weight = float(cfg_get(cfg, "rl.pg_weight", 1.0))
    entropy_weight = float(cfg_get(cfg, "rl.entropy_weight", 0.01))
    baseline_momentum = float(cfg_get(cfg, "rl.reward_baseline_momentum", 0.9))

    baseline_s = 0.0
    baseline_r = 0.0
    monitor_batches = int(cfg_get(cfg, "rl.monitor_eval_batches", 6))
    early_stop_patience = int(cfg_get(cfg, "rl.early_stop_patience", 0))
    early_stop_min_delta = float(cfg_get(cfg, "rl.early_stop_min_delta", 1e-3))
    best_structured_iid = float("-inf")
    bad_epochs = 0

    for epoch in range(epochs):
        sums_s = defaultdict(float)
        sums_r = defaultdict(float)
        for step in range(steps_per_epoch):
            batch = train_env.sample_episode(batch_size=batch_size)
            ms, baseline_s = train_step_structured(
                model=structured,
                optimizer=opt_s,
                batch=batch,
                env=train_env,
                device=device,
                ce_weight=ce_weight,
                pg_weight=pg_weight,
                entropy_weight=entropy_weight,
                baseline=baseline_s,
                baseline_momentum=baseline_momentum,
            )
            mr, baseline_r = train_step_recurrent(
                model=recurrent,
                optimizer=opt_r,
                batch=batch,
                env=train_env,
                device=device,
                ce_weight=ce_weight,
                pg_weight=pg_weight,
                entropy_weight=entropy_weight,
                baseline=baseline_r,
                baseline_momentum=baseline_momentum,
            )
            for k, v in ms.items():
                sums_s[k] += v
            for k, v in mr.items():
                sums_r[k] += v

            if log_every > 0 and (step + 1) % log_every == 0:
                n = float(step + 1)
                msg = (
                    f"seed={seed} epoch={epoch} step={step+1}/{steps_per_epoch} "
                    f"structured(loss={sums_s['loss']/n:.3f}, ce={sums_s['ce']/n:.3f}, reward={sums_s['reward']/n:.3f}) "
                    f"recurrent(loss={sums_r['loss']/n:.3f}, ce={sums_r['ce']/n:.3f}, reward={sums_r['reward']/n:.3f})"
                )
                print(msg, flush=True)
                append_jsonl(
                    log_jsonl_path,
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "step": step + 1,
                        "steps_per_epoch": steps_per_epoch,
                        "phase": "train_step",
                        "structured": {k: float(sums_s[k] / n) for k in ("loss", "ce", "pg", "reward", "probe_entropy")},
                        "recurrent": {k: float(sums_r[k] / n) for k in ("loss", "ce", "pg", "reward", "probe_entropy")},
                    },
                )

        n_epoch = float(max(steps_per_epoch, 1))
        epoch_train = {
            "structured": {k: float(sums_s[k] / n_epoch) for k in ("loss", "ce", "pg", "reward", "probe_entropy")},
            "recurrent": {k: float(sums_r[k] / n_epoch) for k in ("loss", "ce", "pg", "reward", "probe_entropy")},
        }

        structured_iid_monitor = evaluate_model(
            structured, iid_env, device, num_batches=monitor_batches, batch_size=batch_size, structured=True
        )
        recurrent_iid_monitor = evaluate_model(
            recurrent, iid_env, device, num_batches=monitor_batches, batch_size=batch_size, structured=False
        )
        monitor_struct_iid = float(structured_iid_monitor["seq_acc"])
        monitor_recur_iid = float(recurrent_iid_monitor["seq_acc"])
        improved = monitor_struct_iid > (best_structured_iid + early_stop_min_delta)
        if improved:
            best_structured_iid = monitor_struct_iid
            bad_epochs = 0
        else:
            bad_epochs += 1
        should_stop = early_stop_patience > 0 and bad_epochs >= early_stop_patience
        append_jsonl(
            log_jsonl_path,
            {
                "seed": seed,
                "epoch": epoch,
                "phase": "epoch_end",
                "train": epoch_train,
                "monitor": {
                    "structured_iid_seq_acc": monitor_struct_iid,
                    "recurrent_iid_seq_acc": monitor_recur_iid,
                    "best_structured_iid_seq_acc": best_structured_iid,
                    "bad_epochs": bad_epochs,
                    "early_stop_patience": early_stop_patience,
                    "early_stop_min_delta": early_stop_min_delta,
                    "should_stop": should_stop,
                },
            },
        )
        print(
            f"seed={seed} epoch={epoch} monitor(struct_iid={monitor_struct_iid:.3f}, recur_iid={monitor_recur_iid:.3f}, bad_epochs={bad_epochs})",
            flush=True,
        )
        if should_stop:
            print(f"seed={seed} early stop at epoch={epoch} (no IID improvement)", flush=True)
            break

    eval_batches = int(cfg_get(cfg, "rl.eval_batches", 30))
    structured_iid = evaluate_model(structured, iid_env, device, eval_batches, batch_size, structured=True)
    structured_ood = evaluate_model(structured, ood_env, device, eval_batches, batch_size, structured=True)
    recurrent_iid = evaluate_model(recurrent, iid_env, device, eval_batches, batch_size, structured=False)
    recurrent_ood = evaluate_model(recurrent, ood_env, device, eval_batches, batch_size, structured=False)
    return {
        "structured": {"iid": structured_iid, "ood": structured_ood},
        "recurrent_rl": {"iid": recurrent_iid, "ood": recurrent_ood},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bridges/bridge_10_mechanism_addition_rl.yaml")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--out", type=str, default="artifacts/rl/bridge10_rl_compare.json")
    parser.add_argument("--log-jsonl", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = parse_seeds(args.seeds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_seed = {}
    log_jsonl_base: Path | None = Path(args.log_jsonl) if args.log_jsonl else None
    for seed in seeds:
        print(f"=== RL seed {seed} ===")
        seed_log_path = None
        if log_jsonl_base is not None:
            if log_jsonl_base.suffix:
                seed_log_path = log_jsonl_base.with_name(f"{log_jsonl_base.stem}_seed{seed}{log_jsonl_base.suffix}")
            else:
                seed_log_path = log_jsonl_base / f"seed_{seed}.jsonl"
        per_seed[str(seed)] = run_seed(cfg, seed, device, log_jsonl_path=seed_log_path)

    rows = {"structured_iid": [], "structured_ood": [], "recurrent_iid": [], "recurrent_ood": []}
    for blob in per_seed.values():
        rows["structured_iid"].append(blob["structured"]["iid"]["seq_acc"])
        rows["structured_ood"].append(blob["structured"]["ood"]["seq_acc"])
        rows["recurrent_iid"].append(blob["recurrent_rl"]["iid"]["seq_acc"])
        rows["recurrent_ood"].append(blob["recurrent_rl"]["ood"]["seq_acc"])

    summary = {
        "config": args.config,
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {k: aggregate(v) for k, v in rows.items()},
        "delta_structured_minus_recurrent": {
            "iid": aggregate([a - b for a, b in zip(rows["structured_iid"], rows["recurrent_iid"])]),
            "ood": aggregate([a - b for a, b in zip(rows["structured_ood"], rows["recurrent_ood"])]),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote RL comparison to {out_path}", flush=True)


if __name__ == "__main__":
    main()
