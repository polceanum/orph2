import argparse
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

from ood_solver.baselines.no_rules_baseline import NoRulesSolver
from ood_solver.data.dataset import SyntheticEpisodeDataset
from ood_solver.envs.hidden_mechanism_seq import HiddenMechanismSequenceEnv
from ood_solver.envs.config import build_env_from_cfg
from ood_solver.models.approach import ApproachProposer
from ood_solver.models.belief_updater import BeliefUpdater
from ood_solver.models.encoder import SimpleSequenceEncoder
from ood_solver.models.probe_policy import ProbePolicy
from ood_solver.models.rules import RuleProposer
from ood_solver.models.solver_head import SolverHead
from ood_solver.training.learning_curve import (
    LearningCurveMonitor,
    evaluate_fixed_batches,
    format_metrics,
    mean_metrics,
    update_running_sums,
)
from ood_solver.training.trainer import Trainer


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


def build_no_rules_model(cfg: dict, device: str) -> NoRulesSolver:
    d_model = int(cfg_get(cfg, "model.d_model", 64))
    vocab_size = int(cfg_get(cfg, "env.vocab_size", 16))
    nhead = int(cfg_get(cfg, "model.nhead", 4))
    num_layers = int(cfg_get(cfg, "model.num_layers", 2))
    num_approaches = int(cfg_get(cfg, "model.num_approaches", 4))
    num_rules = int(cfg_get(cfg, "model.num_rules", 8))
    archive_size = int(cfg_get(cfg, "model.archive_size", 16))
    seq_len = int(cfg_get(cfg, "env.seq_len", 12))

    model = NoRulesSolver(
        encoder=SimpleSequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=seq_len * 8,
            use_value_features=bool(cfg_get(cfg, "model.use_value_features", False)),
            value_features_only=bool(cfg_get(cfg, "model.value_features_only", False)),
        ),
        approach_proposer=ApproachProposer(
            d_model=d_model,
            num_slots=num_approaches,
        ),
        rule_proposer=RuleProposer(
            d_model=d_model,
            num_rules=num_rules,
        ),
        probe_policy=ProbePolicy(d_model=d_model),
        belief_updater=BeliefUpdater(
            d_model=d_model,
            archive_size=archive_size,
        ),
        solver_head=SolverHead(
            d_model=d_model,
            vocab_size=vocab_size,
            use_local_adapter=bool(cfg_get(cfg, "model.use_local_adapter", False)),
            use_value_features=bool(cfg_get(cfg, "model.use_value_features", False)),
            value_features_only=bool(cfg_get(cfg, "model.value_features_only", False)),
            use_modulo_shift_adapter=bool(cfg_get(cfg, "model.use_modulo_shift_adapter", False)),
            use_demo_shift_prior=bool(cfg_get(cfg, "model.use_demo_shift_prior", False)),
        ),
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--save", type=str, default="artifacts/no_rules_last.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg_get(cfg, "seed", 0))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = build_env_from_cfg(cfg, section="env", seed=seed)

    dataset = SyntheticEpisodeDataset(
        env=env,
        batch_size=int(cfg_get(cfg, "train.batch_size", 32)),
        steps_per_epoch=int(cfg_get(cfg, "train.steps_per_epoch", 100)),
    )
    steps_per_epoch = int(cfg_get(cfg, "train.steps_per_epoch", 100))

    model = build_no_rules_model(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_get(cfg, "train.lr", 3e-4)),
        weight_decay=float(cfg_get(cfg, "train.weight_decay", 1e-4)),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_weights=cfg_get(cfg, "train.loss_weights", {"task": 1.0, "probe": 0.2}),
        grad_clip=cfg_get(cfg, "train.grad_clip", None),
        approach_entropy_weight=float(cfg_get(cfg, "train.approach_entropy_weight", 0.0)),
        rule_entropy_weight=float(cfg_get(cfg, "train.rule_entropy_weight", 0.0)),
        rule_score_spread_weight=float(cfg_get(cfg, "train.rule_score_spread_weight", 0.0)),
        rule_consistency_weight=float(cfg_get(cfg, "train.rule_consistency_weight", 0.0)),
        probe_ig_weight=float(cfg_get(cfg, "train.probe_ig_weight", 0.0)),
        demo_recon_weight=float(cfg_get(cfg, "train.demo_recon_weight", 0.0)),
        approach_diversity_weight=float(cfg_get(cfg, "train.approach_diversity_weight", 0.0)),
        rule_diversity_weight=float(cfg_get(cfg, "train.rule_diversity_weight", 0.0)),
        demo_recon_weight_start=cfg_get(cfg, "train.demo_recon_weight_start", None),
        demo_recon_weight_end=cfg_get(cfg, "train.demo_recon_weight_end", None),
        demo_recon_anneal_steps=int(cfg_get(cfg, "train.demo_recon_anneal_steps", 1)),
        aligned_warmup_steps=int(cfg_get(cfg, "train.aligned_warmup_steps", 0)),
        aligned_ramp_steps=int(cfg_get(cfg, "train.aligned_ramp_steps", 1)),
        aux_loss_ema_decay=float(cfg_get(cfg, "train.aux_loss_ema_decay", 0.98)),
        aux_loss_ema_eps=float(cfg_get(cfg, "train.aux_loss_ema_eps", 1e-6)),
        approach_entropy_target_start=cfg_get(cfg, "train.approach_entropy_target_start", None),
        approach_entropy_target_end=cfg_get(cfg, "train.approach_entropy_target_end", None),
        rule_entropy_target_start=cfg_get(cfg, "train.rule_entropy_target_start", None),
        rule_entropy_target_end=cfg_get(cfg, "train.rule_entropy_target_end", None),
        entropy_target_anneal_steps=int(cfg_get(cfg, "train.entropy_target_anneal_steps", 1)),
        entropy_target_mode=str(cfg_get(cfg, "train.entropy_target_mode", "floor")),
        probe_weight_start=cfg_get(cfg, "train.probe_weight_start", None),
        probe_weight_end=cfg_get(cfg, "train.probe_weight_end", None),
        probe_weight_anneal_steps=int(cfg_get(cfg, "train.probe_weight_anneal_steps", 1)),
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    epochs = int(cfg_get(cfg, "train.epochs", 10))
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_every_steps = int(cfg_get(cfg, "train.monitor.log_every_steps", 50))
    monitor = LearningCurveMonitor(
        metric=str(cfg_get(cfg, "train.monitor.metric", "seq_acc")),
        mode=str(cfg_get(cfg, "train.monitor.mode", "max")),
        min_delta=float(cfg_get(cfg, "train.monitor.min_delta", 1e-4)),
        patience=int(cfg_get(cfg, "train.monitor.patience", 10)),
        warmup_epochs=int(cfg_get(cfg, "train.monitor.warmup_epochs", 0)),
        loss_key=str(cfg_get(cfg, "train.monitor.loss_key", "loss")),
        log_jsonl_path=cfg_get(
            cfg,
            "train.monitor.log_jsonl_path",
            str(save_path.with_suffix(".metrics.jsonl")),
        ),
        instability_window=int(cfg_get(cfg, "train.monitor.instability_window", 5)),
        loss_spike_ratio=float(cfg_get(cfg, "train.monitor.loss_spike_ratio", 0.35)),
        acc_drop_tol=float(cfg_get(cfg, "train.monitor.acc_drop_tol", 0.08)),
        metric_ema_beta=float(cfg_get(cfg, "train.monitor.metric_ema_beta", 0.0)),
    )
    enable_early_stop = bool(cfg_get(cfg, "train.monitor.enable_early_stop", True))
    eval_batches_n = int(cfg_get(cfg, "train.monitor.eval_batches", 0))
    eval_episodes = []
    eval_env = None
    if eval_batches_n > 0:
        eval_section = str(cfg_get(cfg, "train.monitor.eval_section", "eval.id"))
        has_eval_id = cfg_get(cfg, "eval.id", None) is not None
        if eval_section == "eval.id" and not has_eval_id:
            eval_section = "env"
        eval_env = build_env_from_cfg(
            cfg,
            section=eval_section,
            seed=seed,
            seed_offset=int(cfg_get(cfg, "train.monitor.eval_seed_offset", 777)),
        )
        eval_episodes = [
            eval_env.sample_episode(batch_size=int(cfg_get(cfg, "train.batch_size", 32)))
            for _ in range(eval_batches_n)
        ]

    def probe_executor(batch, step, chosen_idx):
        return env.execute_probe_batch(batch, step, chosen_idx, device=device)

    def eval_probe_executor(batch, step, chosen_idx):
        if not eval_episodes or eval_env is None:
            return env.execute_probe_batch(batch, step, chosen_idx, device=device)
        return eval_env.execute_probe_batch(batch, step, chosen_idx, device=device)

    for epoch in range(epochs):
        model.train()
        running_sums: dict[str, float] = defaultdict(float)
        seen_steps = 0

        for step, batch in enumerate(loader):
            step_metrics = trainer.train_step(batch, probe_executor, aux_modules=None)
            update_running_sums(running_sums, step_metrics)
            seen_steps += 1
            if log_every_steps > 0 and ((step + 1) % log_every_steps == 0):
                running_avg = mean_metrics(running_sums, seen_steps)
                msg = format_metrics(
                    running_avg,
                    ["loss", "task_loss", "seq_acc", "probe_loss", "approach_entropy", "rule_entropy"],
                )
                print(f"epoch={epoch} step={step+1}/{steps_per_epoch} avg {msg}")

        epoch_metrics = mean_metrics(running_sums, seen_steps)
        if eval_episodes:
            eval_metrics = evaluate_fixed_batches(
                model=model,
                batches=eval_episodes,
                probe_executor=eval_probe_executor,
                device=device,
                structured=True,
            )
            epoch_metrics.update(eval_metrics)
        event = monitor.evaluate_epoch(epoch, epoch_metrics)
        summary = format_metrics(
            epoch_metrics,
            [
                "loss",
                "task_loss",
                "seq_acc",
                "eval_loss",
                "eval_seq_acc",
                "eval_seq_acc_ci95",
                "eval_seq_acc_lcb",
                "probe_loss",
            ],
        )
        best_value = event["monitor"]["best_value"]
        monitor_value = event["monitor"].get("monitor_value", None)
        monitor_str = "None" if monitor_value is None else f"{monitor_value:.4f}"
        best_str = "None" if best_value is None else f"{best_value:.4f}"
        print(
            f"epoch={epoch} summary {summary} "
            f"monitor_value={monitor_str} "
            f"best_{event['monitor']['metric']}={best_str} "
            f"bad_epochs={event['monitor']['bad_epochs']}"
        )
        if event["monitor"]["warnings"]:
            print(f"epoch={epoch} warnings={event['monitor']['warnings']}")
        if event["monitor"]["suggestions"]:
            print(f"epoch={epoch} suggestions={event['monitor']['suggestions']}")
        if enable_early_stop and event["monitor"]["should_stop"]:
            print(f"Early stopping at epoch={epoch} reason={event['monitor']['stop_reason']}")
            break

    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
        },
        save_path,
    )
    print(f"Saved no-rules checkpoint to {save_path}")


if __name__ == "__main__":
    main()
