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

from ood_solver.envs.hidden_mechanism_seq import HiddenMechanismSequenceEnv
from ood_solver.envs.config import build_env_from_cfg
from ood_solver.data.dataset import SyntheticEpisodeDataset
from ood_solver.baselines.recurrent_baseline import RecurrentBaseline
from ood_solver.losses.losses import task_loss
from ood_solver.training.learning_curve import (
    LearningCurveMonitor,
    evaluate_fixed_batches,
    format_metrics,
    mean_metrics,
    update_running_sums,
)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--save", type=str, default="artifacts/recurrent_last.pt")
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

    model = RecurrentBaseline(
        vocab_size=int(cfg_get(cfg, "env.vocab_size", 16)),
        d_model=int(cfg_get(cfg, "model.d_model", 64)),
        hidden_dim=int(cfg_get(cfg, "model.d_model", 64)),
        seq_len=int(cfg_get(cfg, "env.seq_len", 12)),
        num_probe_steps=int(cfg_get(cfg, "env.num_probe_steps", 4)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_get(cfg, "train.lr", 3e-4)),
        weight_decay=float(cfg_get(cfg, "train.weight_decay", 1e-4)),
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
    grad_clip = cfg_get(cfg, "train.grad_clip", None)
    grad_clip = None if grad_clip is None else float(grad_clip)
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

    for epoch in range(epochs):
        model.train()
        running_sums: dict[str, float] = defaultdict(float)
        seen_steps = 0

        for step, batch in enumerate(loader):
            batch = move_episode_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(batch, probe_executor)
            loss = task_loss(logits, batch.final_target)
            loss.backward()
            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                seq_acc = (pred == batch.final_target).float().mean()

            step_metrics = {
                "loss": float(loss.item()),
                "seq_acc": float(seq_acc.item()),
            }
            update_running_sums(running_sums, step_metrics)
            seen_steps += 1
            if log_every_steps > 0 and ((step + 1) % log_every_steps == 0):
                running_avg = mean_metrics(running_sums, seen_steps)
                msg = format_metrics(running_avg, ["loss", "seq_acc"])
                print(f"epoch={epoch} step={step+1}/{steps_per_epoch} avg {msg}")

        epoch_metrics = mean_metrics(running_sums, seen_steps)
        if eval_episodes:
            def eval_probe_executor(batch, step, chosen_idx):
                if eval_env is None:
                    return env.execute_probe_batch(batch, step, chosen_idx, device=device)
                return eval_env.execute_probe_batch(batch, step, chosen_idx, device=device)

            eval_metrics = evaluate_fixed_batches(
                model=model,
                batches=eval_episodes,
                probe_executor=eval_probe_executor,
                device=device,
                structured=False,
            )
            epoch_metrics.update(eval_metrics)
        event = monitor.evaluate_epoch(epoch, epoch_metrics)
        summary = format_metrics(
            epoch_metrics,
            ["loss", "seq_acc", "eval_loss", "eval_seq_acc", "eval_seq_acc_ci95", "eval_seq_acc_lcb"],
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
    print(f"Saved recurrent checkpoint to {save_path}")


if __name__ == "__main__":
    main()
