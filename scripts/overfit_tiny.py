import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ood_solver.envs.hidden_mechanism_seq import HiddenMechanismSequenceEnv
from ood_solver.envs.config import build_env_from_cfg
from ood_solver.models.encoder import SimpleSequenceEncoder
from ood_solver.models.approach import ApproachProposer
from ood_solver.models.rules import RuleProposer
from ood_solver.models.probe_policy import ProbePolicy
from ood_solver.models.belief_updater import BeliefUpdater
from ood_solver.models.solver_head import SolverHead
from ood_solver.models.hypothesis_solver import HypothesisSolver
from ood_solver.training.trainer import Trainer
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
    """
    Supports both nested and flat configs.
    Example:
      cfg_get(cfg, "model.d_model", 128)
      cfg_get(cfg, "train.batch_size", 32)
    Falls back to flat final key, e.g. "d_model" or "batch_size".
    """
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


def build_model(cfg: dict, device: str) -> HypothesisSolver:
    d_model = cfg_get(cfg, "model.d_model", 128)
    vocab_size = cfg_get(cfg, "env.vocab_size", 16)
    nhead = cfg_get(cfg, "model.nhead", 4)
    num_layers = cfg_get(cfg, "model.num_layers", 2)
    num_approaches = cfg_get(cfg, "model.num_approaches", 4)
    num_rules = cfg_get(cfg, "model.num_rules", 8)
    archive_size = cfg_get(cfg, "model.archive_size", 16)
    seq_len = cfg_get(cfg, "env.seq_len", 12)

    model = HypothesisSolver(
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
        soft_probe_training=bool(cfg_get(cfg, "model.soft_probe_training", False)),
        soft_probe_temp=float(cfg_get(cfg, "model.soft_probe_temp", 1.0)),
    ).to(device)
    return model


def make_fixed_episodes(env: HiddenMechanismSequenceEnv, batch_size: int, num_batches: int):
    return [env.sample_episode(batch_size=batch_size) for _ in range(num_batches)]


@torch.no_grad()
def evaluate_fixed_set(model, episodes, probe_executor, device: str) -> dict:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_app_ent = 0.0
    total_rule_ent = 0.0
    n = 0

    ce = torch.nn.CrossEntropyLoss()

    for batch in episodes:
        batch = move_episode_to_device(batch, device)
        logits, belief, logs = model(batch, probe_executor)
        b, l, v = logits.shape
        loss = ce(logits.reshape(b * l, v), batch.final_target.reshape(b * l))

        pred = logits.argmax(dim=-1)
        acc = (pred == batch.final_target).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()
        total_app_ent += entropy_from_logits(belief.approach_scores).item()
        total_rule_ent += entropy_from_logits(belief.rule_scores).item()
        n += 1

    return {
        "loss": total_loss / max(n, 1),
        "seq_acc": total_acc / max(n, 1),
        "approach_entropy": total_app_ent / max(n, 1),
        "rule_entropy": total_rule_ent / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--num-fixed-batches", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="artifacts/overfit_tiny")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg_get(cfg, "seed", 0)))

    env = build_env_from_cfg(cfg, section="env", seed=int(cfg_get(cfg, "seed", 0)))

    model = build_model(cfg, device)
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
    )

    fixed_episodes = make_fixed_episodes(
        env,
        batch_size=int(cfg_get(cfg, "train.batch_size", 32)),
        num_batches=args.num_fixed_batches,
    )

    def probe_executor(batch, step, chosen_idx):
        return env.execute_probe_batch(batch, step, chosen_idx, device=device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created fixed set with {len(fixed_episodes)} batches.")

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(fixed_episodes):
            metrics = trainer.train_step(batch, probe_executor, aux_modules=None)

        eval_metrics = evaluate_fixed_set(model, fixed_episodes, probe_executor, device)
        print(
            f"epoch={epoch} "
            f"train_loss={metrics['loss']:.4f} "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_seq_acc={eval_metrics['seq_acc']:.4f} "
            f"app_ent={eval_metrics['approach_entropy']:.4f} "
            f"rule_ent={eval_metrics['rule_entropy']:.4f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "eval_metrics": eval_metrics,
        }
        torch.save(ckpt, save_dir / "last.pt")

    print(f"Saved checkpoint to {save_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
