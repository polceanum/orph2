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

from ood_solver.data.dataset import SyntheticEpisodeDataset
from ood_solver.envs.hidden_mechanism_seq import HiddenMechanismSequenceEnv
from ood_solver.envs.config import build_env_from_cfg
from ood_solver.models.approach import ApproachProposer
from ood_solver.models.belief_updater import BeliefUpdater
from ood_solver.models.encoder import SimpleSequenceEncoder
from ood_solver.models.hypothesis_solver import HypothesisSolver
from ood_solver.models.probe_policy import ProbePolicy
from ood_solver.models.rules import RuleProposer
from ood_solver.models.solver_head import SolverHead
from ood_solver.training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_get(cfg: dict, key: str, default=None):
    parts = key.split(".")
    cur = cfg
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return cfg.get(parts[-1], default)
    return cur


def build_model(cfg: dict, device: str) -> HypothesisSolver:
    d_model = int(cfg_get(cfg, "model.d_model", 64))
    vocab_size = int(cfg_get(cfg, "env.vocab_size", 16))
    seq_len = int(cfg_get(cfg, "env.seq_len", 12))

    return HypothesisSolver(
        encoder=SimpleSequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=int(cfg_get(cfg, "model.nhead", 4)),
            num_layers=int(cfg_get(cfg, "model.num_layers", 2)),
            max_len=max(seq_len * 8, int(cfg_get(cfg, "model.max_len", seq_len * 8))),
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
            vocab_size=vocab_size,
            d_model=d_model,
            use_local_adapter=bool(cfg_get(cfg, "model.use_local_adapter", False)),
            use_value_features=bool(cfg_get(cfg, "model.use_value_features", False)),
            value_features_only=bool(cfg_get(cfg, "model.value_features_only", False)),
            use_modulo_shift_adapter=bool(cfg_get(cfg, "model.use_modulo_shift_adapter", False)),
            use_demo_shift_prior=bool(cfg_get(cfg, "model.use_demo_shift_prior", False)),
        ),
        soft_probe_training=bool(cfg_get(cfg, "model.soft_probe_training", False)),
        soft_probe_temp=float(cfg_get(cfg, "model.soft_probe_temp", 1.0)),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the structured OOD solver.")
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--save", type=str, default="artifacts/last.pt")
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=None)

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
    )

    def probe_executor(batch, step, chosen_idx):
        return env.execute_probe_batch(batch, step, chosen_idx, device=device)

    epochs = int(cfg_get(cfg, "train.epochs", 10))
    for epoch in range(epochs):
        last_metrics = None
        for batch in loader:
            last_metrics = trainer.train_step(batch, probe_executor, aux_modules=None)
        print(f"epoch={epoch} metrics={last_metrics}")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
