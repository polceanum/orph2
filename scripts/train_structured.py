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
from ood_solver.data.dataset import SyntheticEpisodeDataset
from ood_solver.models.encoder import SimpleSequenceEncoder
from ood_solver.models.approach import ApproachProposer
from ood_solver.models.rules import RuleProposer
from ood_solver.models.probe_policy import ProbePolicy
from ood_solver.models.belief_updater import BeliefUpdater
from ood_solver.models.solver_head import SolverHead
from ood_solver.models.hypothesis_solver import HypothesisSolver
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


def build_structured_model(cfg: dict, device: str) -> HypothesisSolver:
    d_model = int(cfg_get(cfg, "model.d_model", 64))
    vocab_size = int(cfg_get(cfg, "env.vocab_size", 16))
    nhead = int(cfg_get(cfg, "model.nhead", 4))
    num_layers = int(cfg_get(cfg, "model.num_layers", 2))
    num_approaches = int(cfg_get(cfg, "model.num_approaches", 4))
    num_rules = int(cfg_get(cfg, "model.num_rules", 8))
    archive_size = int(cfg_get(cfg, "model.archive_size", 16))
    seq_len = int(cfg_get(cfg, "env.seq_len", 12))

    model = HypothesisSolver(
        encoder=SimpleSequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=seq_len * 8,
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
        ),
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--save", type=str, default="artifacts/structured_last.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg_get(cfg, "seed", 0))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = HiddenMechanismSequenceEnv(
        vocab_size=int(cfg_get(cfg, "env.vocab_size", 16)),
        seq_len=int(cfg_get(cfg, "env.seq_len", 12)),
        num_probe_steps=int(cfg_get(cfg, "env.num_probe_steps", 4)),
        num_candidate_probes=int(cfg_get(cfg, "env.num_candidate_probes", 8)),
        seed=seed,
    )

    dataset = SyntheticEpisodeDataset(
        env=env,
        batch_size=int(cfg_get(cfg, "train.batch_size", 32)),
        steps_per_epoch=int(cfg_get(cfg, "train.steps_per_epoch", 100)),
    )

    model = build_structured_model(cfg, device)

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
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    epochs = int(cfg_get(cfg, "train.epochs", 10))
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def probe_executor(batch, step, chosen_idx):
        return env.execute_probe_batch(batch, step, chosen_idx, device=device)

    for epoch in range(epochs):
        model.train()
        last_metrics = None

        for step, batch in enumerate(loader):
            last_metrics = trainer.train_step(batch, probe_executor, aux_modules=None)

        print(f"epoch={epoch} metrics={last_metrics}")

    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
        },
        save_path,
    )
    print(f"Saved structured checkpoint to {save_path}")


if __name__ == "__main__":
    main()