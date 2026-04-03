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
from ood_solver.training.metrics import entropy_from_logits, sequence_accuracy
from ood_solver.utils import move_episode_to_device


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
    parser = argparse.ArgumentParser(description="Evaluate a structured solver checkpoint.")
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--ckpt", type=str, default="artifacts/last.pt")
    parser.add_argument("--num-batches", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg_get(cfg, "seed", 0))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = build_env_from_cfg(cfg, section="env", seed=seed, seed_offset=1)
    dataset = SyntheticEpisodeDataset(
        env=env,
        batch_size=int(cfg_get(cfg, "train.batch_size", 32)),
        steps_per_epoch=args.num_batches,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    model = build_model(cfg, device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    def probe_executor(batch, step, chosen_idx):
        return env.execute_probe_batch(batch, step, chosen_idx, device=device)

    accs = []
    approach_entropies = []
    rule_entropies = []

    with torch.no_grad():
        for batch in loader:
            batch = move_episode_to_device(batch, device)
            logits, belief, _ = model(batch, probe_executor)
            accs.append(sequence_accuracy(logits, batch.final_target).item())
            approach_entropies.append(entropy_from_logits(belief.approach_scores).item())
            rule_entropies.append(entropy_from_logits(belief.rule_scores).item())

    print(
        {
            "mean_seq_acc": sum(accs) / len(accs),
            "mean_approach_entropy": sum(approach_entropies) / len(approach_entropies),
            "mean_rule_entropy": sum(rule_entropies) / len(rule_entropies),
        }
    )


if __name__ == "__main__":
    main()
