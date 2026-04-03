import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ood_solver.envs.hidden_mechanism_seq import HiddenMechanismSequenceEnv
from ood_solver.models.approach import ApproachProposer
from ood_solver.models.belief_updater import BeliefUpdater
from ood_solver.models.encoder import SimpleSequenceEncoder
from ood_solver.models.hypothesis_solver import HypothesisSolver
from ood_solver.models.probe_policy import ProbePolicy
from ood_solver.models.rules import RuleProposer
from ood_solver.models.solver_head import SolverHead
from ood_solver.utils import move_episode_to_device


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
    parser = argparse.ArgumentParser(description="Plot belief traces for one sampled episode.")
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--ckpt", type=str, default="artifacts/last.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg_get(cfg, "seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = HiddenMechanismSequenceEnv(
        vocab_size=int(cfg_get(cfg, "env.vocab_size", 16)),
        seq_len=int(cfg_get(cfg, "env.seq_len", 12)),
        num_probe_steps=int(cfg_get(cfg, "env.num_probe_steps", 4)),
        num_candidate_probes=int(cfg_get(cfg, "env.num_candidate_probes", 8)),
        num_demos=int(cfg_get(cfg, "env.num_demos", 3)),
        seed=seed + 2,
    )

    model = build_model(cfg, device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    batch = move_episode_to_device(env.sample_episode(batch_size=1), device)

    def probe_executor(ep, step, chosen_idx):
        return env.execute_probe_batch(ep, step, chosen_idx, device=device)

    with torch.no_grad():
        _, _, logs = model(batch, probe_executor)

    approach_scores = [log["approach_scores"].squeeze(0).cpu() for log in logs]
    rule_scores = [log["rule_scores"].squeeze(0).cpu() for log in logs]
    surprise = [log["surprise"].squeeze().cpu().item() for log in logs]

    plt.figure(figsize=(10, 4))
    for i in range(approach_scores[0].numel()):
        plt.plot([x[i].item() for x in approach_scores], label=f"approach_{i}")
    plt.title("Approach scores over probe steps")
    plt.xlabel("step")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for i in range(rule_scores[0].numel()):
        plt.plot([x[i].item() for x in rule_scores], label=f"rule_{i}")
    plt.title("Rule scores over probe steps")
    plt.xlabel("step")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.plot(surprise, marker="o")
    plt.title("Surprise over probe steps")
    plt.xlabel("step")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
