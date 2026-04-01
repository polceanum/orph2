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
from ood_solver.baselines.recurrent_baseline import RecurrentBaseline
from ood_solver.losses.losses import task_loss
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

    def probe_executor(batch, step, chosen_idx):
        return env.execute_probe_batch(batch, step, chosen_idx, device=device)

    for epoch in range(epochs):
        model.train()
        last_metrics = None

        for step, batch in enumerate(loader):
            batch = move_episode_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(batch, probe_executor)
            loss = task_loss(logits, batch.final_target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                seq_acc = (pred == batch.final_target).float().mean()

            last_metrics = {
                "loss": float(loss.item()),
                "seq_acc": float(seq_acc.item()),
            }

        print(f"epoch={epoch} metrics={last_metrics}")

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