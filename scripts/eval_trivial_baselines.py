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


@torch.no_grad()
def eval_trivial(env, cfg, num_batches: int):
    vocab_size = int(cfg_get(cfg, "env.vocab_size", 16))
    batch_size = int(cfg_get(cfg, "train.batch_size", 32))

    overall = defaultdict(float)
    by_mech = defaultdict(lambda: defaultdict(float))

    for _ in range(num_batches):
        batch = env.sample_episode(batch_size=batch_size)
        target = batch.final_target

        rand_pred = torch.randint(
            low=0,
            high=vocab_size,
            size=target.shape,
        )

        copy_pred = batch.final_query.clone()

        mode_vals = []
        for row in batch.final_query:
            vals, counts = row.unique(return_counts=True)
            mode_vals.append(vals[counts.argmax()])
        mode_vals = torch.stack(mode_vals)
        mode_pred = mode_vals.unsqueeze(1).expand_as(target)

        rand_acc = (rand_pred == target).float().mean(dim=-1)
        copy_acc = (copy_pred == target).float().mean(dim=-1)
        mode_acc = (mode_pred == target).float().mean(dim=-1)

        overall["random_acc"] += rand_acc.mean().item()
        overall["copy_acc"] += copy_acc.mean().item()
        overall["mode_acc"] += mode_acc.mean().item()

        for i in range(target.size(0)):
            mech = int(batch.mechanism_id[i].item())
            by_mech[mech]["random_acc"] += float(rand_acc[i].item())
            by_mech[mech]["copy_acc"] += float(copy_acc[i].item())
            by_mech[mech]["mode_acc"] += float(mode_acc[i].item())
            by_mech[mech]["count"] += 1

    result = {
        "overall": {
            "random_acc": overall["random_acc"] / max(num_batches, 1),
            "copy_acc": overall["copy_acc"] / max(num_batches, 1),
            "mode_acc": overall["mode_acc"] / max(num_batches, 1),
        },
        "by_mechanism": {},
    }

    for mech, vals in by_mech.items():
        cnt = int(vals["count"])
        result["by_mechanism"][str(mech)] = {
            "random_acc": vals["random_acc"] / max(cnt, 1),
            "copy_acc": vals["copy_acc"] / max(cnt, 1),
            "mode_acc": vals["mode_acc"] / max(cnt, 1),
            "count": cnt,
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--num-batches", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg_get(cfg, "seed", 0)))

    env = HiddenMechanismSequenceEnv(
        vocab_size=int(cfg_get(cfg, "env.vocab_size", 16)),
        seq_len=int(cfg_get(cfg, "env.seq_len", 12)),
        num_probe_steps=int(cfg_get(cfg, "env.num_probe_steps", 4)),
        num_candidate_probes=int(cfg_get(cfg, "env.num_candidate_probes", 8)),
        seed=int(cfg_get(cfg, "seed", 0) + 777),
    )

    result = eval_trivial(env, cfg, args.num_batches)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()