import yaml
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ood_solver.envs.config import build_env_from_cfg


def predict_local_from_demos(batch, vocab_size: int, seq_len: int, num_demos: int) -> torch.Tensor:
    x = batch.initial_tokens
    bsz = x.size(0)
    demos = x.view(bsz, num_demos, 2 * seq_len)
    demos_in = demos[:, :, :seq_len]
    demos_out = demos[:, :, seq_len:]
    left = torch.roll(demos_in, shifts=1, dims=-1)
    shifts = (demos_out - demos_in - left) % vocab_size
    flat = shifts.reshape(bsz, -1)

    pred_shift = []
    for i in range(bsz):
        counts = torch.bincount(flat[i], minlength=vocab_size)
        pred_shift.append(int(torch.argmax(counts)))
    pred_shift = torch.tensor(pred_shift, dtype=torch.long).unsqueeze(-1)

    q = batch.final_query
    qleft = torch.roll(q, shifts=1, dims=-1)
    return (q + qleft + pred_shift) % vocab_size


def eval_split(cfg: dict, split: str, num_batches: int = 100, batch_size: int = 64) -> float:
    section = "eval.id" if split == "id" else "eval.ood"
    seed_offset = 123 if split == "id" else 999
    env = build_env_from_cfg(cfg, section=section, seed=int(cfg.get("seed", 0)), seed_offset=seed_offset)
    vocab_size = int(cfg["env"]["vocab_size"])
    seq_len = int(cfg["env"]["seq_len"])
    num_demos = int(cfg["env"]["num_demos"])

    total = 0.0
    for _ in range(num_batches):
        batch = env.sample_episode(batch_size=batch_size)
        pred = predict_local_from_demos(batch, vocab_size=vocab_size, seq_len=seq_len, num_demos=num_demos)
        total += float((pred == batch.final_target).float().mean().item())
    return total / num_batches


def main():
    with open("configs/bridges/bridge_01_local_param.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    id_acc = eval_split(cfg, "id")
    ood_acc = eval_split(cfg, "ood")
    print({"id_rule_solver_seq_acc": id_acc, "ood_rule_solver_seq_acc": ood_acc})


if __name__ == "__main__":
    main()
