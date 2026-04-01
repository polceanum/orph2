from __future__ import annotations

import torch

try:
    from torch.amp import GradScaler, autocast
    _USE_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _USE_NEW_AMP = False

from ood_solver.losses.losses import task_loss, probe_supervision_loss, total_loss
from ood_solver.training.metrics import entropy_from_logits
from ood_solver.utils import move_episode_to_device


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        loss_weights,
        approach_entropy_weight: float = 0.0,
        rule_entropy_weight: float = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_weights = loss_weights
        self.approach_entropy_weight = float(approach_entropy_weight)
        self.rule_entropy_weight = float(rule_entropy_weight)

        if _USE_NEW_AMP:
            self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
        else:
            self.scaler = GradScaler(enabled=torch.cuda.is_available())

    def train_step(self, batch, probe_executor, aux_modules=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = move_episode_to_device(batch, self.device)

        if _USE_NEW_AMP:
            amp_ctx = autocast("cuda", enabled=torch.cuda.is_available())
        else:
            amp_ctx = autocast(enabled=torch.cuda.is_available())

        with amp_ctx:
            logits, belief, logs = self.model(batch, probe_executor)

            losses = {
                "task": task_loss(logits, batch.final_target),
            }

            if batch.diagnostic_probe_targets is not None and len(logs) > 0:
                probe_losses = []
                for step_log, tgt in zip(logs, batch.diagnostic_probe_targets):
                    probe_losses.append(
                        probe_supervision_loss(
                            step_log["probe_logits"],
                            tgt.to(self.device),
                        )
                    )
                losses["probe"] = torch.stack(probe_losses).mean()

            # entropy regularization: subtract entropy penalties from total loss
            app_ent = entropy_from_logits(belief.approach_scores)
            rule_ent = entropy_from_logits(belief.rule_scores)

            total = total_loss(losses, self.loss_weights)
            total = total - self.approach_entropy_weight * app_ent
            total = total - self.rule_entropy_weight * rule_ent

        self.scaler.scale(total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            seq_acc = (pred == batch.final_target).float().mean()

        metrics = {
            "loss": float(total.item()),
            "task_loss": float(losses["task"].item()),
            "seq_acc": float(seq_acc.item()),
            "approach_entropy": float(app_ent.item()),
            "rule_entropy": float(rule_ent.item()),
            "surprise_mean": float(belief.surprise.mean().item()),
            "stagnation_mean": float(belief.stagnation.mean().item()),
        }

        if "probe" in losses:
            metrics["probe_loss"] = float(losses["probe"].item())

        return metrics