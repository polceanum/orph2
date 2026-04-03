from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from torch.amp import GradScaler, autocast
    _USE_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _USE_NEW_AMP = False

from ood_solver.losses.losses import task_loss, probe_supervision_loss, total_loss
from ood_solver.training.metrics import (
    entropy_from_logits,
    score_spread_from_logits,
    top1_margin_from_logits,
)
from ood_solver.utils import move_episode_to_device


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        loss_weights,
        grad_clip: float | None = None,
        approach_entropy_weight: float = 0.0,
        rule_entropy_weight: float = 0.0,
        rule_score_spread_weight: float = 0.0,
        rule_consistency_weight: float = 0.0,
        probe_ig_weight: float = 0.0,
        demo_recon_weight: float = 0.0,
        approach_diversity_weight: float = 0.0,
        rule_diversity_weight: float = 0.0,
        demo_recon_weight_start: float | None = None,
        demo_recon_weight_end: float | None = None,
        demo_recon_anneal_steps: int = 1,
        aligned_warmup_steps: int = 0,
        aligned_ramp_steps: int = 1,
        aux_loss_ema_decay: float = 0.98,
        aux_loss_ema_eps: float = 1e-6,
        approach_entropy_target_start: float | None = None,
        approach_entropy_target_end: float | None = None,
        rule_entropy_target_start: float | None = None,
        rule_entropy_target_end: float | None = None,
        entropy_target_anneal_steps: int = 1,
        entropy_target_mode: str = "floor",
        probe_weight_start: float | None = None,
        probe_weight_end: float | None = None,
        probe_weight_anneal_steps: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_weights = loss_weights
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.approach_entropy_weight = float(approach_entropy_weight)
        self.rule_entropy_weight = float(rule_entropy_weight)
        self.rule_score_spread_weight = float(rule_score_spread_weight)
        self.rule_consistency_weight = float(rule_consistency_weight)
        self.probe_ig_weight = float(probe_ig_weight)
        self.demo_recon_weight = float(demo_recon_weight)
        self.approach_diversity_weight = float(approach_diversity_weight)
        self.rule_diversity_weight = float(rule_diversity_weight)
        self.demo_recon_weight_start = (
            float(demo_recon_weight_start) if demo_recon_weight_start is not None else None
        )
        self.demo_recon_weight_end = (
            float(demo_recon_weight_end) if demo_recon_weight_end is not None else None
        )
        self.demo_recon_anneal_steps = max(1, int(demo_recon_anneal_steps))
        self.aligned_warmup_steps = int(aligned_warmup_steps)
        self.aligned_ramp_steps = max(1, int(aligned_ramp_steps))
        self.global_step = 0
        self.aux_loss_ema_decay = float(aux_loss_ema_decay)
        self.aux_loss_ema_eps = float(aux_loss_ema_eps)
        self.aux_loss_ema: dict[str, torch.Tensor] = {}
        self.approach_entropy_target_start = (
            float(approach_entropy_target_start) if approach_entropy_target_start is not None else None
        )
        self.approach_entropy_target_end = (
            float(approach_entropy_target_end) if approach_entropy_target_end is not None else None
        )
        self.rule_entropy_target_start = (
            float(rule_entropy_target_start) if rule_entropy_target_start is not None else None
        )
        self.rule_entropy_target_end = (
            float(rule_entropy_target_end) if rule_entropy_target_end is not None else None
        )
        self.entropy_target_anneal_steps = max(1, int(entropy_target_anneal_steps))
        if entropy_target_mode not in {"floor", "track"}:
            raise ValueError(f"Unsupported entropy_target_mode={entropy_target_mode!r}; expected 'floor' or 'track'.")
        self.entropy_target_mode = entropy_target_mode
        self.probe_weight_start = float(probe_weight_start) if probe_weight_start is not None else None
        self.probe_weight_end = float(probe_weight_end) if probe_weight_end is not None else None
        self.probe_weight_anneal_steps = max(1, int(probe_weight_anneal_steps))

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

            if self.global_step < self.aligned_warmup_steps:
                aligned_scale = 0.0
            else:
                aligned_scale = min(
                    1.0,
                    (self.global_step - self.aligned_warmup_steps + 1) / float(self.aligned_ramp_steps),
                )

            if self.rule_consistency_weight > 0.0 and len(logs) > 0:
                consistency_losses = []
                for step_log in logs:
                    if "rule_obs_pred" in step_log and "rule_obs_target" in step_log:
                        consistency_losses.append(
                            F.mse_loss(step_log["rule_obs_pred"], step_log["rule_obs_target"])
                        )
                if len(consistency_losses) > 0:
                    losses["rule_consistency"] = torch.stack(consistency_losses).mean()

            if self.probe_ig_weight > 0.0 and len(logs) > 0:
                ig_losses = []
                for step_log in logs:
                    if "approach_disagreement" not in step_log:
                        continue
                    with torch.no_grad():
                        disagreement = step_log["approach_disagreement"].detach()
                        disagreement = disagreement - disagreement.mean(dim=-1, keepdim=True)
                        disagreement = disagreement / disagreement.std(dim=-1, keepdim=True).clamp_min(1e-6)
                        target = torch.softmax(disagreement, dim=-1)

                    log_policy = torch.log_softmax(step_log["probe_logits"], dim=-1)
                    ig_losses.append(F.kl_div(log_policy, target, reduction="batchmean"))

                if len(ig_losses) > 0:
                    losses["probe_ig"] = torch.stack(ig_losses).mean()
            if (
                self.demo_recon_weight > 0.0
                and getattr(belief, "demo_logits", None) is not None
                and getattr(belief, "demo_targets", None) is not None
            ):
                demo_logits = belief.demo_logits
                demo_targets = belief.demo_targets.to(self.device)
                b_demo, l_demo, v_demo = demo_logits.shape
                losses["demo_recon"] = F.cross_entropy(
                    demo_logits.reshape(b_demo * l_demo, v_demo),
                    demo_targets.reshape(b_demo * l_demo),
                )
            if self.approach_diversity_weight > 0.0:
                losses["approach_diversity"] = self._orthogonality_loss(belief.approach_slots)
            if self.rule_diversity_weight > 0.0:
                losses["rule_diversity"] = self._orthogonality_loss(belief.rule_slots)

            # entropy regularization: subtract entropy penalties from total loss
            app_ent = entropy_from_logits(belief.approach_scores)
            rule_ent = entropy_from_logits(belief.rule_scores)
            rule_spread = score_spread_from_logits(belief.rule_scores)

            effective_loss_weights = dict(self.loss_weights)
            probe_weight_applied = effective_loss_weights.get("probe", None)
            if (
                "probe" in losses
                and self.probe_weight_start is not None
                and self.probe_weight_end is not None
            ):
                progress = min(1.0, self.global_step / float(self.probe_weight_anneal_steps))
                probe_weight_applied = (
                    (1.0 - progress) * self.probe_weight_start
                    + progress * self.probe_weight_end
                )
                effective_loss_weights["probe"] = probe_weight_applied
            weighted_losses = {name: value for name, value in losses.items() if name in effective_loss_weights}
            total = total_loss(weighted_losses, effective_loss_weights)
            use_approach_entropy_floor = (
                self.approach_entropy_target_start is not None
                and self.approach_entropy_target_end is not None
            )
            use_rule_entropy_floor = (
                self.rule_entropy_target_start is not None
                and self.rule_entropy_target_end is not None
            )
            if not use_approach_entropy_floor:
                total = total - self.approach_entropy_weight * app_ent
            if not use_rule_entropy_floor:
                total = total - self.rule_entropy_weight * rule_ent
            total = total - self.rule_score_spread_weight * rule_spread
            approach_entropy_target = None
            rule_entropy_target = None
            approach_entropy_floor_penalty = None
            rule_entropy_floor_penalty = None
            if (
                self.approach_entropy_target_start is not None
                and self.approach_entropy_target_end is not None
                and self.approach_entropy_weight > 0.0
            ):
                progress = min(1.0, self.global_step / float(self.entropy_target_anneal_steps))
                approach_entropy_target = (
                    (1.0 - progress) * self.approach_entropy_target_start
                    + progress * self.approach_entropy_target_end
                )
                app_target = torch.tensor(approach_entropy_target, device=app_ent.device, dtype=app_ent.dtype)
                if self.entropy_target_mode == "track":
                    approach_entropy_floor_penalty = (app_ent - app_target).pow(2)
                else:
                    approach_entropy_floor_penalty = torch.relu(app_target - app_ent).pow(2)
                total = total + self.approach_entropy_weight * approach_entropy_floor_penalty
            if (
                self.rule_entropy_target_start is not None
                and self.rule_entropy_target_end is not None
                and self.rule_entropy_weight > 0.0
            ):
                progress = min(1.0, self.global_step / float(self.entropy_target_anneal_steps))
                rule_entropy_target = (
                    (1.0 - progress) * self.rule_entropy_target_start
                    + progress * self.rule_entropy_target_end
                )
                rule_target = torch.tensor(rule_entropy_target, device=rule_ent.device, dtype=rule_ent.dtype)
                if self.entropy_target_mode == "track":
                    rule_entropy_floor_penalty = (rule_ent - rule_target).pow(2)
                else:
                    rule_entropy_floor_penalty = torch.relu(rule_target - rule_ent).pow(2)
                total = total + self.rule_entropy_weight * rule_entropy_floor_penalty
            raw_rule_consistency = None
            raw_probe_ig = None
            norm_rule_consistency = None
            norm_probe_ig = None
            if "rule_consistency" in losses:
                raw_rule_consistency = losses["rule_consistency"]
                norm_rule_consistency = self._normalize_aux_loss("rule_consistency", raw_rule_consistency)
                total = total + (aligned_scale * self.rule_consistency_weight) * norm_rule_consistency
            if "probe_ig" in losses:
                raw_probe_ig = losses["probe_ig"]
                norm_probe_ig = self._normalize_aux_loss("probe_ig", raw_probe_ig)
                total = total + (aligned_scale * self.probe_ig_weight) * norm_probe_ig
            demo_recon_weight_applied = self.demo_recon_weight
            if (
                self.demo_recon_weight_start is not None
                and self.demo_recon_weight_end is not None
            ):
                progress = min(1.0, self.global_step / float(self.demo_recon_anneal_steps))
                demo_recon_weight_applied = (
                    (1.0 - progress) * self.demo_recon_weight_start
                    + progress * self.demo_recon_weight_end
                )
            if "demo_recon" in losses:
                total = total + demo_recon_weight_applied * losses["demo_recon"]
            if "approach_diversity" in losses:
                raw_app_div = losses["approach_diversity"]
                norm_app_div = self._normalize_aux_loss("approach_diversity", raw_app_div)
                total = total + (aligned_scale * self.approach_diversity_weight) * norm_app_div
            if "rule_diversity" in losses:
                raw_rule_div = losses["rule_diversity"]
                norm_rule_div = self._normalize_aux_loss("rule_diversity", raw_rule_div)
                total = total + (aligned_scale * self.rule_diversity_weight) * norm_rule_div

        self.scaler.scale(total).backward()
        if self.grad_clip is not None and self.grad_clip > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            seq_acc = (pred == batch.final_target).float().mean()
            approach_margin = top1_margin_from_logits(belief.approach_scores)
            rule_margin = top1_margin_from_logits(belief.rule_scores)

            step_rule_margin = 0.0
            step_rule_spread = 0.0
            step_probe_margin = 0.0
            step_probe_acc = 0.0
            num_steps = max(len(logs), 1)
            for step_idx, step_log in enumerate(logs):
                step_rule_margin += top1_margin_from_logits(step_log["rule_scores"]).item()
                step_rule_spread += score_spread_from_logits(step_log["rule_scores"]).item()
                step_probe_margin += top1_margin_from_logits(step_log["probe_logits"]).item()
                if batch.diagnostic_probe_targets is not None:
                    tgt = batch.diagnostic_probe_targets[step_idx].to(self.device)
                    step_probe_acc += (step_log["chosen_idx"] == tgt).float().mean().item()

            step_rule_margin /= num_steps
            step_rule_spread /= num_steps
            step_probe_margin /= num_steps
            step_probe_acc = step_probe_acc / num_steps if batch.diagnostic_probe_targets is not None else float("nan")

        metrics = {
            "loss": float(total.item()),
            "task_loss": float(losses["task"].item()),
            "seq_acc": float(seq_acc.item()),
            "approach_entropy": float(app_ent.item()),
            "rule_entropy": float(rule_ent.item()),
            "rule_score_spread": float(rule_spread.item()),
            "approach_top1_margin": float(approach_margin.item()),
            "rule_top1_margin": float(rule_margin.item()),
            "step_rule_top1_margin": float(step_rule_margin),
            "step_rule_score_spread": float(step_rule_spread),
            "step_probe_top1_margin": float(step_probe_margin),
            "surprise_mean": float(belief.surprise.mean().item()),
            "stagnation_mean": float(belief.stagnation.mean().item()),
            "aligned_scale": float(aligned_scale),
        }

        if "probe" in losses:
            metrics["probe_loss"] = float(losses["probe"].item())
            if probe_weight_applied is not None:
                metrics["probe_weight"] = float(probe_weight_applied)
        if "rule_consistency" in losses:
            metrics["rule_consistency_loss"] = float(losses["rule_consistency"].item())
            metrics["rule_consistency_norm"] = float(norm_rule_consistency.item())
        if "probe_ig" in losses:
            metrics["probe_ig_loss"] = float(losses["probe_ig"].item())
            metrics["probe_ig_norm"] = float(norm_probe_ig.item())
        if "demo_recon" in losses:
            metrics["demo_recon_loss"] = float(losses["demo_recon"].item())
            metrics["demo_recon_weight"] = float(demo_recon_weight_applied)
        if "approach_diversity" in losses:
            metrics["approach_diversity_loss"] = float(losses["approach_diversity"].item())
        if "rule_diversity" in losses:
            metrics["rule_diversity_loss"] = float(losses["rule_diversity"].item())
        if approach_entropy_target is not None:
            metrics["approach_entropy_target"] = float(approach_entropy_target)
            metrics["approach_entropy_floor_penalty"] = float(approach_entropy_floor_penalty.item())
        if rule_entropy_target is not None:
            metrics["rule_entropy_target"] = float(rule_entropy_target)
            metrics["rule_entropy_floor_penalty"] = float(rule_entropy_floor_penalty.item())
        metrics["entropy_target_mode"] = self.entropy_target_mode
        if not torch.isnan(torch.tensor(step_probe_acc)):
            metrics["step_diagnostic_probe_acc"] = float(step_probe_acc)

        self.global_step += 1
        return metrics

    def _normalize_aux_loss(self, name: str, raw_loss: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            raw_detached = raw_loss.detach()
            if name not in self.aux_loss_ema:
                self.aux_loss_ema[name] = raw_detached
            else:
                self.aux_loss_ema[name] = (
                    self.aux_loss_ema_decay * self.aux_loss_ema[name]
                    + (1.0 - self.aux_loss_ema_decay) * raw_detached
                )
            denom = self.aux_loss_ema[name].clamp_min(self.aux_loss_ema_eps)
        return raw_loss / denom

    @staticmethod
    def _orthogonality_loss(slots: torch.Tensor) -> torch.Tensor:
        """
        Penalize redundant slot content via squared off-diagonal cosine similarity.
        slots: [B, K, D]
        """
        b, k, _ = slots.shape
        if k < 2:
            return slots.new_zeros(())
        normed = F.normalize(slots, p=2, dim=-1, eps=1e-6)
        sim = torch.matmul(normed, normed.transpose(1, 2))
        eye = torch.eye(k, device=slots.device, dtype=sim.dtype).unsqueeze(0)
        off_diag = sim * (1.0 - eye)
        denom = float(k * (k - 1))
        return off_diag.pow(2).sum(dim=(1, 2)).mean() / denom
