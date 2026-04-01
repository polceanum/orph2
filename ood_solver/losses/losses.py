from __future__ import annotations

import torch
import torch.nn.functional as F


def task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Sequence token cross-entropy.

    logits: [B, L, V]
    targets: [B, L]
    """
    b, l, v = logits.shape
    return F.cross_entropy(
        logits.reshape(b * l, v),
        targets.reshape(b * l),
    )


def probe_supervision_loss(probe_logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over candidate probes.

    probe_logits: [B, P]
    target_idx: [B]
    """
    return F.cross_entropy(probe_logits, target_idx)


def approach_classification_loss(
    pooled_approach_repr: torch.Tensor,
    mechanism_id: torch.Tensor,
    classifier: torch.nn.Module,
) -> torch.Tensor:
    """
    Optional auxiliary loss if you later add a classifier over pooled approach state.

    pooled_approach_repr: [B, D]
    mechanism_id: [B]
    """
    logits = classifier(pooled_approach_repr)
    return F.cross_entropy(logits, mechanism_id)


def surprise_regression_loss(
    pred_surprise: torch.Tensor,
    target_surprise: torch.Tensor,
) -> torch.Tensor:
    """
    Optional auxiliary loss for surprise calibration.

    pred_surprise: [B, 1] or [B]
    target_surprise: [B, 1] or [B]
    """
    return F.mse_loss(pred_surprise.reshape(-1), target_surprise.reshape(-1))


def total_loss(loss_dict: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    total = None
    for name, value in loss_dict.items():
        weight = float(weights.get(name, 1.0))
        weighted = weight * value
        total = weighted if total is None else total + weighted

    if total is None:
        raise ValueError("loss_dict is empty; cannot compute total loss.")

    return total