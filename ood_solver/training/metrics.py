from __future__ import annotations

import torch


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Mean categorical entropy across the last dimension.

    logits: [..., K]
    returns: scalar tensor
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-9))
    ent = -(probs * log_probs).sum(dim=-1)
    return ent.mean()


def sequence_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Token-level mean accuracy for sequence predictions.

    logits: [B, L, V]
    targets: [B, L]
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()


def sequence_accuracy_per_item(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Per-example token-level mean accuracy.

    logits: [B, L, V]
    targets: [B, L]
    returns: [B]
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean(dim=-1)


def top1_margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Mean top-1 minus top-2 probability margin across the last dimension.

    logits: [..., K]
    returns: scalar tensor
    """
    probs = torch.softmax(logits, dim=-1)
    k = probs.size(-1)
    if k < 2:
        return probs.new_zeros(())
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    return margin.mean()


def score_spread_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Mean standard deviation of probabilities across the last dimension.

    logits: [..., K]
    returns: scalar tensor
    """
    probs = torch.softmax(logits, dim=-1)
    if probs.size(-1) < 2:
        return probs.new_zeros(())
    return probs.std(dim=-1).mean()
