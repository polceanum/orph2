from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int = 1, keepdim: bool = False) -> torch.Tensor:
    """
    Compute a mean over `dim`, optionally using a boolean or 0/1 mask.

    x:
        Tensor of shape [..., T, ...]
    mask:
        Tensor broadcastable to x over the reduction dimension.
        Common shapes:
        - [B, T]
        - [B, T, 1]
    """
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)

    mask = mask.to(dtype=x.dtype, device=x.device)

    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)

    weighted = x * mask
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1e-8)
    return weighted.sum(dim=dim, keepdim=keepdim) / denom


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Select one row per batch element.

    values: [B, P, D]
    indices: [B]
    returns: [B, D]
    """
    if values.dim() != 3:
        raise ValueError(f"Expected values to have shape [B, P, D], got {tuple(values.shape)}")
    if indices.dim() != 1:
        raise ValueError(f"Expected indices to have shape [B], got {tuple(indices.shape)}")

    b = values.size(0)
    batch_idx = torch.arange(b, device=values.device)
    return values[batch_idx, indices]


def batched_index_select_2d(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Select multiple rows per batch element.

    values: [B, N, D]
    indices: [B, K]
    returns: [B, K, D]
    """
    if values.dim() != 3:
        raise ValueError(f"Expected values to have shape [B, N, D], got {tuple(values.shape)}")
    if indices.dim() != 2:
        raise ValueError(f"Expected indices to have shape [B, K], got {tuple(indices.shape)}")

    b = values.size(0)
    batch_idx = torch.arange(b, device=values.device).unsqueeze(-1)
    return values[batch_idx, indices]


def move_episode_to_device(batch: Any, device: str | torch.device):
    """
    Recursively move tensors in a dataclass-like episode batch to device.
    """
    if is_dataclass(batch):
        fields = {}
        for k, v in asdict(batch).items():
            fields[k] = _move_value_to_device(v, device)
        return type(batch)(**fields)

    raise TypeError("move_episode_to_device expects a dataclass instance.")


def _move_value_to_device(value: Any, device: str | torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_value_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _move_value_to_device(v, device) for k, v in value.items()}
    return value