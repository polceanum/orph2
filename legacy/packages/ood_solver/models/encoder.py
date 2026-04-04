from __future__ import annotations

import torch
import torch.nn as nn

from ood_solver.utils import masked_mean


class SimpleSequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        max_len: int,
        use_value_features: bool = False,
        value_features_only: bool = False,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.use_value_features = bool(use_value_features)
        self.value_features_only = bool(value_features_only)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.type_emb = nn.Embedding(4, d_model)
        self.value_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ):
        """
        token_ids: [B, T]
        mask: [B, T] with 1/True for valid tokens, 0/False for padding
        """
        b, t = token_ids.shape
        device = token_ids.device

        pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        if self.use_value_features:
            denom = float(max(self.vocab_size - 1, 1))
            x_norm = token_ids.float() / denom
            x_value = torch.stack(
                [
                    x_norm,
                    torch.sin(2.0 * torch.pi * x_norm),
                    torch.cos(2.0 * torch.pi * x_norm),
                ],
                dim=-1,
            )
            x = self.value_proj(x_value)
            if not self.value_features_only:
                x = x + self.token_emb(token_ids)
            x = x + self.pos_emb(pos)
        else:
            x = self.token_emb(token_ids) + self.pos_emb(pos)
        if token_type_ids is not None:
            x = x + self.type_emb(token_type_ids.long().clamp(min=0, max=3))

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.bool()

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        summary = masked_mean(x, mask, dim=1)
        return x, summary
