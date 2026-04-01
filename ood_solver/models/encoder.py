from __future__ import annotations

import torch
import torch.nn as nn

from ood_solver.utils import masked_mean


class SimpleSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, max_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None):
        """
        token_ids: [B, T]
        mask: [B, T] with 1/True for valid tokens, 0/False for padding
        """
        b, t = token_ids.shape
        device = token_ids.device

        pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        x = self.token_emb(token_ids) + self.pos_emb(pos)

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.bool()

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        summary = masked_mean(x, mask, dim=1)
        return x, summary