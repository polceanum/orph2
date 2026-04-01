from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentBaseline(nn.Module):
    """
    Simple baseline with a single recurrent latent state and fixed probe choice.
    Matches the eval/train interface used by the structured model.

    Notes:
    - Uses a lazy projection for probe observations because the environment may
      emit probe/observation embeddings with widths different from d_model.
    - Casts observation tensors to float before projection.
    """

    def __init__(self, vocab_size: int, d_model: int, hidden_dim: int, seq_len: int, num_probe_steps: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=hidden_dim, batch_first=True)

        self.obs_proj = nn.LazyLinear(hidden_dim)
        self.query_emb = nn.Embedding(vocab_size, d_model)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, batch, probe_executor):
        x = self.token_emb(batch.initial_tokens)  # [B, T, D]
        _, h = self.gru(x)
        h = h[-1]  # [B, H]

        # Simple fixed probe policy: always pick candidate 0.
        for step, candidate_probe_embs in enumerate(batch.candidate_probe_embs):
            chosen_idx = torch.zeros(
                candidate_probe_embs.size(0),
                dtype=torch.long,
                device=candidate_probe_embs.device,
            )
            obs_emb = probe_executor(batch, step, chosen_idx)
            h = h + self.obs_proj(obs_emb.float())

        q = self.query_emb(batch.final_query)  # [B, L, D]
        h_exp = h.unsqueeze(1).expand(-1, q.size(1), -1)
        fused = self.fuse(torch.cat([h_exp, q], dim=-1))
        logits = self.out(fused)
        aux = {"hidden": h}
        return logits, aux