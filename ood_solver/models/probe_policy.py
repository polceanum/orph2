from __future__ import annotations

import torch
import torch.nn as nn


class ProbePolicy(nn.Module):
    """
    Scores candidate probes after projecting them into model space.

    Expected shapes:
      context_summary:      [B, D]
      approach_slots:       [B, K, D]
      approach_scores:      [B, K]
      rule_slots:           [B, M, D]
      rule_scores:          [B, M]
      candidate_probe_embs: [B, P, probe_dim]

    Returns:
      probe_logits:         [B, P]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.probe_proj = nn.LazyLinear(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        context_summary: torch.Tensor,
        approach_slots: torch.Tensor,
        approach_scores: torch.Tensor,
        rule_slots: torch.Tensor,
        rule_scores: torch.Tensor,
        candidate_probe_embs: torch.Tensor,
    ) -> torch.Tensor:
        a_w = torch.softmax(approach_scores, dim=-1).unsqueeze(-1)
        r_w = torch.softmax(rule_scores, dim=-1).unsqueeze(-1)

        a_sum = (approach_slots * a_w).sum(dim=1)
        r_sum = (rule_slots * r_w).sum(dim=1)

        probe_embs = self.probe_proj(candidate_probe_embs.float())

        b, p, d = probe_embs.shape
        ctx = context_summary.unsqueeze(1).expand(b, p, d)
        a = a_sum.unsqueeze(1).expand(b, p, d)
        r = r_sum.unsqueeze(1).expand(b, p, d)

        x = torch.cat([ctx, a, r, probe_embs], dim=-1)
        logits = self.mlp(x).squeeze(-1)
        return logits