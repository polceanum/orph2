from __future__ import annotations

import torch
import torch.nn as nn

from ood_solver.types import BeliefState
from ood_solver.utils import batched_index_select_2d


class BeliefUpdater(nn.Module):
    def __init__(self, d_model: int, archive_size: int = 16):
        super().__init__()
        self.archive_size = archive_size
        self.d_model = d_model

        self.probe_proj = nn.LazyLinear(d_model)
        self.obs_proj = nn.LazyLinear(d_model)

        self.approach_delta = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 1),
        )
        self.rule_delta = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 1),
        )

        self.approach_update = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.rule_update = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.surprise_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.stagnation_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
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
        chosen_probe_emb: torch.Tensor,
        observation_emb: torch.Tensor,
        archive_state: tuple[torch.Tensor, torch.Tensor],
    ) -> BeliefState:
        archive_slots, archive_scores = archive_state

        chosen_probe_emb = self.probe_proj(chosen_probe_emb.float())
        observation_emb = self.obs_proj(observation_emb.float())

        b, k, d = approach_slots.shape
        _, m, _ = rule_slots.shape

        probe_a = chosen_probe_emb.unsqueeze(1).expand(b, k, d)
        obs_a = observation_emb.unsqueeze(1).expand(b, k, d)
        ctx_a = context_summary.unsqueeze(1).expand(b, k, d)

        a_in = torch.cat([approach_slots, probe_a, obs_a, ctx_a], dim=-1)
        a_delta = self.approach_delta(a_in).squeeze(-1)
        a_upd = self.approach_update(a_in)

        new_approach_slots = approach_slots + a_upd
        new_approach_scores = approach_scores + a_delta

        probe_r = chosen_probe_emb.unsqueeze(1).expand(b, m, d)
        obs_r = observation_emb.unsqueeze(1).expand(b, m, d)
        ctx_r = context_summary.unsqueeze(1).expand(b, m, d)

        r_in = torch.cat([rule_slots, probe_r, obs_r, ctx_r], dim=-1)
        r_delta = self.rule_delta(r_in).squeeze(-1)
        r_upd = self.rule_update(r_in)

        new_rule_slots = rule_slots + r_upd
        new_rule_scores = rule_scores + r_delta

        pooled_a = (
            new_approach_slots
            * torch.softmax(new_approach_scores, dim=-1).unsqueeze(-1)
        ).sum(dim=1)
        pooled_r = (
            new_rule_slots
            * torch.softmax(new_rule_scores, dim=-1).unsqueeze(-1)
        ).sum(dim=1)

        surprise = self.surprise_head(
            torch.cat([context_summary, chosen_probe_emb, observation_emb], dim=-1)
        )
        stagnation = self.stagnation_head(
            torch.cat([context_summary, pooled_a, pooled_r], dim=-1)
        )

        if archive_slots.numel() == 0:
            combined_slots = new_approach_slots
            combined_scores = new_approach_scores
        else:
            combined_slots = torch.cat([archive_slots, new_approach_slots], dim=1)
            combined_scores = torch.cat([archive_scores, new_approach_scores], dim=1)

        keep = min(self.archive_size, combined_scores.size(1))
        top_vals, top_idx = torch.topk(combined_scores, k=keep, dim=1)
        top_slots = batched_index_select_2d(combined_slots, top_idx)

        return BeliefState(
            approach_slots=new_approach_slots,
            approach_scores=new_approach_scores,
            rule_slots=new_rule_slots,
            rule_scores=new_rule_scores,
            archive_slots=top_slots,
            archive_scores=top_vals,
            surprise=surprise,
            stagnation=stagnation,
        )