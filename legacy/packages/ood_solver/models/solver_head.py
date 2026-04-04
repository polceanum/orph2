from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SolverHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 256,
        dropout: float = 0.0,
        use_local_adapter: bool = False,
        use_value_features: bool = False,
        value_features_only: bool = False,
        use_modulo_shift_adapter: bool = False,
        use_demo_shift_prior: bool = False,
        use_mechanism_router: bool = False,
        num_mechanism_experts: int = 1,
        mechanism_router_temperature: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.use_local_adapter = bool(use_local_adapter)
        self.use_value_features = bool(use_value_features)
        self.value_features_only = bool(value_features_only)
        self.use_modulo_shift_adapter = bool(use_modulo_shift_adapter)
        self.use_demo_shift_prior = bool(use_demo_shift_prior)
        self.use_mechanism_router = bool(use_mechanism_router) and int(num_mechanism_experts) > 1
        self.num_mechanism_experts = max(1, int(num_mechanism_experts))
        self.mechanism_router_temperature = float(max(1e-4, mechanism_router_temperature))
        self.query_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.value_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.fuse = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=dropout,
        )
        self.query_mixer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ctx_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.local_adapter = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        if self.use_mechanism_router:
            self.router = nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.num_mechanism_experts),
            )
            self.expert_out = nn.ModuleList(
                [nn.Linear(d_model, vocab_size) for _ in range(self.num_mechanism_experts)]
            )
        else:
            self.router = None
            self.expert_out = None
        self.shift_out = nn.Linear(d_model, vocab_size)
        self.shift_logit_scale = nn.Parameter(torch.tensor(0.0))
        self.demo_shift_logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        final_query,
        context_summary,
        approach_slots,
        approach_scores,
        rule_slots,
        rule_scores,
        context_tokens: torch.Tensor | None = None,
        demo_shift_logprobs: torch.Tensor | None = None,
    ):
        if self.use_value_features:
            denom = float(max(self.vocab_size - 1, 1))
            q_norm = final_query.float() / denom
            q_feat = torch.stack(
                [
                    q_norm,
                    torch.sin(2.0 * torch.pi * q_norm),
                    torch.cos(2.0 * torch.pi * q_norm),
                ],
                dim=-1,
            )
            q = self.value_proj(q_feat)
            if not self.value_features_only:
                q = q + self.query_emb(final_query)
        else:
            q = self.query_emb(final_query)
        b, l, d = q.shape
        if l > self.pos_emb.num_embeddings:
            raise ValueError(
                f"SolverHead query length {l} exceeds max_len={self.pos_emb.num_embeddings}. "
                "Increase model.max_len in config."
            )

        pos = torch.arange(l, device=final_query.device).unsqueeze(0).expand(b, l)
        q = q + self.pos_emb(pos)

        a_w = torch.softmax(approach_scores, dim=-1).unsqueeze(-1)
        r_w = torch.softmax(rule_scores, dim=-1).unsqueeze(-1)
        a_sum = (approach_slots * a_w).sum(dim=1)
        r_sum = (rule_slots * r_w).sum(dim=1)

        cond = self.fuse(torch.cat([context_summary, a_sum, r_sum], dim=-1))
        h = q + cond.unsqueeze(1)
        if self.use_local_adapter:
            left = torch.cat([q.new_zeros(b, 1, d), q[:, :-1, :]], dim=1)
            local_h = self.local_adapter(
                torch.cat([q, left, cond.unsqueeze(1).expand(b, l, d)], dim=-1)
            )
            h = h + local_h
        h = self.query_mixer(h)
        if context_tokens is not None:
            ctx_out, _ = self.ctx_attn(h, context_tokens, context_tokens, need_weights=False)
            h = h + ctx_out
        h = self.norm(h)
        base_logits = self.out(h)
        if self.use_mechanism_router and self.router is not None and self.expert_out is not None:
            router_in = torch.cat([context_summary, a_sum, r_sum], dim=-1)
            route_logits = self.router(router_in) / self.mechanism_router_temperature
            route = torch.softmax(route_logits, dim=-1)  # [B, E]
            expert_logits = torch.stack([head(h) for head in self.expert_out], dim=1)  # [B, E, L, V]
            base_logits = (route.unsqueeze(-1).unsqueeze(-1) * expert_logits).sum(dim=1)

        out_logits = base_logits

        if self.use_demo_shift_prior and demo_shift_logprobs is not None:
            shift_prior = torch.softmax(demo_shift_logprobs, dim=-1)  # [B, S]
            query_oh = F.one_hot(final_query.long(), num_classes=self.vocab_size).to(base_logits.dtype)
            rolled = []
            for s in range(self.vocab_size):
                rolled.append(torch.roll(query_oh, shifts=s, dims=-1))
            rolled_query = torch.stack(rolled, dim=2)  # [B, L, S, V]
            shifted_probs = (shift_prior.unsqueeze(1).unsqueeze(-1) * rolled_query).sum(dim=2)
            shifted_logits = torch.log(shifted_probs.clamp_min(1e-8))
            out_logits = out_logits + F.softplus(self.demo_shift_logit_scale) * shifted_logits

        if not self.use_modulo_shift_adapter:
            return out_logits

        # Optional arithmetic adapter for shift-like mechanisms.
        # Predict a distribution over modular shifts and apply it to the query token ids.
        shift_logits = self.shift_out(h)
        shift_probs = torch.softmax(shift_logits, dim=-1)
        query_oh = F.one_hot(final_query.long(), num_classes=self.vocab_size).to(base_logits.dtype)
        rolled = []
        for s in range(self.vocab_size):
            rolled.append(torch.roll(query_oh, shifts=s, dims=-1))
        rolled_query = torch.stack(rolled, dim=2)  # [B, L, S, V]
        shifted_probs = (shift_probs.unsqueeze(-1) * rolled_query).sum(dim=2)
        shifted_logits = torch.log(shifted_probs.clamp_min(1e-8))
        mix = F.softplus(self.shift_logit_scale)
        return out_logits + mix * shifted_logits
