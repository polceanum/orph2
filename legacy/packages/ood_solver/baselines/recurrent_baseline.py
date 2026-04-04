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

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        hidden_dim: int,
        seq_len: int,
        num_probe_steps: int,
        use_probe_policy: bool = False,
        sample_probes_during_training: bool = False,
        probe_temperature: float = 1.0,
    ):
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
        self.use_probe_policy = bool(use_probe_policy)
        self.sample_probes_during_training = bool(sample_probes_during_training)
        self.probe_temperature = float(max(1e-4, probe_temperature))
        if self.use_probe_policy:
            self.candidate_probe_proj = nn.LazyLinear(hidden_dim)
            self.probe_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, batch, probe_executor):
        x = self.token_emb(batch.initial_tokens)  # [B, T, D]
        _, h = self.gru(x)
        h = h[-1]  # [B, H]

        step_logs = []
        for step, candidate_probe_embs in enumerate(batch.candidate_probe_embs):
            if self.use_probe_policy:
                cand = self.candidate_probe_proj(candidate_probe_embs.float())
                b, p, _ = cand.shape
                h_exp = h.unsqueeze(1).expand(-1, p, -1)
                probe_logits = self.probe_head(torch.cat([h_exp, cand], dim=-1)).squeeze(-1)
                if self.training and self.sample_probes_during_training:
                    scaled = probe_logits / self.probe_temperature
                    chosen_idx = torch.distributions.Categorical(logits=scaled).sample()
                else:
                    chosen_idx = probe_logits.argmax(dim=-1)
                log_probs = torch.log_softmax(probe_logits, dim=-1)
                chosen_logprob = log_probs.gather(1, chosen_idx.unsqueeze(-1)).squeeze(-1)
                probs = log_probs.exp()
                probe_entropy = -(probs * log_probs).sum(dim=-1)
            else:
                probe_logits = None
                chosen_logprob = None
                probe_entropy = None
                chosen_idx = torch.zeros(
                    candidate_probe_embs.size(0),
                    dtype=torch.long,
                    device=candidate_probe_embs.device,
                )
            obs_emb = probe_executor(batch, step, chosen_idx)
            h = h + self.obs_proj(obs_emb.float())
            step_logs.append(
                {
                    "probe_logits": probe_logits,
                    "chosen_idx": chosen_idx,
                    "chosen_logprob": chosen_logprob,
                    "probe_entropy": probe_entropy,
                }
            )

        q = self.query_emb(batch.final_query)  # [B, L, D]
        h_exp = h.unsqueeze(1).expand(-1, q.size(1), -1)
        fused = self.fuse(torch.cat([h_exp, q], dim=-1))
        logits = self.out(fused)
        aux = {"hidden": h, "step_logs": step_logs}
        return logits, aux
