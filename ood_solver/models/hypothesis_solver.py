from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ood_solver.types import BeliefState, EpisodeBatch
from ood_solver.utils import batched_index_select

class HypothesisSolver(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        approach_proposer: nn.Module,
        rule_proposer: nn.Module,
        probe_policy: nn.Module,
        belief_updater: nn.Module,
        solver_head: nn.Module,
        soft_probe_training: bool = False,
        soft_probe_temp: float = 1.0,
        sample_probes_during_training: bool = False,
        sample_probe_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder=encoder
        self.approach_proposer=approach_proposer
        self.rule_proposer=rule_proposer
        self.probe_policy=probe_policy
        self.belief_updater=belief_updater
        self.solver_head=solver_head
        self.soft_probe_training = bool(soft_probe_training)
        self.soft_probe_temp = float(max(1e-4, soft_probe_temp))
        self.sample_probes_during_training = bool(sample_probes_during_training)
        self.sample_probe_temp = float(max(1e-4, sample_probe_temp))
        self.probe_feature_proj = nn.LazyLinear(self.belief_updater.d_model)
        self.approach_probe_predictor = nn.Sequential(
            nn.Linear(2 * self.belief_updater.d_model, self.belief_updater.d_model),
            nn.GELU(),
            nn.Linear(self.belief_updater.d_model, self.belief_updater.d_model),
        )
        self.rule_probe_predictor = nn.Sequential(
            nn.Linear(3 * self.belief_updater.d_model, self.belief_updater.d_model),
            nn.GELU(),
            nn.Linear(self.belief_updater.d_model, self.belief_updater.d_model),
        )

    def infer_demo_shift_logprobs(self, episode: EpisodeBatch) -> torch.Tensor | None:
        """
        Estimate a shift prior p(s) from demo input/output token pairs.
        For each token pair, s = (y - x) mod V. Aggregate over demos and positions.
        """
        bsz, total = episode.initial_tokens.shape
        q_len = episode.final_query.size(1)
        demo_block = 2 * q_len
        if q_len <= 0 or demo_block <= 0 or total % demo_block != 0:
            return None
        if not hasattr(self.solver_head, "vocab_size"):
            return None
        vocab = int(self.solver_head.vocab_size)
        if vocab <= 1:
            return None

        num_demos = total // demo_block
        demos = episode.initial_tokens.view(bsz, num_demos, demo_block)
        demo_in = demos[:, :, :q_len]
        demo_out = demos[:, :, q_len:]
        shifts = (demo_out.long() - demo_in.long()) % vocab  # [B, D, L]
        shift_one_hot = torch.nn.functional.one_hot(shifts, num_classes=vocab).float()
        counts = shift_one_hot.sum(dim=(1, 2))  # [B, V]
        probs = (counts + 1e-3) / (counts.sum(dim=-1, keepdim=True) + 1e-3 * vocab)
        return torch.log(probs)
    def initialize_beliefs(self, context_tokens:torch.Tensor, context_summary:torch.Tensor)->BeliefState:
        approach_slots, approach_scores=self.approach_proposer(context_tokens, context_summary)
        rule_slots, rule_scores,_=self.rule_proposer(context_tokens, context_summary, approach_slots, approach_scores)
        b,_,d=approach_slots.shape
        return BeliefState(approach_slots, approach_scores, rule_slots, rule_scores, approach_slots.new_zeros((b,0,d)), approach_scores.new_zeros((b,0)), approach_scores.new_zeros((b,1)), approach_scores.new_zeros((b,1)))

    def select_probe_index(self, probe_logits: torch.Tensor, step: int, episode: EpisodeBatch) -> torch.Tensor:
        if self.training and self.sample_probes_during_training:
            scaled = probe_logits / self.sample_probe_temp
            return torch.distributions.Categorical(logits=scaled).sample()
        return probe_logits.argmax(dim=-1)

    @staticmethod
    def build_demo_token_types(episode: EpisodeBatch) -> torch.Tensor | None:
        bsz, total = episode.initial_tokens.shape
        q_len = episode.final_query.size(1)
        demo_block = 2 * q_len
        if q_len <= 0 or demo_block <= 0 or total % demo_block != 0:
            return None
        num_demos = total // demo_block
        types = episode.initial_tokens.new_zeros((bsz, total))
        block_types = torch.cat(
            [
                types.new_zeros((q_len,)),
                types.new_ones((q_len,)),
            ],
            dim=0,
        ).unsqueeze(0).expand(num_demos, demo_block).reshape(-1)
        types[:] = block_types.unsqueeze(0)
        return types

    def forward(self, episode:EpisodeBatch, probe_executor):
        token_types = self.build_demo_token_types(episode)
        context_tokens, context_summary=self.encoder(
            episode.initial_tokens,
            episode.initial_mask,
            token_type_ids=token_types,
        )
        belief=self.initialize_beliefs(context_tokens, context_summary)
        logs=[]
        for step, candidate_probe_embs in enumerate(episode.candidate_probe_embs):
            probe_features = self.probe_feature_proj(candidate_probe_embs.float())
            probe_logits=self.probe_policy(context_summary, belief.approach_slots, belief.approach_scores, belief.rule_slots, belief.rule_scores, candidate_probe_embs)

            b, k, d = belief.approach_slots.shape
            p = probe_features.size(1)
            approach_expanded = belief.approach_slots.unsqueeze(2).expand(b, k, p, d)
            probe_expanded = probe_features.unsqueeze(1).expand(b, k, p, d)
            approach_probe_pred = self.approach_probe_predictor(torch.cat([approach_expanded, probe_expanded], dim=-1))
            approach_disagreement = approach_probe_pred.var(dim=1, unbiased=False).mean(dim=-1)

            chosen_idx=self.select_probe_index(probe_logits, step, episode)
            probe_log_probs = F.log_softmax(probe_logits, dim=-1)
            chosen_logprob = probe_log_probs.gather(1, chosen_idx.unsqueeze(-1)).squeeze(-1)
            probe_probs = probe_log_probs.exp()
            probe_entropy = -(probe_probs * probe_log_probs).sum(dim=-1)
            if self.training and self.soft_probe_training:
                probe_weights = torch.softmax(probe_logits / self.soft_probe_temp, dim=-1)
                probe_weights_u = probe_weights.unsqueeze(-1)
                chosen_probe_tokens = (episode.candidate_probe_tokens[step].float() * probe_weights_u).sum(dim=1)
                chosen_probe_features = (probe_features * probe_weights_u).sum(dim=1)
            else:
                chosen_probe_tokens = batched_index_select(episode.candidate_probe_tokens[step], chosen_idx)
                chosen_probe_features = batched_index_select(probe_features, chosen_idx)
            rule_weights = torch.softmax(belief.rule_scores, dim=-1).unsqueeze(-1)
            pooled_rule = (belief.rule_slots * rule_weights).sum(dim=1)
            if self.training and self.soft_probe_training:
                all_observation_tokens = probe_executor(episode, step, None)
                observation_tokens = (all_observation_tokens.float() * probe_weights_u).sum(dim=1)
            else:
                observation_tokens=probe_executor(episode, step, chosen_idx)
            rule_obs_pred = self.rule_probe_predictor(
                torch.cat([context_summary, pooled_rule, chosen_probe_features], dim=-1)
            )
            with torch.no_grad():
                rule_obs_target = self.belief_updater.obs_proj(observation_tokens.float())
            belief=self.belief_updater(context_summary, belief.approach_slots, belief.approach_scores, belief.rule_slots, belief.rule_scores, chosen_probe_tokens, observation_tokens, (belief.archive_slots, belief.archive_scores))
            logs.append({
                'probe_logits':probe_logits,
                'chosen_idx':chosen_idx,
                'chosen_logprob':chosen_logprob,
                'probe_entropy':probe_entropy,
                'approach_scores':belief.approach_scores,
                'rule_scores':belief.rule_scores,
                'surprise':belief.surprise,
                'stagnation':belief.stagnation,
                'approach_disagreement':approach_disagreement,
                'rule_obs_pred':rule_obs_pred,
                'rule_obs_target':rule_obs_target,
            })
        demo_shift_logprobs = self.infer_demo_shift_logprobs(episode)

        logits=self.solver_head(
            episode.final_query,
            context_summary,
            belief.approach_slots,
            belief.approach_scores,
            belief.rule_slots,
            belief.rule_scores,
            context_tokens=context_tokens,
            demo_shift_logprobs=demo_shift_logprobs,
        )
        # Auxiliary supervised signal: reconstruct demo outputs from demo inputs.
        # This directly teaches the latent state to encode the task mechanism.
        bsz, total = episode.initial_tokens.shape
        q_len = episode.final_query.size(1)
        demo_block = 2 * q_len
        if demo_block > 0 and total % demo_block == 0:
            num_demos = total // demo_block
            demos = episode.initial_tokens.view(bsz, num_demos, demo_block)
            demo_query = demos[:, :, :q_len].reshape(bsz * num_demos, q_len)
            demo_target = demos[:, :, q_len:].reshape(bsz * num_demos, q_len)

            demo_ctx = context_summary.repeat_interleave(num_demos, dim=0)
            demo_ctx_tokens = context_tokens.repeat_interleave(num_demos, dim=0)
            demo_approach_slots = belief.approach_slots.repeat_interleave(num_demos, dim=0)
            demo_approach_scores = belief.approach_scores.repeat_interleave(num_demos, dim=0)
            demo_rule_slots = belief.rule_slots.repeat_interleave(num_demos, dim=0)
            demo_rule_scores = belief.rule_scores.repeat_interleave(num_demos, dim=0)
            demo_logits = self.solver_head(
                demo_query,
                demo_ctx,
                demo_approach_slots,
                demo_approach_scores,
                demo_rule_slots,
                demo_rule_scores,
                context_tokens=demo_ctx_tokens,
                demo_shift_logprobs=demo_shift_logprobs.repeat_interleave(num_demos, dim=0) if demo_shift_logprobs is not None else None,
            )
            belief.demo_logits = demo_logits
            belief.demo_targets = demo_target
        return logits, belief, logs
