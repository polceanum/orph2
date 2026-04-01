from __future__ import annotations
import torch
import torch.nn as nn
from ood_solver.types import BeliefState, EpisodeBatch
from ood_solver.utils import batched_index_select

class HypothesisSolver(nn.Module):
    def __init__(self, encoder:nn.Module, approach_proposer:nn.Module, rule_proposer:nn.Module, probe_policy:nn.Module, belief_updater:nn.Module, solver_head:nn.Module)->None:
        super().__init__()
        self.encoder=encoder
        self.approach_proposer=approach_proposer
        self.rule_proposer=rule_proposer
        self.probe_policy=probe_policy
        self.belief_updater=belief_updater
        self.solver_head=solver_head
    def initialize_beliefs(self, context_tokens:torch.Tensor, context_summary:torch.Tensor)->BeliefState:
        approach_slots, approach_scores=self.approach_proposer(context_tokens, context_summary)
        rule_slots, rule_scores,_=self.rule_proposer(context_tokens, context_summary, approach_slots, approach_scores)
        b,_,d=approach_slots.shape
        return BeliefState(approach_slots, approach_scores, rule_slots, rule_scores, approach_slots.new_zeros((b,0,d)), approach_scores.new_zeros((b,0)), approach_scores.new_zeros((b,1)), approach_scores.new_zeros((b,1)))
    def forward(self, episode:EpisodeBatch, probe_executor):
        context_tokens, context_summary=self.encoder(episode.initial_tokens, episode.initial_mask)
        belief=self.initialize_beliefs(context_tokens, context_summary)
        logs=[]
        for step, candidate_probe_embs in enumerate(episode.candidate_probe_embs):
            probe_logits=self.probe_policy(context_summary, belief.approach_slots, belief.approach_scores, belief.rule_slots, belief.rule_scores, candidate_probe_embs)
            chosen_idx=probe_logits.argmax(dim=-1)
            chosen_probe_tokens=batched_index_select(episode.candidate_probe_tokens[step], chosen_idx)
            observation_tokens=probe_executor(episode, step, chosen_idx)
            belief=self.belief_updater(context_summary, belief.approach_slots, belief.approach_scores, belief.rule_slots, belief.rule_scores, chosen_probe_tokens, observation_tokens, (belief.archive_slots, belief.archive_scores))
            logs.append({'probe_logits':probe_logits,'chosen_idx':chosen_idx,'approach_scores':belief.approach_scores,'rule_scores':belief.rule_scores,'surprise':belief.surprise,'stagnation':belief.stagnation})
        logits=self.solver_head(episode.final_query, context_summary, belief.approach_slots, belief.approach_scores, belief.rule_slots, belief.rule_scores)
        return logits, belief, logs
