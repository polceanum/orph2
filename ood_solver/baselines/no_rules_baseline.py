from __future__ import annotations

import torch

from ood_solver.models.hypothesis_solver import HypothesisSolver
from ood_solver.types import BeliefState, EpisodeBatch
from ood_solver.utils import batched_index_select


class NoRulesSolver(HypothesisSolver):
    """
    Structured solver ablation that preserves approach reasoning and probe
    updates while removing rule information from the active belief state.
    """

    def _empty_rule_state(self, approach_slots: torch.Tensor, approach_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, d_model = approach_slots.shape
        rule_slots = approach_slots.new_zeros((batch_size, 1, d_model))
        rule_scores = approach_scores.new_zeros((batch_size, 1))
        return rule_slots, rule_scores

    def initialize_beliefs(self, context_tokens: torch.Tensor, context_summary: torch.Tensor) -> BeliefState:
        approach_slots, approach_scores = self.approach_proposer(context_tokens, context_summary)
        rule_slots, rule_scores = self._empty_rule_state(approach_slots, approach_scores)
        batch_size, _, d_model = approach_slots.shape
        return BeliefState(
            approach_slots=approach_slots,
            approach_scores=approach_scores,
            rule_slots=rule_slots,
            rule_scores=rule_scores,
            archive_slots=approach_slots.new_zeros((batch_size, 0, d_model)),
            archive_scores=approach_scores.new_zeros((batch_size, 0)),
            surprise=approach_scores.new_zeros((batch_size, 1)),
            stagnation=approach_scores.new_zeros((batch_size, 1)),
        )

    def forward(self, episode: EpisodeBatch, probe_executor):
        context_tokens, context_summary = self.encoder(episode.initial_tokens, episode.initial_mask)
        belief = self.initialize_beliefs(context_tokens, context_summary)
        logs = []

        for step, candidate_probe_embs in enumerate(episode.candidate_probe_embs):
            probe_logits = self.probe_policy(
                context_summary,
                belief.approach_slots,
                belief.approach_scores,
                belief.rule_slots,
                belief.rule_scores,
                candidate_probe_embs,
            )
            chosen_idx = self.select_probe_index(probe_logits, step, episode)
            chosen_probe_tokens = batched_index_select(episode.candidate_probe_tokens[step], chosen_idx)
            observation_tokens = probe_executor(episode, step, chosen_idx)

            updated = self.belief_updater(
                context_summary,
                belief.approach_slots,
                belief.approach_scores,
                belief.rule_slots,
                belief.rule_scores,
                chosen_probe_tokens,
                observation_tokens,
                (belief.archive_slots, belief.archive_scores),
            )
            rule_slots, rule_scores = self._empty_rule_state(updated.approach_slots, updated.approach_scores)
            belief = BeliefState(
                approach_slots=updated.approach_slots,
                approach_scores=updated.approach_scores,
                rule_slots=rule_slots,
                rule_scores=rule_scores,
                archive_slots=updated.archive_slots,
                archive_scores=updated.archive_scores,
                surprise=updated.surprise,
                stagnation=updated.stagnation,
            )
            logs.append(
                {
                    "probe_logits": probe_logits,
                    "chosen_idx": chosen_idx,
                    "approach_scores": belief.approach_scores,
                    "rule_scores": belief.rule_scores,
                    "surprise": belief.surprise,
                    "stagnation": belief.stagnation,
                }
            )

        logits = self.solver_head(
            episode.final_query,
            context_summary,
            belief.approach_slots,
            belief.approach_scores,
            belief.rule_slots,
            belief.rule_scores,
        )
        return logits, belief, logs
