from __future__ import annotations

import torch

from ood_solver.models.hypothesis_solver import HypothesisSolver
from ood_solver.types import EpisodeBatch


class RandomProbeSolver(HypothesisSolver):
    """
    Structured solver ablation that keeps the same architecture but replaces
    learned probe choice with uniformly random probe selection.
    """

    def select_probe_index(self, probe_logits: torch.Tensor, step: int, episode: EpisodeBatch) -> torch.Tensor:
        batch_size, num_candidates = probe_logits.shape
        return torch.randint(
            low=0,
            high=num_candidates,
            size=(batch_size,),
            device=probe_logits.device,
        )
