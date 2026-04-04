from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class EpisodeBatch:
    initial_tokens: torch.Tensor
    initial_mask: Optional[torch.Tensor]
    candidate_probe_tokens: list[torch.Tensor]
    candidate_probe_embs: list[torch.Tensor]
    final_query: torch.Tensor
    final_target: torch.Tensor
    mechanism_id: torch.Tensor
    mechanism_params: Optional[torch.Tensor]
    diagnostic_probe_targets: Optional[list[torch.Tensor]] = None

@dataclass
class BeliefState:
    approach_slots: torch.Tensor
    approach_scores: torch.Tensor
    rule_slots: torch.Tensor
    rule_scores: torch.Tensor
    archive_slots: torch.Tensor
    archive_scores: torch.Tensor
    surprise: torch.Tensor
    stagnation: torch.Tensor
    demo_logits: Optional[torch.Tensor] = None
    demo_targets: Optional[torch.Tensor] = None
