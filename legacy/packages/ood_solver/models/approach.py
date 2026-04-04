from __future__ import annotations
import torch
import torch.nn as nn

class ApproachProposer(nn.Module):
    def __init__(self, d_model:int, num_slots:int):
        super().__init__()
        self.num_slots=num_slots
        self.slot_queries=nn.Parameter(torch.randn(num_slots, d_model))
        self.attn=nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.slot_mlp=nn.Sequential(nn.Linear(d_model,d_model), nn.GELU(), nn.Linear(d_model,d_model))
        self.score_head=nn.Linear(d_model,1)
    def forward(self, context_tokens:torch.Tensor, context_summary:torch.Tensor):
        b,_,d=context_tokens.shape
        q=self.slot_queries.unsqueeze(0).expand(b,self.num_slots,d)
        slots,_=self.attn(q, context_tokens, context_tokens)
        slots=slots+self.slot_mlp(slots)
        scores=self.score_head(slots).squeeze(-1)
        return slots, scores
