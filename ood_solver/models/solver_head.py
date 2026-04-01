from __future__ import annotations
import torch
import torch.nn as nn

class SolverHead(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.query_emb=nn.Embedding(vocab_size, d_model)
        self.fuse=nn.Sequential(nn.Linear(4*d_model,2*d_model), nn.GELU(), nn.Linear(2*d_model,d_model))
        self.out=nn.Linear(d_model, vocab_size)
    def forward(self, final_query, context_summary, approach_slots, approach_scores, rule_slots, rule_scores):
        q=self.query_emb(final_query)
        a_w=torch.softmax(approach_scores, dim=-1).unsqueeze(-1)
        r_w=torch.softmax(rule_scores, dim=-1).unsqueeze(-1)
        a_sum=(approach_slots*a_w).sum(dim=1)
        r_sum=(rule_slots*r_w).sum(dim=1)
        b,l,d=q.shape
        ctx=context_summary.unsqueeze(1).expand(b,l,d)
        a=a_sum.unsqueeze(1).expand(b,l,d)
        r=r_sum.unsqueeze(1).expand(b,l,d)
        h=self.fuse(torch.cat([q,ctx,a,r], dim=-1))
        return self.out(h)
