from __future__ import annotations
import torch
import torch.nn as nn

class RuleProposer(nn.Module):
    def __init__(self, d_model:int, num_rules:int):
        super().__init__()
        self.num_rules=num_rules
        self.rule_queries=nn.Parameter(torch.randn(num_rules, d_model))
        self.attn_ctx=nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn_app=nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.rule_mlp=nn.Sequential(nn.Linear(d_model,d_model), nn.GELU(), nn.Linear(d_model,d_model))
        self.rule_score=nn.Linear(d_model,1)
        self.rule_to_approach=nn.Linear(d_model,d_model)
    def forward(self, context_tokens, context_summary, approach_slots, approach_scores):
        b,_,d=context_tokens.shape
        q=self.rule_queries.unsqueeze(0).expand(b,self.num_rules,d)
        r_ctx,_=self.attn_ctx(q, context_tokens, context_tokens)
        r_app,_=self.attn_app(r_ctx, approach_slots, approach_slots)
        rules=r_ctx+r_app
        rules=rules+self.rule_mlp(rules)
        rule_scores=self.rule_score(rules).squeeze(-1)
        link_logits=torch.einsum('bmd,bkd->bmk', self.rule_to_approach(rules), approach_slots)
        return rules, rule_scores, link_logits
