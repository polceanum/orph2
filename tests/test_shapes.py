import torch
from ood_solver.models.encoder import SimpleSequenceEncoder
from ood_solver.models.approach import ApproachProposer
from ood_solver.models.rules import RuleProposer

def test_model_shapes():
    enc=SimpleSequenceEncoder(vocab_size=16, d_model=32, nhead=4, num_layers=1, max_len=64)
    x=torch.randint(0,16,(2,12))
    tok,summ=enc(x)
    assert tok.shape==(2,12,32)
    assert summ.shape==(2,32)
    app=ApproachProposer(d_model=32, num_slots=4)
    a,s=app(tok,summ)
    assert a.shape==(2,4,32)
    assert s.shape==(2,4)
    rules=RuleProposer(d_model=32, num_rules=8)
    r,rs,links=rules(tok,summ,a,s)
    assert r.shape==(2,8,32)
    assert rs.shape==(2,8)
    assert links.shape==(2,8,4)
