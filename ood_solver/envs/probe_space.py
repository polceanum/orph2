from __future__ import annotations
import torch

def sample_probe_sequences(batch_size:int,num_candidate_probes:int,seq_len:int,vocab_size:int,generator:torch.Generator)->torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, num_candidate_probes, seq_len), generator=generator)
