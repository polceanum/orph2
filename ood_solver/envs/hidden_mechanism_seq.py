from __future__ import annotations
import torch
from ood_solver.envs.probe_space import sample_probe_sequences
from ood_solver.types import EpisodeBatch

class HiddenMechanismSequenceEnv:
    LOCAL=0
    RECURSIVE=1
    SWITCH=2
    STOCHASTIC=3

    def __init__(self, vocab_size:int=16, seq_len:int=12, num_probe_steps:int=4, num_candidate_probes:int=8, num_demos:int=3, seed:int=0)->None:
        self.vocab_size=vocab_size
        self.seq_len=seq_len
        self.num_probe_steps=num_probe_steps
        self.num_candidate_probes=num_candidate_probes
        self.num_demos=num_demos
        self.generator=torch.Generator().manual_seed(seed)

    def _sample_mechanism(self, batch_size:int)->torch.Tensor:
        return torch.randint(0, 4, (batch_size,), generator=self.generator)

    def _sample_params(self, batch_size:int)->torch.Tensor:
        return torch.randint(0, self.vocab_size, (batch_size, 3), generator=self.generator)

    def _sample_sequences(self, batch_size:int, num_seq:int)->torch.Tensor:
        return torch.randint(0, self.vocab_size, (batch_size, num_seq, self.seq_len), generator=self.generator)

    def _transform_local(self, x:torch.Tensor, params:torch.Tensor)->torch.Tensor:
        shift=params[:,0].unsqueeze(-1)
        left=torch.roll(x, shifts=1, dims=-1)
        return (x+left+shift)%self.vocab_size

    def _transform_recursive(self, x:torch.Tensor, params:torch.Tensor)->torch.Tensor:
        half=x.size(-1)//2
        left=x[:,:half].sum(dim=-1, keepdim=True)%self.vocab_size
        right=x[:,half:].sum(dim=-1, keepdim=True)%self.vocab_size
        global_ctx=(left+right+params[:,1].unsqueeze(-1))%self.vocab_size
        return (x+global_ctx)%self.vocab_size

    def _transform_switch(self, x:torch.Tensor, params:torch.Tensor)->torch.Tensor:
        mode=(params[:,0]%2).unsqueeze(-1)
        a=(torch.flip(x, dims=[-1])+params[:,1].unsqueeze(-1))%self.vocab_size
        b=(x+torch.arange(x.size(-1), device=x.device).unsqueeze(0)+params[:,2].unsqueeze(-1))%self.vocab_size
        return torch.where(mode.bool(), a, b)

    def _transform_stochastic(self, x:torch.Tensor, params:torch.Tensor)->torch.Tensor:
        noise=(x.sum(dim=-1, keepdim=True)+params[:,0:1]+3*params[:,2:3])%3
        return (x+noise)%self.vocab_size

    def transform(self, mechanism_id:torch.Tensor, x:torch.Tensor, params:torch.Tensor)->torch.Tensor:
        y=torch.zeros_like(x)
        for mech in range(4):
            mask=mechanism_id==mech
            if not mask.any():
                continue
            xx=x[mask]
            pp=params[mask]
            if mech==self.LOCAL:
                yy=self._transform_local(xx, pp)
            elif mech==self.RECURSIVE:
                yy=self._transform_recursive(xx, pp)
            elif mech==self.SWITCH:
                yy=self._transform_switch(xx, pp)
            else:
                yy=self._transform_stochastic(xx, pp)
            y[mask]=yy
        return y

    def _make_initial_tokens(self, demos_in:torch.Tensor, demos_out:torch.Tensor)->torch.Tensor:
        return torch.cat([demos_in, demos_out], dim=-1).reshape(demos_in.size(0), -1)

    def _diagnostic_probe_target(self, candidate_probe_tokens:torch.Tensor, params:torch.Tensor)->torch.Tensor:
        b,p,l=candidate_probe_tokens.shape
        scores=torch.zeros(b,p)
        flat=candidate_probe_tokens.reshape(b*p,l)
        for mech in range(4):
            mech_ids=torch.full((b*p,), mech, dtype=torch.long)
            out=self.transform(mech_ids, flat, params.repeat_interleave(p, dim=0)).view(b,p,l)
            scores += out.float().std(dim=-1)
        return scores.argmax(dim=-1)

    def sample_episode(self, batch_size:int)->EpisodeBatch:
        mechanism_id=self._sample_mechanism(batch_size)
        params=self._sample_params(batch_size)
        demos_in=self._sample_sequences(batch_size, self.num_demos)
        demos_out=self.transform(mechanism_id.repeat_interleave(self.num_demos), demos_in.view(-1, self.seq_len), params.repeat_interleave(self.num_demos, dim=0)).view(batch_size, self.num_demos, self.seq_len)
        initial_tokens=self._make_initial_tokens(demos_in, demos_out)
        initial_mask=torch.ones_like(initial_tokens, dtype=torch.bool)
        final_query=self._sample_sequences(batch_size, 1).squeeze(1)
        final_target=self.transform(mechanism_id, final_query, params)
        candidate_probe_tokens=[]
        candidate_probe_embs=[]
        diagnostic_probe_targets=[]
        for _ in range(self.num_probe_steps):
            probes=sample_probe_sequences(batch_size, self.num_candidate_probes, self.seq_len, self.vocab_size, self.generator)
            candidate_probe_tokens.append(probes)
            candidate_probe_embs.append(probes.clone())
            diagnostic_probe_targets.append(self._diagnostic_probe_target(probes, params))
        return EpisodeBatch(initial_tokens=initial_tokens, initial_mask=initial_mask, candidate_probe_tokens=candidate_probe_tokens, candidate_probe_embs=candidate_probe_embs, final_query=final_query, final_target=final_target, mechanism_id=mechanism_id, mechanism_params=params, diagnostic_probe_targets=diagnostic_probe_targets)

    def execute_probe_batch(self, batch:EpisodeBatch, step:int, chosen_idx:torch.Tensor, device=None)->torch.Tensor:
        probes=batch.candidate_probe_tokens[step]
        b=probes.size(0)
        chosen=probes[torch.arange(b), chosen_idx]
        out=self.transform(batch.mechanism_id.cpu(), chosen.cpu(), batch.mechanism_params.cpu())
        return out.to(device) if device is not None else out
