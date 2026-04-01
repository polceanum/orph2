import torch

from ood_solver.envs.hidden_mechanism_seq import HiddenMechanismSequenceEnv
from ood_solver.models.approach import ApproachProposer
from ood_solver.models.belief_updater import BeliefUpdater
from ood_solver.models.encoder import SimpleSequenceEncoder
from ood_solver.models.hypothesis_solver import HypothesisSolver
from ood_solver.models.probe_policy import ProbePolicy
from ood_solver.models.rules import RuleProposer
from ood_solver.models.solver_head import SolverHead
from ood_solver.training.trainer import Trainer

def test_single_train_step_runs():
    device = "cpu"
    env = HiddenMechanismSequenceEnv(seq_len=8, num_probe_steps=2, num_candidate_probes=4, num_demos=2)
    batch = env.sample_episode(batch_size=2)
    model = HypothesisSolver(
        SimpleSequenceEncoder(vocab_size=16, d_model=32, nhead=4, num_layers=1, max_len=128),
        ApproachProposer(d_model=32, num_slots=4),
        RuleProposer(d_model=32, num_rules=8),
        ProbePolicy(d_model=32),
        BeliefUpdater(d_model=32, archive_size=8),
        SolverHead(vocab_size=16, d_model=32),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, opt, device=device, loss_weights={"task": 1.0, "probe": 0.2})

    def probe_executor(ep, step, chosen_idx):
        return env.execute_probe_batch(ep, step, chosen_idx, device=device)

    metrics = trainer.train_step(batch, probe_executor, aux_modules=None)
    assert "loss" in metrics
