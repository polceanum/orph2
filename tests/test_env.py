from ood_solver.envs.hidden_mechanism_seq import HiddenMechanismSequenceEnv

def test_env_shapes():
    env=HiddenMechanismSequenceEnv(seq_len=8, num_probe_steps=2, num_candidate_probes=4, num_demos=2)
    batch=env.sample_episode(batch_size=3)
    assert batch.initial_tokens.shape[0]==3
    assert len(batch.candidate_probe_tokens)==2
    assert batch.final_query.shape==(3,8)
    assert batch.final_target.shape==(3,8)
