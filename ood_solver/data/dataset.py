from __future__ import annotations
from torch.utils.data import IterableDataset

class SyntheticEpisodeDataset(IterableDataset):
    def __init__(self, env, batch_size:int, steps_per_epoch:int):
        self.env=env
        self.batch_size=batch_size
        self.steps_per_epoch=steps_per_epoch
    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            yield self.env.sample_episode(self.batch_size)
