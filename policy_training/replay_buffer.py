import io
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def _worker_init_fn(worker_id):
    # seed = np.random.get_state()[1][0] + worker_id
    # np.random.seed(seed)
    # random.seed(seed)
    np.random.seed(worker_id)
    random.seed(worker_id)


def make_expert_replay_loader(iterable, batch_size):
    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader
