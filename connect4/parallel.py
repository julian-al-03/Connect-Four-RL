"""Process pool shared by self-play and evaluation."""
import multiprocessing

import torch


def _init_worker():
    # Each process handles one game at a time, so extra threads per process would only compete
    torch.set_num_threads(1)


def make_pool(workers):
    # "spawn" starts clean processes; "fork" can deadlock once PyTorch has started its threads
    return multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker)
