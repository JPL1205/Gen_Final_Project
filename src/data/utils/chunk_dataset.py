from typing import *

import random

import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import get_worker_info

from src.data.utils.chunk_data_source import ParquetChunkDataSource


class ChunkedDataset(IterableDataset):
    """
    Iterable dataset backed by ParquetChunkDataSource.
    Subclasses must implement `get_trainable_data_from_raw_data`.
    """
    def __init__(
        self,
        data_source: ParquetChunkDataSource,
        shuffle: bool = False,
        shuffle_buffer_size: int = -1,
        chunks_queue_max_size: int = 1,  # kept for API compatibility
        seed: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.data_source = data_source
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.chunks_queue_max_size = chunks_queue_max_size
        self.seed = seed
        self.epoch = 0

        # Keep compatibility with previous callsites that forward extra kwargs.
        self.extra_kwargs = kwargs

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def get_trainable_data_from_raw_data(self, raw_data_list) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _get_shard_spec(self) -> Tuple[int, int]:
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        # Global shard across distributed processes and dataloader workers.
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        return shard_id, num_shards

    @staticmethod
    def _seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)

    def _iter_raw_samples(self) -> Iterator[Dict[str, Any]]:
        shard_id, num_shards = self._get_shard_spec()
        local_seed = self.seed + self.epoch * 100003 + shard_id * 9176
        self._seed_everything(local_seed)

        raw_iter = self.data_source.iter_rows(
            shard_id=shard_id,
            num_shards=num_shards,
            shuffle=self.shuffle,
            seed=self.seed,
            epoch=self.epoch,
        )

        # Optional deterministic buffered shuffle.
        if self.shuffle and self.shuffle_buffer_size is not None and self.shuffle_buffer_size > 1:
            rng = random.Random(local_seed)
            buffer: List[Dict[str, Any]] = []

            for sample in raw_iter:
                buffer.append(sample)
                if len(buffer) >= self.shuffle_buffer_size:
                    idx = rng.randrange(len(buffer))
                    yield buffer.pop(idx)

            while len(buffer) > 0:
                idx = rng.randrange(len(buffer))
                yield buffer.pop(idx)
        else:
            yield from raw_iter

    def __iter__(self):
        for raw_sample in self._iter_raw_samples():
            yield self.get_trainable_data_from_raw_data([raw_sample])


class ChunkedDataLoader(DataLoader):
    """
    Thin DataLoader wrapper to provide deterministic worker seeding while keeping
    compatibility with existing training scripts.
    """
    def __init__(self, *args, **kwargs):
        user_worker_init_fn = kwargs.pop("worker_init_fn", None)

        def _worker_init_fn(worker_id: int):
            # Keep numpy/python/torch RNGs aligned per worker.
            worker_seed = torch.initial_seed() % (2 ** 32)
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

            if user_worker_init_fn is not None:
                user_worker_init_fn(worker_id)

        kwargs["worker_init_fn"] = _worker_init_fn
        super().__init__(*args, **kwargs)
