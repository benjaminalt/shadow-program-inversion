"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import math
import os

import tables
import torch
from natsort import natsorted
from torch.utils.data import IterableDataset
from itertools import cycle, chain, islice


class DirectoryDataset(IterableDataset):
    """
    Adapted from https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """
    def __init__(self, directory: str, start: int = 0, end: int = None):
        super(DirectoryDataset, self).__init__()
        self.start = start
        self.data_filepaths = list(map(lambda filename: os.path.join(directory, filename), natsorted(filter(lambda filename: filename.endswith(".h5"), os.listdir(directory)))))
        self.size = sum(map(self._count_data_in_file, self.data_filepaths))
        if end is None:
            self.end = self.size
        else:
            assert end <= self.size
            self.end = end
        assert self.start < self.end

    @staticmethod
    def _read_file(filepath: str):
        with tables.open_file(filepath) as data_file:
            for batch in data_file.list_nodes("/inputs", classname="Array"):
                inputs_arr = data_file.get_node(f"/inputs/{batch.name}").read()
                start_states_arr = data_file.get_node(f"/point_start/{batch.name}").read()
                sim_arr = data_file.get_node(f"/sim/{batch.name}").read()
                real_arr = data_file.get_node(f"/real/{batch.name}").read()
                for inputs, start_state, sim, real in zip(inputs_arr, start_states_arr, sim_arr, real_arr):
                    yield torch.from_numpy(inputs).float(), torch.from_numpy(start_state).float(),\
                          torch.from_numpy(sim).float(), torch.from_numpy(real).float()

    @staticmethod
    def _count_data_in_file(filepath: str):
        length = 0
        with tables.open_file(filepath) as data_file:
            for batch in data_file.list_nodes("/inputs", classname="Array"):
                length += len(batch)
        return length

    def _get_stream(self):
        paths = cycle(self.data_filepaths)
        return chain.from_iterable(map(self._read_file, paths))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:     # Single-process data loading
            iter_start = self.start
            iter_end = self.end
        else:                       # In a worker process
            per_worker = int(math.ceil(self.end - self.start) / float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        # Now make an iterator for data between iter_start and iter_end
        return islice(self._get_stream(), iter_start, iter_end)

    def __len__(self):
        return self.end - self.start
