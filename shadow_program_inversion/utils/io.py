"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import multiprocessing
import os
import uuid

import tables
import torch
from natsort import natsorted


def save_simulation_data_to_file(data_filepath: str, inputs: torch.Tensor, points_start: torch.Tensor,
                                 sim: torch.Tensor = None, real: torch.Tensor = None, batch_id: str = None):
    batch_id = batch_id if batch_id is not None else "a" + str(uuid.uuid4()).replace("-", "")
    if not os.path.exists(data_filepath):
        print(f"{data_filepath} does not exist, creating...")
        with tables.open_file(data_filepath, "w", title="Simulation data") as output_file:
            output_file.create_group("/", "inputs")
            output_file.create_group("/", "point_start")
            if sim is not None:
                output_file.create_group("/", "sim")
            if real is not None:
                output_file.create_group("/", "real")
    with tables.open_file(data_filepath, "a", title="Simulation data") as output_file:
        nodes = output_file.list_nodes("/inputs")
        node_names = natsorted([node.name for node in nodes])
        batch_id = str(int(node_names[-1]) + 1) if len(node_names) > 0 else "0"
        output_file.create_array("/inputs", batch_id, inputs.numpy())
        output_file.create_array("/point_start", batch_id, points_start.numpy())
        if sim is not None:
            output_file.create_array("/sim", batch_id, sim.numpy())
        if real is not None:
            output_file.create_array("/real", batch_id, real.numpy())


def _load_group(data_filepath, group_name, return_list):
    with tables.open_file(data_filepath) as data_file:
        for batch in natsorted(data_file.list_nodes(group_name, classname="Array"), key=lambda node: node.name):
            return_list.append(torch.from_numpy(batch.read()))


def load_data_file(data_filepath: str):
    manager = multiprocessing.Manager()

    inputs = manager.list()
    points_from = manager.list()
    sim = manager.list()
    real = manager.list()

    processes = [
        multiprocessing.Process(target=_load_group, args=(data_filepath, "/inputs", inputs)),
        multiprocessing.Process(target=_load_group, args=(data_filepath, "/point_start", points_from)),
        multiprocessing.Process(target=_load_group, args=(data_filepath, "/sim", sim)),
        multiprocessing.Process(target=_load_group, args=(data_filepath, "/real", real))
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    inputs_tensor = torch.cat(list(inputs), dim=0).float()
    points_from_tensor = torch.cat(list(points_from), dim=0).float()
    sim_tensor = torch.cat(list(sim), dim=0).float() if len(sim) > 0 else None
    real_tensor = torch.cat(list(real), dim=0).float() if len(real) > 0 else None
    return inputs_tensor, points_from_tensor, sim_tensor, real_tensor
