"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import json
import os
import random
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from shadow_program_inversion.experiments.contact.common import ScenarioFactory
from shadow_program_inversion.priors.path_generator import PathGeneratorMoveLinear
from shadow_program_inversion.priors.static_prior import StaticPrior
from shadow_program_inversion.utils.io import save_simulation_data_to_file
from shadow_program_inversion.utils.sequence_utils import chunk
import shadow_program_inversion.utils.config as cfg

TRAJECTORY_LENGTH = 250


def main(args):
    scenario = ScenarioFactory.make_scenario(args.material)
    config_path = os.path.join(cfg.DATA_DIR, "config", f"config_urscript_{args.material}.json")
    ml_simulator = StaticPrior(PathGeneratorMoveLinear(), sampling_interval=0.016, multiproc=True)
    with open(config_path) as config_file:
        config = json.load(config_file)
    batch_size = (torch.multiprocessing.cpu_count() - 1) * 8
    for kind in ["train", "test"]:
        output_filepath = os.path.join(cfg.DATA_DIR, "urscript", args.material, f"sim_{kind}",
                                       f"urscript_{args.material}_{kind}.h5")
        num_data = int(0.8 * args.num_data) if kind == "train" else int(0.2 * args.num_data)
        for batch in tqdm(chunk(range(num_data), batch_size), total=int(num_data / batch_size)):
            inputs = []
            points_from = []
            for _ in batch:
                approach_pose = scenario.approach_pose()
                vel = random.uniform(*config["velocity"]["limits"]["cartesian"])
                acc = random.uniform(*config["acceleration"]["limits"]["cartesian"])
                z_offset = random.uniform(*config["point_to_offset"]["limits"]["z"])
                goal_pose = scenario.goal_pose()
                goal_pose.position.z += z_offset
                input_tensor = torch.cat((torch.as_tensor(goal_pose.parameters()), torch.as_tensor([vel, acc])), dim=-1)
                inputs.append(input_tensor)
                point_from_tensor = torch.as_tensor(approach_pose.parameters())
                points_from.append(point_from_tensor)
            input_batch = torch.stack(inputs)
            points_from_batch = torch.stack(points_from)
            simulations = ml_simulator.simulate(input_batch, points_from_batch,
                                                max_trajectory_len=TRAJECTORY_LENGTH, cache=False)
            save_simulation_data_to_file(output_filepath, input_batch, points_from_batch, simulations, None)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("material", type=str, choices=["foam", "pcb", "rubber"])
    parser.add_argument("num_data", type=int)
    main(parser.parse_args())
