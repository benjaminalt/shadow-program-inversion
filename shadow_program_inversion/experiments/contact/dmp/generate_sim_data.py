"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import argparse
import json
import multiprocessing
import os
import random

import torch
from tqdm import tqdm

from shadow_program_inversion.common.pose import Pose
from shadow_program_inversion.common.trajectory import Trajectory
from shadow_program_inversion.experiments.contact.common import ScenarioFactory
from shadow_program_inversion.utils.io import save_simulation_data_to_file
from shadow_program_inversion.utils.sequence_utils import chunk
from shadow_program_inversion.utils.ur.ur_dmp_program import URLinearMotionDMP
import shadow_program_inversion.utils.config as cfg


MAX_TRAJECTORY_LEN = 500


def _generate_sim_data_concurrent(input_tensor: torch.Tensor, point_from_tensor: torch.Tensor):
    approach_pose = Pose.from_parameters(point_from_tensor.tolist())
    tau = input_tensor[-1].item()
    goal_pose = Pose.from_parameters(input_tensor[:7].tolist())
    dmp = URLinearMotionDMP(approach_pose, goal_pose, tau=tau, sampling_interval=0.016)
    cartesian_positions = dmp.rollout()[0]

    # Build trajectory
    fixed_orientations = torch.as_tensor(approach_pose.orientation.parameters()).unsqueeze(0).repeat(
        cartesian_positions.size(0), 1)
    forces = torch.zeros((cartesian_positions.size(0), 6), dtype=torch.float32)
    traj_without_meta = torch.cat((cartesian_positions, fixed_orientations, forces), dim=-1)
    traj_padded_with_meta = Trajectory.from_tensor(traj_without_meta).to_tensor(meta_inf=True,
                                                                                pad_to=MAX_TRAJECTORY_LEN)
    return traj_padded_with_meta


def main(args):
    scenario = ScenarioFactory.make_scenario(args.material)
    config_path = os.path.join(cfg.DATA_DIR, "config", f"config_dmp_{args.material}.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    batch_size = (multiprocessing.cpu_count() - 2) * 8
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 4)
    for kind in ["train", "test"]:
        output_filepath = os.path.join(cfg.DATA_DIR, "dmp", args.material, f"sim_{kind}",
                                       f"dmp_{args.material}_{kind}.h5")
        for batch in tqdm(chunk(range(args.num_data), batch_size), total=int(args.num_data / batch_size)):
            inputs = []
            points_from = []

            for _ in batch:
                approach_pose = scenario.approach_pose()
                tau = random.uniform(*config["tau"]["limits"])
                z_offset = random.uniform(*config["point_to_offset"]["limits"]["z"])
                goal_pose = scenario.goal_pose()
                goal_pose.position.z += z_offset
                input_tensor = torch.cat((torch.as_tensor(goal_pose.parameters()), torch.as_tensor([tau])), dim=-1)
                inputs.append(input_tensor)
                point_from_tensor = torch.as_tensor(approach_pose.parameters())
                points_from.append(point_from_tensor)
            args = [(inputs[i], points_from[i]) for i in range(len(inputs))]
            simulations = pool.starmap(_generate_sim_data_concurrent, args)
            save_simulation_data_to_file(output_filepath, torch.stack(inputs), torch.stack(points_from),
                                         torch.stack(simulations), None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, choices=["foam", "pcb", "rubber"])
    parser.add_argument("num_data", type=int)
    main(parser.parse_args())
