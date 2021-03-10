"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import json
import os
import random
from argparse import ArgumentParser

import torch

from shadow_program_inversion.experiments.contact.common import mlrc_result_label, ScenarioFactory
from shadow_program_inversion.priors.neural_prior import NeuralPrior
from shadow_program_inversion.utils.io import save_simulation_data_to_file
from shadow_program_inversion.utils.sequence_utils import chunk
from shadow_program_inversion.utils.ur.rtde_recorder import RTDERecorder
import shadow_program_inversion.utils.config as cfg
from shadow_program_inversion.utils.ur.ur_script_program import execute_move_linear

TRAJECTORY_LENGTH = 250


def main(args):
    scenario = ScenarioFactory.make_scenario(args.material)
    config_path = os.path.join(cfg.DATA_DIR, "config", f"config_urscript_{args.material}.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    model_dir = os.path.join(cfg.TRAINED_MODELS_DIR, "urscript", args.material)
    prior_dir = os.path.join(model_dir, list(filter(lambda dirname: dirname.startswith("NeuralPrior"),
                                                    os.listdir(model_dir)))[0])
    simulator = NeuralPrior.load(prior_dir)
    rec = RTDERecorder(cfg.ROBOT_IP, 30004, mlrc_result_label, sampling_interval=0.016, respect_ur_runtime_state=True)

    batch_size = 100
    for kind in ["train", "test"]:
        num_data = int(0.8 * args.num_data) if kind == "train" else int(0.2 * args.num_data)
        output_filepath = os.path.join(cfg.DATA_DIR, "urscript", args.material, kind,
                                       f"urscript_{args.material}_{kind}.h5")
        for batch in chunk(range(num_data), batch_size):
            inputs = []
            points_from = []
            simulations = []
            trajectories = []
            for _ in batch:
                # Run single experiment
                approach_pose = scenario.approach_pose()
                vel = random.uniform(*config["velocity"]["limits"]["cartesian"])
                acc = random.uniform(*config["acceleration"]["limits"]["cartesian"])
                z_offset = random.uniform(*config["point_to_offset"]["limits"]["z"])
                goal_pose = scenario.goal_pose()
                goal_pose.position.z += z_offset
                trajectory = execute_move_linear(cfg.ROBOT_IP, rec, approach_pose, goal_pose, vel, acc)
                if trajectory is None:
                    continue

                # Save data as tensors
                input_tensor = torch.cat((torch.as_tensor(goal_pose.parameters()), torch.as_tensor([vel, acc])), dim=-1)
                inputs.append(input_tensor)
                point_from_tensor = torch.as_tensor(approach_pose.parameters())
                points_from.append(point_from_tensor)
                sim = simulator.simulate(input_tensor, point_from_tensor, max_trajectory_len=TRAJECTORY_LENGTH)
                simulations.append(sim.squeeze())
                trajectories.append(trajectory.to_tensor(pad_to=TRAJECTORY_LENGTH))

            save_simulation_data_to_file(output_filepath, torch.stack(inputs), torch.stack(points_from),
                                         sim=torch.stack(simulations), real=torch.stack(trajectories))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("material", type=str, choices=["foam", "pcb", "rubber"])
    parser.add_argument("num_data", type=int)
    main(parser.parse_args())
