"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import json
import os
import random
from argparse import ArgumentParser

import pyrobolearn as prl
import torch

from shadow_program_inversion.experiments.contact.common import Scenario, mlrc_result_label, debias_forces, \
    ScenarioFactory
from shadow_program_inversion.priors.dummy_prior import DummyPrior
from shadow_program_inversion.priors.neural_prior import NeuralPrior
from shadow_program_inversion.utils.io import save_simulation_data_to_file
from shadow_program_inversion.utils.sequence_utils import chunk
from shadow_program_inversion.utils.ur.rtde_recorder import RTDERecorder
from shadow_program_inversion.utils.ur.ur_dmp_program import DMPDataCollector, execute_move_linear
import shadow_program_inversion.utils.config as cfg

MAX_TRAJECTORY_LEN = 500


def main(args):
    scenario = ScenarioFactory.make_scenario(args.material)
    start_pose = scenario.approach_pose()

    sim = prl.simulators.Bullet(render=False)
    world = prl.worlds.BasicWorld(sim)
    robot = world.load_robot(
        prl.robots.UR5e(sim, urdf="/home/lab002/Projects/pyrobolearn/pyrobolearn/robots/urdfs/ur/ur5e_poke_tcp.urdf"))
    data_collector = DMPDataCollector(world, robot)

    model_dir = os.path.join(cfg.TRAINED_MODELS_DIR, "dmp", args.material)
    prior_dir = os.path.join(model_dir, list(filter(lambda dirname: dirname.startswith("NeuralPrior"),
                                                    os.listdir(model_dir)))[0])
    ml_simulator = NeuralPrior.load(prior_dir)
    config_path = os.path.join(cfg.DATA_DIR, "config", f"config_dmp_{args.material}.json")
    with open(config_path) as config_file:
        config = json.load(config_file)

    batch_size = 100
    for kind in ["train", "test"]:
        num_data = int(0.8 * args.num_data) if kind == "train" else int(0.2 * args.num_data)
        output_filepath = os.path.join(cfg.DATA_DIR, "dmp", args.material, kind,
                                       f"dmp_{args.material}_{kind}.h5")
        for batch in chunk(range(num_data), batch_size):
            inputs = []
            points_from = []
            simulations = []
            trajectories = []

            for _ in batch:
                tau = random.uniform(*config["tau"]["limits"])
                z_offset = random.uniform(*config["point_to_offset"]["limits"]["z"])
                goal_pose = scenario.goal_pose()
                goal_pose.position.z += z_offset

                rtde_recorder = RTDERecorder(cfg.ROBOT_IP, 30004, mlrc_result_label, sampling_interval=0.016)
                trajectory = execute_move_linear(data_collector, rtde_recorder, start_pose, goal_pose, tau)
                if trajectory is None:
                    raise RuntimeError()
                print(f"Recorded trajectory of length {len(trajectory)}")
                trajectory = debias_forces(trajectory)

                input_tensor = torch.cat((torch.as_tensor(goal_pose.parameters()), torch.as_tensor([tau])), dim=-1)
                inputs.append(input_tensor)
                point_from_tensor = torch.as_tensor(start_pose.parameters())
                points_from.append(point_from_tensor)
                sim = ml_simulator.simulate(input_tensor.unsqueeze(0), point_from_tensor.unsqueeze(0), max_trajectory_len=MAX_TRAJECTORY_LEN).squeeze()
                simulations.append(sim.squeeze())
                trajectories.append(trajectory.to_tensor(pad_to=MAX_TRAJECTORY_LEN))

            save_simulation_data_to_file(output_filepath, torch.stack(inputs), torch.stack(points_from),
                                         sim=torch.stack(simulations), real=torch.stack(trajectories))

    data_collector.move_to_pose(start_pose)     # Move to home when done


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("material", type=str, choices=["foam", "pcb", "rubber"])
    parser.add_argument("num_data", type=int)
    main(parser.parse_args())
