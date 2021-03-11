"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import os
from argparse import ArgumentParser

import numpy as np
import pyrobolearn as prl
import torch

from shadow_program_inversion.common.pose import Pose
from shadow_program_inversion.common.trajectory import Trajectory
from shadow_program_inversion.experiments.contact.common import Scenario, mlrc_result_label, debias_forces, \
    ScenarioFactory
from shadow_program_inversion.utils.io import load_data_file
from shadow_program_inversion.utils.ur.rtde_recorder import RTDERecorder
from shadow_program_inversion.utils.ur.ur_dmp_program import DMPDataCollector, execute_move_linear
from shadow_program_inversion.utils.viz import plot_multiple_trajectories_vertical
import shadow_program_inversion.utils.config as cfg

MAX_TRAJECTORY_LEN = 500


def main(args):
    scenario = ScenarioFactory.make_scenario(args.material)
    optimized_param_filepath = os.path.join(cfg.OPTIMIZED_PARAM_DIR, f"dmp_{args.material}_{args.target_force}N.npy")
    optimized_parameters = np.load(optimized_param_filepath)

    sim = prl.simulators.Bullet(render=False)
    world = prl.worlds.BasicWorld(sim)
    robot = world.load_robot(prl.robots.UR5e(sim))
    data_collector = DMPDataCollector(world, robot)
    rec = RTDERecorder(cfg.ROBOT_IP, 30004, mlrc_result_label, sampling_interval=0.016)

    result_trajectories = []
    test_data_dir = os.path.join(cfg.DATA_DIR, "dmp", args.material, "test")
    test_data_filepath = os.path.join(test_data_dir, list(os.listdir(test_data_dir))[0])
    inputs_tensor, points_from_tensor, sim_tensor, real_tensor = load_data_file(test_data_filepath)
    for i, optimized_params in enumerate(optimized_parameters):
        param_tensor = torch.from_numpy(optimized_params).float()
        goal_pose = Pose.from_parameters(param_tensor[:7])
        tau = param_tensor[7].item()
        real_trajectory = execute_move_linear(data_collector, rec, scenario.approach_pose(), goal_pose, tau)
        if real_trajectory is None:
            raise RuntimeError("Could not execute trajectory!!!")
        real_trajectory = debias_forces(real_trajectory)
        if args.show_plots:
            plot_multiple_trajectories_vertical([Trajectory.from_tensor(real_tensor[i]),
                                                 real_trajectory],
                                                show=True, include_forces=True, ms_per_sample=16,
                                                colors=["black", "green"],
                                                labels=["Before", "Optimized (Real)"])
        result_trajectories.append(real_trajectory)
    result_trajectories = torch.stack([traj.to_tensor(pad_to=MAX_TRAJECTORY_LEN) for traj in result_trajectories])
    output_filepath = os.path.join(cfg.RESULTS_DIR, f"dmp_{args.material}_{args.target_force}N.npy")
    np.save(output_filepath, result_trajectories.numpy())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("material", type=str, choices=["foam", "pcb", "rubber"])
    parser.add_argument("target_force", type=float, help="Desired contact force in N")
    parser.add_argument("--show_plots", action="store_true")
    main(parser.parse_args())
