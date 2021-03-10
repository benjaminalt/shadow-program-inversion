"""
Copyright (C) 2021 ArtiMinds Robotics GmbH

Analyze optimization results.
Print the median absolute deviation of the measured contact forces from the intended contact force for each series (set of measurements).
Optionally plot the force trajectories before and after parameter optimization.

Example:
    $ python analyze_results.py /home/demo/shadow-program-inversion/results --plot-series=urscript_pcb_5N
"""

import argparse
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from shadow_program_inversion.utils.io import load_data_file
from shadow_program_inversion.utils.sequence_utils import unpad_padded_sequence

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "data"))


def median_absolute_force_deviation(results_arr: np.ndarray, goal_force: float) -> float:
    force_z_dim = 11
    losses = []
    for traj in results_arr:
        unpadded = unpad_padded_sequence(torch.from_numpy(traj))
        loss = torch.abs(goal_force - unpadded[-3:, force_z_dim].median())
        losses.append(loss)
    return torch.stack(losses).median().item()


def plot_forces_before_after(trajectories_before: np.ndarray, trajectories_after: np.ndarray, goal_force: float):
    fig, ax = plt.subplots(figsize=(3.76, 2.76))
    force_z_dim = 11
    sampling_interval = 0.016
    for traj_before in trajectories_before:
        unpadded = unpad_padded_sequence(torch.from_numpy(traj_before))
        ax.plot([t * sampling_interval for t in range(len(unpadded))], unpadded[:, force_z_dim], color="gray", linewidth=.5, alpha=0.5)
    for traj_after in trajectories_after:
        unpadded = unpad_padded_sequence(torch.from_numpy(traj_after))
        ax.plot([t * sampling_interval for t in range(len(unpadded))], unpadded[:, force_z_dim], color="red", linewidth=.5)
    ax.legend([Line2D([0], [0], color="gray"), Line2D([0], [0], color="red")], ["Baseline", "Optimized"], loc="lower right")
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Force (Z) [$N$]")
    ax.axhline(goal_force, linestyle="--", color="black")
    ax.text(0, goal_force + 0.2, "goal force")
    plt.subplots_adjust(left=0.129, bottom=0.145, right=1.0, top=1.0)
    plt.show()


def main(args):
    results = {}

    for filename in os.listdir(args.results_dir):
        match = re.match(r"([a-z]+)_([a-z]+)_([0-9\.]+)N.npy", filename)
        goal_force = float(match.group(3))
        program_type = match.group(1)
        material = match.group(2)
        test_data_filepath = os.path.join(DATA_DIR, program_type, material, "test", f"{program_type}_{material}_test.h5")
        _, _, _, test_real = load_data_file(test_data_filepath)
        results_arr = np.load(os.path.join(args.results_dir, filename))
        if program_type not in results.keys():
            results[program_type] = {}
        if material not in results[program_type].keys():
            results[program_type][material] = {}
        force_deviation = median_absolute_force_deviation(results_arr, goal_force)
        force_deviation_before = median_absolute_force_deviation(test_real[:len(results_arr)].numpy(), goal_force)
        improvement = (force_deviation - force_deviation_before) / force_deviation_before
        print(f"{filename}: {len(results_arr)}, metric: {force_deviation:.4f}, improvement: {improvement:.4f}")
        results[program_type][material][f"{goal_force} N"] = force_deviation

        if args.plot_series is not None:
            plot_program_type, plot_material, plot_goal_force = args.plot_series.split("_")
            plot_goal_force = float(plot_goal_force[:-1])
            if goal_force == plot_goal_force and program_type == plot_program_type and material == plot_material:
                plot_forces_before_after(test_real[:len(results_arr)].numpy(), results_arr, goal_force)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Path to dir containing measured trajectories, e.g. '/home/demo/shadow-program-inversion/results'")
    parser.add_argument("--plot-series", type=str, help="Name of the series to plot, e.g. 'urscript_pcb_5N'")
    main(parser.parse_args())
