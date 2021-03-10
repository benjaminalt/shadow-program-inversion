"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import os
from argparse import ArgumentParser

import numpy as np
import torch

from shadow_program_inversion.common.trajectory import Trajectory
from shadow_program_inversion.objectives.contact_objective import ContactObjective
from shadow_program_inversion.priors.neural_prior import NeuralPrior
from shadow_program_inversion.shadow_program import ShadowProgram
from shadow_program_inversion.shadow_skill import ShadowSkill
from shadow_program_inversion.utils.io import load_data_file
from shadow_program_inversion.utils.viz import plot_multiple_trajectories_vertical
import shadow_program_inversion.utils.config as cfg


def _optimize_concurrent(target_force: float, inputs_tensor: torch.Tensor, points_from_tensor: torch.Tensor,
                         shadow_skill: ShadowSkill):
    loss = ContactObjective(torch.tensor(target_force), dimension=11)
    neural_program = ShadowProgram([shadow_skill])
    optimized_param_history, intermediate_trajectories, loss_history, _ = neural_program.optimize(
        num_iterations=1000,
        loss_fn=loss,
        inputs_world=[inputs_tensor],
        start_state_world=points_from_tensor, learning_rates=[5e-5],
        param_limits=[shadow_skill.input_limits])
    min_loss_idx = np.argmin(loss_history)
    optimal_parameters = optimized_param_history[0][min_loss_idx]

    optimal_trajectory = intermediate_trajectories[min_loss_idx]
    return optimal_parameters, optimal_trajectory, torch.stack(optimized_param_history[0])


def main(args):
    model_dir = os.path.join(cfg.TRAINED_MODELS_DIR, "urscript", args.material)
    prior_dir = os.path.join(model_dir,
                             list(filter(lambda dirname: dirname.startswith("NeuralPrior"), os.listdir(model_dir)))[0])
    sim = NeuralPrior.load(prior_dir)
    skill_dir = os.path.join(model_dir,
                             list(filter(lambda dirname: dirname.startswith("ShadowSkill"), os.listdir(model_dir)))[0])
    shadow_skill = ShadowSkill.load(skill_dir, sim)
    test_data_dir = os.path.join(cfg.DATA_DIR, "urscript", args.material, "test")
    test_data_filepath = os.path.join(test_data_dir, list(os.listdir(test_data_dir))[0])
    inputs_tensor, points_from_tensor, sim_tensor, real_tensor = load_data_file(test_data_filepath)
    if args.show_plots:
        results = []
        optimized_param_histories = []
        for i in range(args.n):
            trajectory_world, simulation, _ = shadow_skill.predict(inputs_tensor[i], points_from_tensor[i],
                                                                   max_trajectory_len=real_tensor.size(1))

            plot_multiple_trajectories_vertical([Trajectory.from_tensor(real_tensor[i]),
                                                 Trajectory.from_tensor(simulation),
                                                 Trajectory.from_tensor(trajectory_world)],
                                                show=True, include_forces=True, ms_per_sample=16,
                                                colors=["black", "orange", "red"],
                                                labels=["Label", "Sim", "Prediction"])
            optimal_parameters, optimal_trajectory, optimized_param_history = _optimize_concurrent(args.target_force,
                                                                                                   inputs_tensor[i],
                                                                                                   points_from_tensor[
                                                                                                       i], shadow_skill)
            plot_multiple_trajectories_vertical([Trajectory.from_tensor(real_tensor[i]),
                                                 Trajectory.from_tensor(optimal_trajectory)],
                                                show=True, include_forces=True, ms_per_sample=16,
                                                colors=["blue", "red"],
                                                labels=["Before", "Optimized"])
            results.append(optimal_parameters)
            optimized_param_histories.append(optimized_param_history)
    else:
        pool = torch.multiprocessing.Pool(3)
        result_tuples = pool.starmap(_optimize_concurrent, [(args.target_force, inputs_tensor[i], points_from_tensor[i],
                                                             shadow_skill) for i in range(args.n)])
        results, optimal_trajectories, optimized_param_histories = zip(*result_tuples)

    output_filepath = os.path.join(cfg.OPTIMIZED_PARAM_DIR, f"urscript_{args.material}_{args.target_force}N.npy")
    np.save(output_filepath, torch.stack(results).numpy())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("material", type=str, choices=["foam", "pcb", "rubber"])
    parser.add_argument("target_force", type=float, help="Desired contact force in N")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--show_plots", action="store_true")
    main(parser.parse_args())
