import time
from typing import List

import numpy as np
import torch

import shadow_program_inversion.experiments.contact.urscript.train_shadow_skill
from shadow_program_inversion.utils.sequence_utils import unpad_padded_sequence

device = torch.device("cpu")


class ShadowProgram(object):
    """
    A neural program has several neural templates.
    Provides functionality to optimize the learnable parameters of the constituent templates.
    """

    def __init__(self, neural_templates):
        """
        :param neural_templates: A list of NeuralTemplate objects
        """
        self.neural_templates = neural_templates

    def predict(self, inputs_world: List[torch.Tensor], start_state_world: torch.Tensor, max_trajectory_len: int = 500):
        """
        Do a forward pass and compute the resulting outputs.
        """
        partial_trajectories = []
        internal_simulations = []
        state_from = start_state_world
        for i, nt in enumerate(self.neural_templates):
            partial_trajectory, internal_simulation, _ = nt.atomic_forward(inputs_world[i].unsqueeze(0),
                                                                           state_from.unsqueeze(0),
                                                                           simulation_world=None, denormalize_out=True,
                                                                           cache_sim=False)
            unpadded_partial_trajectory = unpad_padded_sequence(partial_trajectory.squeeze())
            unpadded_internal_simulation = unpad_padded_sequence(internal_simulation.squeeze())
            partial_trajectories.append(unpadded_partial_trajectory.cpu())
            internal_simulations.append(unpadded_internal_simulation.cpu())

            has_gripper_info = state_from.size(-1) == 8
            state_from = partial_trajectory[0, -1, 2:9]
            if has_gripper_info:
                state_from = torch.cat((state_from,  partial_trajectory[0, -1, 15].unsqueeze(0)), dim=-1)
        combined_trajectory = torch.cat(partial_trajectories, dim=0)
        return combined_trajectory, internal_simulations

    def _optimize(self, loss_fn, inputs_world: List[torch.Tensor], start_state_world: torch.Tensor, max_iterations: int,
                  learning_rates: List[float], param_limits: List[torch.Tensor] = None, callback_fn=None, patience: int = 10):
        assert len(inputs_world) == len(self.neural_templates)

        for i in range(len(inputs_world)):
            inputs_world[i] = inputs_world[i].to(device)
            inputs_world[i].requires_grad = True
        start_state_world = start_state_world.to(device)
        for i in range(len(param_limits)):
            param_limits[i] = param_limits[i].to(device)

        # Prepare models
        for i, nt in enumerate(self.neural_templates):
            if nt.model is not None:
                nt.model = nt.model.to(device)
                shadow_program_inversion.experiments.contact.urscript.train_shadow_skill.train()  # Enable train for backprop
                for param in nt.model.parameters():
                    param.requires_grad = False  # Disable gradients: Don't change network parameters
                nt.learnable_parameter_gradient_mask = nt.learnable_parameter_gradient_mask.to(device)

        if len(learning_rates) == 1:
            optimizers = [torch.optim.Adam(inputs_world, lr=learning_rates[0])]
        else:
            optimizers = [torch.optim.Adam([inputs_world[i]], lr=learning_rates[i]) for i in range(len(learning_rates))]
        # schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt) for opt in optimizers]
        # schedulers = [torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda epoch: 0.95) for opt in optimizers]

        min_loss = np.inf
        patience_count = 0

        res = []
        for i in range(max_iterations):
            start_time = time.time()

            for optimizer in optimizers:
                optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            complete_trajectory_world, internal_simulations = self.predict(inputs_world, start_state_world)
            loss = loss_fn(complete_trajectory_world.unsqueeze(0))

            # Backward pass and optimization
            loss.backward()

            # Apply gradient masks (depending on the neural template, some input parameters must remain fixed)
            for j, nt in enumerate(self.neural_templates):
                if hasattr(nt, "learnable_parameter_gradient_mask"):
                    inputs_world[j].grad *= nt.learnable_parameter_gradient_mask

            for j in range(len(optimizers)):
                optimizers[j].step()
                # schedulers[j].step()

            # Clip
            if param_limits is not None:
                for j, param_tensor in enumerate(inputs_world):
                    inputs_clipped = torch.empty(len(param_tensor))
                    for k in range(len(param_tensor)):
                        inputs_clipped[k] = param_tensor[k].clamp(param_limits[j][0, k], param_limits[j][1, k])
                    inputs_world[j].data = inputs_clipped.data

            print("[{: >3}] Loss:  {:.4f}, Time: {:.2f}".format(i, loss.item(), time.time() - start_time))
            epoch_res = [[inp.detach().cpu().clone() for inp in inputs_world], complete_trajectory_world.detach().cpu().clone(),
                         [sim.detach().cpu().clone() for sim in internal_simulations], loss.item()]
            if callback_fn is not None:
                callback_fn(*epoch_res)
            res.append(epoch_res)

            if patience_count == patience:
                print("Stopping optimization: Converged")
                break
            if loss < min_loss:     # Improved: Reset patience
                patience_count = 0
                min_loss = loss.item()
            else:                   # Did not improve: Consume patience
                patience_count += 1
        return res

    def optimize(self, num_iterations, loss_fn, inputs_world: List[torch.Tensor], start_state_world: torch.Tensor,
                 learning_rates: List[float], param_limits: List[torch.Tensor] = None, callback_fn=None, patience: int = 10):
        print("NeuralProgram::optimize: Starting optimization...")
        start_time = time.time()
        optimized_param_history, intermediate_trajectories, internal_sims, loss_history = zip(*self._optimize(loss_fn,
                                                                                                              inputs_world,
                                                                                                              start_state_world,
                                                                                                              num_iterations,
                                                                                                              learning_rates,
                                                                                                              param_limits,
                                                                                                              callback_fn,
                                                                                                              patience))
        print("NeuralProgram::optimize: Optimization finished, took {0:.2f}s".format(time.time() - start_time))
        optimized_param_history_per_template = [[] for _ in range(len(inputs_world))]
        for iteration in range(len(optimized_param_history)):
            for template_nr in range(len(inputs_world)):
                optimized_param_history_per_template[template_nr].append(optimized_param_history[iteration][template_nr])
        return optimized_param_history_per_template, intermediate_trajectories, np.array(loss_history), internal_sims
