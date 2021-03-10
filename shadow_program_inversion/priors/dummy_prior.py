"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import torch

from shadow_program_inversion.priors.differentiable_prior import DifferentiablePrior
from shadow_program_inversion.priors.static_prior import TrajectoryProperties


class DummyPrior(DifferentiablePrior):
    def __init__(self, trajectory_properties: TrajectoryProperties = None):
        self.trajectory_properties = trajectory_properties if trajectory_properties is not None else TrajectoryProperties()

    def simulate(self, inputs_world: torch.Tensor, point_start_world: torch.Tensor,
                 max_trajectory_len: int = 500) -> torch.Tensor:
        batch_size = inputs_world.size(0)

        # Build trajectory according to group mask
        eos_labels = torch.zeros(batch_size, max_trajectory_len, 1, device=inputs_world.device)
        eos_labels[:, -1] = 1.0
        success_probs = torch.zeros(batch_size, max_trajectory_len, 1, device=inputs_world.device)
        traj = torch.cat((eos_labels, success_probs), dim=-1)
        if self.trajectory_properties.cartesian:        # Fake trajectory data: Stay at start point
            traj = torch.cat((traj, point_start_world[:, :7].unsqueeze(1).repeat(1, max_trajectory_len, 1)), dim=-1)
        if self.trajectory_properties.wrench:
            traj = torch.cat((traj, torch.zeros(batch_size, max_trajectory_len, 6, device=inputs_world.device)), dim=-1)
        if self.trajectory_properties.gripper:          # Fake gripper data: Stay at start state
            traj = torch.cat((traj, point_start_world[:, 7].unsqueeze(1).repeat(1, max_trajectory_len, 1)), dim=-1)
        return traj
