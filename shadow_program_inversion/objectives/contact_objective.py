"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import torch


class ContactObjective(torch.nn.Module):
    def __init__(self, target_force: torch.Tensor, dimension: int = 11, weight: float = 1.0):
        super().__init__()
        self.target_force = target_force
        self.dimension = dimension
        self.weight = weight
        self.duration = 3

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        median_force = torch.median(trajectory[:, -self.duration:, self.dimension])
        return torch.nn.L1Loss()(median_force, self.target_force)
