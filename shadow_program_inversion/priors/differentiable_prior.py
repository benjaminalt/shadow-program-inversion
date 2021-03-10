"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""


from abc import ABC, abstractmethod

import torch


class DifferentiablePrior(ABC):
    @abstractmethod
    def simulate(self, inputs_world: torch.Tensor, point_start_world: torch.Tensor, max_trajectory_len: int = 500) -> torch.Tensor:
        pass
