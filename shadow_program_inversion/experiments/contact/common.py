"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pyquaternion import Quaternion

from shadow_program_inversion.common.wrench import Wrench
from shadow_program_inversion.common.orientation import Orientation
from shadow_program_inversion.common.pose import Pose
from shadow_program_inversion.common.position import Position
from shadow_program_inversion.common.trajectory import Trajectory


def debias_forces(trajectory: Trajectory) -> Trajectory:
    bias = np.array(trajectory.wrenches[0].parameters())
    for i in range(len(trajectory)):
        trajectory.wrenches[i] = Wrench.from_parameters(np.array(trajectory.wrenches[i].parameters()) - bias)
    return trajectory


def mlrc_result_label(datapoints: List[List[float]]):
    # MLRC successful if goal_force in Z execeeded
    goal_force = 5.0
    max_force_z = max([dp[9] for dp in datapoints])
    return max_force_z >= goal_force


class Scenario(ABC):
    @abstractmethod
    def approach_pose(self):
        pass

    @abstractmethod
    def goal_pose(self):
        pass

    @abstractmethod
    def tcp(self):
        pass


class RubberScenario(Scenario):
    def approach_pose(self):
        ori = Orientation(Quaternion(axis=[1, 0, 0], angle=0))
        return Pose(Position(0.292, -0.290, 0.007), ori, reference=None)

    def goal_pose(self):
        ori = Orientation(Quaternion(axis=[1, 0, 0], angle=0))
        return Pose(Position(0.292, -0.290, -0.006), ori, reference=None)

    def tcp(self):
        ori = Orientation(Quaternion(axis=[2.0543, 0.8551, -0.4639], angle=np.linalg.norm([2.0543, 0.8551, -0.4639])))
        return Pose(Position(0.001, 0.088, 0.04), ori, reference=None)


class PCBScenario(Scenario):
    def approach_pose(self):
        ori = Orientation(Quaternion(axis=[1, 0, 0], angle=0))
        return Pose(Position(0.194, -0.344, 0.043), ori, reference=None)

    def goal_pose(self):
        ori = Orientation(Quaternion(axis=[1, 0, 0], angle=0))
        return Pose(Position(0.194, -0.344, 0.033), ori, reference=None)

    def tcp(self):
        ori = Orientation(Quaternion(axis=[2.0543, 0.8551, -0.4639], angle=np.linalg.norm([2.0543, 0.8551, -0.4639])))
        return Pose(Position(0.001, 0.088, 0.04), ori, reference=None)


class FoamScenario(Scenario):
    def approach_pose(self):
        ori = Orientation(Quaternion(axis=[1, 0, 0], angle=0))
        return Pose(Position(0.087, -0.449, 0.028), ori, reference=None)

    def goal_pose(self):
        ori = Orientation(Quaternion(axis=[1, 0, 0], angle=0))
        return Pose(Position(0.087, -0.449, 0.010), ori, reference=None)

    def tcp(self):
        ori = Orientation(Quaternion(axis=[2.0543, 0.8551, -0.4639], angle=np.linalg.norm([2.0543, 0.8551, -0.4639])))
        return Pose(Position(0.001, 0.088, 0.04), ori, reference=None)


class ScenarioFactory(object):
    keys = ["rubber", "pcb", "foam"]

    @staticmethod
    def make_scenario(key: str) -> Scenario:
        if key == "rubber":
            return RubberScenario()
        elif key == "pcb":
            return PCBScenario()
        elif key == "foam":
            return FoamScenario()
        else:
            raise ValueError(f"Unknown scenario: {key}")
