"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

from typing import List

import torch
from pyquaternion import Quaternion

from shadow_program_inversion.common.wrench import Wrench
from shadow_program_inversion.common.orientation import Orientation
from shadow_program_inversion.common.pose import Pose
from shadow_program_inversion.shadow_skill import ShadowSkill
from shadow_program_inversion.utils.sequence_utils import unpad_padded_sequence, pad_padded_sequence


class Trajectory(object):
    def __init__(self, poses, wrenches, gripper_states: List[float] = None, success_label: float = 1.0):
        self.poses = poses
        self.wrenches = wrenches
        self.gripper_states = gripper_states
        self.success_label = success_label

    @staticmethod
    def from_raw(raw_trajectory: list, reference_pose: Pose, success_label: float = 1.0):
        """
        :param raw_trajectory: List of RPS poses (xyz_rxryrz) in meters and radians
        Does NOT contain gripper states
        """
        poses = []
        wrenches = []
        for raw_dp in raw_trajectory:
            if reference_pose is not None:
                poses.append(Pose.make_relative_from_xyz_rxryrz(raw_dp[:6], reference_pose))
            else:
                poses.append(Pose.make_absolute_from_xyz_rxryrz(raw_dp[:6]))
            wrenches.append(Wrench.from_parameters(raw_dp[6:12]))
        return Trajectory(poses, wrenches, success_label=success_label)

    def to_raw(self):
        """
        Does NOT contain gripper states
        """
        return [self.poses[i].to_absolute_xyz_rxryrz() + self.wrenches[i].parameters() for i in range(len(self.poses))]

    @staticmethod
    def from_list(arr, success_label: float = 1.0):
        poses = []
        wrenches = []
        gripper_states = []
        for dp_params in arr:
            poses.append(Pose.from_parameters(dp_params[:7]))
            wrenches.append(Wrench.from_parameters(dp_params[7:13]))
            if len(dp_params) == 14:
                gripper_states.append(dp_params[13])
        return Trajectory(poses, wrenches, gripper_states if len(gripper_states) > 0 else None,
                          success_label=success_label)

    def to_list(self):
        lst = []
        for i in range(len(self.poses)):
            dp = self.poses[i].parameters() + self.wrenches[i].parameters()
            if self.gripper_states is not None:
                dp.append(self.gripper_states[i])
            lst.append(dp)
        return lst

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        if tensor.size(-1) > 7 + 6 + 1: # Pose + force + gripper
            # Trajectory contains meta information and may be padded
            unpadded = unpad_padded_sequence(tensor)
            success_label = ShadowSkill.success_probability(unpadded)
            return Trajectory.from_list(unpadded[:, 2:].tolist(), success_label=success_label.item())
        return Trajectory.from_list(tensor.tolist())

    def to_tensor(self, meta_inf: bool = True, pad_to: int = None):
        if not meta_inf and pad_to is not None:
            raise RuntimeError("Cannot pad trajectory without meta information")
        base_tensor = torch.tensor(self.to_list(), dtype=torch.float32)
        if meta_inf:
            eos = torch.zeros(base_tensor.size(0), dtype=torch.float32).unsqueeze(-1)
            eos[-1, 0] = 1.0
            success = torch.tensor(self.success_label, dtype=torch.float32).view(1, 1).repeat(base_tensor.size(0), 1)
            traj_unpadded = torch.cat((eos, success, base_tensor), dim=-1)
            if pad_to is None:
                return traj_unpadded
            return pad_padded_sequence(traj_unpadded, pad_to)
        return base_tensor

    def smoothen_orientations(self):
        # Set first orientation to have positive w component
        if self.poses[0].orientation.q.w < 0:
            self.poses[0].orientation = Orientation(Quaternion(-self.poses[0].orientation.q))
        # Homogenize remaining orientations to avoid jumps
        for i in range(1, len(self.poses)):
            self.poses[i].orientation.smoothen(self.poses[i-1].orientation)

    def normalize_orientations(self):
        for pose in self.poses:
            pose.orientation.normalize()

    def scale(self, from_min, from_max, to_min, to_max):
        """
        Scale both poses and forces from the given range to the given range.
        :param from_min: 9D-vector (3 pose dimensions, 6 FT dimensions)
        :param from_max: 9D-vector (3 pose dimensions, 6 FT dimensions)
        :param to_min: 9D-vector (3 pose dimensions, 6 FT dimensions)
        :param to_max: 9D-vector (3 pose dimensions, 6 FT dimensions)
        :return:
        """
        for i in range(len(self.poses)):
            self.poses[i].position.scale(from_min[:3], from_max[:3], to_min[:3], to_max[:3])
            self.wrenches[i].scale(from_min[3:], from_max[3:], to_min[3:], to_max[3:])

    def transform(self, affine_transformation):
        """
        Transform each pose in the trajectory by the given transformation.
        Forces are not transformed.
        """
        for i in range(len(self.poses)):
            self.poses[i].transform(affine_transformation)

    def __len__(self):
        return len(self.poses)
