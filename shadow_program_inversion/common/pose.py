"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import numpy as np

from shadow_program_inversion.common.orientation import Orientation
from shadow_program_inversion.common.position import Position
from shadow_program_inversion.utils.conversions import quaternion_from_rotation_matrix, rotation_matrix_from_quaternion


class Pose(object):
    def __init__(self, position, orientation, reference=None):
        self.position = position
        self.orientation = orientation
        self.reference = reference

    @staticmethod
    def make_absolute_from_xyz_rxryrz(xyz_rxryrz, degrees=False):
        return Pose(Position(*xyz_rxryrz[:3]),
                    Orientation.from_euler_zyx(*xyz_rxryrz[3:6], degrees),
                    reference=None)

    @staticmethod
    def make_relative_from_xyz_rxryrz(xyz_rxryrz, reference):
        relative_position = Position(*xyz_rxryrz[:3]).to_relative(reference.position)
        relative_orientation = Orientation.from_euler_zyx(*xyz_rxryrz[3:6]).to_relative(reference.orientation)
        return Pose(relative_position, relative_orientation, reference)

    def to_absolute_xyz_rxryrz(self):
        if self.is_relative():
            absolute_position = self.position.to_absolute(self.reference.position).to_xyz()
            absolute_orientation = self.orientation.to_absolute(self.reference.orientation).to_euler_zyx()
        else:
            absolute_position = self.position.to_xyz()
            absolute_orientation = self.orientation.to_euler_zyx()
        return absolute_position + absolute_orientation

    def to_absolute_xyz_quaternion(self):
        if self.is_relative():
            absolute_position = self.position.to_absolute(self.reference.position).to_xyz()
            absolute_orientation = self.orientation.to_absolute(self.reference.orientation).parameters()
        else:
            absolute_position = self.position.to_xyz()
            absolute_orientation = self.orientation.parameters()
        return absolute_position + absolute_orientation

    @staticmethod
    def make_absolute_from_xyz_quaternion(xyz_quaternion):
        return Pose(Position(*xyz_quaternion[:3]),
                    Orientation.from_parameters(xyz_quaternion[3:7]),
                    reference=None)

    @staticmethod
    def make_relative_from_xyz_quaternion(xyz_quaternion, reference):
        return Pose(Position(*xyz_quaternion[:3]).to_relative(reference.position),
                    Orientation.from_parameters(xyz_quaternion[3:7]).to_relative(reference.orientation),
                    reference=reference)

    def is_relative(self):
        return self.reference is not None


    @staticmethod
    def from_parameters(parameters, reference=None):
        return Pose(Position.from_parameters(parameters[:3]),
                    Orientation.from_parameters(parameters[3:7]),
                    reference)

    def parameters(self):
        return self.position.parameters() + self.orientation.parameters()

    @staticmethod
    def parameter_names():
        return Position.parameter_names() + Orientation.parameter_names()

    # def scale(self, from_min, from_max, to_min, to_max):
    #     self.position.scale(from_min[:3], from_max[:3], to_min[:3], to_max[:3])
    #     # self.orientation.scale(from_min[3:7], from_max[3:7], to_min[3:7], to_max[3:7])

    def transform(self, affine_transformation):
        result_affine = np.matmul(affine_transformation, self.to_affine())
        result = Pose.from_affine(result_affine, None)
        self.position = result.position
        self.orientation = result.orientation

    @staticmethod
    def from_affine(affine, reference):
        position = Position(*affine[:3,3])
        orientation = Orientation(quaternion_from_rotation_matrix(affine[:3,:3]))
        return Pose(position, orientation, reference)

    def to_affine(self):
        # Rotation matrix from quaternion
        rotation = rotation_matrix_from_quaternion(self.orientation.q)
        translation = np.array(self.position.parameters()).reshape((3,1))
        top = np.hstack([rotation, translation])
        bottom = np.expand_dims([0,0,0,1], 0)
        return np.vstack([top, bottom])

    def __str__(self):
        return str(self.parameters())

    def __repr__(self):
        return self.parameters()
