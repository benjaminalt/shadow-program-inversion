"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

from pyquaternion.quaternion import Quaternion
import numpy as np
from pytorch_utils import transformations

from shadow_program_inversion.utils.conversions import quaternion_from_euler_zyx, euler_zyx_from_quaternion


class Orientation(object):

    def __init__(self, q):
        self.q = q

    @staticmethod
    def from_parameters(parameters):
        return Orientation(Quaternion(w=parameters[0], x=parameters[1], y=parameters[2], z=parameters[3]))

    def parameters(self):
        return [self.q.w, self.q.x, self.q.y, self.q.z]

    @staticmethod
    def from_euler_zyx(rx, ry, rz, degrees=False):
        return Orientation(quaternion_from_euler_zyx(rx, ry, rz, degrees))

    def to_euler_zyx(self):
        return euler_zyx_from_quaternion(self.q)

    def to_qxyzw(self):
        return [self.q.x, self.q.y, self.q.z, self.q.w]

    def to_relative(self, reference_orientation):
        return Orientation(self.q * reference_orientation.q.inverse)

    def to_absolute(self, reference_orientation):
        return Orientation(self.q * reference_orientation.q)

    @staticmethod
    def parameter_names():
        return ["qw", "qx", "qy", "qz"]

    def normalize(self):
        self.q = self.q.normalised

    def scale(self, from_min, from_max, to_min, to_max):
        w = transformations.scale(self.q.w, from_min[0], from_max[0], to_min[0], to_max[0])
        x = transformations.scale(self.q.x, from_min[1], from_max[1], to_min[1], to_max[1])
        y = transformations.scale(self.q.y, from_min[2], from_max[2], to_min[2], to_max[2])
        z = transformations.scale(self.q.z, from_min[3], from_max[3], to_min[3], to_max[3])
        self.q = Quaternion(w=w, x=x, y=y, z=z)

    def smoothen(self, other):
        if np.linalg.norm((other.q - self.q).elements) > np.linalg.norm((other.q + self.q).elements):
            self.q = -self.q
