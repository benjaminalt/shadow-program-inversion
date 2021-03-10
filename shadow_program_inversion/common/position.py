"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""
from pytorch_utils import transformations


class Position(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_relative(self, reference_position):
        return Position(
            self.x - reference_position.x,
            self.y - reference_position.y,
            self.z - reference_position.z
        )
    
    def to_absolute(self, reference_position):
        return Position(
            reference_position.x + self.x,
            reference_position.y + self.y,
            reference_position.z + self.z
        )

    def to_xyz(self):
        return [self.x, self.y, self.z]
    
    def parameters(self):
        return [self.x, self.y, self.z]

    @staticmethod
    def from_parameters(parameters):
        return Position(*parameters)

    @staticmethod
    def parameter_names():
        return ["x", "y", "z"]

    def scale(self, from_min, from_max, to_min, to_max):
        self.x = transformations.scale(self.x, from_min[0], from_max[0], to_min[0], to_max[0])
        self.y = transformations.scale(self.y, from_min[1], from_max[1], to_min[1], to_max[1])
        self.z = transformations.scale(self.z, from_min[2], from_max[2], to_min[2], to_max[2])
