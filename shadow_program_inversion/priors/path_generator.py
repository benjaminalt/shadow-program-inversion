"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

from abc import ABC, abstractmethod

import torch
from pytorch_utils.transformations import affine_transform, absolute_to_relative, axis_angle_from_rotation, \
    rotation_from_axis_angle, quaternion_distance, pose_to_affine, affine_to_pose


class DifferentiablePathGenerator(ABC):
    @abstractmethod
    def simulate(self, inputs: torch.Tensor, point_from: torch.Tensor) -> torch.Tensor:
        pass


class PathGeneratorMoveLinearRelative(DifferentiablePathGenerator):
    def simulate(self, inputs: torch.Tensor, point_from: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: point_to|velocity|acceleration. point_to here encodes a relative motion
        :param point_from:
        :return:
        """
        path = self.generate_path(inputs)
        if len(point_from.size()) == 1:
            point_from = point_from.unsqueeze(0)
        path_absolute = affine_transform(point_from[:, :7], path)
        # path_absolute = torch.cat((point_from, path_absolute), dim=0)
        return path_absolute

    def generate_path(self, inputs: torch.Tensor) -> torch.Tensor:
        point_to_relative = inputs[:7]
        vel, acc = inputs[7:]
        points = torch.stack((torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=point_to_relative.dtype,
                                           device=point_to_relative.device), point_to_relative[:7]))
        path = MotionTrajectoryOptimizerTimeCartesian(vel, acc).optimize(points)
        return path


class PathGeneratorMoveLinear(DifferentiablePathGenerator):
    def simulate(self, inputs: torch.Tensor, point_from: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: point_to|velocity|acceleration
        :param point_from:
        :return:
        """
        point_to_relative = absolute_to_relative(inputs[:7].unsqueeze(0), point_from[:7].unsqueeze(0)).squeeze()
        path = self.generate_path(torch.cat((point_to_relative, inputs[7:]), dim=-1))
        if len(point_from.size()) == 1:
            point_from = point_from.unsqueeze(0)
        path_absolute = affine_transform(point_from[:, :7], path)
        # path_absolute = torch.cat((point_from, path_absolute), dim=0)
        return path_absolute

    def generate_path(self, inputs: torch.Tensor) -> torch.Tensor:
        return PathGeneratorMoveLinearRelative().generate_path(inputs)


class VelocityProfileTrapezoidal(object):
    def __init__(self, max_vel, max_acc):
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.a1, self.a2, self.a3, self.b1, self.b2, self.b3, self.c1, self.c2, self.c3 = [0] * 9
        self.duration = 0
        self.t1 = 0
        self.t2 = 0
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.start_pos = 0
        self.end_pos = 0

    def set_profile(self, pos1: float, pos2: float):
        self.start_pos = pos1
        self.end_pos = pos2
        self.t1 = self.max_vel / self.max_acc
        s = torch.sign(self.end_pos - self.start_pos)
        deltax1 = s * self.max_acc * self.t1 ** 2 / 2.0
        deltat = (self.end_pos - self.start_pos - 2.0 * deltax1) / (s * self.max_vel)
        if deltat > 0:
            self.duration = 2 * self.t1 + deltat
            self.t2 = self.duration - self.t1
        else:
            self.t1 = torch.sqrt((self.end_pos - self.start_pos) / s / self.max_acc)
            self.duration = self.t1 * 2.0
            self.t2 = self.t1
        self.a3 = s * self.max_acc / 2.0
        self.a2 = 0
        self.a1 = self.start_pos
        self.b3 = 0
        self.b2 = self.a2 + 2 * self.a3 * self.t1 - 2 * self.b3 * self.t1
        self.b1 = self.a1 + self.t1 * (self.a2 + self.a3 * self.t1) - self.t1 * (self.b2 + self.t1 * self.b3)
        self.c3 = -s * self.max_acc / 2.0
        self.c2 = self.b2 + 2 * self.b3 * self.t2 - 2.0 * self.c3 * self.t2
        self.c1 = self.b1 + self.t2 * (self.b2 + self.b3 * self.t2) - self.t2 * (self.c2 + self.t2 * self.c3)

    def pos(self, time: torch.Tensor) -> float:
        if time < 0:
            return self.start_pos
        elif time < self.t1:
            return self.a1 + time * (self.a2 + self.a3 * time)
        elif time < self.t2:
            return self.b1 + time * (self.b2 + self.b3 * time)
        elif time <= self.duration:
            return self.c1 + time * (self.c2 + self.c3 * time)
        return self.end_pos


class PathComposite(object):
    def __init__(self):
        self.path_length = 0
        self.cached_starts = 0
        self.cached_ends = 0
        self.cached_index = 0
        self.gv = []
        self.dv = []

    def add(self, geom, aggregate=True):
        self.path_length += geom.path_length
        self.dv.append(self.path_length)
        self.gv.append((geom, aggregate))

    def pos(self, s):
        s = self.lookup(s)
        return self.gv[self.cached_index][0].pos(s)

    def lookup(self, s):
        assert s >= -1e-12
        # assert s <= self.path_length + 1e-12
        if self.cached_starts <= s <= self.cached_ends:
            return s - self.cached_starts
        previous_s = 0
        for i in range(len(self.dv)):
            if s <= self.dv[i] or i == len(self.dv) - 1:
                self.cached_index = i
                self.cached_starts = previous_s
                self.cached_ends = self.dv[i]
                return s - previous_s
            previous_s = self.dv[i]
        return 0


class Frame(object):
    def __init__(self, homogeneous_matrix: torch.Tensor):
        self.M = homogeneous_matrix

    def p(self):
        return self.M[:3, -1]

    def rotation_matrix(self):
        return self.M[:3, :3]

    @staticmethod
    def from_components(ori: torch.Tensor, pos: torch.Tensor):
        """
        :param ori: 3x3 rotation matrix
        :param pos: 3D position vector
        :return: Frame
        """
        top_block = torch.cat((ori, pos.view(3, 1)), dim=1)
        return Frame(torch.cat((top_block, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=ori.device)), dim=0))


class RotationalInterpolation(object):
    def __init__(self):
        self.r_base_start = None
        self.r_base_end = None
        self.r_start_end_axis = None
        self.angle = None

    def set_start_end(self, start: torch.Tensor, end: torch.Tensor):
        self.r_base_start = start
        self.r_base_end = end
        r_start_end = start.inverse().matmul(end)
        self.r_start_end_axis, self.angle = axis_angle_from_rotation(r_start_end)

    def pos(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.r_base_start, rotation_from_axis_angle(self.r_start_end_axis, theta))


class PathLine(object):
    def __init__(self, start_pos: Frame, end_pos: Frame, orient: RotationalInterpolation, eqradius: float,
                 aggregate=True):
        self.orient = orient
        self.v_base_start = start_pos.p()
        self.v_base_end = end_pos.p()
        self.eqradius = eqradius
        self.aggregate = aggregate

        diff = self.v_base_end - self.v_base_start
        dist = torch.norm(diff)
        self.v_start_end = diff / dist

        self.orient.set_start_end(start_pos.rotation_matrix(), end_pos.rotation_matrix())
        alpha = self.orient.angle

        if alpha != 0 and alpha * eqradius > dist:
            # rotational_interpolation is the limitation
            self.path_length = alpha * eqradius
            self.scale_rot = 1 / eqradius
            self.scale_lin = dist / self.path_length
        elif dist != 0:
            # translation is the limitation
            self.path_length = dist
            self.scale_rot = alpha / self.path_length
            self.scale_lin = 1
        else:
            # both were zero
            self.path_length = torch.zeros(1, dtype=dist.dtype, device=dist.device)
            self.scale_rot = 1
            self.scale_lin = 1

    def pos(self, s: torch.Tensor) -> Frame:
        return Frame.from_components(self.orient.pos(s * self.scale_rot),
                                     self.v_base_start + self.v_start_end * s * self.scale_lin)


class TrajectorySegment(object):
    def __init__(self, path, velocity_profile):
        self.velocity_profile = velocity_profile
        self.path = path

    def pos(self, time) -> Frame:
        return self.path.pos(self.velocity_profile.pos(time))

    def duration(self):
        return self.velocity_profile.duration


class MotionTrajectoryOptimizerTimeCartesian(object):
    def __init__(self, max_vel_pos, max_acc_pos, sampling_interval=0.016):
        self.sampling_interval = sampling_interval
        self.max_vel_pos = max_vel_pos
        self.max_acc_pos = max_acc_pos

    def optimize(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points: A sampled path of 7D points
        :return:
        """
        if points[0, :3].allclose(points[1, :3]) and quaternion_distance(points[0, 3:7], points[1, 3:7]) < 0.1:
            return points      # Degenerated path just consisting of start and end points to avoid NaNs
        path_composite = PathComposite()
        for i in range(points.size(0) - 1):
            point = Frame(pose_to_affine(points[i].unsqueeze(0)).squeeze())
            next_point = Frame(pose_to_affine(points[i + 1].unsqueeze(0)).squeeze())
            path_composite.add(PathLine(point, next_point, RotationalInterpolation(), eqradius=0.25))
        vel_prof = VelocityProfileTrapezoidal(self.max_vel_pos, self.max_acc_pos)
        vel_prof.set_profile(0, path_composite.path_length)
        traject = TrajectorySegment(path_composite, vel_prof)
        dur = traject.duration()
        steps = dur / self.sampling_interval + 1
        steps_rounded = torch.round(steps).int()
        path = []
        for i in range(steps_rounded + 1):
            t = dur * i / steps
            current_pose = traject.pos(t)
            # if current_pose.M.requires_grad:
            #     current_pose.M.register_hook(lambda x: print(f"MotionTrajectoryOptimizerTimeCartesian::optimize: current_pose.M: {x}"))
            path.append(current_pose.M)
        if len(path) == 0:
            return points
        return affine_to_pose(torch.stack(path))