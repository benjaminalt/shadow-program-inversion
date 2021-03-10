"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""
import math

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform.rotation import Rotation


def quaternion_from_euler_zyx(rx, ry, rz, degrees=False):
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_Angles_to_Quaternion_Conversion
    rx, ry, rz are floats (unit: rad) representing an orientation in Euler angles with ZYX rotation order (Tait-Bryan)
    :return:
    """
    heading = np.deg2rad(rz) if degrees else rz
    attitude = np.deg2rad(ry) if degrees else ry
    bank = np.deg2rad(rx) if degrees else rx
    ch = math.cos(heading / 2)
    sh = math.sin(heading / 2)
    ca = math.cos(attitude / 2)
    sa = math.sin(attitude / 2)
    cb = math.cos(bank / 2)
    sb = math.sin(bank / 2)
    w = cb * ca * ch + sb * sa * sh
    x = sb * ca * ch - cb * sa * sh
    y = cb * sa * ch + sb * ca * sh
    z = cb * ca * sh - sb * sa * ch
    return Quaternion(w=w, x=x, y=y, z=z).normalised


def euler_zyx_from_quaternion(q):
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_Angles_Conversion
    :param q: Quaternion
    :return:
    """
    bank = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x ** 2 + q.y ** 2))
    attitude = math.asin(2 * (q.w * q.y - q.z * q.x))
    heading = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
    return [bank, attitude, heading]


def rotation_matrix_from_quaternion(q):
    """
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    :param q: Quaternion
    :return:
    """
    qn = q.normalised
    return np.array(Rotation.from_quat([qn.x, qn.y, qn.z, qn.w]).as_matrix())


def quaternion_from_rotation_matrix(mat):
    """
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    :param q: 3x3 rotation matrix
    :return: Unit quaternion
    """
    q = Rotation.from_dcm(mat).as_quat()  # xyzw
    return Quaternion(w=q[3], x=q[0], y=q[1], z=q[2]).normalised  # wxyz
