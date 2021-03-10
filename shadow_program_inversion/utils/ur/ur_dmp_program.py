"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""
import time
from abc import ABC, abstractmethod
from queue import Empty
from threading import Lock
from typing import List, Tuple, Iterable

import numpy as np
import roslibpy

import torch
import pyrobolearn as prl
from pyrobolearn.models.dmp.dmpytorch import DiscreteDMP

from shadow_program_inversion.common.orientation import Orientation
from shadow_program_inversion.common.pose import Pose
from shadow_program_inversion.common.trajectory import Trajectory

from shadow_program_inversion.priors.path_generator import PathGeneratorMoveLinear
from shadow_program_inversion.priors.static_prior import TrajectoryProperties, StaticPrior
from shadow_program_inversion.utils.sequence_utils import unpad_padded_sequence
from shadow_program_inversion.utils.ur.rtde_recorder import RTDERecorder

MAX_TRAJECTORY_LEN = 250


def servo_sim_robot_to_joint_position(robot: prl.robots.Robot, world: prl.worlds.World, joint_position: List):
    robot.set_joint_positions(joint_position)
    while True:
        current_pos = robot.get_joint_positions()
        reached = True
        for i in range(len(current_pos)):
            reached &= abs(joint_position[i] - current_pos[i]) < 0.0001
        if reached:
            break
        world.step(sleep_dt=1. / 240)


class URDMP(ABC):
    def __init__(self, fixed_orientation: Orientation, sampling_interval: float):
        self.sampling_interval = sampling_interval
        self.fixed_orientation = fixed_orientation

    @abstractmethod
    def rollout(self) -> torch.Tensor:
        pass

    def rollout_with_ik(self, robot: prl.robots.Robot, world: prl.worlds.World,
                        initial_joint_position: List) -> Tuple[List, List]:
        states = []
        poses = []

        servo_sim_robot_to_joint_position(robot, world, initial_joint_position)
        joint_pos = robot.get_joint_positions()
        print(f"Simulated joint positions: {joint_pos}")
        ee_link_id = robot.get_link_ids("tool1")
        sim_tcp_pose = robot.get_link_poses(ee_link_id)
        print(f"Simulated TCP pose: {sim_tcp_pose}")
        pos_vel_acc = zip(*self.rollout())
        for pos, vel, acc in pos_vel_acc:
            joint_state = robot.calculate_inverse_kinematics(ee_link_id, position=pos.tolist(),
                                                             orientation=self.fixed_orientation.to_qxyzw(),
                                                             max_iters=250)
            states.append(joint_state.tolist())
            poses.append(pos.tolist())
        return states, poses


class URLinearMotionDMP(URDMP):
    def __init__(self, start_pose: Pose, goal_pose: Pose, tau: float, sampling_interval: float = 0.032):
        super(URLinearMotionDMP, self).__init__(start_pose.orientation, sampling_interval)
        # Learn weights from ARTM simulator
        self.dmp = DiscreteDMP(num_dmps=3, num_basis=20)
        sim = StaticPrior(PathGeneratorMoveLinear(), sampling_interval=sampling_interval, multiproc=False,
                              trajectory_properties=TrajectoryProperties(wrench=False, gripper=False))
        point_to = torch.as_tensor(goal_pose.parameters(), dtype=torch.float32)
        point_start = torch.as_tensor(start_pose.parameters(), dtype=torch.float32)
        vel = torch.tensor(0.05)
        acc = torch.tensor(0.05)
        traj = sim.simulate(torch.cat((point_to, torch.stack((vel, acc))), dim=-1),
                            point_start, max_trajectory_len=MAX_TRAJECTORY_LEN, cache=False).squeeze()
        traj_unpadded = unpad_padded_sequence(traj)
        y_target = traj_unpadded[:, 2:5]
        y_target = y_target.transpose(0, 1).numpy()
        self.dmp.imitate(y_target)
        self.tau = tau
        self.goal_pose = goal_pose
        # fig, axes = plt.subplots(3, 1)
        # for i in range(3):
        #     axes[i].plot(range(len(y_target[i])), y_target[i], color="green")
        # for tau in [0.5, 1.0, 1.5, 2.0]:
        #     y, dy, ddy = self.dmp.rollout(tau=tau)
        #     for i in range(3):
        #         axes[i].plot(range(len(y[i])), y[i].detach(), color="red")
        # y, dy, ddy = self.dmp.rollout(tau=self.tau)
        # for i in range(3):
        #     axes[i].plot(range(len(y[i])), y[i].detach(), color="black")
        # plt.grid(True, axis="both")
        # plt.show()

    def rollout(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, dy, ddy = self.dmp.rollout(tau=self.tau)
        return y.transpose(0, 1), dy.transpose(0, 1), ddy.transpose(0, 1)


class DMPDataCollector(object):
    def __init__(self, world: prl.worlds.World, robot: prl.robots.Robot):
        self.robot = robot
        self.world = world
        # Rosbridge client
        self.ros_client = roslibpy.Ros(host='localhost', port=9090)
        self.ros_client.run()
        self.action_client = roslibpy.actionlib.ActionClient(self.ros_client,
                                                        '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                                                        'control_msgs/FollowJointTrajectoryAction')
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                            "wrist_2_joint", "wrist_3_joint"]
        # Subscribe to joint states
        self.joint_state_listener = roslibpy.Topic(self.ros_client, "/joint_states", "sensor_msgs/JointState")
        self.current_joint_states = None
        self.joint_state_lock = Lock()
        self.joint_state_listener.subscribe(self.on_joint_state_received)

    def __del__(self):
        self.ros_client.terminate()

    def execute_dmp(self, dmp: URDMP, initial_joint_position: List[float]):
        states, poses = dmp.rollout_with_ik(self.robot, self.world, initial_joint_position)
        times = np.arange(0.0, len(states) * dmp.sampling_interval, dmp.sampling_interval)
        self.execute_joint_trajectory(states, times)

    def execute_joint_trajectory(self, joints: List[List[float]], times: Iterable[float]):
        durations = [seconds_to_duration(seconds) for seconds in times]
        points = []
        for i in range(len(joints)):
            points.append({
                "positions": joints[i],
                "velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "accelerations": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "time_from_start": durations[i]
            })
        goal = roslibpy.actionlib.Goal(self.action_client, roslibpy.Message({
            "trajectory": {
                "header": {
                    "stamp": 1.0
                },
                "joint_names": self.joint_names,
                "points": points,
            }
        }))
        goal.send()
        res = goal.wait()
        assert res["error_code"] == 0

    def move_to_pose(self, goal_pose: Pose):
        ee_link_id = self.robot.get_link_ids("tool1")

        # Align simulated robot with real robot
        servo_sim_robot_to_joint_position(self.robot, self.world, self.get_current_joint_pos())

        # Calculate IK and move there
        goal_state = self.robot.calculate_inverse_kinematics(ee_link_id,
                                                             position=goal_pose.position.parameters(),
                                                             orientation=goal_pose.orientation.to_qxyzw(),
                                                             max_iters=250)
        self.move_to_state(goal_state.tolist())

    def move_to_state(self, goal_state: List[float]):
        self.execute_joint_trajectory([goal_state], [1.0])

    def on_joint_state_received(self, message):
        with self.joint_state_lock:
            self.current_joint_states = message

    def get_current_joint_pos(self) -> List[float]:
        while self.current_joint_states is None:
            time.sleep(0.1)
        with self.joint_state_lock:
            current_joints = []
            for joint_name in self.joint_names:
                joint_idx = self.current_joint_states["name"].index(joint_name)
                current_joints.append(self.current_joint_states["position"][joint_idx])
            return current_joints


def seconds_to_duration(seconds):
    secs = int(seconds)
    remainder = seconds - secs
    nanosecs = int(remainder * 1e+9)
    return {"secs": secs, "nsecs": nanosecs}


def execute_move_linear(data_collector: DMPDataCollector, rec: RTDERecorder, start_pose: Pose, goal_pose: Pose, tau: float) -> Trajectory:
    data_collector.move_to_pose(start_pose)
    dmp = URLinearMotionDMP(start_pose, goal_pose, tau=tau, sampling_interval=0.016)
    rec.start_recording()
    data_collector.execute_dmp(dmp, data_collector.get_current_joint_pos())
    rec.stop_recording()
    try:
        return rec.queue.get(timeout=10)
    except Empty:
        print("Could not receive trajectory")
        return None