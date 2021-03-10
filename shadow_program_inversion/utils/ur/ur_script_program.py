"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import socket
from abc import ABC, abstractmethod
from queue import Empty
from typing import List

from shadow_program_inversion.common.pose import Pose
from shadow_program_inversion.common.trajectory import Trajectory
from shadow_program_inversion.experiments.contact.common import debias_forces
from shadow_program_inversion.utils.ur.rtde_recorder import RTDERecorder


class URScriptMotionCommand(ABC):
    def __init__(self, arguments: List):
        self.arguments = arguments

    def serialize(self):
        return f"{self._function_name()}({self._serialize_arguments()})"

    @abstractmethod
    def _function_name(self):
        pass

    def _serialize_arguments(self):
        return ", ".join([self._serialize_argument(arg) for arg in self.arguments])

    @staticmethod
    def _serialize_argument(arg):
        if type(arg) == Pose:
            ax, ay, az = arg.orientation.q.axis * arg.orientation.q.angle
            return f"p[{arg.position.x}, {arg.position.y}, {arg.position.z}, {ax}, {ay}, {az}]"
        else:
            return str(arg)


class URScriptProgram(object):
    def __init__(self, commands: List[URScriptMotionCommand] = None):
        self.commands = commands if commands is not None else []

    def serialize(self):
        script = "def main():\n"
        for command in self.commands:
            command_script = command.serialize()
            for line in command_script.split("\n"):
                script += "\t" + line + "\n"
        script += "end\n"
        return script

    def execute(self, ip: str, port: int = 30002):
        script = self.serialize()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ip, port))
        s.send(script.encode("utf8"))
        s.close()


class URScriptMoveLinear(URScriptMotionCommand):
    def _function_name(self):
        return "movel"


class URScriptMoveProcess(URScriptMotionCommand):
    def _function_name(self):
        return "movep"


def execute_synchronously(program, ur_ip, rec: RTDERecorder = None) -> Trajectory:
    trajectory = None
    if rec is not None:
        rec.start_recording()
    program.execute(ur_ip)
    if rec is not None:
        try:
            trajectory = rec.queue.get(timeout=10)
        except Empty:
            pass
        rec.stop_recording()
    return trajectory


def execute_move_linear(ur_ip: str, rec: RTDERecorder, approach_pose: Pose, goal_pose: Pose, vel: float,
                        acc: float) -> Trajectory:
    move_to_start = URScriptProgram([URScriptMoveLinear([approach_pose, 0.2, 0.25])])
    execute_synchronously(move_to_start, ur_ip, rec)
    prog = URScriptProgram([URScriptMoveLinear([goal_pose, vel, acc])])
    trajectory = execute_synchronously(prog, ur_ip, rec)
    if trajectory is None:
        print("Empty trajectory!!!")
        return None
    print(f"Trajectory length: {len(trajectory)}")
    trajectory = debias_forces(trajectory)
    return trajectory