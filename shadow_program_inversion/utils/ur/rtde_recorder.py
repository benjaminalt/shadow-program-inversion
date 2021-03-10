"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import logging
import os
import sys
import time
from enum import Enum
from multiprocessing import Process, Event, Queue
from typing import List

import numpy as np
from rtde import RTDE, rtde_config, RTDEException
from pyquaternion import Quaternion

from shadow_program_inversion.common.trajectory import Trajectory

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


class URRuntimeState(Enum):
    STOPPED = 1
    RUNNING = 2


def rtde_datapoint_to_tensor(state):
    datapoint = []
    position = state.actual_TCP_pose[:3]
    axangle = state.actual_TCP_pose[3:]
    orientation = Quaternion(axis=axangle, angle=np.linalg.norm(axangle))
    datapoint.extend(position)
    datapoint.extend(orientation.elements)
    datapoint.extend(state.actual_TCP_force)
    return datapoint


class RTDERecorderWorker(Process):
    def __init__(self, host: str, port: int, success_label_fn, sampling_interval: float,
                 record_event: Event, kill_event: Event, queue: Queue, respect_ur_runtime_state: bool = False):
        super(RTDERecorderWorker, self).__init__()
        self.con = RTDE(host, port)
        self.success_label_fn = success_label_fn
        self.sampling_interval = sampling_interval
        self.kill_event = kill_event
        self.record_event = record_event
        self.queue = queue
        self.respect_ur_runtime_state = respect_ur_runtime_state

    def run(self):
        conf = rtde_config.ConfigFile(os.path.join(SCRIPT_DIR, "rtde_trajectory_config.xml"))
        output_names, output_types = conf.get_recipe('out')

        self.con.connect()

        # setup recipes
        if not self.con.send_output_setup(output_names, output_types, frequency=1/self.sampling_interval):
            print('Unable to configure output', file=sys.stderr)
            sys.exit()

        # start data synchronization
        if not self.con.send_start():
            logging.error('Unable to start synchronization')
            sys.exit()

        keep_running = True
        traj = []
        while keep_running:
            try:
                state = self.con.receive()
                if state is not None:
                    # print(f"Recording: {self.record_event.is_set()}, runtime: {state.runtime_state}")
                    if self.record_event.is_set() and (not self.respect_ur_runtime_state or (state.runtime_state == URRuntimeState.RUNNING.value and self.respect_ur_runtime_state)):
                        traj.append(rtde_datapoint_to_tensor(state))
                    if len(traj) != 0 and ((not self.record_event.is_set() and not self.respect_ur_runtime_state) or (state.runtime_state == URRuntimeState.STOPPED.value and self.respect_ur_runtime_state)):
                        # I was recording and got stopped
                        trajectory = Trajectory.from_list(traj, self.success_label_fn(traj))
                        self.queue.put(trajectory)
                        print("RTDERecorderWorker::run: Recorded trajectory")
                        traj = []

            except (KeyboardInterrupt, RTDEException) as e:
                keep_running = False

            if self.kill_event.is_set():
                keep_running = False

        self.con.send_pause()
        self.con.disconnect()
        print("RTDERecorderWorker::run: Worker finished")


class RTDERecorder(object):
    def __init__(self, host: str, port: int, success_label_fn, sampling_interval: float = 0.032,
                 respect_ur_runtime_state=False):
        self.kill_event = Event()
        self.record_event = Event()
        self.queue = Queue()
        self.worker = RTDERecorderWorker(host, port, success_label_fn, sampling_interval,
                                         self.record_event, self.kill_event, self.queue, respect_ur_runtime_state)
        self.worker.start()

    def __del__(self):
        self.kill_event.set()

    def start_recording(self):
        if not self.worker.is_alive():
            self.worker.start()
            time.sleep(0.1)  # Ensure worker is up!
            self.record_event.set()
        else:
            print("RTDERecorder::start_recording: Worker already running")
            self.record_event.set()

    def stop_recording(self):
        print("RTDERecorder::stop_recording")
        self.record_event.clear()

    def trajectories(self) -> List[Trajectory]:
        if self.worker.is_alive():
            print("RTDERecorder::trajectories: Worker is still alive, stopping...")
            self.stop_recording()
        trajectories = []
        while self.queue.qsize() != 0:
            trajectories.append(self.queue.get())
        return trajectories
