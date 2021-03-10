"""
Copyright (C) 2021 ArtiMinds Robotics GmbH
"""

import os
from typing import List

from rtde import RTDE, rtde_config

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_current_joint_states(ip: str) -> List:
    con = RTDE(ip, 30004)
    conf = rtde_config.ConfigFile(os.path.join(SCRIPT_DIR, "rtde_joint_state_config.xml"))
    output_names, output_types = conf.get_recipe('out')
    con.connect()
    if not con.send_output_setup(output_names, output_types):
        raise RuntimeError('Unable to configure output')
    if not con.send_start():
        raise RuntimeError('Unable to start synchronization')
    state = con.receive()
    if state is None:
        raise RuntimeError("Could not read state from robot")
    joint_states = state.actual_q
    con.send_pause()
    con.disconnect()
    return joint_states
