import numpy as np


def unpack_planner_request(msg):
    q_0 = np.array(msg.q_0)
    q_dot_0 = np.array(msg.q_dot_0)
    q_ddot_0 = np.array(msg.q_ddot_0)
    q_d = np.array(msg.q_d)
    q_dot_d = np.array(msg.q_dot_d)
    q_ddot_d = np.array(msg.q_ddot_d)
    return q_0, q_dot_0, q_ddot_0, q_d, q_dot_d, q_ddot_d
