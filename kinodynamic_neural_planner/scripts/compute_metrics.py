import os
import os.path
import pickle
from copy import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
import rosbag
from glob import glob

import pinocchio as pino
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as Rot

from manifold_planning.utils.manipulator import Iiwa
from manifold_planning.utils.feasibility import check_if_plan_valid, compute_cartesian_losses

from manifold_planning.utils.constants import Table1, Table2, Cup, Limits

root_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(root_dir)

robot_file = os.path.join(root_dir, "manifold_planning", "iiwa_cup.urdf")
pino_model = pino.buildModelFromUrdf(robot_file)
pino_data = pino_model.createData()
joint_id = pino_model.getFrameId("F_link_cup")

package_path = os.path.join(os.path.dirname(__file__), "..")
file_name = os.path.join(package_path, "data/kinodynamic7fixed_12kg_validated/test/data.tsv")
data = np.loadtxt(file_name, delimiter="\t").astype(np.float32)#[:3]


def compute_object_coordinates(xyz, R):
    h = Cup.height
    w = Cup.width
    xyz_cuboid = np.array([
        # corners
        [w, w, h], [w, w, -h], [w, -w, h], [w, -w, -h],
        [-w, w, h], [-w, w, -h], [-w, -w, h], [-w, -w, -h],
        ## middle points on the edges
        # [w, w, 0], [w, -w, 0], [-w, w, 0], [-w, -w, 0],
        # [w, 0, h], [w, 0, -h], [-w, 0, h], [-w, 0, -h],
        # [0, w, h], [0, w, -h], [0, -w, h], [0, -w, -h],
        ## middle points on the faces
        # [w, 0, 0], [-w, 0, 0],
        # [0, w, 0], [0, -w, 0],
        # [0, 0, h], [0, 0, -h],
    ])[np.newaxis, :, :, np.newaxis]
    xyz_object = xyz[:, np.newaxis] + (R[:, np.newaxis] @ xyz_cuboid)[..., 0]
    return xyz_object


def cal_d(fs, dts):
    dfs = np.zeros_like(fs)
    dfs[1: -1] = (fs[2:, :] - fs[:-2, :]) / (dts[2:] - dts[:-2])
    dfs[0] = dfs[1]
    dfs[-1] = dfs[-2]
    return dfs


def rnea(q, dq, ddq):
    taus = []
    for i in range(len(q)):
        taus.append(pino.rnea(pino_model, pino_data, q[i], dq[i], ddq[i]))
    return np.array(taus)


def get_vel_acc(t, q_m):
    # Position
    bv, av = butter(6, 40, fs=1000)
    b, a = butter(6, 40, fs=1000)
    bt, at = butter(6, 30, fs=1000)
    # bv, av = butter(6, 4, fs=1000)
    # b, a = butter(6, 4, fs=1000)
    q_m_filter = q_m

    # Velocity
    dq_m = cal_d(q_m, t)
    dq_m_filter = filtfilt(bv, av, dq_m.copy(), axis=0)

    # Acceleration
    ddq_m = cal_d(dq_m_filter, t)
    ddq_m_filter = filtfilt(b, a, ddq_m.copy(), axis=0)

    torque = rnea(q_m.astype(np.float32), dq_m.astype(np.float32), ddq_m.astype(np.float32))
    # Torque
    tau_m = torque
    tau_m_filter = filtfilt(bt, at, tau_m.copy(), axis=0)

    return q_m_filter, dq_m_filter, ddq_m_filter, tau_m_filter, q_m, dq_m, ddq_m, tau_m


def compute_vel_acc_tau(t, q, qd, qd_dot, qd_ddot):
    desired_torque = rnea(qd, qd_dot, qd_ddot)
    q, dq, ddq, torque, q_, dq_, ddq_, torque_ = get_vel_acc(t[:, np.newaxis], q)
    return q, dq, ddq, torque, qd, qd_dot, qd_ddot, desired_torque


def forwardKinematics(q):
    xyz = []
    R = []
    for i in range(len(q)):
        pino.forwardKinematics(pino_model, pino_data, q[i])
        pino.updateFramePlacements(pino_model, pino_data)
        xyz_pino = pino_data.oMf[joint_id].translation
        R_pino = pino_data.oMf[joint_id].rotation
        xyz.append(copy(xyz_pino))
        R.append(copy(R_pino))
    xyz = np.stack(xyz)
    R = np.stack(R)
    return xyz, R


def compute_vertical_constraints(q):
    xyz, R = forwardKinematics(q)
    vertical_loss = 1. - R[:, 2, 2]
    angles = []
    for i in range(len(R)):
        angles.append(Rot.from_matrix(R[i]).as_euler("xyz"))
    return vertical_loss, np.array(angles)


def if_valid_box_constraints(q, zh1, zh2):
    allowed_violation = 0.02
    xyz, R = forwardKinematics(q)

    def dist2box(xyz, xl, yl, zl, xh, yh, zh):
        l = np.stack([xl, yl, zl], axis=-1)
        h = np.stack([xh, yh, zh], axis=-1)
        xyz_dist = np.max(np.stack([l - xyz, np.zeros_like(xyz), xyz - h], axis=-1), axis=-1)
        dist = np.sqrt(np.sum(np.square(xyz_dist), axis=-1) + 1e-8)
        return dist

    def inside_box(xyz, xl, yl, zl, xh, yh, zh):
        pxl = xyz[..., 0] > xl + allowed_violation
        pxh = xyz[..., 0] < xh - allowed_violation
        pyl = xyz[..., 1] > yl + allowed_violation
        pyh = xyz[..., 1] < yh - allowed_violation
        pzl = xyz[..., 2] > zl + allowed_violation
        pzh = xyz[..., 2] < zh - allowed_violation
        return np.all(np.stack([pxl, pxh, pyl, pyh, pzl, pzh], axis=-1), axis=-1)

    robot_dist2box_1 = dist2box(xyz, Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh, zh1 - Cup.height)
    robot_dist2box_2 = dist2box(xyz, Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh, zh2 - Cup.height)
    inside_box_1 = inside_box(xyz, Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh, zh1 - Cup.height)
    inside_box_2 = inside_box(xyz, Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh, zh2 - Cup.height)

    xyz_object = compute_object_coordinates(xyz, R)

    object_inside_box_1 = inside_box(xyz_object, Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh,
                                     zh1 - Cup.height)
    object_inside_box_2 = inside_box(xyz_object, Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh,
                                     zh2 - Cup.height)

    return not np.any(inside_box_1) and not np.any(inside_box_2) and \
           np.all(robot_dist2box_1 > 0.15 - allowed_violation) and \
           np.all(robot_dist2box_2 > 0.15 - allowed_violation) and \
           not np.any(object_inside_box_1) and not np.any(object_inside_box_2)


def read_bag(bag):
    time = []
    joint_state_desired = []
    joint_state_actual = []
    joint_state_error = []
    planning_time = None
    success = None
    planned_motion_time = None

    for topic, msg, t_bag in bag.read_messages():
        if topic in ["/bspline_ff_kino_joint_trajectory_controller/state"]:
            n_joints = len(msg.joint_names)
            time.append(msg.header.stamp.to_sec())
            joint_state_desired.append(np.concatenate(
                [msg.desired.positions, msg.desired.velocities, msg.desired.accelerations, msg.desired.effort]))
            joint_state_actual.append(np.concatenate(
                [msg.actual.positions, msg.actual.velocities, msg.actual.accelerations, msg.actual.effort]))
            joint_state_error.append(np.concatenate(
                [msg.error.positions, msg.error.velocities, msg.error.accelerations, msg.error.effort]))
        elif topic == "/neural_planner/status":
            planning_time = msg.planning_time
            planned_motion_time = msg.planned_motion_time
            success = msg.success
    desired = np.array(joint_state_desired).astype(np.float32)
    actual = np.array(joint_state_actual).astype(np.float32)
    error = np.array(joint_state_error).astype(np.float32)
    t = np.array(time)
    t -= t[0]
    t = t.astype(np.float32)
    return dict(t=t, actual=actual, desired=desired, error=error, planning_time=planning_time, success=success,
                planned_motion_time=planned_motion_time)


def compute_metrics(bag_path):
    # bag_path = os.path.join(package_dir, "bags", bag_path)
    # bag_path = os.path.join(package_dir, "bags/ours_nn/K9.bag")
    bag_file = rosbag.Bag(bag_path)
    i = int(bag_path.split("/")[-1][:-4])
    data_i = data[i]
    zh1 = data_i[16]
    zh2 = data_i[19]

    result = {}

    bag_dict = read_bag(bag_file)
    q, dq, ddq, torque, qd, qd_dot, qd_ddot, torqued = compute_vel_acc_tau(bag_dict["t"],
                                                                           bag_dict["actual"][:, :7],
                                                                           bag_dict["desired"][:, :7],
                                                                           bag_dict["desired"][:, 7:14],
                                                                           bag_dict["desired"][:, 14:21])
    #plt.plot(bag_dict["t"], qd_dot[..., :2], 'r')
    # plt.plot(bag_dict["t"], dq[..., :2], 'm')
    moving = np.linalg.norm(qd_dot, axis=-1) > 5e-2
    if not np.any(moving):
        result["valid"] = 0
        result["finished"] = 0
        result["planning_time"] = bag_dict["planning_time"]
        return result
    q = q[moving]
    dq = dq[moving]
    ddq = ddq[moving]
    torque = torque[moving]
    qd = qd[moving]
    qd_dot = qd_dot[moving]
    qd_ddot = qd_ddot[moving]
    torqued = torqued[moving]
    t = bag_dict["t"][moving]

    #plt.plot(t, qd_dot[..., :2], 'b')
    #plt.show()

    vertical_loss, angles = compute_vertical_constraints(q)

    box_constraints = if_valid_box_constraints(q, zh1, zh2)
    vertical_constraints = np.all(vertical_loss < 0.05)
    alpha_beta = np.sum(np.abs(angles[:, :2]), axis=-1)

    torque_limits = 1.0 * np.array([320, 320, 176, 176, 110, 40, 40], dtype=np.float32)[np.newaxis]
    q_dot_limits = 1.0 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562], dtype=np.float32)[np.newaxis]
    q_ddot_limits = 10. * q_dot_limits
    torque_constraints = np.all(np.abs(torque) < torque_limits)
    dq_constraints = np.all(np.abs(dq) < q_dot_limits)
    ddq_constraints = np.all(np.abs(ddq) < q_ddot_limits)
    #for i in range(7):
    #    plt.subplot(331 + i)
    #    plt.plot(np.abs(dq[:, i]))
    #    plt.plot([0, dq.shape[0]], [q_dot_limits[0, i], q_dot_limits[0, i]])
    #plt.show()
    #for i in range(7):
    #    plt.subplot(331 + i)
    #    plt.plot(np.abs(ddq[:, i]))
    #    plt.plot([0, ddq.shape[0]], [q_ddot_limits[0, i], q_ddot_limits[0, i]])
    #plt.show()
    #for i in range(7):
    #    plt.subplot(331 + i)
    #    plt.plot(np.abs(torque[:, i]))
    #    plt.plot([0, torque.shape[0]], [torque_limits[0, i], torque_limits[0, i]])
    #plt.show()

    #dist_2_goal = np.linalg.norm(data_i[7:14] - q[-1])
    #finished = dist_2_goal < 0.2
    xyz_final, _ = forwardKinematics([q[-1]])
    xyz_desired, _ = forwardKinematics([data_i[7:14]])
    dist_2_goal = np.linalg.norm(xyz_final[0] - xyz_desired[0])
    finished = dist_2_goal < 0.2
    #valid = box_constraints and vertical_constraints and torque_constraints and dq_constraints and ddq_constraints and finished
    valid = box_constraints and vertical_constraints and torque_constraints and dq_constraints and finished
    #valid = box_constraints and vertical_constraints and finished

    ee, _ = forwardKinematics(q)
    ee_d, _ = forwardKinematics(qd)

    # plt.plot(ee[..., 0], ee[..., 1], 'r')
    # plt.plot(ee_d[..., 0], ee_d[..., 1], 'g')
    # plt.plot(bag_dict['puck_pose'][:, 0], bag_dict['puck_pose'][:, 1], 'b')
    # plt.show()

    #for i in range(6):
    #   plt.subplot(321 + i)
    #   plt.plot(t, q[..., i], 'r')
    #   plt.plot(t, qd[..., i], 'g')
    #plt.show()

    # for i in range(6):
    #    plt.subplot(321 + i)
    #    plt.plot(t, np.abs(q[..., i] - qd[..., i]))
    #    plt.plot(t, np.linalg.norm(ee_d[..., :2] - ee[..., :2], axis=-1))
    # plt.show()

    movement_time = t[-1] - t[0]

    dt = np.diff(t)
    integral = lambda x: np.sum(np.abs(x)[1:] * dt)
    # reduce_integral = lambda x: integral(np.sum(np.abs(x), axis=-1))
    reduce_integral = lambda x: integral(np.linalg.norm(x, axis=-1))

    result["valid"] = int(valid)
    result["finished"] = int(finished)
    result["vertical_loss"] = integral(vertical_loss)
    result["planning_time"] = bag_dict["planning_time"]
    result["motion_time"] = movement_time


    result["alpha_beta"] = integral(alpha_beta)
    result["joint_trajectory_error"] = reduce_integral(qd - q)
    result["cartesian_trajectory_error"] = reduce_integral(ee_d - ee)

    print(result)
    return result


#planners = ["ours", "nlopt", "sst", "cbirrt", "mpcmpnet"]
#planners = ["nlopt", "sst", "cbirrt", "mpcmpnet"]
#planners = ["ours"]
#planners = ["ours_long"]
#planners = ["cbirrt"]
#planners = ["nlopt"]
#planners = ["mpcmpnet"]
#planners = ["sst"]
#planners = ["ours_n10"]
#planners = ["ours_n20"]
#planners = ["ours_l256", "ours_l512", "ours_l1024"]
#planners = ["ours_l64"]
#planners = ["ours_l128"]
#planners = ["ours_l128_long", "ours_l256_long", "ours_l3072_long"]
#planners = ["ours_l1024_long"]
#planners = ["ours_l512_long"]
#planners = ["ours_l64_long"]
#planners = ["ours_l2048s"]
#planners = ["ours_10kg"]
#planners = ["ours_12kg"]
planners = ["ours_13kg"]
#planners = ["ours_16kg"]
#planners = ["ours_18kg"]
#planners = ["ours_l2048_long"]
package_dir = "/home/piotr/b8/ah_ws/data"
for planner in planners:
    print(planner)
    dir_path = os.path.join(package_dir, "kino_exp", planner)
    sp = dir_path.replace("data", "results")
    os.makedirs(sp, exist_ok=True)
    for i, p in enumerate(glob(os.path.join(dir_path, "*.bag"))):
    #for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [16, 23, 24, 42, 44, 67, 72]]):
    #for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [1, 12, 16, 24, 39, 42, 44, 65, 74, 91]]):
    #for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [0, 58]]):
    #for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [24, 44, 53, 80, 93, 94]]):
    #for i, p in enumerate(glob(os.path.join(dir_path, "00*.bag"))):
    #for i, p in enumerate(glob(os.path.join(dir_path, "*42.bag"))):
    #for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [0, 6, 16, 21, 23, 24, 42, 44, 65, 66, 67, 72, 92]]):
        print(i)
        d = compute_metrics(p)
        save_path = p[:-3] + "res"
        save_path = save_path.replace("data", "results")
        with open(save_path, 'wb') as fh:
            pickle.dump(d, fh)
