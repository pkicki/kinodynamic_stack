from copy import copy
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
import pinocchio as pino

import sys, os


BASELINES_DIR = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.dirname(BASELINES_DIR)
PACKAGE_DIR = os.path.dirname(SCRIPTS_DIR)
PLANNING_MODULE_DIR = os.path.join(SCRIPTS_DIR, "manifold_planning")
sys.path.append(PLANNING_MODULE_DIR)
sys.path.append(SCRIPTS_DIR)

from manifold_planning.utils.constants import Limits
from manifold_planning.utils.constants import Table1, Table2, Cup


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


class NloptPlanner:
    def __init__(self, n_pts, pino_model, joint_id):
        self.N = n_pts
        self.M = self.N - 2
        self.D = 7
        self.pino_model = pino_model
        self.pino_data = self.pino_model.createData()
        self.joint_id = joint_id
        self.diff_diag, self.diag, self.zeros = self.prepare_utils()

    def prepare_utils(self):
        diff_diag = np.zeros(((self.N - 1) * self.D, self.M * self.D))
        diag = np.zeros(((self.N - 1) * self.D, self.M * self.D))
        zeros = np.zeros(((self.N - 1) * self.D, self.M * self.D))
        for i in range(self.D):
            s = np.arange(self.M)
            diff_diag[i * (self.N - 1) + s + 1, i * self.M + s] = -1.
            diff_diag[i * (self.N - 1) + s, i * self.M + s] = 1.
            diag[i * (self.N - 1) + s + 1, i * self.M + s] = -1.
        return diff_diag, diag, zeros

    def objective(self, x):
        return x[-1]

    def jac(self, x):
        grad = np.zeros_like(x)
        grad[-1] = 1.
        return grad

    def forwardKinematics(self, q):
        xyz = []
        R = []
        for i in range(len(q)):
            pino.forwardKinematics(self.pino_model, self.pino_data, q[i])
            pino.updateFramePlacements(self.pino_model, self.pino_data)
            xyz_pino = self.pino_data.oMf[self.joint_id].translation
            R_pino = self.pino_data.oMf[self.joint_id].rotation
            xyz.append(copy(xyz_pino))
            R.append(copy(R_pino))
        xyz = np.stack(xyz)
        R = np.stack(R)
        return xyz, R

    def vertical_constraint(self, x):
        q = self.get_q(x)
        xyz, R = self.forwardKinematics(q)
        vertical_loss = R[:, 2, 2] - 1.
        return vertical_loss

    def vertical_constraint_jac(self, x):
        q = self.get_q(x)
        Js = []
        for i in range(self.M):
            J = pino.computeFrameJacobian(self.pino_model, self.pino_data, q[i], self.joint_id, pino.LOCAL_WORLD_ALIGNED)[3:5, :]
            Js.append(J)
        Js = np.stack(Js)
        xyz, R = self.forwardKinematics(q)
        rpy = []
        for i in range(self.M):
            rpy.append(pino.rpy.matrixToRpy(R[i]))
        rpy = np.stack(rpy)
        alpha = rpy[:, 0]
        beta = rpy[:, 1]
        dL_dalpha = np.cos(beta) * (-np.sin(alpha))
        dL_dbeta = np.cos(alpha) * (-np.sin(beta))
        dL_dalphabeta = np.stack([dL_dalpha, dL_dbeta], axis=-1)
        dL_dq = dL_dalphabeta[:, np.newaxis] @ Js
        grad = np.zeros((self.M, x.shape[0]))
        for i in range(self.M):
            grad[i, np.arange(0, self.D * self.M, self.M) + (i % self.M)] = dL_dq[i, 0]
        return grad


    def box_constraint(self, x, q0, qk, zh1, zh2):
        q = self.get_q(x)
        xyz, R = self.forwardKinematics(q)

        def dist2box(xyz, xl, yl, zl, xh, yh, zh):
            l = np.reshape(np.stack([xl, yl, zl], axis=-1), (-1,) + (1,) * (len(xyz.shape) - 2) + (3,))
            h = np.reshape(np.stack([xh, yh, zh], axis=-1), (-1,) + (1,) * (len(xyz.shape) - 2) + (3,))
            xyz_dist = np.max(np.stack([l - xyz, np.zeros_like(xyz), xyz - h], axis=-1), axis=-1)
            dist = np.sqrt(np.sum(np.square(xyz_dist), axis=-1) + 1e-8)
            return dist

        def inside_box(xyz, xl, yl, zl, xh, yh, zh):
            pxl = xyz[..., 0] > xl
            pxh = xyz[..., 0] < xh
            pyl = xyz[..., 1] > yl
            pyh = xyz[..., 1] < yh
            pzl = xyz[..., 2] > zl
            pzh = xyz[..., 2] < zh
            return np.all(np.stack([pxl, pxh, pyl, pyh, pzl, pzh], axis=-1), axis=-1)

        def dist_point_2_box_inside(xyz, xl, yl, zl, xh, yh, zh):
            dist = np.min(np.abs(np.stack([xyz[..., 0] - xl, xyz[..., 0] - xh,
                                           xyz[..., 1] - yl, xyz[..., 1] - yh,
                                           xyz[..., 2] - zl, xyz[..., 2] - zh,
                                           ], axis=-1)), axis=-1)
            return dist

        robot_dist2box_1 = dist2box(xyz, Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh, zh1 - Cup.height)
        robot_dist2box_2 = dist2box(xyz, Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh, zh2 - Cup.height)
        # inside_box_1 = inside_box(xyz, Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh, zh1 - Cup.height)
        # inside_box_2 = inside_box(xyz, Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh, zh2 - Cup.height)
        robot_box1violation = np.maximum(0.15 - robot_dist2box_1, 0.)
        robot_box2violation = np.maximum(0.15 - robot_dist2box_2, 0.)

        xyz_object = compute_object_coordinates(xyz, R)

        object_dist2box_1 = dist_point_2_box_inside(xyz_object, Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh,
                                                    zh1 - Cup.height)
        object_dist2box_2 = dist_point_2_box_inside(xyz_object, Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh,
                                                    zh2 - Cup.height)
        object_inside_box_1 = inside_box(xyz_object, Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh,
                                         zh1 - Cup.height)
        object_inside_box_2 = inside_box(xyz_object, Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh,
                                         zh2 - Cup.height)
        object_box1violation = object_dist2box_1 * object_inside_box_1.astype(np.float32)
        object_box2violation = object_dist2box_2 * object_inside_box_2.astype(np.float32)
        # box_violations = -np.concatenate([robot_box1violation[:, np.newaxis], robot_box2violation[:, np.newaxis],
        #                                  object_box1violation, object_box2violation,
        #                                  ], axis=-1)
        box_violations = -np.concatenate([robot_box1violation, robot_box2violation,
                                          object_box1violation.reshape(-1), object_box2violation.reshape(-1),
                                          ], axis=-1)
        box_violations = np.reshape(box_violations, (-1))
        return box_violations

    def box_constraint_jac(self, x, q0, qk, zh1, zh2):
        c = self.box_constraint(x, q0, qk, zh1, zh2)
        q = self.get_q(x)
        Js = []
        for i in range(self.M):
            J = pino.computeFrameJacobian(self.pino_model, self.pino_data, q[i], self.joint_id, pino.LOCAL_WORLD_ALIGNED)[:3, :]
            Js.append(J)
        Js = np.stack(Js)

        def dist2box(xyz, xl, yl, zl, xh, yh, zh):
            l = np.reshape(np.stack([xl, yl, zl], axis=-1), (-1,) + (1,) * (len(xyz.shape) - 2) + (3,))
            h = np.reshape(np.stack([xh, yh, zh], axis=-1), (-1,) + (1,) * (len(xyz.shape) - 2) + (3,))
            xyz_sign = np.argmax(np.stack([l - xyz, np.zeros_like(xyz), xyz - h], axis=-1), axis=-1)
            xyz_dist = np.max(np.stack([l - xyz, np.zeros_like(xyz), xyz - h], axis=-1), axis=-1)
            return xyz_sign, xyz_dist

        def dist_point_2_box_inside(xyz, xl, yl, zl, xh, yh, zh):
            dist = np.argmin(np.abs(np.stack([xyz[..., 0] - xl, xyz[..., 0] - xh,
                                              xyz[..., 1] - yl, xyz[..., 1] - yh,
                                              xyz[..., 2] - zl, xyz[..., 2] - zh,
                                              ], axis=-1)), axis=-1)
            return dist

        xyz, R = self.forwardKinematics(q)

        sign2box_1, dist2box_1 = dist2box(xyz, Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh, zh1 - Cup.height)
        sign2box_2, dist2box_2 = dist2box(xyz, Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh, zh2 - Cup.height)
        dsign_box_1 = np.sign(sign2box_1 - 1.)
        dsign_box_2 = np.sign(sign2box_2 - 1.)

        dist_der_box_1 = dist2box_1 / np.linalg.norm(dist2box_1, axis=-1, keepdims=True)
        dist_der_box_2 = dist2box_2 / np.linalg.norm(dist2box_2, axis=-1, keepdims=True)
        grad_box1 = ((dsign_box_1 * dist_der_box_1)[:, np.newaxis] @ Js)[:, 0]
        grad_box2 = ((dsign_box_2 * dist_der_box_2)[:, np.newaxis] @ Js)[:, 0]

        xyz_object = compute_object_coordinates(xyz, R)
        object_dist2box_1 = dist_point_2_box_inside(xyz_object, Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh,
                                                    zh1 - Cup.height)
        object_dist2box_2 = dist_point_2_box_inside(xyz_object, Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh,
                                                    zh2 - Cup.height)

        def compute_gradient_dir(object_dist2box):
            o = np.ones_like(object_dist2box)
            z = np.zeros_like(object_dist2box)
            object_dist2box_x = np.where(object_dist2box == 0, -o, np.where(object_dist2box == 1, o, z))
            object_dist2box_y = np.where(object_dist2box == 2, -o, np.where(object_dist2box == 3, o, z))
            object_dist2box_z = np.where(object_dist2box == 4, -o, np.where(object_dist2box == 5, o, z))
            return np.stack([object_dist2box_x, object_dist2box_y, object_dist2box_z], axis=-1)

        grad_object_dist2box1 = compute_gradient_dir(object_dist2box_1)
        grad_object_dist2box2 = compute_gradient_dir(object_dist2box_2)
        grad_box3 = (grad_object_dist2box1[:, :, np.newaxis] @ Js[:, np.newaxis])[..., 0, :]
        grad_box4 = (grad_object_dist2box2[:, :, np.newaxis] @ Js[:, np.newaxis])[..., 0, :]

        grad_boxes = np.concatenate([grad_box1, grad_box2, grad_box3.reshape([-1, 7]), grad_box4.reshape([-1, 7])],
                                    axis=0)
        grad_boxes = np.where(c[:, np.newaxis] < 0, grad_boxes, np.zeros_like(grad_boxes))
        grad = np.zeros((grad_boxes.shape[0], x.shape[0]))
        for i in range(grad_boxes.shape[0]):
            grad[i, np.arange(0, self.D * self.M, self.M) + (i % self.M)] = grad_boxes[i]
        return grad

    def dq_constraint(self, x, q0, qk, dq0, dqk):
        dt = x[-1] / self.N
        q = self.get_q(x)
        q_dot = self.get_dq(x)
        q = np.concatenate([q0[np.newaxis], q, qk[np.newaxis]], axis=0)
        q_dot = np.concatenate([dq0[np.newaxis], q_dot, dqk[np.newaxis]], axis=0)
        q_diff = q[1:] - q[:-1]
        delta_q = q_dot[:-1] * dt
        r = q_diff - delta_q
        r = np.reshape(r.T, -1)
        return r

    def dq_constraint_jac(self, x, q0, qk, dq0, dqk):
        dt = x[-1] / self.N
        q_dot = self.get_dq(x)
        q_dot = np.concatenate([dq0[np.newaxis], q_dot, dqk[np.newaxis]], axis=0)
        q_dot_grad = np.reshape(q_dot[:-1].T, -1)
        grad = np.concatenate([self.diff_diag, self.diag * dt, self.zeros, -q_dot_grad[:, np.newaxis] / self.N], axis=-1)
        return grad

    def ddq_constraint(self, x, q0, qk, dq0, dqk, ddq0, ddqk):
        dt = x[-1] / self.N
        q_dot = self.get_dq(x)
        q_dot = np.concatenate([dq0[np.newaxis], q_dot, dqk[np.newaxis]], axis=0)
        q_ddot = self.get_ddq(x)
        q_ddot = np.concatenate([ddq0[np.newaxis], q_ddot, ddqk[np.newaxis]], axis=0)
        q_dot_diff = q_dot[1:] - q_dot[:-1]
        delta_q_dot = q_ddot[:-1] * dt
        r = q_dot_diff - delta_q_dot
        r = np.reshape(r.T, -1)
        return r

    def ddq_constraint_jac(self, x, q0, qk, dq0, dqk, ddq0, ddqk):
        dt = x[-1] / self.N
        q_ddot = self.get_ddq(x)
        q_ddot = np.concatenate([ddq0[np.newaxis], q_ddot, ddqk[np.newaxis]], axis=0)
        q_ddot_grad = np.reshape(q_ddot[:-1].T, -1)
        grad = np.concatenate([self.zeros, self.diff_diag, self.diag * dt, -q_ddot_grad[:, np.newaxis] / self.N], axis=-1)
        return grad

    def get_q(self, x):
        return np.reshape(x[:self.M * self.D], (self.D, self.M)).T

    def get_dq(self, x):
        return np.reshape(x[self.M * self.D:2 * self.M * self.D], (self.D, self.M)).T

    def get_ddq(self, x):
        return np.reshape(x[2 * self.M * self.D:3 * self.M * self.D], (self.D, self.M)).T

    def tau_constraint(self, x):
        q = self.get_q(x)
        q_dot = self.get_dq(x)
        q_ddot = self.get_ddq(x)
        taus_computed = []
        for i in range(self.M):
            tau_computed = pino.rnea(self.pino_model, self.pino_data, q[i], q_dot[i], q_ddot[i])
            taus_computed.append(tau_computed)
        taus_computed = np.stack(taus_computed, axis=0)
        # r = Limits.tau7[np.newaxis] - np.abs(taus_computed)
        r = taus_computed
        # r = np.reshape(r.T, -1)
        return r

    def abs_tau_constraint(self, x):
        diff = Limits.tau7[np.newaxis] - np.abs(self.tau_constraint(x))
        diff = np.reshape(diff.T, -1)
        return diff

    def tau_constraint_jac(self, x):
        q = self.get_q(x)
        q_dot = self.get_dq(x)
        q_ddot = self.get_ddq(x)
        dtau_dqs = []
        dtau_dvs = []
        dtau_das = []
        for i in range(self.M):
            pino.computeRNEADerivatives(self.pino_model, self.pino_data, q[i], q_dot[i], q_ddot[i])
            dtau_dqs.append(copy(self.pino_data.dtau_dq))
            dtau_dvs.append(copy(self.pino_data.dtau_dv))
            Mtriu = copy(self.pino_data.M)
            mm = Mtriu + Mtriu.T
            diag_idx = np.arange(mm.shape[0])
            mm[diag_idx, diag_idx] = Mtriu[diag_idx, diag_idx]
            dtau_das.append(mm)
        dtau_dqs = np.stack(dtau_dqs, axis=0)
        dtau_dvs = np.stack(dtau_dvs, axis=0)
        dtau_das = np.stack(dtau_das, axis=0)
        grad = np.zeros((self.M, self.D, x.shape[0]))
        for i in range(self.M):
            grad[i, :, np.arange(0, self.D * self.M, self.M) + i] = dtau_dqs[i]
            grad[i, :, self.D * self.M + np.arange(0, self.D * self.M, self.M) + i] = dtau_dvs[i]
            grad[i, :, 2 * self.D * self.M + np.arange(0, self.D * self.M, self.M) + i] = dtau_das[i]
        grad = np.reshape(np.transpose(grad, (1, 0, 2)), (self.M * self.D, -1))
        c_val = self.tau_constraint(x)
        c_val = np.reshape(c_val.T, -1)
        dcdf = np.where(c_val > 0., -np.ones_like(c_val), np.ones_like(c_val))
        return grad * dcdf[:, np.newaxis]

    def solve(self, q0, qk):
        q0 = q0[:self.D]
        qk = qk[:self.D]
        dq0 = np.zeros_like(q0)
        dqk = np.zeros_like(q0)
        ddq0 = np.zeros_like(q0)
        ddqk = np.zeros_like(q0)

        xyz_, _ = self.forwardKinematics([q0])
        zh1 = xyz_[0, -1]

        xyz_, _ = self.forwardKinematics([qk])
        zh2 = xyz_[0, -1]

        a_0 = q0[np.newaxis]
        a_1 = (dq0 + 3 * q0)[np.newaxis]
        a_3 = qk[np.newaxis]
        a_2 = (3 * qk - dqk)[np.newaxis]
        t = np.linspace(0., 1., self.N)[:, np.newaxis]
        q_ = a_3 * t ** 3 + a_2 * t ** 2 * (1 - t) + a_1 * t * (1 - t) ** 2 + a_0 * (1 - t) ** 3
        q_dot_ = 3 * a_3 * t ** 2 + a_2 * (-3 * t ** 2 + 2 * t) + a_1 * (
                3 * t ** 2 - 4 * t + 1) - a_0 * 3 * (1 - t) ** 2
        q_ddot_ = 6 * a_3 * t ** 1 + a_2 * (-6 * t + 2) + \
                  a_1 * (6 * t - 4) + a_0 * 6 * (1 - t)

        init_q = q_  # + 0.01 * (2*np.random.random((self.N, 1)) - 1.)
        init_dq = q_dot_
        init_ddq = q_ddot_

        q_dot_mul = np.max(np.abs(q_dot_) / Limits.q_dot7[np.newaxis])
        q_ddot_mul = np.max(np.abs(q_ddot_) / Limits.q_ddot7[np.newaxis])
        T0 = np.maximum(q_dot_mul, np.sqrt(q_ddot_mul))


        init_x = np.zeros((3 * (self.M * self.D) + 1))
        init_x[:self.M * self.D] = np.reshape(init_q[1:-1].T, -1)
        init_x[self.M * self.D:2 * self.M * self.D] = np.reshape(init_dq[1:-1].T, -1)
        init_x[2 * self.M * self.D:3 * self.M * self.D] = np.reshape(init_ddq[1:-1].T, -1)
        init_x[-1] = T0

        qLimit = np.tile(Limits.q7[:self.D, np.newaxis], (1, self.M))
        qLimit = np.reshape(qLimit, -1)
        dqLimit = np.tile(Limits.q_dot7[:self.D, np.newaxis], (1, self.M))
        dqLimit = np.reshape(dqLimit, -1)
        ddqLimit = np.tile(Limits.q_ddot7[:self.D, np.newaxis], (1, self.M))
        ddqLimit = np.reshape(ddqLimit, -1)
        lb = (-qLimit).tolist() + (-dqLimit).tolist() + (-ddqLimit).tolist() + [0.1]
        ub = qLimit.tolist() + dqLimit.tolist() + ddqLimit.tolist() + [None]
        bounds = list(zip(lb, ub))

        constrs = []
        constrs.append(self.vertical_constraint(init_x))
        constrs.append(self.box_constraint(init_x, q0, qk, zh1, zh2))
        constrs.append(self.dq_constraint(init_x, q0, qk, dq0, dqk))
        constrs.append(self.ddq_constraint(init_x, q0, qk, dq0, dqk, ddq0, ddqk))
        constrs.append(self.abs_tau_constraint(init_x))
        c = np.concatenate(constrs, axis=0)

        grads = []
        grads.append(self.vertical_constraint_jac(init_x))
        grads.append(self.box_constraint_jac(init_x, q0, qk, zh1, zh2))
        grads.append(self.dq_constraint_jac(init_x, q0, qk, dq0, dqk))
        grads.append(self.ddq_constraint_jac(init_x, q0, qk, dq0, dqk, ddq0, ddqk))
        grads.append(self.tau_constraint_jac(init_x))
        for g in grads:
            print(g.shape)
        g = np.concatenate(grads, axis=0)
        gs = np.sum(np.abs(g), axis=0)

        dq_constraint_dict = {
            "type": "eq",
            "fun": self.dq_constraint,
            "jac": self.dq_constraint_jac,
            "args": [q0, qk, dq0, dqk],
        }

        ddq_constraint_dict = {
            "type": "eq",
            "fun": self.ddq_constraint,
            "jac": self.ddq_constraint_jac,
            "args": [q0, qk, dq0, dqk, ddq0, ddqk],
        }

        tau_constraint_dict = {
            "type": "ineq",
            "fun": self.abs_tau_constraint,
            "jac": self.tau_constraint_jac,
        }

        box_constraint_dict = {
            "type": "ineq",
            "fun": self.box_constraint,
            "jac": self.box_constraint_jac,
            "args": [q0, qk, zh1, zh2],
        }

        vertical_constraint_dict = {
            "type": "ineq",
            "fun": self.vertical_constraint,
            "jac": self.vertical_constraint_jac,
        }

        init_x_ = copy(init_x)
        tic = perf_counter()
        res = minimize(self.objective, init_x_, method="SLSQP", jac=self.jac, bounds=bounds,
                       constraints=[dq_constraint_dict, ddq_constraint_dict, tau_constraint_dict,
                                    box_constraint_dict, vertical_constraint_dict],
                       options={"disp": False, "maxiter": 100})
        planning_time = perf_counter() - tic
        x = res.x

        q = self.get_q(x)
        q = np.concatenate([q0[np.newaxis], q, qk[np.newaxis]], axis=0)
        q_dot = self.get_dq(x)
        dq = np.concatenate([dq0[np.newaxis], q_dot, dqk[np.newaxis]], axis=0)
        q_ddot = self.get_ddq(x)
        ddq = np.concatenate([ddq0[np.newaxis], q_ddot, ddqk[np.newaxis]], axis=0)
        t = np.linspace(0., x[-1], self.N)
        return q, dq, ddq, t, planning_time
