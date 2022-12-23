from copy import copy
from time import perf_counter

import numpy as np
import pinocchio as pino
from ompl import base as ob
from ompl import geometric as og

from manifold_planning.utils.constants import Limits
from manifold_planning.utils.constants import Table1, Table2, Cup


def forward_kinematics(pino_model, pino_data, q):
    pino.forwardKinematics(pino_model, pino_data, q)
    pino.updateFramePlacements(pino_model, pino_data)
    xyz_pino = pino_data.oMf[-1].translation
    R_pino = pino_data.oMf[-1].rotation
    return copy(xyz_pino), copy(R_pino)


class VerticalConstraint(ob.Constraint):

    def __init__(self, pino_model, pino_data, joint_id):
        super(VerticalConstraint, self).__init__(7, 1)
        self.pino_model = pino_model
        self.pino_data = pino_data
        self.joint_id = joint_id

    def function(self, x, out):
        xyz, R = forward_kinematics(self.pino_model, self.pino_data, x)
        vertical_loss = R[2, 2] - 1.
        out[:] = vertical_loss

    def jacobian(self, x, out):
        J = pino.computeFrameJacobian(self.pino_model, self.pino_data, x, self.joint_id, pino.LOCAL_WORLD_ALIGNED)[3:5,
            :]
        xyz, R = forward_kinematics(self.pino_model, self.pino_data, x)
        rpy = pino.rpy.matrixToRpy(R)
        alpha = rpy[0]
        beta = rpy[1]
        dL_dalpha = np.cos(beta) * (-np.sin(alpha))
        dL_dbeta = np.cos(alpha) * (-np.sin(beta))
        dL_dalphabeta = np.stack([dL_dalpha, dL_dbeta], axis=-1)
        dL_dq = dL_dalphabeta[np.newaxis] @ J
        out[:] = dL_dq[0]


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
    ])[:, :, np.newaxis]
    xyz_object = xyz[np.newaxis] + (R[np.newaxis] @ xyz_cuboid)[..., 0]
    return xyz_object


class CBiRRTPlanner:
    def __init__(self, n_pts, pino_model, joint_id):
        self.N = n_pts
        self.M = self.N - 2
        self.D = 7
        self.pino_model = pino_model
        self.pino_data = self.pino_model.createData()
        self.joint_id = joint_id

        self.time = 60.
        tolerance = 0.01#ob.CONSTRAINT_PROJECTION_TOLERANCE
        tries = ob.CONSTRAINT_PROJECTION_MAX_ITERATIONS
        lambda_ = ob.CONSTRAINED_STATE_SPACE_LAMBDA
        delta = ob.CONSTRAINED_STATE_SPACE_DELTA

        rvss = ob.RealVectorStateSpace(self.D)
        bounds = ob.RealVectorBounds(self.D)
        lb = pino_model.lowerPositionLimit[:self.D]
        ub = pino_model.upperPositionLimit[:self.D]
        for i in range(self.D):
            bounds.setLow(i, lb[i])
            bounds.setHigh(i, ub[i])
        rvss.setBounds(bounds)

        # Create our constraint.
        constraint = VerticalConstraint(self.pino_model, self.pino_data, self.joint_id)
        constraint.setTolerance(tolerance)
        constraint.setMaxIterations(tries)
        self.css = ob.ProjectedStateSpace(rvss, constraint)
        self.csi = ob.ConstrainedSpaceInformation(self.css)
        self.ss = og.SimpleSetup(self.csi)

        self.css.setDelta(delta)
        self.css.setLambda(lambda_)

        self.planner = og.RRTConnect(self.csi)
        self.ss.setPlanner(self.planner)

        self.css.setup()
        self.ss.setup()

    def obstacles(self, x, zh1, zh2):
        allowed_violation = 0.01
        q = np.zeros((self.D))
        for i in range(self.D):
            q[i] = x[i]
        xyz, R = forward_kinematics(self.pino_model, self.pino_data, q)

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

        tau = pino.rnea(self.pino_model, self.pino_data, q, np.zeros_like(q), np.zeros_like(q))
        return not inside_box_1 and not inside_box_2 and robot_dist2box_1 > 0.15 - allowed_violation and \
               robot_dist2box_2 > 0.15 - allowed_violation and \
               not np.any(object_inside_box_1) and not np.any(object_inside_box_2) and np.all(np.abs(tau) < Limits.tau7)

    def solve(self, q0, qk):
        self.ss.clear()
        start = ob.State(self.css)
        goal = ob.State(self.css)
        for i in range(self.D):
            start[i] = q0[i]
        for i in range(self.D):
            goal[i] = qk[i]
        self.ss.setStartAndGoalStates(start, goal, 0.01)
        xyz0, _ = forward_kinematics(self.pino_model, self.pino_data, q0)
        xyzk, _ = forward_kinematics(self.pino_model, self.pino_data, qk)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(lambda x: self.obstacles(x, xyz0[-1], xyzk[-1])))
        stat = self.ss.solve(self.time)
        planning_time = self.ss.getLastPlanComputationTime()
        success = False
        q = np.array([q0])
        t = np.array([0.])
        if stat:
            # Get solution and validate
            path = self.ss.getSolutionPath()
            path.interpolate()
            states = [[x[i] for i in range(self.D)] for x in path.getStates()]
            q = np.array(states)
            success = True
            q_diff = q[1:] - q[:-1]
            diff = np.sum(np.abs(q_diff), axis=-1)
            include = np.concatenate([diff > 0, [True]])
            q = q[include]
            q_diff = q_diff[include[:-1]]
            ts = np.abs(q_diff) / Limits.q_dot7
            t = np.max(ts, axis=-1)
            t = np.concatenate([[0.], t + 1e-4])
            t = np.cumsum(t)
        return q, [], [], t, planning_time
