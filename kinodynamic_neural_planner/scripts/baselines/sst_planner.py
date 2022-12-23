from time import perf_counter, time

import numpy as np
import pinocchio as pino
from copy import copy
from _sst_module import SSTWrapper, IiwaAcc, IiwaAccDistance, IiwaKinoRectangleObs3DSystem, IiwaKinoDistance

from manifold_planning.utils.constants import Table1, Table2


class IiwaKinoRectangleObs3D(IiwaKinoRectangleObs3DSystem):
    def __init__(self, obstacle_list):
        super().__init__(obstacle_list)

    def distance_computer(self):
        # return euclidean_distance(np.array(self.is_circular_topology()))
        return IiwaKinoDistance()
        # return IiwaAccDistance()


def forward_kinematics(pino_model, pino_data, q):
    pino.forwardKinematics(pino_model, pino_data, q)
    pino.updateFramePlacements(pino_model, pino_data)
    xyz_pino = pino_data.oMf[-1].translation
    R_pino = pino_data.oMf[-1].rotation
    return copy(xyz_pino), copy(R_pino)


class SSTPlanner:
    def __init__(self, pino_model, joint_id):
        self.pino_model = pino_model
        self.pino_data = self.pino_model.createData()
        self.joint_id = joint_id
        self.D = 7
        self.config = dict(
            random_seed=0,
            goal_radius=0.2,
            sst_delta_near=0.2,
            sst_delta_drain=0.1,
            integration_step=0.005,
            min_time_steps=1,
            max_time_steps=20,
            number_of_iterations=300000,
            max_planning_time=60.,
        )

    def solve(self, q0, qk):
        start_state = q0[:self.D].tolist() + np.zeros(self.D).tolist()
        goal_state = qk[:self.D].tolist() + np.zeros(self.D).tolist()

        xyz_0, _ = forward_kinematics(self.pino_model, self.pino_data, q0)
        xyz_d, _ = forward_kinematics(self.pino_model, self.pino_data, qk)

        obs = np.array([[Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh, xyz_0[-1]],
                        [Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh, xyz_d[-1]]])
        system = IiwaKinoRectangleObs3D(obs)

        planner = SSTWrapper(
            state_bounds=system.get_state_bounds(),
            control_bounds=system.get_control_bounds(),
            distance=system.distance_computer(),
            start_state=start_state,
            goal_state=goal_state,
            goal_radius=self.config['goal_radius'],
            random_seed=0,
            sst_delta_near=self.config['sst_delta_near'],
            sst_delta_drain=self.config['sst_delta_drain']
        )

        tic = perf_counter()
        for iteration in range(self.config["number_of_iterations"]):
            planner.step(system, self.config["min_time_steps"], self.config["max_time_steps"],
                         self.config["integration_step"])
            if perf_counter() - tic > self.config["max_planning_time"] or planner.get_solution() is not None:
                break

        planning_time = perf_counter() - tic

        solution = planner.get_solution()
        solution = solution if solution is not None else planner.get_approximate_solution()
        solution = solution if solution is not None else []

        q = np.array([q0])
        dq = np.zeros_like(q)
        t = np.zeros((1,))
        if len(solution):
            q = solution[0][:, :self.D]
            dq = solution[0][:, self.D:]
            t = np.concatenate([[0.], solution[2]])
            t = np.cumsum(t)
        return q, dq, [], t, planning_time
