from copy import copy
from time import perf_counter

import numpy as np
from mpcmpnet_params import get_params
from _mpc_mpnet_module import MPCMPNetWrapper
import pinocchio as pino

from manifold_planning.utils.constants import Table1, Table2


def forward_kinematics(pino_model, pino_data, q):
    pino.forwardKinematics(pino_model, pino_data, q)
    pino.updateFramePlacements(pino_model, pino_data)
    xyz_pino = pino_data.oMf[-1].translation
    R_pino = pino_data.oMf[-1].rotation
    return copy(xyz_pino), copy(R_pino)


class MPCMPNetPlanner:
    def __init__(self, pino_model, joint_id):
        self.params = get_params()
        self.pino_model = pino_model
        self.pino_data = self.pino_model.createData()
        self.joint_id = joint_id



    def solve(self, q0, qk):
        xyz_0, _ = forward_kinematics(self.pino_model, self.pino_data, q0)
        xyz_d, _ = forward_kinematics(self.pino_model, self.pino_data, qk)

        obs = np.array([[Table1.xl, Table1.yl, -1e10, Table1.xh, Table1.yh, xyz_0[-1]],
                        [Table2.xl, Table2.yl, -1e10, Table2.xh, Table2.yh, xyz_d[-1]]])
        start_state = q0.tolist() + np.zeros(7).tolist()
        end_state = qk.tolist() + np.zeros(7).tolist()
        planner = MPCMPNetWrapper(system_type="iiwa_kino", start_state=start_state, goal_state=end_state,
                                  random_seed=0, goal_radius=self.params['goal_radius'],
                                  sst_delta_near=self.params['sst_delta_near'],
                                  sst_delta_drain=self.params['sst_delta_drain'],
                                  obs_list=obs, width=self.params['width'], verbose=self.params['verbose'],
                                  mpnet_weight_path=self.params['mpnet_weight_path'],
                                  cost_predictor_weight_path=self.params['cost_predictor_weight_path'],
                                  cost_to_go_predictor_weight_path=self.params[
                                      'cost_to_go_predictor_weight_path'],
                                  num_sample=self.params['cost_samples'],
                                  shm_max_step=self.params['shm_max_steps'],
                                  np=self.params['n_problem'], ns=self.params['n_sample'], nt=self.params['n_t'],
                                  ne=self.params['n_elite'], max_it=self.params['max_it'], converge_r=self.params['converge_r'],
                                  mu_u=self.params['mu_u'], std_u=self.params['sigma_u'], mu_t=self.params['mu_t'],
                                  std_t=self.params['sigma_t'], t_max=self.params['t_max'],
                                  step_size=self.params['step_size'], integration_step=self.params['dt'],
                                  device_id=self.params['device_id'], refine_lr=self.params['refine_lr'],
                                  weights_array=self.params['weights_array'],
                                  obs_voxel_array=np.zeros((10, 10, 3)).reshape(-1)
                                  )
        solution = planner.get_solution()

        tic = perf_counter()
        for iteration in range(int(1e10)):
            planner.mp_path_step(self.params['refine'],
                                 refine_threshold=self.params['refine_threshold'],
                                 using_one_step_cost=self.params['using_one_step_cost'],
                                 cost_reselection=self.params['cost_reselection'],
                                 goal_bias=self.params['goal_bias'],
                                 num_of_problem=self.params['n_problem'])
            solution = planner.get_solution()
            # and np.sum(solution[2]) < th:
            if solution is not None or perf_counter()-tic > self.params['max_planning_time']:
                break
        planning_time = perf_counter() - tic
        q = None
        dq = None
        t = None
        print(solution)
        print(planner.get_approximate_solution())
        solution = solution if solution is not None else planner.get_approximate_solution()
        q = q0[np.newaxis, :7]
        dq = np.zeros_like(q)
        t = np.zeros(1)
        if solution is not None:
            q = solution[0][:, :7]
            dq = solution[0][:, 7:]
            t = np.concatenate([[0.], solution[2]])
            t = np.cumsum(t)
        return q, dq, [], t, planning_time
