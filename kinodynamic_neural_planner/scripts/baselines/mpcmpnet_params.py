import numpy as np

def get_params():
    params = {
        'solver_type': "cem",
        'n_problem': 1,
        'n_sample': 64,
        'n_elite': 4,
        'n_t': 1,
        'max_it': 30,
        # 'converge_r': 1e-1,
        'converge_r': 2e-2,

        'dt': 1e-2,
        'mu_u': [0, 0, 0, 0, 0, 0, 0],
        #'sigma_u': [1, 1, 1, 1, 1, 1],
        'sigma_u': np.array([256.,  256.,  140.8, 140.8,  88.,   32.,   32.]),
        #'sigma_u': 0.8 * np.array([256.,  256.,  140.8, 140.8,  88.,   32.]),
        #'sigma_u': 0.5 * np.array([256.,  256.,  140.8, 140.8,  88.,   32.]),
        #'sigma_u': 0.1 * np.array([256.,  256.,  140.8, 140.8,  88.,   32.]),
        'mu_t': 5e-2,
        'sigma_t': 0.1,
        't_max': 0.1,
        'verbose': False,  # True, #
        #'step_size': 0.75,
        'step_size': 0.1,

        "goal_radius": 0.2,
        "sst_delta_near": 0.2,
        "sst_delta_drain": 0.1,
        #"sst_delta_near": 0.05,
        #"sst_delta_drain": 0.02,
        "goal_bias": 0.05,

        "width": 6,
        "hybrid": True,
        "hybrid_p": 0.3,
        "cost_samples": 5,
        "mpnet_weight_path": "/home/piotr/b8/new_airhockey/baselines/mpc-mpnet-py/mpnet/output/iiwa_kino_paper/mpnet_kino_cpu_skip1_dropout05/ep01550.pth",
        "cost_to_go_predictor_weight_path": "/home/piotr/b8/new_airhockey/baselines/mpc-mpnet-py/mpnet/output/iiwa_kino_paper/c2g/ep01600.pth",
        "cost_predictor_weight_path": "",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cpu:0",
        #"device_id": "cuda:3",

        "cost_reselection": False,
        "number_of_iterations": 40,
        #"weights_array": [1, 1, 1., 1.],
        "weights_array": [1, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
        #"weights_array": [1, 1, 1, 1, 1, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        # "weights_array": [],
        #"weights_array": [1, 1, .2, .2],
        'max_planning_time': 60,
        'shm_max_steps': 1
    }

    return params
