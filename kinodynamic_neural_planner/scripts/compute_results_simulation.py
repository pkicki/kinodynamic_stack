import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import namedtuple
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

root_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(root_dir)

#planners_names_map = dict(ours="ours", mpcmpnet="MPC-MPNet", nlopt="TrajOpt", cbirrt="CBiRRT")
#planners_names_map = dict(ours="ours", sst="SST", mpcmpnet="MPC-MPNet", nlopt="TrajOpt", cbirrt="CBiRRT",
#                          ours_n10="ours_n10", ours_n20="ours_n20", ours_l512="ours_l512")
order = [
    ("ours_l2048_long", "CNP-B (ours)"),
    ("cbirrt", "CBiRRT [11]"),
    ("nlopt", "TrajOpt [25]"),
    ("mpcmpnet", "MPC-MPNet [54]"),
    ("sst", "SST [41]"),
]
planners_names_map = {x: y for x, y in order}
planners_names_order = {x: i for i, (x, y) in enumerate(order)}
#planners_names_map = dict(ours_l2048_long="ours", sst="SST", mpcmpnet="MPC-MPNet", nlopt="TrajOpt", cbirrt="CBiRRT")
#planners_names_map = dict(ours="ours", sst="SST", mpcmpnet="MPC-MPNet", nlopt="TrajOpt", cbirrt="CBiRRT")
#planners_names_map = dict(ours="ours_l2048", ours_l64="ours_l0064", ours_l256="ours_l0256", ours_l512="ours_l0512",
#                          ours_l1024="ours_l1024", ours_l128="ours_l0128")
#planners_names_map = dict(ours_l64_long="ours_l0064", ours_l128_long="ours_l0128",
#                          ours_l256_long="ours_l0256", ours_l512_long="ours_l0512",
#                          ours_l1024_long="ours_l1024", ours_l2048_long="ours_l2048",
#                          ours_l3072_long="ours_l3072", ours_l2048s="ours_l2048s",
#                          ours_l2048sl="ours_l2048sl")
#planners_names_map = dict(ours="ours", ours_l512="ours_l512", ours_l1024="ours_l1024")

def mean(r, k, filter_finished):
    if filter_finished:
        s = [v[k] for _, v in r.items() if (k in v and v[k] is not None and v["finished"])]
    else:
        s = [v[k] for _, v in r.items() if k in v and v[k] is not None]
    return np.mean(s)

def bar_chart(data, categories):
    plt.rc('font', size=15)
    plt.rc('legend', fontsize=17)
    n_cat = len(list(categories.keys()))
    planners = list(data[list(categories.keys())[0]].keys())
    #planners.sort(key=lambda x: planners_names_map[x])
    planners.sort(key=lambda x: planners_names_order[x])

    #data = [(k, v) for k, v in data]
    #data.sort(lambda )

    #planners_results = {}
    #for planner in planners:
    #    results = [data[k][planner] for k, v, in categories.items()]
    #    planners_results[planner] = results
    width = 0.1
    #fig, ax = plt.subplots(1, n_cat, figsize=(18, 8))
    fig, ax = plt.subplots(1, n_cat, figsize=(18, 4))
    for i, (k, v) in enumerate(data.items()):
        for j, p in enumerate(planners):
            ax[i].bar(i + width * j, v[p] * categories[k].scale, width, label=planners_names_map[p])
    for i, (k, v) in enumerate(categories.items()):
        if v.log:
            ax[i].set_yscale('log')
        if k == "valid":
        #if "ratio" in v.title:
            ax[i].set_ylim(0., 100.)
        ax[i].set_title(v.title)
        ax[i].set_xticks([])
        ax[i].set_xticks([], minor=True)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=n_cat+1, frameon=False)
    plt.show()
    a = 0






def read_results(path):
    results = {}
    for p in glob(os.path.join(path, "*.res")):
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
            name = p.split("/")[-1][:-4]
            idx = int(name)
            results[idx] = d
    return results


results = {}
for path in glob("/home/piotr/b8/ah_ws/results/kino_exp/*"):
    data = read_results(path)
    name = path.split("/")[-1]
    if name in planners_names_map.keys():
        results[name] = data

description = namedtuple("Description", "title scale log filter_finished")
categories = {
    "valid": description("Success ratio [%]", 100., False, False),
    "finished": description("Success ratio \n(no constraints) [%]", 100., False, False),
    "planning_time": description("Mean planning time [ms]", 1., True, False),
    #"planning_time": description("Mean planning time [ms]", 1., False, False),
    "motion_time": description("Mean motion time [s]", 1., False, False),
    "alpha_beta": description("Mean vertical error [rad]", 1., False, False),
    #"joint_trajectory_error": description("Mean trajectory tracking error [rad]", 1., True, True),
}
summary = {k: {} for k, v in categories.items()}
for name, data in results.items():
    for cat_k, cat_v in categories.items():
        summary[cat_k][name] = mean(data, cat_k, cat_v.filter_finished)
a = 0

bar_chart(summary, categories)

# box_plot(results, categories)

# box_plot((ours, baseline))
# plot_scatter(ours, "scored")
# plot_scatter(ours, "planned_puck_velocity_magnitude")
# plot_scatter(baseline, "scored")
# plot_scatter(baseline, "planned_puck_velocity_magnitude")
## plot_hist(ours, "puck_actual_vs_planned_velocity_magnitude_error", abs=False, name="ours")
## plot_hist(baseline, "puck_actual_vs_planned_velocity_magnitude_error", abs=False, name="baseline")
# plot_hist(ours, "puck_velocity_magnitude", abs=False, name="ours")
# plot_hist(baseline, "puck_velocity_magnitude", abs=False, name="baseline", alpha=0.5)
# plt.legend()
# plt.show()
## for i in range(6):
##    plt.plot(t, qd_ddot[:, i], label=f"q_ddot_{i}")
## plt.legend()
## plt.show()


# plt.plot(puck[:, 0], puck[:, 1])
# plt.show()
