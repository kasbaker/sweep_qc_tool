import json

import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = np.around(rect.get_height() + .75, 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom', size=8)


if __name__ == "__main__":
    path = r'C:\Users\Katie\GitHub\sweep_qc_tool\src\benchmarks\qc_times\200721_21.26.50.json'

    with open(path, 'r') as file:
        load_times = json.load(file)

    num_trials = len(load_times)
    num_cells = len(load_times[0])

    file_names = [file for file in load_times[0]]
    load_methods = list(load_times[0][file_names[0]].keys())

    mono_slow = [[load_times[trial][file]['mono_slow'] for file in file_names]
                  for trial in range(num_trials)]
    mono_slow = np.array(mono_slow, dtype=np.float)
    mono_slow_means = np.nanmean(mono_slow, axis=0)
    mono_slow_std = np.nanstd(mono_slow, axis=0)

    mono_fast = [[load_times[trial][file]['mono_fast'] for file in file_names]
                  for trial in range(num_trials)]
    mono_fast = np.array(mono_fast, dtype=np.float)
    mono_fast_means = np.nanmean(mono_fast, axis=0)
    mono_fast_std = np.std(mono_fast, axis=0)

    dual_slow = [[load_times[trial][file]['dual_slow'] for file in file_names]
                  for trial in range(num_trials)]
    dual_slow = np.array(dual_slow, dtype=np.float)
    dual_slow_means = np.nanmean(dual_slow, axis=0)
    dual_slow_std = np.nanstd(dual_slow, axis=0)

    dual_fast = [[load_times[trial][file]['dual_fast'] for file in file_names]
                  for trial in range(num_trials)]
    dual_fast = np.array(dual_fast, dtype=np.float)
    dual_fast_means = np.nanmean(dual_fast, axis=0)
    dual_fast_std = np.std(dual_fast, axis=0)


    ind = np.arange(len(mono_slow_means))  # the x locations for the groups

    width = 0.2  # the width of the bars
    spacing = width/2

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)

    rects1 = ax.bar(ind - 3*spacing, mono_slow_means, width, yerr=mono_slow_std,
                    label='single process - slow QC')
    rects2 = ax.bar(ind - spacing, mono_fast_means, width, yerr=mono_fast_std,
                    label='single process - fast QC')
    rects3 = ax.bar(ind + spacing, dual_slow_means, width, yerr=dual_slow_std,
                    label='dual process - slow QC')
    rects4 = ax.bar(ind + 3*spacing, dual_fast_means, width, yerr=dual_fast_std,
                    label='dual process - fast QC')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Load time (s)')
    ax.set_title('Load times by processes and QC method (N=3)')

    ax.set_xticks(ind)
    ax.set_xticklabels((f"file {x}" for x in range(1,len(file_names)+1))) #, rotation='vertical')
    ax.legend()

    autolabel(rects1)   #, "left")
    autolabel(rects2)   #, "center")
    autolabel(rects3)   #, "left")
    autolabel(rects4)   #, "center")

    fig.tight_layout()

    plt.show()