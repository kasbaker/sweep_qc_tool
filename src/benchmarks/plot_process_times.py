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
    path = r'C:\Users\Katie\GitHub\sweep_qc_tool\src\benchmarks\process_times\200713_21.11.13.json'
    with open(path, 'r') as file:
        load_times = json.load(file)

    num_trials = len(load_times)
    num_cells = len(load_times[0])

    file_names = [file for file in load_times[0]]
    processes = list(load_times[0][file_names[0]].keys())

    mono_times = [[load_times[trial][file]['mono'] for file in file_names]
                  for trial in range(num_trials)]
    mono_times = np.array(mono_times, dtype=np.float)
    mono_means = np.nanmean(mono_times, axis=0)
    mono_std = np.nanstd(mono_times, axis=0)

    dual_times = [[load_times[trial][file]['dual'] for file in file_names]
                  for trial in range(num_trials)]
    dual_times = np.array(dual_times, dtype=np.float)
    dual_means = np.nanmean(dual_times, axis=0)
    dual_std = np.std(dual_times, axis=0)


    tri_times = [[load_times[trial][file]['tri'] for file in file_names]
                  for trial in range(num_trials)]
    tri_times = np.array(tri_times, dtype=np.float)
    tri_means = np.nanmean(tri_times, axis=0)
    tri_std = np.std(tri_times, axis=0)

    ind = np.arange(len(mono_means))  # the x locations for the groups
    width = 0.6  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)

    rects1 = ax.bar(ind - width / 3, mono_means, width/3, yerr=mono_std,
                    label='Mono')
    rects2 = ax.bar(ind, dual_means, width/3, yerr=dual_std,
                    label='Dual')
    rect3 = ax.bar(ind + width / 3, tri_means, width/3, yerr=tri_std,
                    label='Tri')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Load time (s)')
    ax.set_title('Load times by file number and processing type (N=20)')

    ax.set_xticks(ind)
    ax.set_xticklabels((f"file {x}" for x in range(1,len(file_names)+1))) #, rotation='vertical')
    ax.legend()

    autolabel(rects1)   #, "left")
    autolabel(rects2)   #, "center")
    autolabel(rect3)    #, "right")

    fig.tight_layout()

    plt.show()