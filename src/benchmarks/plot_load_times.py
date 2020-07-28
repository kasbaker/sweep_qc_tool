import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk
from tkinter import filedialog


def autolabel(rects, height_offsets, xpos='center', ):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for index, rect in enumerate(rects):
        height = np.around(rect.get_height(), 2)
        label_height = height + height_offsets[index]
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, label_height),
                    xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom', size=8)


if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    base_dir = Path(__file__).parent

    file_selected = filedialog.askopenfile()

    with open(str(file_selected.name), 'r') as file:
        load_times = json.load(file)

    num_trials = len(load_times)
    num_cells = len(load_times[0])

    file_names = [file for file in load_times[0]]

    plot_data = [{
        'method': method,
        'times': np.array([[load_times[trial][file][f'{method}'] for file in file_names]
                           for trial in range(num_trials)], dtype=np.float),
        'rects': None
    } for method in load_times[0][file_names[0]].keys()]

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    ind = np.arange(len(plot_data[0]['times'][1]))  # the x locations for the groups
    width = 0.2  # the width of the bars
    spacing = width/2

    x_offset = -3
    for index, method in enumerate(plot_data):

        means = np.nanmean(method['times'], axis=0)
        std = np.nanstd(method['times'], axis=0)

        plot_data[index]['rects'] = ax.bar(ind - x_offset * spacing, means, width, yerr=std,
                                           label=f"{method['method']}")
        autolabel(rects=plot_data[index]['rects'], height_offsets=std, xpos='center')

        x_offset += 2

    ax.set_ylabel('Load time (s)')
    ax.set_title(f'Load times by processes and QC method (N={num_trials})')

    ax.set_xticks(ind)
    ax.set_xticklabels((f"file {x}" for x in range(1, len(file_names)+1)))
    ax.legend()

    fig.tight_layout()

    plt.show()