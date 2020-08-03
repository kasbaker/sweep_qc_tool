import json
import pathlib
from typing import List, Dict


import matplotlib.pyplot as plt
import numpy as np
# from tkinter import Tk
# from tkinter import filedialog


class ComparisonPlotter(object):
    def __init__(self, figzie=(16, 9)):
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(figzie)

    def autolabel(self, rects, yerr, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for index, rect in enumerate(rects):
            height = np.around(rect.get_height(), 2)
            self.ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height + yerr[index]),
                        xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom', size=8)

    def plot_bars(self, data: List[Dict[str, Dict[str, float]]], title: str, y_label: str):
        """ Plots clusted bar charts group by 'group name' with different 'methods' associated with them. """
        # number of trials
        num_trials = len(data)

        # names of groups
        group_names = [group for group in data[0]]
        # number of groups
        num_grouops = len(group_names)

        # list of methods
        methods = list(data[0][group_names[0]].keys())


        # array of indicies for methods
        num_groups = len(group_names)
        group_idx = np.arange(num_groups)

        num_methods = len(methods)
        method_nums = np.arange(num_methods)

        # spacing of indices: np.arrange(n) - .5n + .5
        bar_width = 1/(num_methods+1)

        for idx, method in enumerate(methods):
            load_times = np.array(
                [[data[trial][group][f'{method}'] for group in group_names] for trial in range(num_trials)],
                dtype=np.float
            )
            method_means = np.nanmean(load_times, axis=0)
            method_errs = np.nanstd(load_times, axis=0)

            # not sure if this is entirely correct spacing, but it works for 4 methods
            spacing = bar_width*(method_nums[idx]*(-num_methods + 1)/(num_methods-1)+1.5)

            rects = self.ax.bar(
                x=group_idx - spacing, height=method_means,
                width=bar_width, yerr=method_errs, label=method
            )
            self.autolabel(rects, yerr=method_errs)

        # set title and y label
        self.ax.set_title(f"{title} (N={num_trials})")
        self.ax.set_ylabel(y_label)

        # set x ticks and group labels
        self.ax.set_xticks(group_idx)
        self.ax.set_xticklabels(group_names)

        # legend of methods
        self.ax.legend()

        self.fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    # root = Tk()
    # root.withdraw()

    base_dir = pathlib.Path(__file__).parent

    # file_selected = filedialog.askopenfile()

    path = 'qc_times/200803_00.18.30.json'  # ubuntu pen drive
    # path = 'qc_times/200730_14.37.02.json'  # rig 1

    # path = 'qc_times/200727_21.32.23.json'    # laptop
    # path = 'qc_times/200727_22.24.04.json'  # laptop netflix

    # path = 'qc_times/200727_12.49.13.json'  # home windows rig


    # with open(str(file_selected.name), 'r') as file:
    with open(path, 'r') as file:
        load_times = json.load(file)

    file_names = sorted([file for file in load_times[0]], key=str.lower)
    group_names = [str(pathlib.PureWindowsPath(file).stem[0:7]) for file in file_names]

    num_trials = len(load_times)
    bar_data = [{group_names[idx]: load_times[trial][key] for idx, key in enumerate(file_names)} for trial in range(num_trials)]

    comparison_plotter = ComparisonPlotter()
    comparison_plotter.plot_bars(bar_data, f"Load times for Linux pen drive desktop", 'time (s)')
    # num_trials = len(load_times)
    # num_cells = len(load_times[0])
    #
    # file_names = [file for file in load_times[0]]
    # load_methods = list(load_times[0][file_names[0]].keys())
    #
    # method_0 = [[load_times[trial][file][f'{load_methods[0]}'] for file in file_names]
    #             for trial in range(num_trials)]
    # method_0 = np.array(method_0, dtype=np.float)
    # method_0_means = np.nanmean(method_0, axis=0)
    # method_0_std = np.nanstd(method_0, axis=0)
    #
    # method_1 = [[load_times[trial][file][f'{load_methods[1]}'] for file in file_names]
    #             for trial in range(num_trials)]
    # method_1 = np.array(method_1, dtype=np.float)
    # method_1_means = np.nanmean(method_1, axis=0)
    # method_1_std = np.std(method_1, axis=0)
    #
    # method_2 = [[load_times[trial][file][f'{load_methods[2]}'] for file in file_names]
    #             for trial in range(num_trials)]
    # method_2 = np.array(method_2, dtype=np.float)
    # method_2_means = np.nanmean(method_2, axis=0)
    # method_2_std = np.nanstd(method_2, axis=0)
    #
    # method_3 = [[load_times[trial][file][f'{load_methods[3]}'] for file in file_names]
    #             for trial in range(num_trials)]
    # method_3 = np.array(method_3, dtype=np.float)
    # method_3_means = np.nanmean(method_3, axis=0)
    # method_3_std = np.std(method_3, axis=0)
    #
    # ind = np.arange(len(method_0_means))  # the x locations for the groups
    #
    # width = 0.2  # the width of the bars
    # spacing = width/2
    #
    # fig, ax = plt.subplots()
    # fig.set_size_inches(16, 9)
    #
    # rects1 = ax.bar(ind - 3 * spacing, method_0_means, width, yerr=method_0_std,
    #                 label=f'{load_methods[0]}')
    # rects2 = ax.bar(ind - spacing, method_1_means, width, yerr=method_1_std,
    #                 label=f'{load_methods[1]}')
    # rects3 = ax.bar(ind + spacing, method_2_means, width, yerr=method_2_std,
    #                 label=f'{load_methods[2]}')
    # rects4 = ax.bar(ind + 3 * spacing, method_3_means, width, yerr=method_3_std,
    #                 label=f'{load_methods[3]}')
    #
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Load time (s)')
    # ax.set_title(f'Load times by processes and QC method (N={num_trials})')
    #
    # ax.set_xticks(ind)
    #
    # path_list = [Path(file) for file in file_names]
    # ax.set_xticklabels((str(path.stem)[-15:] for path in path_list)) #, rotation='vertical')
    # ax.legend()
    #
    # autolabel(rects1)   #, "left")
    # autolabel(rects2)   #, "center")
    # autolabel(rects3)   #, "left")
    # autolabel(rects4)   #, "center")
    #
    # fig.tight_layout()
    #
    # plt.show()