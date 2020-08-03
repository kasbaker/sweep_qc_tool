from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


class GroupedBarPlotter(object):
    def __init__(self, fig_size: Tuple[float, float] = (16, 9)):
        """ A plotter which automatically creates and labels grouped bar graphs 
        
        Parameters
        ----------
        fig_size : Tuple[float, float]
            size of the figure [x,y] in inches
        
        """
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(fig_size)

    def label_bars(self, bars: plt.Axes, yerr: np.ndarray):
        """ Takes an axes object containing a bar chart and places a label
        at the top of each bar displaying the mean.
        
        Parameters
        ----------
        
        bars : plt.Axes
            axes object containing the bar chart to be labeled
        yerr : np.ndarray
            array of y errors to offset labels by so they do not overlap

        """

        for index, bar in enumerate(bars):
            # get the height of each bar
            height = np.around(bar.get_height(), 2)
            # annotate each bar with height of the mean placed above error bar
            self.ax.annotate(
                f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height + yerr[index]),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', size=8
            )

    def plot_bars(self, data: List[Dict[str, Dict[str, float]]], title: str, y_label: str):
        """ Takes in a data structure of format:
        [
            # trial 1
            {
                "group1": {"method1": float, "method2": float, "method3": float},
                "group2": {"method1": float, "method2": float, "method3": float},
                "group3": {"method1": float, "method2": float, "method3": float}
            },

            # trial 2
            {
                "group1": {"method1": float, "method2": float, "method3": float},
                "group2": {"method1": float, "method2": float, "method3": float},
                "group3": {"method1": float, "method2": float, "method3": float}
            },

            # trial 3
            ...
        ]
        Then plots bar grouped bar charts associated with these data

        Parameters
        ----------
        data : List[Dict[str, Dict[str, float]]]
            data structure with format shown above
        title : str
            title for the figure
        y_label : str
            label for the y-axis

        """

        # number of trials
        num_trials = len(data)

        # names of groups
        group_names = [group for group in data[0]]

        # list of methods
        methods = list(data[0][group_names[0]].keys())

        # array of indicies for methods
        num_groups = len(group_names)

        # spacing of indices should be: np.arrange(n) - .5n + .5
        group_idx = np.arange(num_groups)

        num_methods = len(methods)
        method_nums = np.arange(num_methods)

        # one extra bar width in between each group
        bar_width = 1/(num_methods+1)

        for idx, method in enumerate(methods):
            load_times = np.array(
                [[data[trial][group][f'{method}'] for group in group_names] for trial in range(num_trials)],
                dtype=np.float
            )
            method_means = np.nanmean(load_times, axis=0)
            method_errs = np.nanstd(load_times, axis=0)

            # not sure if this is entirely correct spacing, but it works for 4 methods
            spacing = bar_width*(method_nums[idx]*(-num_methods + 1)/(num_methods-1)+(num_methods-1)/2)

            bars = self.ax.bar(
                x=group_idx - spacing, height=method_means,
                width=bar_width, yerr=method_errs, label=method
            )
            self.label_bars(bars, yerr=method_errs)

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