import json
import pathlib

import numpy as np
from optimization.grouped_bar_plotter import GroupedBarPlotter


if __name__ == "__main__":

    # path = 'qc_times/200803_00.18.30.json'  # ubuntu pen drive
    # path = 'qc_times/200730_14.37.02.json'  # rig 1

    # path = 'qc_times/200727_21.32.23.json'    # laptop
    # path = 'qc_times/200727_22.24.04.json'  # laptop netflix

    # path = 'qc_times/200727_12.49.13.json'  # home windows rig

    rig_files = {
        'rig1': "qc_times/200730_14.37.02.json",
        'rig1_no_onboard': "qc_times/200804_14.27.08.json",
        # 'ubuntu': "qc_times/200803_00.18.30.json",
        # 'windows': "qc_times/200727_12.49.13.json",
        # 'laptop': "qc_times/200727_21.32.23.json",
        # 'laptop_streaming': "qc_times/200727_22.24.04.json"
    }

    method = "qc_worker_pickle"

    rig_names = rig_files.keys()

    rig_times = {rig: [] for rig in rig_names}

    # data = []

    for rig, path in rig_files.items():
        with open(path, 'r') as file:
            load_times = json.load(file)

        file_names = sorted([file for file in load_times[0]], key=str.lower)
        group_names = [str(pathlib.PureWindowsPath(file).stem[0:7]) for file in file_names]

        rig_times[rig] = [
            {
                group_names[idx]: {rig: load_times[trial][file_name][method]}
                for idx, file_name in enumerate(file_names)
            } for trial in range(len(load_times))
        ]

    # find maximum number of trials for data structure initialization
    trial_nums = {key: len(value) for key, value in rig_times.items()}
    max_trial_rig = max(trial_nums, key=trial_nums.get)
    max_trial_num = trial_nums[max_trial_rig]

    # initialize data structure for comparison plotter
    data = [
        {
            group: {
                rig: np.nan for rig in rig_names
            } for group in group_names
        } for trial in range(max_trial_num)
    ]

    # a very messy way of rearranging the data
    for rig in rig_times:
        for trial in range(trial_nums[rig]):
            for group in rig_times[rig][trial]:
                data[trial][group][rig] = rig_times[rig][trial][group][rig]

    comparison_plotter = GroupedBarPlotter()
    comparison_plotter.plot_bars(data, f"Rig comparison for {method} (N=10)", 'time (s)')