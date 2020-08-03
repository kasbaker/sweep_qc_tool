import json
import pathlib

from optimization.grouped_bar_plotter import GroupedBarPlotter


if __name__ == "__main__":

    # path = 'qc_times/200803_00.18.30.json'  # ubuntu pen drive
    # path = 'qc_times/200730_14.37.02.json'  # rig 1

    # path = 'qc_times/200727_21.32.23.json'    # laptop
    # path = 'qc_times/200727_22.24.04.json'  # laptop netflix

    # path = 'qc_times/200727_12.49.13.json'  # home windows rig

    rig_files = {
        'rig1': "qc_times/200730_14.37.02.json",
        'ubuntu': "qc_times/200803_00.18.30.json",
        'windows': "qc_times/200727_12.49.13.json"
    }

    data = []

    for rig, path in rig_files.items():
        with open(path, 'r') as file:
            load_times = json.load(file)




    # with open(str(file_selected.name), 'r') as file:
    # with open(path, 'r') as file:
    #     load_times = json.load(file)

    file_names = sorted([file for file in load_times[0]], key=str.lower)
    group_names = [str(pathlib.PureWindowsPath(file).stem[0:7]) for file in file_names]

    num_trials = len(load_times)
    bar_data = [{group_names[idx]: load_times[trial][key] for idx, key in enumerate(file_names)} for trial in
                range(num_trials)]

    comparison_plotter = GroupedBarPlotter()
    comparison_plotter.plot_bars(bar_data, f"Load times for Linux pen drive desktop", 'time (s)')