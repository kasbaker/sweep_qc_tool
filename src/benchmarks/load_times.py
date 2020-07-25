import io
import os
from pathlib import Path
import multiprocessing as mp
from copy import deepcopy
import datetime as dt
import json
from warnings import filterwarnings
from timeit import default_timer
import cProfile
import pstats

from PyQt5.QtCore import QByteArray
import matplotlib.pyplot as plt
import numpy as np

from ipfx.dataset.create import create_ephys_data_set
from ipfx.qc_feature_extractor import cell_qc_features, sweep_qc_features
from ipfx.qc_feature_evaluator import qc_experiment, DEFAULT_QC_CRITERIA_FILE
from ipfx.bin.run_qc import qc_summary
from ipfx.sweep_props import drop_tagged_sweeps
from ipfx.stimulus import StimulusOntology
from benchmarks.fast_qc import QCOperator, slow_qc

# ignore warnings during loading .nwb files
filterwarnings('ignore')


def svg_from_mpl_axes(fig) -> QByteArray:
    """ Convert a matplotlib figure to SVG and store it in a Qt byte array.

    Parameters
    ----------
    fig: mpl.figure.Figure
        a matplotlib figure containing the plot to be turned into a thumbnail

    Returns
    -------
    thumbnail : QByteArray
        a QByteArray used as a thumbnail for the given plot

    """

    data = io.BytesIO()
    fig.savefig(data, format="svg")
    plt.close(fig)

    return QByteArray(data.getvalue())


class MockPlotter:
    def __init__(self):
        self.fig, self. ax = plt.subplots()

    def make_plot(self, sweep_data):
        num_pts = len(sweep_data['response'])
        self.ax.plot(
            np.linspace(0, num_pts/sweep_data['sampling_rate'], num_pts),
            sweep_data['response']
        )
        thumbnail = svg_from_mpl_axes(self.fig)
        self.ax.clear()
        return thumbnail


def run_auto_qc(nwb_file: str, experiment_qc_pipe: mp.Pipe, fast_qc: bool):

    if fast_qc:
        qc_operator = QCOperator(nwb_file=nwb_file)
        qc_results = qc_operator.fast_experiment_qc()
    else:
        qc_results = slow_qc(nwb_file, return_data_set=False)

    _, qc_out = experiment_qc_pipe
    qc_out.send(qc_results)
    qc_out.close()


def make_plots(nwb_file: str, thumb_pipe: mp.Pipe = None, sweep_datas=None):
    """ Generate thumbnail plots for all sweeps. """
    if sweep_datas:
        sweep_datas = sweep_datas
    else:
        data_set = create_ephys_data_set(nwb_file=nwb_file)
        sweep_datas = list(map(
            data_set.get_sweep_data,
            list(range(len(data_set._data.sweep_numbers)))
        ))  # grab sweep numbers from ._data.sweep_numbers (impolite, but fast)

    # initialize a plotter
    plotter = MockPlotter()
    thumbs = list(map(plotter.make_plot, sweep_datas))

    if thumb_pipe:
        # pipe to send thumbnails out
        _, thumb_out = thumb_pipe
        thumb_out.send(thumbs)
        thumb_out.close()
        # print(f"plotting took {default_timer() - start_time} seconds")
    else:
        # print(f"plotting took {default_timer() - start_time} seconds")
        return thumbs


def plot_process(nwb_file: str):
    """ Makes plots using single process.

    Parameters:
        nwb_file (str): string of nwb path

    Returns:
        thumbs (list[QByteArray]): a list of plot thumbnails

    """
    plot_pipe = mp.Pipe(duplex=False)
    plot_worker = mp.Process(
        name="plot_worker", target=make_plots, args=(nwb_file, plot_pipe)
    )
    return plot_pipe, plot_worker


def single_process(nwb_file: str, fast_qc: bool):
    """ Does Auto QC and makes plots using single process.

    Parameters:
        nwb_file (str): string of nwb path

    Returns:
        thumbs (list[QByteArray]): a list of plot thumbnails

        qc_results (tuple): cell_features, cell_tags,
            sweep_features, cell_state, sweep_states

    """
    if fast_qc:
        qc_operator = QCOperator(nwb_file=nwb_file)
        qc_results = qc_operator.fast_experiment_qc()
        data_set = qc_operator.data_set
    else:
        qc_results, data_set = slow_qc(nwb_file=nwb_file, return_data_set=True)

    sweep_datas = list(map(
        data_set.get_sweep_data,
        list(range(len(data_set._data.sweep_numbers)))
    ))  # grab sweep numbers from ._data.sweep_numbers (impolite, but fast)

    thumbs = make_plots(nwb_file=nwb_file, sweep_datas=sweep_datas)

    return thumbs, qc_results


def plot_worker(nwb_file: str, fast_qc: bool):
    plot_pipe, plot_worker = plot_process(nwb_file=nwb_file)
    plot_worker.daemon = True

    plot_worker.start()
    if fast_qc:
        qc_operator = QCOperator(nwb_file=nwb_file)
        qc_results = qc_operator.fast_experiment_qc()
    else:
        qc_results = slow_qc(nwb_file=nwb_file, return_data_set=False)

    plot_pipe[1].close()
    thumbs = plot_pipe[0].recv()
    plot_worker.join()
    plot_worker.terminate()

    return thumbs, qc_results


def qc_worker(nwb_file: str, fast_qc: bool):

    qc_pipe = mp.Pipe(duplex=False)
    # worker to do auto-qc
    qc_worker = mp.Process(
        name="qc_worker",
        target=run_auto_qc, args=(nwb_file, qc_pipe, fast_qc)
    )
    qc_worker.daemon = True

    qc_worker.start()

    data_set = create_ephys_data_set(nwb_file)
    sweep_datas = map(data_set.get_sweep_data,
        data_set._data.sweep_numbers.tolist())
    # grab sweep numbers from ._data.sweep_numbers (impolite, but fast)
    thumbs = make_plots(nwb_file=nwb_file, sweep_datas=sweep_datas)

    qc_pipe[1].close()
    qc_results = qc_pipe[0].recv()
    qc_worker.join()
    qc_worker.terminate()

    return thumbs, qc_results


def dual_process(nwb_file: str, fast_qc: bool):
    """ Does Auto QC and makes plot using two processes.

    Parameters:
        nwb_file (str): string of nwb path

    Returns:
        thumbs (list[QByteArray]): a list of plot thumbnails

        qc_results (tuple): cell_features, cell_tags,
            sweep_features, cell_state, sweep_states

    """

    # spawn qc pipe and worker
    # pipe for qc data
    qc_pipe = mp.Pipe(duplex=False)
    # worker to do auto-qc
    qc_worker = mp.Process(
        name="qc_worker",
        target=run_auto_qc, args=(nwb_file, qc_pipe, fast_qc)
    )

    # spawn plot pipe and worker
    plot_pipe, plot_worker = plot_process(nwb_file=nwb_file)

    # start workers
    qc_worker.start()
    plot_worker.start()

    # close pipes
    qc_pipe[1].close()
    plot_pipe[1].close()

    # receive datas
    thumbs = plot_pipe[0].recv()
    qc_results = qc_pipe[0].recv()

    # join workers
    qc_worker.join()
    plot_worker.join()

    # kill workers
    qc_worker.terminate()
    plot_worker.terminate()

    return thumbs, qc_results


def main(nwb_file, load_method: int, dual=False, fast_qc=False):
    start_time = default_timer()

    if load_method == 0:
        thumbs, qc_results = qc_worker(nwb_file=nwb_file, fast_qc=False)
        elapsed_time = default_timer() - start_time
    elif load_method == 1:
        thumbs, qc_results = qc_worker(nwb_file=nwb_file, fast_qc=True)
        elapsed_time = default_timer() - start_time
    elif load_method == 2:
        thumbs, qc_results = plot_worker(nwb_file=nwb_file, fast_qc=False)
        elapsed_time = default_timer() - start_time
    elif load_method == 3:
        thumbs, qc_results = plot_worker(nwb_file=nwb_file, fast_qc=True)
        elapsed_time = default_timer() - start_time
    else:
        thumbs, qc_results = single_process(nwb_file, fast_qc=fast_qc)
        elapsed_time = default_timer() - start_time

    # if dual:
    #     thumbs, qc_results = dual_process(nwb_file=nwb_file, fast_qc=fast_qc)
    #     elapsed_time = default_timer() - start_time
    # else:
    #     thumbs, qc_results = plot_worker(nwb_file=nwb_file, fast_qc=fast_qc)
    #     elapsed_time = default_timer() - start_time

    # for thumb in thumbs:
    #     print(thumb)
    # for result in qc_results:
    #     print(result)

    return elapsed_time


if __name__ == '__main__':

    num_trials = 2
    profile = False

    files = list(Path("data/nwb").glob("*.nwb"))
    base_dir = Path(__file__).parent

    today = dt.datetime.now().strftime('%y%m%d')
    now = dt.datetime.now().strftime('%H.%M.%S')

    profile_dir = base_dir.joinpath(f'fast_qc_profiles/{today}_{now}')

    time_file = base_dir.joinpath(f'qc_times/{today}_{now}.json')

    load_methods = ('qc_worker-slow', 'qc_worker-fast',
                    'plot_worker-slow', 'plot_worker-fast')

    times = [
        {str(files[x]): dict.fromkeys(
            load_methods
        ) for x in range(len(files))} for _ in range(num_trials)
    ]

    if profile:
        profile_dir.mkdir(parents=True)
        for file in files:
            nwb_file = str(base_dir.joinpath(file))
            fast_qc_file = str(profile_dir.joinpath(f'fast_qc_{file.stem}.prof'))
            cProfile.run('main(str(nwb_file), load_method=-1, fast_qc=True)', filename=fast_qc_file)
            p = pstats.Stats(fast_qc_file)
            p.sort_stats('cumtime').print_stats(10)

            slow_qc_file = str(profile_dir.joinpath(f'slow_qc_{file.stem}.prof'))
            cProfile.run('main(str(nwb_file), load_method=-1, fast_qc=False)', filename=slow_qc_file)
            p = pstats.Stats(slow_qc_file)
            p.sort_stats('cumtime').print_stats(10)

    else:
        for trial in range(num_trials):
            print(f"--------TRIAL {trial}--------")
            for index, file in enumerate(files):
                nwb_file = str(base_dir.joinpath(file))

                # qc_0 = main(
                #     nwb_file=nwb_file, load_method=0
                # )
                # print(f"{load_methods[0]}: {file} took {qc_0} time to load")
                # times[trial][str(files[index])][load_methods[0]] = qc_0
                # with open(time_file, 'w') as save_loc:
                #     json.dump(times, save_loc, indent=4)

                qc_1 = main(
                    nwb_file=nwb_file, load_method=1
                )
                print(f"{load_methods[1]}: {file} took {qc_1} time to load")
                times[trial][str(files[index])][load_methods[1]] = qc_1
                with open(time_file, 'w') as save_loc:
                    json.dump(times, save_loc, indent=4)

                # qc_2 = main(
                #     nwb_file=nwb_file, load_method=2
                # )
                # print(f"{load_methods[2]}: {file} took {qc_2} time to load")
                # times[trial][str(files[index])][load_methods[2]] = qc_2
                # with open(time_file, 'w') as save_loc:
                #     json.dump(times, save_loc, indent=4)

                qc_3 = main(
                    nwb_file=nwb_file, load_method=3
                )
                print(f"{load_methods[3]}: {file} took {qc_3} time to load")
                times[trial][str(files[index])][load_methods[3]] = qc_3
                with open(time_file, 'w') as save_loc:
                    json.dump(times, save_loc, indent=4)

        for file in times[0]:
            print(f"Elapsed times for {file}")
            for cpu in times[0][file].keys():
                print(f"    {cpu} times: ")
                temp_time = 0
                for trial in range(num_trials):
                    try:
                        temp_time += times[trial][file][cpu]
                        print(f"            {times[trial][file][cpu]}")
                    except TypeError:
                        print(f"            N/A")
                print(f"       avg: {temp_time/num_trials}")