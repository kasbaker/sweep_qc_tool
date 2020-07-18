import io
import os
from pathlib import Path
import multiprocessing as mp
from copy import deepcopy
import datetime as dt
import json
from warnings import filterwarnings
from timeit import default_timer

from PyQt5.QtCore import QByteArray
import matplotlib.pyplot as plt
import numpy as np

from ipfx.dataset.create import create_ephys_data_set
from ipfx.qc_feature_extractor import cell_qc_features, sweep_qc_features
from ipfx.qc_feature_evaluator import qc_experiment, DEFAULT_QC_CRITERIA_FILE
from ipfx.bin.run_qc import qc_summary
from ipfx.sweep_props import drop_tagged_sweeps
from ipfx.stimulus import StimulusOntology


import cProfile
import pstats

# CONFIG = SweepPlotConfig(
#     test_pulse_plot_start= 0.04,
#     test_pulse_plot_end=0.1,
#     test_pulse_baseline_samples=100,
#     backup_experiment_start_index=5000,
#     experiment_baseline_start_index=5000,
#     experiment_baseline_end_index=9000,
#     thumbnail_step=20
# )

NUM_TRIALS = 20

filterwarnings('ignore')

with open(DEFAULT_QC_CRITERIA_FILE, "r") as path:
    QC_CRITERIA = json.load(path)

with open(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE, "r") as path:
    STIMULUS_ONTOLOGY = StimulusOntology(json.load(path))


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


def run_auto_qc(nwb_file: str, experiment_qc_pipe: mp.Pipe = None):
    # create data set
    data_set = create_ephys_data_set(nwb_file=nwb_file)

    # sweep QC worker
    sweep_features = sweep_qc_features(data_set)
    sweep_features = deepcopy(sweep_features)
    drop_tagged_sweeps(sweep_features)

    # cell QC worker
    cell_features, cell_tags = cell_qc_features(data_set)
    cell_features = deepcopy(cell_features)

    # experiment QC worker
    cell_state, sweep_states = qc_experiment(
        ontology=STIMULUS_ONTOLOGY,
        cell_features=cell_features,
        sweep_features=sweep_features,
        qc_criteria=QC_CRITERIA
    )

    qc_summary(
        sweep_features=sweep_features,
        sweep_states=sweep_states,
        cell_features=cell_features,
        cell_state=cell_state
    )

    qc_results = (
        cell_features, cell_tags, sweep_features, cell_state, sweep_states
    )

    _, qc_out = experiment_qc_pipe
    qc_out.send(qc_results)
    qc_out.close()


def run_cell_qc(nwb_file: str, cell_qc_pipe: mp.Pipe = None):
    """ Creates data set, runs cell QC, and pipes out sweep features. """
    # start_time = default_timer()
    # create data set
    data_set = create_ephys_data_set(nwb_file=nwb_file)

    # run cell QC
    cell_features, cell_tags = cell_qc_features(data_set)
    cell_features = deepcopy(cell_features)
    # delete deepcopy(cell features)? I don't know why we needed it

    # pipe out cell QC results
    _, cell_qc_out = cell_qc_pipe
    cell_qc_out.send((cell_features, cell_tags))
    cell_qc_out.close()
    # print(f"cell qc took {default_timer() - start_time} seconds")


def run_sweep_qc(nwb_file: str, sweep_qc_pipe: mp.Pipe = None):
    """ Creates data set, runs sweep QC, and pipes out sweep features. """
    # start_time = default_timer()
    # create data set
    data_set = create_ephys_data_set(nwb_file=nwb_file)

    # run sweep QC
    sweep_features = sweep_qc_features(data_set)

    # copy this so we keep dropped sweep info???
    sweep_features = deepcopy(sweep_features)

    # drop any auto-failed sweep (e.g. MIES fail)
    drop_tagged_sweeps(sweep_features)

    # pipe out sweep features
    _, sweep_qc_out = sweep_qc_pipe
    sweep_qc_out.send(sweep_features)
    sweep_qc_out.close()
    # print(f"sweep qc took {default_timer() - start_time} seconds")


def run_experiment_qc(cell_qc_pipe: mp.Pipe, sweep_qc_pipe: mp.Pipe, qc_pipe: mp.Pipe):
    """" Run experiment qc after receiving cell and sweep qc data"""
    # unpack cell qc pipe
    cell_qc_input, cell_qc_output = cell_qc_pipe
    cell_qc_output.close()  # close other end of cell_qc pipe
    cell_features, cell_tags = cell_qc_input.recv()  # receive cell qc features

    print(cell_features)
    print(cell_tags)

    # unpack sweep qc pipe
    sweep_qc_input, sweep_qc_output = sweep_qc_pipe
    sweep_qc_output.close()  # close other end of sweep_qc pipe
    sweep_features = sweep_qc_input.recv()  # receive sweep qc features

    print(sweep_features)

    cell_state, sweep_states = qc_experiment(
        ontology=STIMULUS_ONTOLOGY,
        cell_features=cell_features,
        sweep_features=sweep_features,
        qc_criteria=QC_CRITERIA
    )

    qc_summary(
        sweep_features=sweep_features,
        sweep_states=sweep_states,
        cell_features=cell_features,
        cell_state=cell_state
    )

    qc_results = (
        cell_features, cell_tags, sweep_features, cell_state, sweep_states
    )

    print(qc_results)

    _, qc_out = qc_pipe
    qc_out.send(qc_results)
    qc_out.close()


def make_plots(nwb_file: str, thumb_pipe: mp.Pipe = None, sweep_datas=None):
    """ Generate thumbnail plots for all sweeps. """
    # start_time = default_timer()
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


def single_process(nwb_file: str):
    """ Does Auto QC and makes plots using single process.

    Parameters:
        nwb_file (str): string of nwb path

    Returns:
        thumbs (list[QByteArray]): a list of plot thumbnails

        qc_results (tuple): cell_features, cell_tags,
            sweep_features, cell_state, sweep_states

    """
    data_set = create_ephys_data_set(nwb_file=nwb_file)

    # cell QC worker
    cell_features, cell_tags = cell_qc_features(data_set)
    cell_features = deepcopy(cell_features)

    # sweep QC worker
    sweep_features = sweep_qc_features(data_set)
    sweep_features = deepcopy(sweep_features)
    drop_tagged_sweeps(sweep_features)

    # experiment QC worker
    cell_state, sweep_states = qc_experiment(
        ontology=STIMULUS_ONTOLOGY,
        cell_features=cell_features,
        sweep_features=sweep_features,
        qc_criteria=QC_CRITERIA
    )

    qc_summary(
        sweep_features=sweep_features,
        sweep_states=sweep_states,
        cell_features=cell_features,
        cell_state=cell_state
    )

    qc_results = (
        cell_features, cell_tags, sweep_features, cell_state, sweep_states
    )

    sweep_datas = list(map(
        data_set.get_sweep_data,
        list(range(len(data_set._data.sweep_numbers)))
    ))  # grab sweep numbers from ._data.sweep_numbers (impolite, but fast)

    thumbs = make_plots(nwb_file=nwb_file, sweep_datas=sweep_datas)

    return thumbs, qc_results


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


def dual_process(nwb_file: str):
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
        target=run_auto_qc, args=(nwb_file, qc_pipe)
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


def tri_process(nwb_file: str):
    """ Does Auto QC and makes plot using three processes.

    Parameters:
        nwb_file (str): string of nwb path

    Returns:
        thumbs (list[QByteArray]): a list of plot thumbnails

        qc_results (tuple): cell_features, cell_tags,
            sweep_features, cell_state, sweep_states

    """
    # pipe and worker for sweep qc data
    sweep_qc_pipe = mp.Pipe(duplex=False)
    sweep_qc_worker = mp.Process(
        name="sweep_qc_worker",
        target=run_sweep_qc, args=(nwb_file, sweep_qc_pipe)
    )

    # pipe and worker for cell qc data
    cell_qc_pipe = mp.Pipe(duplex=False)
    cell_qc_worker = mp.Process(
        name="cell_qc_worker",
        target=run_cell_qc, args=(nwb_file, cell_qc_pipe)
    )

    # pipe and worker for plot data
    plot_pipe, plot_worker = plot_process(nwb_file)

    # start workers
    cell_qc_worker.start()
    sweep_qc_worker.start()
    plot_worker.start()

    # close pipes
    sweep_qc_pipe[1].close()
    cell_qc_pipe[1].close()
    plot_pipe[1].close()

    # receive datas
    thumbs = plot_pipe[0].recv()
    sweep_features = sweep_qc_pipe[0].recv()
    cell_features, cell_tags = cell_qc_pipe[0].recv()

    cell_state, sweep_states = qc_experiment(
        ontology=STIMULUS_ONTOLOGY,
        cell_features=cell_features,
        sweep_features=sweep_features,
        qc_criteria=QC_CRITERIA
    )

    qc_summary(
        sweep_features=sweep_features,
        sweep_states=sweep_states,
        cell_features=cell_features,
        cell_state=cell_state
    )

    qc_results = (
        cell_features, cell_tags, sweep_features, cell_state, sweep_states
    )

    # join workers
    cell_qc_worker.join()
    sweep_qc_worker.join()
    plot_worker.join()

    # kill workers
    cell_qc_worker.terminate()
    sweep_qc_worker.terminate()
    plot_worker.terminate()

    return thumbs, qc_results


def quad_process(nwb_file: str):
    """ Does Auto QC and makes plot using four processes.

    Parameters:
        nwb_file (str): string of nwb path

    Returns:
        thumbs (list[QByteArray]): a list of plot thumbnails

        qc_results (tuple): cell_features, cell_tags,
            sweep_features, cell_state, sweep_states

    """

    # pipe and worker for sweep qc data
    sweep_qc_pipe = mp.Pipe(duplex=False)
    sweep_qc_worker = mp.Process(
        name="sweep_qc_worker",
        target=run_sweep_qc, args=(nwb_file, sweep_qc_pipe)
    )

    # pipe and worker for cell qc data
    cell_qc_pipe = mp.Pipe(duplex=False)
    cell_qc_worker = mp.Process(
        name="cell_qc_worker",
        target=run_cell_qc, args=(nwb_file, cell_qc_pipe)
    )

    # pipe and worker for experiment qc
    experiment_qc_pipe = mp.Pipe(duplex=False)
    experiment_qc_worker = mp.Process(
        name="experiment_qc_worker",
        target=run_experiment_qc, args=(cell_qc_pipe, sweep_qc_pipe, cell_qc_pipe)
    )

    # pipe and worker for plot data
    plot_pipe, plot_worker = plot_process(nwb_file)

    # start workers
    cell_qc_worker.start()
    sweep_qc_worker.start()
    experiment_qc_worker.start()
    plot_worker.start()

    # close pipes
    sweep_qc_pipe[1].close()
    cell_qc_pipe[1].close()
    experiment_qc_pipe[1].close()
    plot_pipe[1].close()

    # receive data
    qc_results = experiment_qc_pipe[0].recv()
    thumbs = plot_pipe[0].recv()

    # join workers
    cell_qc_worker.join()
    sweep_qc_worker.join()
    experiment_qc_worker.join()
    plot_worker.join()

    # kill workers
    cell_qc_worker.terminate()
    sweep_qc_worker.terminate()
    experiment_qc_worker.terminate()
    plot_worker.terminate()

    return thumbs, qc_results


def main(nwb_file, dual=False, tri=False, quad=False):
    start_time = default_timer()

    if dual:
        thumbs, qc_results = dual_process(nwb_file=nwb_file)
        elapsed_time = default_timer() - start_time
        print(f"Dual processing of {nwb_file} took {elapsed_time} to complete")
    elif tri:
        thumbs, qc_results = tri_process(nwb_file=nwb_file)
        elapsed_time = default_timer() - start_time
        print(f"Tri processing of {nwb_file} took {elapsed_time} to complete")
    elif quad:
        thumbs, qc_results = quad_process(nwb_file=nwb_file)
        elapsed_time = default_timer() - start_time
        print(f"Quad processing of {nwb_file} took {elapsed_time} to complete")
    else:
        thumbs, qc_results = single_process(nwb_file=nwb_file)
        elapsed_time = default_timer() - start_time
        print(f"Single processing of {nwb_file} took {elapsed_time} to complete")

    # for thumb in thumbs:
    #     print(thumb)
    # for result in qc_results:
    #     print(result)

    return elapsed_time


if __name__ == '__main__':
    files = os.listdir("data/nwb")
    base_dir = Path(__file__).parent

    today = dt.datetime.now().strftime('%y%m%d')
    now = dt.datetime.now().strftime('%H.%M.%S')

    # profile_dir = base_dir.joinpath(f'profiles/{today}/{now}/')
    # profile_dir.mkdir(parents=True)

    time_file = base_dir.joinpath(f'process_times/{today}_{now}.json')

    times = [
        {files[x]: dict.fromkeys(['mono', 'dual', 'tri']) for x in range(len(files))}
        for _ in range(NUM_TRIALS)
    ]
    for trial in range(NUM_TRIALS):
        print(f"--------TRIAL {trial}--------")
        for index in range(len(files)):
            nwb_file = str(base_dir.joinpath(f'data/nwb/{files[index]}'))

            times[trial][files[index]]['mono'] = main(nwb_file=nwb_file)
            times[trial][files[index]]['dual'] = main(nwb_file=nwb_file, dual=True)
            times[trial][files[index]]['tri'] = main(nwb_file=nwb_file, tri=True)
            # times[files[index]]['quad'] = main(nwb_file=nwb_file, quad=True)

            # main(nwb_file=nwb_file, tri=True)

            # # benchmark dual processing
            # dual_profile_file = str(profile_dir.joinpath(f'dual_{files[index][0:-4]}.prof'))
            # cProfile.run('main(nwb_file, dual=True)', filename=dual_profile_file)
            # p = pstats.Stats(dual_profile_file)
            # p.sort_stats('cumtime').print_stats(2)
            #
            # # benchmark quad processing
            # quad_profile_file = str(profile_dir.joinpath(f'quad_{files[index][0:-4]}.prof'))
            # cProfile.run(
            #     f'main(nwb_file, quad=True)',
            #     filename=quad_profile_file
            # )
            # p = pstats.Stats(quad_profile_file)
            # p.sort_stats('cumtime').print_stats(2)

            # # benchmark single processing
            # profile_file = str(profile_dir.joinpath(f'single_{files[index][0:-4]}.prof'))
            # cProfile.run('main(nwb_file, multi=False)', filename=profile_file)
            # p = pstats.Stats(profile_file)
            # p.sort_stats('cumtime').print_stats(20)
            #
            # multi_profile_file = str(profile_dir.joinpath(f'multi_{files[index][0:-4]}.prof'))
            # cProfile.run(
            #     f'main(nwb_file, multi=True)',
            #     filename=multi_profile_file
            # )
            # p = pstats.Stats(multi_profile_file)
            # p.sort_stats('cumtime').print_stats(20)

        with open(time_file, 'w') as file:
            json.dump(times, file, indent=4)

    for file in times[0]:
        print(f"Elapsed times for {file}")
        for cpu in times[0][file].keys():
            print(f"    {cpu} times: ")
            temp_time = 0
            for trial in range(NUM_TRIALS):
                try:
                    temp_time += times[trial][file][cpu]
                    print(f"            {times[trial][file][cpu]}")
                except TypeError:
                    print(f"            N/A")
            print(f"       avg: {temp_time/NUM_TRIALS}")