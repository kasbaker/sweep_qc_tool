import io
import os
from pathlib import Path
import multiprocessing as mp
from copy import deepcopy
import datetime as dt
from json import load
from warnings import filterwarnings

# from concurrent.futures import ProcessPoolExecutor
from PyQt5.QtCore import QByteArray #QRectF
# from PyQt5.QtSvg import QSvgRenderer
# from PyQt5.QtGui import QPainter, QPixmap
# from PyQt5.QtWidgets import QStyleOptionViewItem
import matplotlib.pyplot as plt
import numpy as np

from ipfx.dataset.create import create_ephys_data_set
from ipfx.qc_feature_extractor import cell_qc_features, sweep_qc_features
from ipfx.qc_feature_evaluator import qc_experiment, DEFAULT_QC_CRITERIA_FILE
from ipfx.bin.run_qc import qc_summary
from ipfx.sweep_props import drop_tagged_sweeps
from ipfx.ephys_data_set import EphysDataSet
from ipfx.stimulus import StimulusOntology

# from main import MainWindow, PlotPage, SweepPage
# from sweep_table_model import SweepTableModel
# from sweep_table_view import SweepTableView
# from cell_feature_page import CellFeaturePage
# from sweep_plotter import SweepPlotConfig

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

filterwarnings('ignore')

with open(DEFAULT_QC_CRITERIA_FILE, "r") as path:
    QC_CRITERIA = load(path)

with open(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE, "r") as path:
    STIMULUS_ONTOLOGY = StimulusOntology(load(path))


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


def run_sweep_qc(nwb_file: str, sweep_qc_pipe: mp.Pipe = None):
    """ Creates data set, runs sweep QC, and pipes out sweep features. """
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


def run_experiment_qc(cell_qc_pipe: mp.Pipe, sweep_qc_pipe: mp.Pipe, qc_pipe: mp.Pipe):
    """" Run experiment qc after receiving cell and sweep qc data"""
    # unpack cell qc pipe
    cell_qc_input, cell_qc_output = cell_qc_pipe
    cell_qc_output.close()  # close other end of cell_qc pipe
    cell_features, cell_tags = cell_qc_input.recv()  # receive cell qc features

    # unpack sweep qc pipe
    sweep_qc_input, sweep_qc_output = sweep_qc_pipe
    sweep_qc_output.close()  # close other end of sweep_qc pipe
    sweep_features = sweep_qc_input.recv()  # receive sweep qc features

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

    _, qc_out = qc_pipe
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
    else:
        return thumbs


def main(nwb_file, dual=False, tri=False, quad=False):
    # multiprocessing == True if dual or quad == True
    multi = dual or tri or quad
    # do this if we are using multiprocessing
    if multi:

        if dual:
            # pipe for qc data
            qc_pipe = mp.Pipe(duplex=False)
            # worker to do auto-qc
            qc_worker = mp.Process(
                name="qc_worker",
                target=run_auto_qc, args=(nwb_file, qc_pipe)
            )
            qc_worker.start()   # start qc worker

            qc_pipe[1].close()

            try:
                qc_results = qc_pipe[0].recv()
                for result in qc_results:
                    print(result)
            except EOFError:
                print("error no qc data")

        if tri:
            # pipe and worker for sweep qc data
            sweep_qc_pipe = mp.Pipe(duplex=False)
            sweep_qc_worker = mp.Process(
                name="sweep_qc_worker",
                target=run_sweep_qc, args=(nwb_file, sweep_qc_pipe)
            )
            sweep_qc_worker.start()  # start sweep qc worker

            # pipe and worker for cell qc data
            cell_qc_pipe = mp.Pipe(duplex=False)
            cell_qc_worker = mp.Process(
                name="cell_qc_worker",
                target=run_cell_qc, args=(nwb_file, cell_qc_pipe)
            )
            cell_qc_worker.start()  # start cell qc worker

            # close sweep and cell qc pipes
            sweep_qc_pipe[1].close()
            cell_qc_pipe[1].close()

        if quad:
            # pipe for qc data
            experiment_qc_pipe = mp.Pipe(duplex=False)
            # pipe and worker for sweep qc data
            sweep_qc_pipe = mp.Pipe(duplex=False)
            sweep_qc_worker = mp.Process(
                name="sweep_qc_worker",
                target=run_sweep_qc, args=(nwb_file, sweep_qc_pipe)
            )
            sweep_qc_worker.start()  # start sweep qc worker

            # pipe and worker for cell qc data
            cell_qc_pipe = mp.Pipe(duplex=False)
            cell_qc_worker = mp.Process(
                name="cell_qc_worker",
                target=run_cell_qc, args=(nwb_file, cell_qc_pipe)
            )
            cell_qc_worker.start()  # start cell qc worker

            # worker to process qc results
            experiment_qc_worker = mp.Process(
                name="experiment_qc_worker",
                target=run_experiment_qc, args=(cell_qc_pipe, sweep_qc_pipe, qc_pipe)
            )
            experiment_qc_worker.start()    # start experiment qc worker

            # close sweep and cell qc pipes
            sweep_qc_pipe[1].close()
            cell_qc_pipe[1].close()
            experiment_qc_pipe[1].close()
            try:
                qc_results = experiment_qc_pipe[0].recv()
                for result in qc_results:
                    print(result)
            except EOFError:
                print("error no qc data")

        # pipe and worker for plot data
        plot_pipe = mp.Pipe(duplex=False)
        plot_worker = mp.Process(
            name="plot_worker", target=make_plots, args=(nwb_file, plot_pipe)
        )
        plot_worker.start()  # start plot worker

        # close plot pipe

        # close qc pipe

        try:
            thumbnails = plot_pipe[0].recv()
        except EOFError:
            print("error no plot data")


        # join and terminate workers
        plot_worker.join()
        if dual:
            qc_worker.join()
            qc_worker.terminate()
        if quad:
            sweep_qc_worker.join()
            experiment_qc_worker.join()
            sweep_qc_worker.terminate()
            experiment_qc_worker.terminate()
        plot_worker.terminate()

    else:
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
        for result in qc_results:
            print(result)


if __name__ == '__main__':
    files = os.listdir("data/nwb")
    base_dir = Path(__file__).parent

    today = dt.datetime.now().strftime('%y%m%d')
    now = dt.datetime.now().strftime('%H.%M.%S')

    profile_dir = base_dir.joinpath(f'profiles/{today}/{now}/')
    profile_dir.mkdir(parents=True)

    for index in range(len(files)):
        nwb_file = str(base_dir.joinpath(f'data/nwb/{files[index]}'))

        # main(nwb_file=nwb_file, quad=True)

        # benchmark dual processing
        dual_profile_file = str(profile_dir.joinpath(f'dual_{files[index][0:-4]}.prof'))
        cProfile.run('main(nwb_file, dual=True)', filename=dual_profile_file)
        p = pstats.Stats(dual_profile_file)
        p.sort_stats('cumtime').print_stats(2)

        # benchmark quad processing
        quad_profile_file = str(profile_dir.joinpath(f'quad_{files[index][0:-4]}.prof'))
        cProfile.run(
            f'main(nwb_file, quad=True)',
            filename=quad_profile_file
        )
        p = pstats.Stats(quad_profile_file)
        p.sort_stats('cumtime').print_stats(2)

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
