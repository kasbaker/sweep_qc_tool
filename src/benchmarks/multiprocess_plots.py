import io
import os
from pathlib import Path
import multiprocessing as mp
from copy import deepcopy
import datetime as dt
from json import load


from concurrent.futures import ProcessPoolExecutor
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
from sweep_plotter import SweepPlotConfig

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


# class PlotProcess(mp.Process):
#     def __init__(self, sweep_datas):
#         super().__init__()
#         self.sweep_datas =
#         self.plotter = MockPlotter
#
#     def run(self):

# def plot_worker(sweep_datas, plotter):
#     for sweep in sweep_datas:
#         plotter.make_plot(sweep)


def run_auto_qc(nwb_file: str, qc_pipe: mp.Pipe = None):

    data_set = create_ephys_data_set(nwb_file=nwb_file)
    cell_features, cell_tags = cell_qc_features(data_set)
    sweep_features = sweep_qc_features(data_set)

    post_qc_sweep_features = deepcopy(sweep_features)
    cell_features = deepcopy(cell_features)

    drop_tagged_sweeps(post_qc_sweep_features)

    cell_state, sweep_states = qc_experiment(
        ontology=STIMULUS_ONTOLOGY,
        cell_features=cell_features,
        sweep_features=post_qc_sweep_features,
        qc_criteria=QC_CRITERIA
    )

    qc_summary(
        sweep_features=post_qc_sweep_features,
        sweep_states=sweep_states,
        cell_features=cell_features,
        cell_state=cell_state
    )

    qc_results = (cell_features, cell_tags, sweep_features,
                  cell_state, sweep_states, post_qc_sweep_features)

    if qc_pipe:
        _, qc_out = qc_pipe
        qc_out.send(qc_results)
        qc_out.close()
    else:
        return data_set, qc_results


def make_plots(nwb_file: str, thumb_pipe: mp.Pipe = None, sweep_datas=None):

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


def main(nwb_file, multi=False):
    if multi:
        plot_pipe = mp.Pipe(duplex=False)
        qc_pipe = mp.Pipe(duplex=False)

        # worker to do auto-qc
        qc_worker = mp.Process(
            name="qc_worker", target=run_auto_qc, args=(nwb_file, qc_pipe)
        )

        # worker to make plots
        plot_worker = mp.Process(
            name="plot_worker", target=make_plots, args=(nwb_file, plot_pipe)
        )

        # start workers
        qc_worker.start()
        plot_worker.start()

        # close pipes
        qc_pipe[1].close()
        plot_pipe[1].close()

        try:
            foo = plot_pipe[0].recv()
        except EOFError:
            print("error no plot data")
        try:
            bar = qc_pipe[0].recv()
        except EOFError:
            print("error no qc data")


        # join workers
        qc_worker.join()
        plot_worker.join()

        # kill workers
        qc_worker.terminate()
        plot_worker.terminate()


    else:
        data_set, qc_results = run_auto_qc(nwb_file)
        sweep_datas = list(map(
            data_set.get_sweep_data,
            list(range(len(data_set._data.sweep_numbers)))
        ))  # grab sweep numbers from ._data.sweep_numbers (impolite, but fast)
        thumbs = make_plots(nwb_file=nwb_file, sweep_datas=sweep_datas)


if __name__ == '__main__':
    files = os.listdir("data/nwb")
    base_dir = Path(__file__).parent

    today = dt.datetime.now().strftime('%y%m%d')
    now = dt.datetime.now().strftime('%H.%M.%S')

    profile_dir = base_dir.joinpath(f'profiles/{today}/{now}/')
    profile_dir.mkdir(parents=True)

    plotter = MockPlotter()
    for index in range(len(files)):
        nwb_file = str(base_dir.joinpath(f'data/nwb/{files[index]}'))

        # main(nwb_file=nwb_file, multi=True)

        # benchmark single processing
        profile_file = str(profile_dir.joinpath(f'single_{files[index][0:-4]}.prof'))
        cProfile.run('main(nwb_file, multi=False)', filename=profile_file)
        p = pstats.Stats(profile_file)
        p.sort_stats('cumtime').print_stats(20)

        multi_profile_file = str(profile_dir.joinpath(f'multi_{files[index][0:-4]}.prof'))
        cProfile.run(
            f'main(nwb_file, multi=True)',
            filename=multi_profile_file
        )
        p = pstats.Stats(multi_profile_file)
        p.sort_stats('cumtime').print_stats(20)

        # # benchmark concurrent_futures
        # concurrent_profile_file = str(profile_dir.joinpath(f'concurrent_{files[index][0:-4]}.prof'))
        # cProfile.run(
        #     f'main(nwb_file, plotter, multi=True, concurrent=True, num_procs={cpu})',
        #     filename=concurrent_profile_file
        # )
        # p = pstats.Stats(concurrent_profile_file)
        # p.sort_stats('cumtime').print_stats(20)

