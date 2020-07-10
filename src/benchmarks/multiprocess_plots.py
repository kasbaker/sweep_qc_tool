import io
import os
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


from PyQt5.QtCore import QByteArray #QRectF
# from PyQt5.QtSvg import QSvgRenderer
# from PyQt5.QtGui import QPainter, QPixmap
# from PyQt5.QtWidgets import QStyleOptionViewItem

from ipfx.dataset.create import create_ephys_data_set
import matplotlib.pyplot as plt
import numpy as np


# from main import MainWindow, PlotPage, SweepPage
# from sweep_table_model import SweepTableModel
# from sweep_table_view import SweepTableView
# from cell_feature_page import CellFeaturePage
from sweep_plotter import SweepPlotConfig

import cProfile
import pstats
import datetime as dt

CPU_COUNT = mp.cpu_count()

# CONFIG = SweepPlotConfig(
#     test_pulse_plot_start= 0.04,
#     test_pulse_plot_end=0.1,
#     test_pulse_baseline_samples=100,
#     backup_experiment_start_index=5000,
#     experiment_baseline_start_index=5000,
#     experiment_baseline_end_index=9000,
#     thumbnail_step=20
# )


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

def plot_worker(sweep_datas, plotter):
    for sweep in sweep_datas:
        plotter.make_plot(sweep)


def main(nwb_file, sweep_datas, multi=False, concurrent=False, num_procs=CPU_COUNT):

    data_set = create_ephys_data_set(nwb_file=nwb_file)
    num_sweeps = len(data_set._data.sweep_numbers)
    # if multi:
    #     if concurrent:
    #         # with ProcessPoolExecutor(max_workers=num_procs) as executor:
    #         #     executor.map(plotter.make_plot, sweep_datas)
    #         ...
    #
    #     else:
    #         # workers = [mp.Process(name=x,target=plot_worker, args = len)]
    #         tasks = np.array_split(sweep_datas, num_procs)
    #         workers = [
    #             mp.Process(name = f"p{x}", target=plot_worker, args=(tasks[x], MockPlotter()))
    #             for x in range(num_procs)
    #         ]
    #
    #         for worker in workers:
    #             worker.start()
    #         for worker in workers:
    #             worker.join()
    # else:
    #     plotter = MockPlotter()
    #     for sweep in sweep_datas:
    #         plotter.make_plot(sweep)


if __name__ == '__main__':
    files = os.listdir("data/nwb")
    base_dir = Path(__file__).parent

    today = dt.datetime.now().strftime('%y%m%d')
    now = dt.datetime.now().strftime('%H.%M.%S')

    # plotter = MockPlotter()
    # for index in range(len(files)):
    for index in range(len(files)):
        nwb_file = str(base_dir.joinpath(f'data/nwb/{files[index]}'))

        # main(nwb_file, sweep_datas=None)

        profile_dir = base_dir.joinpath(f'profiles/plotter/{today}/{now}/')
        profile_dir.mkdir(parents=True)

        # data_set = create_ephys_data_set(nwb_file=nwb_file)
        # num_sweeps = len(data_set.sweep_table)
        # sweep_datas = list(map(data_set.get_sweep_data, list(range(num_sweeps))))


        # benchmark single processing
        profile_file = str(profile_dir.joinpath(f'{files[index][0:-4]}.prof'))
        cProfile.run('main(nwb_file, sweep_datas=None)', filename=profile_file)
        p = pstats.Stats(profile_file)
        p.sort_stats('cumtime').print_stats(20)

        # benchmark multiprocessing
        # for cpu in range(1, CPU_COUNT):
        #     multi_profile_file = str(profile_dir.joinpath(f'cpu{cpu}_{files[index][0:-4]}.prof'))
        #     cProfile.run(
        #         f'main(nwb_file, sweep_datas=None, multi=True, num_procs={cpu})',
        #         filename=multi_profile_file
        #     )
        #     p = pstats.Stats(multi_profile_file)
        #     p.sort_stats('cumtime').print_stats(20)
        #
        # # benchmark concurrent_futures
        # concurrent_profile_file = str(profile_dir.joinpath(f'concurrent_{files[index][0:-4]}.prof'))
        # cProfile.run(
        #     f'main(nwb_file, plotter, multi=True, concurrent=True, num_procs={cpu})',
        #     filename=concurrent_profile_file
        # )
        # p = pstats.Stats(concurrent_profile_file)
        # p.sort_stats('cumtime').print_stats(20)

