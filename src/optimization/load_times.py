import io
from pathlib import Path
import multiprocessing as mp
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
from ipfx.qc_feature_evaluator import DEFAULT_QC_CRITERIA_FILE
from ipfx.stimulus import StimulusOntology
from data_extractor import DataExtractor
from optimization.fast_qc import QCOperatorLite

# ignore warnings during loading .nwb files
filterwarnings('ignore')


with open(DEFAULT_QC_CRITERIA_FILE, "r") as path:
    QC_CRITERIA = json.load(path)

with open(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE, "r") \
        as path:
    ONTOLOGY = StimulusOntology(json.load(path))


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


def run_auto_qc(nwb_file: str, qc_output):

    data_extractor = DataExtractorLite(nwb_file, ONTOLOGY)
    recording_date = data_extractor.recording_date
    sweep_data_list = list(data_extractor.data_iter)

    qc_operator = QCOperatorLite(sweep_data_list, ONTOLOGY, QC_CRITERIA, recording_date)
    qc_results = qc_operator.fast_experiment_qc()

    qc_out = qc_output
    qc_out.send(qc_results)
    qc_out.close()


def run_auto_qc_pickle(
        sweep_data_list, ontology, qc_criteria, recording_date, qc_output
):
    qc_operator = QCOperatorLite(
        sweep_data_list, ontology, qc_criteria, recording_date
    )
    qc_results = qc_operator.fast_experiment_qc()

    qc_out = qc_output
    qc_out.send(qc_results)
    qc_out.close()


def run_plot_worker(nwb_file: str, thumb_out):
    data_set = create_ephys_data_set(nwb_file)
    sweep_datas = map(data_set.get_sweep_data,
        data_set._data.sweep_numbers.tolist())
    # grab sweep numbers from ._data.sweep_numbers (impolite, but fast)
    thumbs = make_plots(sweep_datas=sweep_datas)

    # connection to send thumbnails out
    thumb_out.send(thumbs)
    thumb_out.close()


def run_plot_worker_pickle(sweep_data_list, thumb_out):
    thumbs = make_plots(sweep_datas=sweep_data_list)

    # connection to send thumbnails out
    thumb_out.send(thumbs)
    thumb_out.close()


def make_plots(sweep_datas):
    """ Generate thumbnail plots for all sweeps. """

    # initialize a plotter
    plotter = MockPlotter()
    thumbs = list(map(plotter.make_plot, sweep_datas))
    return thumbs


def plot_process(nwb_file: str, series_iter):
    """ Makes plots using single process.

    Parameters:
        nwb_file (str): string of nwb path

    Returns:
        thumbs (list[QByteArray]): a list of plot thumbnails

    """
    series_iter = series_iter
    plot_pipe = mp.Pipe(duplex=False)
    plot_worker = mp.Process(
        name="plot_worker", target=make_plots, args=(nwb_file, plot_pipe, series_iter)
    )
    return plot_pipe, plot_worker


def qc_worker(nwb_file: str):

    qc_pipe = mp.Pipe(duplex=False)
    # worker to do auto-qc
    qc_worker = mp.Process(
        name="qc_worker",
        target=run_auto_qc, args=(nwb_file, qc_pipe[1])
    )
    qc_worker.daemon = True

    qc_worker.start()

    data_extractor = DataExtractor(nwb_file=nwb_file, ontology=ONTOLOGY)
    sweep_data_iter = data_extractor.data_iter

    # data_set = create_ephys_data_set(nwb_file)
    # sweep_datas = map(data_set.get_sweep_data,
    #     data_set._data.sweep_numbers.tolist())
    # grab sweep numbers from ._data.sweep_numbers (impolite, but fast)
    thumbs = make_plots(sweep_datas=sweep_data_iter)

    qc_pipe[1].close()
    qc_results = qc_pipe[0].recv()
    qc_worker.join()
    qc_worker.terminate()

    return thumbs, qc_results


def qc_worker_pickle(nwb_file):
    data_extractor = DataExtractor(nwb_file=nwb_file, ontology=ONTOLOGY)
    sweep_data_iter = data_extractor.data_iter
    sweep_data_list = list(sweep_data_iter)
    recording_date = data_extractor.recording_date

    qc_pipe = mp.Pipe(duplex=False)
    # worker to do auto-qc
    qc_worker = mp.Process(
        name="qc_worker",
        target=run_auto_qc_pickle, args=(
            sweep_data_list, ONTOLOGY, QC_CRITERIA, recording_date, qc_pipe[1]
        )
    )
    qc_worker.daemon = True

    qc_worker.start()

    # grab sweep numbers from ._data.sweep_numbers (impolite, but fast)
    thumbs = make_plots(sweep_datas=sweep_data_iter)

    qc_pipe[1].close()
    qc_results = qc_pipe[0].recv()
    qc_worker.join()
    qc_worker.terminate()

    return thumbs, qc_results


def plot_worker(nwb_file: str):
    plot_pipe = mp.Pipe(duplex=False)
    # worker to do make thumbnails
    plot_worker = mp.Process(
        name="qc_worker",
        target=run_plot_worker, args=(nwb_file, plot_pipe[1])
    )
    plot_worker.daemon = True
    plot_worker.start()

    data_extractor = DataExtractor(nwb_file, ONTOLOGY)
    recording_date = data_extractor.recording_date
    sweep_data_list = list(data_extractor.data_iter)

    qc_operator = QCOperatorLite(sweep_data_list, ONTOLOGY, QC_CRITERIA, recording_date)
    qc_results = qc_operator.fast_experiment_qc()

    plot_pipe[1].close()
    thumbs = plot_pipe[0].recv()
    plot_worker.join()
    plot_worker.terminate()

    return thumbs, qc_results


def plot_worker_pickle(nwb_file: str):
    data_extractor = DataExtractor(nwb_file=nwb_file, ontology=ONTOLOGY)
    sweep_data_iter = data_extractor.data_iter
    sweep_data_list = list(sweep_data_iter)

    plot_pipe = mp.Pipe(duplex=False)
    # worker to do make thumbnails
    plot_worker = mp.Process(
        name="qc_worker",
        target=run_plot_worker_pickle, args=(sweep_data_list, plot_pipe[1])
    )
    plot_worker.daemon = True

    plot_worker.start()

    recording_date = data_extractor.recording_date

    qc_operator = QCOperatorLite(sweep_data_list, ONTOLOGY, QC_CRITERIA, recording_date)
    qc_results = qc_operator.fast_experiment_qc()

    plot_pipe[1].close()
    thumbs = plot_pipe[0].recv()
    plot_worker.join()
    plot_worker.terminate()

    return thumbs, qc_results


def main(nwb_file, load_method: int):
    start_time = default_timer()

    if load_method == 0:
        thumbs, qc_results = qc_worker(nwb_file=nwb_file)
        elapsed_time = default_timer() - start_time
    elif load_method == 1:
        thumbs, qc_results = qc_worker_pickle(nwb_file=nwb_file)
        elapsed_time = default_timer() - start_time
    elif load_method == 2:
        thumbs, qc_results = plot_worker(nwb_file=nwb_file)
        elapsed_time = default_timer() - start_time
    elif load_method == 3:
        thumbs, qc_results = plot_worker_pickle(nwb_file=nwb_file)
        elapsed_time = default_timer() - start_time
    else:
    #     thumbs, qc_results = single_process(nwb_file)
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

    load_methods = ('qc_worker', 'qc_worker_pickle',
                    'plot_worker', 'plot_worker_pickle')

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

                qc_0 = main(
                    nwb_file=nwb_file, load_method=0
                )
                print(f"{load_methods[0]}: {file} took {qc_0} time to load")
                times[trial][str(files[index])][load_methods[0]] = qc_0
                with open(time_file, 'w') as save_loc:
                    json.dump(times, save_loc, indent=4)

                qc_1 = main(
                    nwb_file=nwb_file, load_method=1
                )
                print(f"{load_methods[1]}: {file} took {qc_1} time to load")
                times[trial][str(files[index])][load_methods[1]] = qc_1
                with open(time_file, 'w') as save_loc:
                    json.dump(times, save_loc, indent=4)

                qc_2 = main(
                    nwb_file=nwb_file, load_method=2
                )
                print(f"{load_methods[2]}: {file} took {qc_2} time to load")
                times[trial][str(files[index])][load_methods[2]] = qc_2
                with open(time_file, 'w') as save_loc:
                    json.dump(times, save_loc, indent=4)

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