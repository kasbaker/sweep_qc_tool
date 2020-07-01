import sys
import argparse
import os
from pathlib import Path
from pyqtgraph import setConfigOption
from main import Application

import cProfile
import pstats
import datetime as dt


def main(nwb_file):

    import logging
    logging.getLogger().setLevel(logging.INFO)

    setConfigOption("background", "w")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.getcwd(), type=str,
                        help="output path for manual states")
    parser.add_argument("--backup_experiment_start_index", type=int, default=5000,
                        help="when plotting experiment pulses, where to set the start index if it is erroneously stored as <= 0"
                        )
    parser.add_argument("--experiment_baseline_start_index", type=int, default=5000,
                        help="when plotting experiment pulses, where to start the baseline assessment epoch"
                        )
    parser.add_argument("--experiment_baseline_end_index", type=int, default=9000,
                        help="when plotting experiment pulses, where to end the baseline assessment epoch"
                        )
    parser.add_argument("--test_pulse_plot_start", type=float, default=0.04,
                        help="where in time (s) to start the test pulse plot"
                        )
    parser.add_argument("--test_pulse_plot_end", type=float, default=0.1,
                        help="in seconds, the end time of the test pulse plot's domain"
                        )
    parser.add_argument("--test_pulse_baseline_samples", type=int, default=100,
                        help="when plotting test pulses, how many samples to use for baseline assessment"
                        )
    parser.add_argument("--thumbnail_step", type=float, default=20,
                        help="step size for generating decimated thumbnail images for individual sweeps."
                        )
    parser.add_argument("--initial_nwb_path", type=str, default=nwb_file,
                        help="upon start, immediately load an nwb file from here"
                        )
    parser.add_argument("--initial_stimulus_ontology_path", type=str, default=None,
                        help="upon start, immediately load a stimulus ontology from here"
                        )
    parser.add_argument("--initial_qc_criteria_path", type=str, default=None,
                        help="upon start, immediately load qc criteria from here"
                        )

    args = parser.parse_args()

    app = Application(**args.__dict__)

    exit_code = app.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    files = os.listdir("data/nwb")
    base_dir = Path(__file__).parent
    today = dt.datetime.now().strftime('%y%m%d')
    now = dt.datetime.now().strftime('%H.%M.%S')
    profile_dir = base_dir.joinpath(f'profiles/{today}/{now}')
    profile_dir.mkdir(parents=True)
    for index in range(len(files)):
        nwb_file = str(base_dir.joinpath(f'data/nwb/{files[index]}'))
        profile_file = str(profile_dir.joinpath(f'{files[index][0:-4]}.prof'))
        cProfile.run('main(nwb_file)', filename=profile_file)
        p = pstats.Stats(profile_file)
        p.sort_stats('cumtime').print_stats(20)
