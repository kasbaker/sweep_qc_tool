from pathlib import Path
from timeit import default_timer
from warnings import filterwarnings
import datetime as dt
from copy import deepcopy
import json

from ipfx.qc_feature_extractor import cell_qc_features, sweep_qc_features
from ipfx.qc_feature_evaluator import qc_experiment, DEFAULT_QC_CRITERIA_FILE
from ipfx.dataset.create import create_ephys_data_set

from ipfx.sweep_props import drop_tagged_sweeps
from ipfx.bin.run_qc import qc_summary
from ipfx.stimulus import StimulusOntology

from data_extractor import DataExtractor
from qc_operator import QCOperator, QCResults

with open(DEFAULT_QC_CRITERIA_FILE, "r") as path:
    QC_CRITERIA = json.load(path)

with open(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE, "r") \
        as path:
    ONTOLOGY = StimulusOntology(json.load(path))


def slow_qc(nwb_file: str, return_data_set = False):
    """ Does Auto QC and makes plots using single process.

    Parameters:
        nwb_file (str): string of nwb path

    Returns:
        thumbs (list[QByteArray]): a list of plot thumbnails

        qc_results (tuple): cell_features, cell_tags,
            sweep_features, cell_state, sweep_states

    """

    data_set = create_ephys_data_set(nwb_file=nwb_file, ontology=ONTOLOGY)

    # cell QC worker
    cell_features, cell_tags = cell_qc_features(data_set)
    cell_features = deepcopy(cell_features)

    # sweep QC worker
    sweep_features = sweep_qc_features(data_set)
    sweep_features = deepcopy(sweep_features)
    drop_tagged_sweeps(sweep_features)

    # experiment QC worker
    cell_state, sweep_states = qc_experiment(
        ontology=ONTOLOGY,
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

    qc_results = QCResults(
        cell_features=cell_features,
        cell_tags=cell_tags,
        cell_state=cell_state,
        sweep_features=sweep_features,
        sweep_states=sweep_states
    )
    if return_data_set:
        return qc_results, data_set
    else:
        return qc_results


if __name__ == "__main__":

    num_trials = 1

    # ignore warnings during loading .nwb files
    filterwarnings('ignore')

    files = list(Path("data/nwb").glob("*.nwb"))
    base_dir = Path(__file__).parent

    today = dt.datetime.now().strftime('%y%m%d')
    now = dt.datetime.now().strftime('%H.%M.%S')

    # profile_dir = base_dir.joinpath(f'fast_qc_profiles/{today}_{now}')
    # profile_dir.mkdir(parents=True)

    time_file = base_dir.joinpath(f'qc_times/{today}_{now}.json')

    times = [
        {str(files[x]): dict.fromkeys(['slow_qc', 'fast_qc']) for x in range(len(files))}
        for _ in range(num_trials)
    ]
    for trial in range(num_trials):
        print(f"-----------------TRIAL {trial}-----------------")
        for index, file in enumerate(files):
            nwb_file = str(base_dir.joinpath(file))
            start_time = default_timer()
            data_extractor = DataExtractor(nwb_file=nwb_file, ontology=ONTOLOGY)
            sweep_data_tuple = tuple(data_extractor.data_iter)
            recording_date = data_extractor.recording_date
            qc_operator = QCOperator(
                sweep_data_tuple=sweep_data_tuple, ontology=ONTOLOGY,
                qc_criteria=QC_CRITERIA, recording_date=recording_date
            )

            qc_results, full_sweep_qc_info, sweep_types = qc_operator.fast_experiment_qc()

            fast_qc_time = default_timer()-start_time
            print(f'Fast QC: {file} took {fast_qc_time} to load')
            times[trial][str(files[index])]['fast_qc'] = fast_qc_time

            start_time = default_timer()
            slow_qc_results = slow_qc(nwb_file=nwb_file)
            slow_qc_time = default_timer()-start_time
            print(f'Slow QC: {file} took {slow_qc_time} to load')
            times[trial][str(files[index])]['slow_qc'] = slow_qc_time

            fast_qc_keys = set(qc_results.sweep_features[0].keys())
            slow_qc_keys = set(slow_qc_results.sweep_features[0].keys())

            pop_keys = fast_qc_keys.symmetric_difference(slow_qc_keys)

            slow_qc_sweep_features = list(slow_qc_results.sweep_features)

            for idx, sweep in enumerate(slow_qc_sweep_features):
                for key in pop_keys:
                    slow_qc_sweep_features[idx].pop(key)

            print(f"Cell features equal? "
                  f"{slow_qc_results.cell_features == qc_results.cell_features}")
            print(f"Cell tags equal? "
                  f"{slow_qc_results.cell_tags == qc_results.cell_tags}")
            print(f"Cell state equal? "
                  f"{slow_qc_results.cell_state == qc_results.cell_state}")
            print(f"Sweep Features equal? "
                  f" {slow_qc_sweep_features == qc_results.sweep_features}")
            print(f"Sweep_states equal? "
                  f"{slow_qc_results.sweep_states == qc_results.sweep_states}")
            print('----------------------------------------------------------')

            with open(time_file, 'w') as save_loc:
                json.dump(times, save_loc, indent=4)

    for file in times[0]:
        print(f"Elapsed times for {file}")
        for qc_mode in times[0][file].keys():
            print(f"    {qc_mode} times: ")
            temp_time = 0
            for trial in range(num_trials):
                try:
                    temp_time += times[trial][file][qc_mode]
                    print(f"            {times[trial][file][qc_mode]}")
                except TypeError:
                    print(f"            N/A")
                    print(f"       avg: {temp_time/num_trials}")
