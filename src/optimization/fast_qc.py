from pathlib import Path
from timeit import default_timer
from warnings import filterwarnings
import datetime as dt
import logging
from copy import copy, deepcopy
import json
from typing import Tuple, List, Dict, Union
from multiprocessing import Process

import numpy as np
from pynwb.icephys import (
    CurrentClampSeries, CurrentClampStimulusSeries,
    VoltageClampSeries, VoltageClampStimulusSeries,
    IZeroClampSeries
)
from ipfx.qc_feature_extractor import cell_qc_features, sweep_qc_features
from ipfx.qc_feature_evaluator import qc_experiment, DEFAULT_QC_CRITERIA_FILE
from ipfx.dataset.create import create_ephys_data_set
import ipfx.epochs as ep
import ipfx.qc_features as qcf
import ipfx.stim_features as stf
from ipfx import error as er
from ipfx.qc_feature_extractor \
    import extract_recording_date, compute_input_access_resistance_ratio
from ipfx.sweep_props import drop_tagged_sweeps
from ipfx.bin.run_qc import qc_summary
from ipfx.stimulus import StimulusOntology
from data_extractor import DataExtractor

with open(DEFAULT_QC_CRITERIA_FILE, "r") as path:
    QC_CRITERIA = json.load(path)

with open(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE, "r") \
        as path:
    ONTOLOGY = StimulusOntology(json.load(path))


class QCOperatorLite(object):
    __slots__ = [
        '_sweep_data_tuple', '_ontology', '_qc_criteria', '_recording_date'
    ]

    def __init__(
            self,
            sweep_data_tuple: Tuple[Dict[str, Union[int, str, np.ndarray, float, Dict[str, tuple]]]],
            ontology: StimulusOntology, qc_criteria: list, recording_date: str
    ):
        self._sweep_data_tuple = sweep_data_tuple
        self._ontology = ontology
        self._qc_criteria = qc_criteria
        self._recording_date = recording_date

    @property
    def sweep_data_tuple(self):
        return self._sweep_data_tuple

    @property
    def ontology(self):
        return self._ontology

    @property
    def qc_criteria(self):
        return self._qc_criteria

    @property
    def recording_date(self):
        return self._recording_date

    def fast_cell_qc(self, sweep_types, manual_values=None):

        if manual_values is None:
            manual_values = {}

        features = {}
        tags = []

        features['blowout_mv'] = self.fast_extract_blowout(sweep_types['blowouts'], tags)

        features['electrode_0_pa'] = self.fast_extract_electrode_0(sweep_types['baths'], tags)

        if self.recording_date is None:
            tags.append("Recording date is missing")
        features['recording_date'] = self.recording_date

        features["seal_gohm"] = self.fast_extract_clamp_seal(sweep_types['seals'], tags,
                                                        manual_values)

        input_resistance, access_resistance = \
            self.fast_extract_input_and_acess_resistance(sweep_types['breakins'], tags)

        features['input_resistance_mohm'] = input_resistance
        features["initial_access_resistance_mohm"] = access_resistance

        features['input_access_resistance_ratio'] = \
            compute_input_access_resistance_ratio(input_resistance, access_resistance)

        return features, tags

    def fast_experiment_qc(self):

        # initialize a list of dictionaries to be used in sweep table model

        # get sweep_types and update sweep table data

        sweep_types = self.get_sweep_types()

        cell_features, cell_tags = self.fast_cell_qc(sweep_types)
        cell_features = deepcopy(cell_features)

        pre_qc_sweep_features = self.fast_sweep_qc(sweep_types)
        post_qc_sweep_features = deepcopy(pre_qc_sweep_features)
        drop_tagged_sweeps(post_qc_sweep_features)

        cell_state, sweep_states = qc_experiment(
            ontology=self.ontology,
            cell_features=cell_features,
            sweep_features=post_qc_sweep_features,
            qc_criteria=self.qc_criteria
        )
        qc_summary(
            sweep_features=post_qc_sweep_features,
            sweep_states=sweep_states,
            cell_features=cell_features,
            cell_state=cell_state
        )

        sweep_table_data = [{
            'sweep_number': sweep['sweep_number'],
            'stimulus_code': sweep['stimulus_code'],
            'stimulus_name': sweep['stimulus_name'],
            'auto_qc_state': "n/a", 'manual_qc_state': "default", 'tags': []
        } for sweep in self.sweep_data_tuple]

        sweep_table_data = self.get_sweep_table_data(
            sweep_table_data=sweep_table_data,
            pre_qc_sweep_features=pre_qc_sweep_features,
            post_qc_sweep_features=post_qc_sweep_features,
            sweep_states=sweep_states
        )

        qc_results = (
            cell_features, cell_tags, pre_qc_sweep_features, cell_state, sweep_states
        )

        return qc_results, sweep_table_data, sweep_types

    def get_sweep_types(self):
        sweep_keys = (
            'v_clamps', 'i_clamps', 'blowouts', 'baths', 'seals', 'breakins',
            'ramps', 'long_squares', 'coarse_long_squares', 'short_square_triples',
            'short_squares', 'searches', 'tests', 'extps'
        )

        sweep_types = {key: [] for key in sweep_keys}

        ontology = self.ontology

        for sweep in self.sweep_data_tuple:
            sweep_num = sweep['sweep_number']
            stim_unit = sweep['stimulus_unit']

            if stim_unit == "Volts":
                sweep_types['v_clamps'].append(sweep_num)
            if stim_unit == "Amps":
                sweep_types['i_clamps'].append(sweep_num)

            stim_code = sweep['stimulus_code']
            stim_name = sweep['stimulus_name']
            if stim_name in ontology.extp_names or ontology.extp_names[0] in stim_code:
                sweep_types['extps'].append(sweep_num)
            if stim_name in ontology.test_names:
                sweep_types['tests'].append(sweep_num)
            if stim_name in ontology.blowout_names or ontology.blowout_names[0] in stim_code:
                sweep_types['blowouts'].append(sweep_num)
            if stim_name in ontology.bath_names or ontology.bath_names[0] in stim_code:
                sweep_types['baths'].append(sweep_num)
            if stim_name in ontology.seal_names or ontology.seal_names[0] in stim_code:
                sweep_types['seals'].append(sweep_num)
            if stim_name in ontology.breakin_names or ontology.breakin_names[0] in stim_code:
                sweep_types['breakins'].append(sweep_num)

            if stim_name in ontology.ramp_names:
                sweep_types['ramps'].append(sweep_num)
            if stim_name in ontology.long_square_names:
                sweep_types['long_squares'].append(sweep_num)
            if stim_name in ontology.coarse_long_square_names:
                sweep_types['coarse_long_squares'].append(sweep_num)
            if stim_name in ontology.short_square_triple_names:
                sweep_types['short_square_triples'].append(sweep_num)
            if stim_name in ontology.short_square_names:
                sweep_types['short_squares'].append(sweep_num)
            if stim_name in ontology.search_names:
                sweep_types['searches'].append(sweep_num)

        return sweep_types

    def fast_extract_blowout(self, blowout_sweeps, tags):
        if blowout_sweeps:
            last_blowout_sweep = self.sweep_data_tuple[blowout_sweeps[-1]]
            blowout_mv = qcf.measure_blowout(
                last_blowout_sweep['response'], last_blowout_sweep['epochs']['test'][1]
            )
        else:
            tags.append("Blowout is not available")
            blowout_mv = None
        return blowout_mv

    def fast_extract_electrode_0(self, bath_sweeps: list, tags):
        if bath_sweeps:
            last_bath_sweep = self.sweep_data_tuple[bath_sweeps[-1]]
            e0 = qcf.measure_electrode_0(last_bath_sweep['response'], last_bath_sweep['sampling_rate'])
        else:
            tags.append("Electrode 0 is not available")
            e0 = None
        return e0

    def fast_extract_clamp_seal(self, seal_sweeps: list, tags, manual_values=None):
        if seal_sweeps:
            last_seal_sweep = self.sweep_data_tuple[seal_sweeps[-1]]

            # num_pts = len(last_seal_sweep['stimulus'])

            time = np.arange(
                len(last_seal_sweep['stimulus'])
            ) / last_seal_sweep['sampling_rate']

            # time = np.linspace(0, num_pts/last_seal_sweep['sampling_rate'], num_pts)

            seal_gohm = qcf.measure_seal(
                last_seal_sweep['stimulus'], last_seal_sweep['response'], time
            )
            if seal_gohm is None or not np.isfinite(seal_gohm):
                raise er.FeatureError("Could not compute seal")
        else:
            tags.append("Seal is not available")
            seal_gohm = manual_values.get('manual_seal_gohm', None)
            if seal_gohm is not None:
                tags.append("Using manual seal value: %f" % seal_gohm)

        return seal_gohm

    def fast_extract_input_and_acess_resistance(self, breakin_sweeps: list, tags):
        if breakin_sweeps:
            last_breakin_sweep = self.sweep_data_tuple[breakin_sweeps[-1]]
            # num_pts = len(last_breakin_sweep['stimulus'])
            # time = np.linspace(0, num_pts/last_breakin_sweep['sampling_rate'], num_pts)
            time = np.arange(
                len(last_breakin_sweep['stimulus'])
            ) / last_breakin_sweep['sampling_rate']
            try:
                input_resistance = qcf.measure_input_resistance(
                    last_breakin_sweep['stimulus'], last_breakin_sweep['response'], time
                )
            except Exception as e:
                logging.warning("Error reading input resistance.")
                raise

            try:
                access_resistance = qcf.measure_initial_access_resistance(
                    last_breakin_sweep['stimulus'], last_breakin_sweep['response'], time
                )
            except Exception as e:
                logging.warning("Error reading initial access resistance.")
                raise

        else:
            tags.append("Breakin sweep not found")
            input_resistance = None
            access_resistance = None

        return input_resistance, access_resistance

    def fast_sweep_qc(self, sweep_types):
        if len(sweep_types['i_clamps']) == 0:
            logging.warning("No current clamp sweeps available to compute QC features")

        qc_sweeps = list(set(
            sweep_types['i_clamps']).difference(
            set(sweep_types['tests']), set(sweep_types['searches'])
        ))
        qc_sweeps.sort()
        # qc_sweeps.sort()
        sweep_gen = (self.sweep_data_tuple[idx] for idx in qc_sweeps)
        sweep_qc_results = []

        for sweep in sweep_gen:
            sweep_num = sweep['sweep_number']
            sweep_features = {
                'sweep_number': sweep_num, 'stimulus_code': sweep['stimulus_code'],
                'stimulus_name': sweep['stimulus_name']
            }
            is_ramp = False

            if sweep_num in sweep_types['ramps']:
                is_ramp = True

            tags = self.fast_check_sweep_integrity(sweep, is_ramp)

            sweep_features['tags'] = tags

            stim_features = self.fast_current_clamp_stim_features(sweep)
            sweep_features.update(stim_features)

            if not tags:
                qc_features = self.fast_current_clamp_sweep_qc_features(sweep, is_ramp)
                sweep_features.update(qc_features)
            else:
                logging.warning("sweep {}: {}".format(sweep_num, tags))

            sweep_qc_results.append(sweep_features)

        return sweep_qc_results

    @staticmethod
    def fast_current_clamp_stim_features(sweep):
        stim_features = {}

        i = sweep['stimulus']
        hz = sweep['sampling_rate']

        time = np.arange(len(sweep['stimulus'])) / hz

        # num_pts = len(i)
        # t = np.linspace(0, num_pts/hz, num_pts)

        start_time, dur, amp, start_idx, end_idx = stf.get_stim_characteristics(i, time)

        stim_features['stimulus_start_time'] = start_time
        stim_features['stimulus_amplitude'] = amp
        stim_features['stimulus_duration'] = dur

        if sweep['epochs']['experiment']:
            expt_start_idx, _ = sweep['epochs']['experiment']
            interval = stf.find_stim_interval(expt_start_idx, i, hz)
        else:
            interval = None

        stim_features['stimulus_interval'] = interval

        return stim_features

    @staticmethod
    def fast_check_sweep_integrity(sweep, is_ramp):

        tags = []

        for k, v in sweep['epochs'].items():
            if not v:
                tags.append(f"{k} epoch is missing")

        if not is_ramp:
            if sweep['epochs']['recording'] and sweep['epochs']['experiment']:
                if sweep['epochs']['recording'][1] < sweep['epochs']['experiment'][1]:
                    tags.append("Recording stopped before completing the experiment epoch")

        return tags

    @staticmethod
    def fast_current_clamp_sweep_qc_features(sweep, is_ramp):
        qc_features = {}

        voltage = sweep['response']
        hz = sweep['sampling_rate']
        # measure noise before stimulus
        idx0, idx1 = ep.get_first_noise_epoch(sweep['epochs']['experiment'][0], hz)
        # count from the beginning of the experiment
        _, qc_features["pre_noise_rms_mv"] = qcf.measure_vm(voltage[idx0:idx1])

        # measure mean and rms of Vm at end of recording
        # do not check for ramps, because they do not have enough time to recover

        rec_end_idx = sweep['epochs']['recording'][1]
        if not is_ramp:
            idx0, idx1 = ep.get_last_stability_epoch(rec_end_idx, hz)
            mean_last_stability_epoch, _ = qcf.measure_vm(voltage[idx0:idx1])

            idx0, idx1 = ep.get_last_noise_epoch(rec_end_idx, hz)
            _, rms_last_noise_epoch = qcf.measure_vm(voltage[idx0:idx1])
        else:
            rms_last_noise_epoch = None
            mean_last_stability_epoch = None

        qc_features["post_vm_mv"] = mean_last_stability_epoch
        qc_features["post_noise_rms_mv"] = rms_last_noise_epoch

        # measure mean and rms of Vm and over extended interval before stimulus, to check stability

        stim_start_idx = sweep['epochs']['stim'][0]

        idx0, idx1 = ep.get_first_stability_epoch(stim_start_idx, hz)
        mean_first_stability_epoch, rms_first_stability_epoch = qcf.measure_vm(voltage[idx0:idx1])

        qc_features["pre_vm_mv"] = mean_first_stability_epoch
        qc_features["slow_vm_mv"] = mean_first_stability_epoch
        qc_features["slow_noise_rms_mv"] = rms_first_stability_epoch

        qc_features["vm_delta_mv"] = qcf.measure_vm_delta(mean_first_stability_epoch, mean_last_stability_epoch)

        return qc_features

    @staticmethod
    def get_epochs(sampling_rate, stimulus, response):
        test_epoch = ep.get_test_epoch(stimulus, sampling_rate)
        if test_epoch:
            test_pulse = True
        else:
            test_pulse = False

        sweep_epoch = ep.get_sweep_epoch(response)
        recording_epoch = ep.get_recording_epoch(response)
        stimulus_epoch = ep.get_stim_epoch(stimulus, test_pulse=test_pulse)
        experiment_epoch = ep.get_experiment_epoch(
            stimulus, sampling_rate, test_pulse=test_pulse
        )

        return {
            'test': test_epoch,
            'sweep': sweep_epoch,
            'recording': recording_epoch,
            'experiment': experiment_epoch,
            'stim': stimulus_epoch
        }

    @staticmethod
    def get_sweep_table_data(
            sweep_table_data: List[dict],
            pre_qc_sweep_features: List[dict],
            post_qc_sweep_features: List[dict],
            sweep_states: List[dict]
    ):
        """ foo
        """

        for idx, state in enumerate(sweep_states):
            if state['passed']:
                sweep_table_data[state['sweep_number']]['auto_qc_state'] = "passed"
            else:
                sweep_table_data[state['sweep_number']]['auto_qc_state'] = "failed"

            sweep_table_data[state['sweep_number']]['tags'] += post_qc_sweep_features[idx]['tags']
            sweep_table_data[state['sweep_number']]['tags'] += state['reasons']

        for feature in pre_qc_sweep_features:
            if sweep_table_data[feature['sweep_number']]['auto_qc_state'] \
                    not in {"passed", "failed"}:
                sweep_table_data[feature['sweep_number']]['auto_qc_state'] = "failed"
            sweep_table_data[feature['sweep_number']]['tags'] += feature['tags']

        for idx, feature in enumerate(sweep_table_data):
            if sweep_table_data[idx]['auto_qc_state'] not in ("passed", "failed"):
                sweep_table_data[idx]['tags'] += ["no auto qc"]

        return sweep_table_data

# def run_auto_qc(sweep_data_list: list, ontology: StimulusOntology,
#                 qc_criteria: list, recording_date: str, qc_output: Connection):
#
#     qc_operator = QCOperator(
#         sweep_data_list, ontology, qc_criteria, recording_date
#     )
#     qc_results = qc_operator.fast_experiment_qc()
#
#     qc_out = qc_output
#     qc_out.send(qc_results)
#     qc_out.close()


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

    qc_results = (
        cell_features, cell_tags, sweep_features, cell_state, sweep_states
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
            # nwb_file = str(base_dir.joinpath(file))
            nwb_file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/H20.03.302.11.52.01.03.nwb"

            start_time = default_timer()
            data_extractor = DataExtractor(nwb_file=nwb_file, ontology=ONTOLOGY)
            sweep_data_tuple = tuple(data_extractor.data_iter)
            recording_date = data_extractor.recording_date
            qc_operator = QCOperatorLite(
                sweep_data_tuple=sweep_data_tuple, ontology=ONTOLOGY,
                qc_criteria=QC_CRITERIA, recording_date=recording_date
            )

            qc_results, sweep_table_data, sweep_types = qc_operator.fast_experiment_qc()
            fast_qc_time = default_timer()-start_time
            print(f'Fast QC: {file} took {fast_qc_time} to load')
            times[trial][str(files[index])]['fast_qc'] = fast_qc_time

            start_time = default_timer()
            slow_qc_results = slow_qc(nwb_file=nwb_file)
            slow_qc_time = default_timer()-start_time
            print(f'Slow QC: {file} took {slow_qc_time} to load')
            times[trial][str(files[index])]['slow_qc'] = slow_qc_time


            for idx, sweep_dict in enumerate(qc_results[2]):
                if len(sweep_dict) < 15:
                    qc_results[2].pop(idx)

            fast_qc_keys = set(qc_results[2][0].keys())
            slow_qc_keys = set(slow_qc_results[2][0].keys())
            pop_keys = fast_qc_keys.symmetric_difference(slow_qc_keys)

            for idx, sweep in enumerate(slow_qc_results[2]):
                for key in pop_keys:
                    slow_qc_results[2][idx].pop(key)


            print(f"Cell features difference? "
                  f"{set(slow_qc_results[0]).symmetric_difference(qc_results[0])}")
            print(f"Cell tags difference? "
                  f"{set(slow_qc_results[1]).symmetric_difference(qc_results[1])}")
            print(f"Cell features equal? "
                  f"{slow_qc_results[2]==qc_results[2]}")
            print(f"Cell state difference?"
                  f" {set(slow_qc_results[3]).symmetric_difference(qc_results[3])}")
            print(f"Sweep_states equal? "
                  f"{slow_qc_results[4] == qc_results[4]}")
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
