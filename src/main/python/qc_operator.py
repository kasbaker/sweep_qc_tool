import logging
from copy import deepcopy
from multiprocessing.connection import Connection
from typing import List

import numpy as np
from ipfx.qc_feature_evaluator import qc_experiment
import ipfx.epochs as ep
import ipfx.qc_features as qcf
import ipfx.stim_features as stf
from ipfx import error as er
from ipfx.qc_feature_extractor import compute_input_access_resistance_ratio
from ipfx.sweep_props import drop_tagged_sweeps
from ipfx.bin.run_qc import qc_summary
from ipfx.stimulus import StimulusOntology


class QCOperator(object):
    __slots__ = [
        '_sweep_data_list', '_ontology', '_qc_criteria', '_recording_date'
    ]

    def __init__(self, sweep_data_list: list, ontology: StimulusOntology,
                 qc_criteria: list, recording_date: str):
        self._sweep_data_list = sweep_data_list
        self._ontology = ontology
        self._qc_criteria = qc_criteria
        self._recording_date = recording_date

    @property
    def sweep_data_list(self):
        return self._sweep_data_list

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

        features['blowout_mv'] = fast_extract_blowout(sweep_types['blowouts'], tags)

        features['electrode_0_pa'] = fast_extract_electrode_0(sweep_types['baths'], tags)

        if self.recording_date is None:
            tags.append("Recording date is missing")
        features['recording_date'] = self.recording_date

        features["seal_gohm"] = fast_extract_clamp_seal(sweep_types['seals'], tags,
                                                        manual_values)

        input_resistance, access_resistance = \
            fast_extract_input_and_acess_resistance(sweep_types['breakins'], tags)

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

        pre_qc_sweep_features = fast_sweep_qc(sweep_types)
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
        } for sweep in self.sweep_data_list]

        sweep_table_data = get_sweep_table_data(
            sweep_table_data=sweep_table_data,
            pre_qc_sweep_features=pre_qc_sweep_features,
            post_qc_sweep_features=post_qc_sweep_features,
            sweep_states=sweep_states
        )

        qc_results = (
            cell_features, cell_tags, pre_qc_sweep_features, cell_state, sweep_states
        )

        return qc_results,  sweep_table_data

    def get_sweep_types(self):
        sweep_keys = (
            'v_clamps', 'i_clamps', 'blowouts', 'baths', 'seals', 'breakins',
            'ramps', 'long_squares', 'coarse_long_squares', 'short_square_triples',
            'short_squares', 'searches', 'tests', 'extps'
        )

        sweep_types = {key: [] for key in sweep_keys}

        ontology = self.ontology

        for sweep in self.sweep_data_list:

            stim_unit = sweep['stimulus_unit']

            if stim_unit == "Volts":
                sweep_types['v_clamps'].append(sweep)
            if stim_unit == "Amps":
                sweep_types['i_clamps'].append(sweep)

            stim_code = sweep['stimulus_code']
            stim_name = sweep['stimulus_name']
            if stim_name in ontology.extp_names or ontology.extp_names[0] in stim_code:
                sweep_types['extps'].append(sweep)
            if stim_name in ontology.test_names:
                sweep_types['tests'].append(sweep)
            if stim_name in ontology.blowout_names or ontology.blowout_names[0] in stim_code:
                sweep_types['blowouts'].append(sweep)
            if stim_name in ontology.bath_names or ontology.bath_names[0] in stim_code:
                sweep_types['baths'].append(sweep)
            if stim_name in ontology.seal_names or ontology.seal_names[0] in stim_code:
                sweep_types['seals'].append(sweep)
            if stim_name in ontology.breakin_names or ontology.breakin_names[0] in stim_code:
                sweep_types['breakins'].append(sweep)

            if stim_name in ontology.ramp_names:
                sweep_types['ramps'].append(sweep)
            if stim_name in ontology.long_square_names:
                sweep_types['long_squares'].append(sweep)
            if stim_name in ontology.coarse_long_square_names:
                sweep_types['coarse_long_squares'].append(sweep)
            if stim_name in ontology.short_square_triple_names:
                sweep_types['short_square_triples'].append(sweep)
            if stim_name in ontology.short_square_names:
                sweep_types['short_squares'].append(sweep)
            if stim_name in ontology.search_names:
                sweep_types['searches'].append(sweep)

        return sweep_types


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


def fast_extract_blowout(blowout_sweeps, tags):
    if blowout_sweeps:
        blowout_mv = qcf.measure_blowout(
            blowout_sweeps[-1]['response'], blowout_sweeps[-1]['epochs']['test'][1]
        )
    else:
        tags.append("Blowout is not available")
        blowout_mv = None
    return blowout_mv


def fast_extract_electrode_0(bath_sweeps: list, tags):
    if bath_sweeps:
        e0 = qcf.measure_electrode_0(
            bath_sweeps[-1]['response'], bath_sweeps[-1]['sampling_rate']
        )
    else:
        tags.append("Electrode 0 is not available")
        e0 = None
    return e0


def fast_extract_clamp_seal(seal_sweeps: list, tags, manual_values=None):
    if seal_sweeps:
        num_pts = len(seal_sweeps[-1]['stimulus'])
        time = np.linspace(0, num_pts/seal_sweeps[-1]['sampling_rate'], num_pts)
        seal_gohm = qcf.measure_seal(
            seal_sweeps[-1]['stimulus'], seal_sweeps[-1]['response'], time
        )
        if seal_gohm is None or not np.isfinite(seal_gohm):
            raise er.FeatureError("Could not compute seal")
    else:
        tags.append("Seal is not available")
        seal_gohm = manual_values.get('manual_seal_gohm', None)
        if seal_gohm is not None:
            tags.append("Using manual seal value: %f" % seal_gohm)

    return seal_gohm


def fast_extract_input_and_acess_resistance(breakin_sweeps: list, tags):
    if breakin_sweeps:
        num_pts = len(breakin_sweeps[-1]['stimulus'])
        time = np.linspace(0, num_pts/breakin_sweeps[-1]['sampling_rate'], num_pts)
        try:
            input_resistance = qcf.measure_input_resistance(
                breakin_sweeps[-1]['stimulus'], breakin_sweeps[-1]['response'], time
            )
        except Exception as e:
            logging.warning("Error reading input resistance.")
            raise

        try:
            access_resistance = qcf.measure_initial_access_resistance(
                breakin_sweeps[-1]['stimulus'], breakin_sweeps[-1]['response'], time
            )
        except Exception as e:
            logging.warning("Error reading initial access resistance.")
            raise

    else:
        tags.append("Breakin sweep not found")
        input_resistance = None
        access_resistance = None

    return input_resistance, access_resistance


def fast_sweep_qc(sweep_types):
    if len(sweep_types['i_clamps']) == 0:
        logging.warning("No current clamp sweeps available to compute QC features")

    sweep_gen = (
        sweep for sweep in sweep_types['i_clamps']
        if sweep not in sweep_types['tests'] and sweep not in sweep_types['searches']
    )
    sweep_qc_results = []

    for sweep in sweep_gen:
        sweep_num = sweep['sweep_number']
        sweep_features = {
            'sweep_number': sweep_num, 'stimulus_code': sweep['stimulus_code'],
            'stimulus_name': sweep['stimulus_name']
        }
        is_ramp = False

        if sweep in sweep_types['ramps']:
            is_ramp = True

        tags = fast_check_sweep_integrity(sweep, is_ramp)

        sweep_features['tags'] = tags

        stim_features = fast_current_clamp_stim_features(sweep)
        sweep_features.update(stim_features)

        if not tags:
            qc_features = fast_current_clamp_sweep_qc_features(sweep, is_ramp)
            sweep_features.update(qc_features)
        else:
            logging.warning("sweep {}: {}".format(sweep_num, tags))

        sweep_qc_results.append(sweep_features)

    return sweep_qc_results


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


def fast_current_clamp_stim_features(sweep):
    stim_features = {}

    i = sweep['stimulus']
    hz = sweep['sampling_rate']
    num_pts = len(i)
    t = np.linspace(0, num_pts/hz, num_pts)

    start_time, dur, amp, start_idx, end_idx = stf.get_stim_characteristics(i, t)

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


def run_auto_qc(sweep_data_list: list, ontology: StimulusOntology,
                qc_criteria: list, recording_date: str, qc_output: Connection):

    qc_operator = QCOperator(
        sweep_data_list, ontology, qc_criteria, recording_date
    )
    qc_results = qc_operator.fast_experiment_qc()

    qc_out = qc_output
    qc_out.send(qc_results)
    qc_out.close()
