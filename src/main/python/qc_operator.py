import logging
from copy import copy, deepcopy
from multiprocessing.connection import Connection
from warnings import filterwarnings
from typing import List

import numpy as np
from pynwb.icephys import (
    CurrentClampSeries, CurrentClampStimulusSeries,
    VoltageClampSeries, VoltageClampStimulusSeries,
    IZeroClampSeries
)
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


class QCOperator(object):
    __slots__ = [
        'nwb_file', '_data_set', 'ontology_file',
        '_series_iter', '_data_iter', '_qc_criteria', '_ontology'
    ]

    def __init__(
            self, nwb_file: str, ontology: StimulusOntology,
            qc_criteria: dict, warnings="ignore"
    ):
        self.nwb_file = nwb_file
        self._ontology = ontology
        self._qc_criteria = qc_criteria

        self._data_set = None
        self._series_iter = None
        self._data_iter = None

        if not warnings:
            filterwarnings(warnings)

    @property
    def ontology(self):
        return self._ontology

    @property
    def qc_criteria(self):
        return self._qc_criteria

    @property
    def data_set(self):
        if self._data_set:
            return self._data_set
        else:
            self._data_set = create_ephys_data_set(
                nwb_file=self.nwb_file, ontology=self.ontology
            )
            return self._data_set

    def _extract_series_data(self, series):
        sweep_number = series[0].sweep_number

        stimulus_code = ""
        stimulus_name = ""
        response = None
        stimulus = None
        stimulus_unit = None
        sampling_rate = float(series[0].rate)

        for s in series:
            if isinstance(s, (VoltageClampSeries, CurrentClampSeries, IZeroClampSeries)):
                response = copy(s.data[:] * float(s.conversion))

                stim_code = s.stimulus_description
                if stim_code[-5:] == "_DA_0":
                    stim_code = stim_code[:-5]
                stimulus_code = stim_code.split("[")[0]
                stimulus_name = self.data_set._data.get_stimulus_name(stimulus_code)

            elif isinstance(s, (VoltageClampStimulusSeries, CurrentClampStimulusSeries)):
                stimulus = copy(s.data[:] * float(s.conversion))
                unit = s.unit
                if not unit:
                    stimulus_unit = "Unknown"
                elif unit in {"Amps", "A", "amps", "amperes"}:
                    stimulus_unit = "Amps"
                elif unit in {"Volts", "V", "volts"}:
                    stimulus_unit = "Volts"
                else:
                    stimulus_unit = unit

        if stimulus_unit == "Volts":
            stimulus = stimulus * 1.0e3
            response = response * 1.0e12
        elif stimulus_unit == "Amps":
            stimulus = stimulus * 1.0e12
            response = response * 1.0e3

        nonzero = np.flatnonzero(response)
        if len(nonzero) == 0:
            recording_end_idx = 0
        else:
            recording_end_idx = nonzero[-1] + 1
        response = response[:recording_end_idx]
        stimulus = stimulus[:recording_end_idx]

        epochs = self.get_epochs(
            sampling_rate=sampling_rate, stimulus=stimulus, response=response
        )

        return {
            'sweep_number': sweep_number,
            'stimulus_code': stimulus_code,
            'stimulus_name': stimulus_name,
            'stimulus': stimulus,
            'response': response,
            'stimulus_unit': stimulus_unit,
            'sampling_rate': sampling_rate,
            'epochs': epochs
        }

    def fast_cell_qc(self, sweep_types, manual_values=None):

        if manual_values is None:
            manual_values = {}

        features = {}
        tags = []

        features['blowout_mv'] = self.fast_extract_blowout(sweep_types['blowouts'], tags)

        features['electrode_0_pa'] = self.fast_extract_electrode_0(sweep_types['baths'], tags)

        features['recording_date'] = extract_recording_date(self.data_set, tags)

        features["seal_gohm"] = self.fast_extract_clamp_seal(sweep_types['seals'], tags,
                                                             manual_values)

        input_resistance, access_resistance = \
            self.fast_extract_input_and_acess_resistance(sweep_types['breakins'], tags)

        features['input_resistance_mohm'] = input_resistance
        features["initial_access_resistance_mohm"] = access_resistance

        features['input_access_resistance_ratio'] = \
            compute_input_access_resistance_ratio(input_resistance, access_resistance)

        return features, tags

    def fast_sweep_qc(self, sweep_types):
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
                'sweep_number': sweep_num,
                'stimulus_code': sweep['stimulus_code'],
                'stimulus_name': sweep['stimulus_name']
            }
            is_ramp = False

            if sweep in sweep_types['ramps']:
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

    def fast_experiment_qc(self):
        sweep_table = self.data_set._data.nwb.sweep_table

        # number of sweeps is half the shape of the sweep table here because
        # each sweep has two series associated with it (stimulus and response)
        num_sweeps = sweep_table.series.shape[0]//2
        sweep_range = range(num_sweeps)

        # initialize a list of dictionaries to be used in sweep table model
        sweep_table_data = [{
            'sweep_number': idx, 'stimulus_code': "", 'stimulus_name': "",
            'auto_qc_state': "n/a", 'manual_qc_state': "default", 'tags': []
        } for idx in sweep_range]

        self._series_iter = map(sweep_table.get_series, sweep_range)
        # iterator with all the necessary sweep data
        self._data_iter = map(self._extract_series_data, self._series_iter)

        # get sweep_types and update sweep table data
        sweep_types, sweep_table_data = self.get_sweep_types_and_info(sweep_table_data)

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

        sweep_table_data = self.populate_qc_info(
            sweep_table_data=sweep_table_data,
            pre_qc_sweep_features=pre_qc_sweep_features,
            post_qc_sweep_features=post_qc_sweep_features,
            sweep_states=sweep_states
        )


        qc_results = (
            cell_features, cell_tags, pre_qc_sweep_features, cell_state, sweep_states
        )

        return qc_results, sweep_table_data

    def populate_qc_info(
        self,
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
                    not in ("passed", "failed"):
                sweep_table_data[feature['sweep_number']]['auto_qc_state'] = "failed"
            sweep_table_data[feature['sweep_number']]['tags'] += feature['tags']

        for idx, feature in enumerate(sweep_table_data):
            if sweep_table_data[idx]['auto_qc_state'] not in ("passed", "failed"):
                sweep_table_data[idx]['tags'] += ["no auto qc"]

        return sweep_table_data

    def get_sweep_types_and_info(self, sweep_table_data):
        sweep_keys = (
            'v_clamps', 'i_clamps', 'blowouts', 'baths', 'seals', 'breakins',
            'ramps', 'long_squares', 'coarse_long_squares', 'short_square_triples',
            'short_squares', 'searches', 'tests', 'extps'
        )

        sweep_types = {key: [] for key in sweep_keys}

        ontology = self.data_set.ontology

        for idx, sweep in enumerate(self._data_iter):

            stim_unit = sweep['stimulus_unit']

            if stim_unit == "Volts":
                sweep_types['v_clamps'].append(sweep)
            if stim_unit == "Amps":
                sweep_types['i_clamps'].append(sweep)

            # find stim code and add it to list of sweep table data
            stim_code = sweep['stimulus_code']
            sweep_table_data[idx]['stimulus_code'] = stim_code

            # find stim name and append it to list of sweep table data
            stim_name = sweep['stimulus_name']
            sweep_table_data[idx]['stimulus_name'] = stim_name

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

        return sweep_types, sweep_table_data

    @staticmethod
    def fast_extract_blowout(blowout_sweeps, tags):
        if blowout_sweeps:
            blowout_mv = qcf.measure_blowout(
                blowout_sweeps[-1]['response'], blowout_sweeps[-1]['epochs']['test'][1]
            )
        else:
            tags.append("Blowout is not available")
            blowout_mv = None
        return blowout_mv

    @staticmethod
    def fast_extract_electrode_0(bath_sweeps: list, tags):
        if bath_sweeps:
            e0 = qcf.measure_electrode_0(
                bath_sweeps[-1]['response'], bath_sweeps[-1]['sampling_rate']
            )
        else:
            tags.append("Electrode 0 is not available")
            e0 = None
        return e0

    @staticmethod
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

    @staticmethod
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


def run_auto_qc(nwb_file: str, ontology: StimulusOntology, qc_criteria: dict,
                output_conn: Connection):

    qc_operator = QCOperator(nwb_file=nwb_file, ontology=ontology, qc_criteria=qc_criteria)
    qc_results = qc_operator.fast_experiment_qc()
    output_conn.send(qc_results)
    output_conn.close()


