import logging
from copy import deepcopy
from multiprocessing.connection import Connection
from typing import List, Tuple, Dict, Union, NamedTuple

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


class QCResults(NamedTuple):
    cell_features: dict
    cell_tags: list
    cell_state: dict
    sweep_features: List[dict]
    sweep_states: List[dict]


class QCOperator(object):
    """ Docstrings copy/pasted from pre_fx_data:

    def run_qc(stimulus_ontology, cell_features, sweep_features, qc_criteria):
        Adds qc status to sweep features and outputs a qc summary to the log.

        Parameters
        ----------
        stimulus_ontology : StimulusOntology
            stimulus ontology used for this data set
        cell_features : dict
            dictionary of qc info for the cell (overall experiment level info)
        sweep_features : list[dict]
            a list of dictionaries containing qc info for each individual sweep
        qc_criteria : dict
            a dictionary containing the criteria used for auto QC

        Returns
        -------
        cell_state : dict
            a dictionary of qc states for various cell level qc criteria
        cell_features : dict
            dictionary of qc info for the cell (overall experiment level info)
        sweep_states : List[dict]
            a list of dictionaries containing auto QC states
        post_qc_sweep_features : List[dict]
            similar to sweep_features input, but with rows removed for most sweeps
            that failed auto QC and new column containing the auto QC states

    def extract_qc_features(data_set):
        extracts QC information for the cell and the sweeps using ipfx.

        Parameters
        ----------
        data_set : EphysDataSet
            raw data used in qc feature extraction

        Returns
        -------
        cell_features : dict
            dictionary of qc info for the cell (overall experiment level info)
        cell_tags : list
            a list of qc tags for the cell (e.g. 'Blowout is not available')
        sweep_features : list[dict]
            a list of dictionaries containing qc info for each individual sweep

    def populate_qc_info(
        self,
        sweep_table,
        pre_qc_sweep_features: List[dict],
        post_qc_sweep_features: List[dict],
        auto_qc_states: List[dict]
    ):
        Uses pre and post sweep qc features to populate initial and current
        sweep QC features and states. Sweep features and states use values of
        True, False, or None to indicate their auto QC states.

         For sweep_features['passed']:
            True = Passed all auto qc
            False = Failed in second round of auto qc when run_qc() was called.
                These sweeps exist in post_qc_sweep_features
            None = Dropped in first round of auto qc due to having a fail tag
                or no auto QC was performed.
                Sweeps with None in this column are dropped before feature
                extraction so that extract_data_set_features() doesn't break.
                These sweeps exist in pre_qc_sweep_features, but do not exist
                in post_qc_sweep_features.

        For sweep_states['passed']:
            True = Passed all auto qc
            False = Failed in first or second round of auto qc.
            None = No auto QC. These sweeps exist in the sweep table, but do
                not exist in pre_qc_sweep_features or post_qc_sweep_features

        Parameters
        ----------
        pre_qc_sweep_features : List[dict]
            Contains sweep features that went through qc feature extraction.
            The ['passed'] column does not exist in this list.
        post_qc_sweep_features : List[dict]
            Contains sweep features that went through the second round of
            auto QC. Sweeps that had a fail tag in pre_qc_sweep_features are
            dropped and not present in this list.
        sweep_states : List[dict]
            Contains auto QC states obtained in the second round of auto QC
            Again, sweeps that were dropped because they had a fail tag
            are not present in this list.

    """

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

        features['blowout_mv'] = self.fast_extract_blowout(sweep_types['blowout'], tags)

        features['electrode_0_pa'] = self.fast_extract_electrode_0(sweep_types['in_bath'], tags)

        if self.recording_date is None:
            tags.append("Recording date is missing")
        features['recording_date'] = self.recording_date

        features["seal_gohm"] = self.fast_extract_clamp_seal(sweep_types['seal'], tags,
                                                        manual_values)

        input_resistance, access_resistance = \
            self.fast_extract_input_and_acess_resistance(sweep_types['break_in'], tags)

        features['input_resistance_mohm'] = input_resistance
        features["initial_access_resistance_mohm"] = access_resistance

        features['input_access_resistance_ratio'] = \
            compute_input_access_resistance_ratio(input_resistance, access_resistance)

        return features, tags

    def fast_experiment_qc(self):

        # initialize a list of dictionaries to be used in sweep table model

        # get sweep_types and update sweep table data

        sweep_types = self.get_sweep_types()

        nuc_vc_features = self.nuc_vc_sweep_qc(sweep_types['nuc_vc'])

        # if nuc_vc_features:
        #     for feature in nuc_vc_features:
        #         print(feature)

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

        # grab set of sweeps that made it through auto qc and add to sweep types
        sweep_types['pipeline'] = {state['sweep_number'] for state in sweep_states}

        qc_summary(
            sweep_features=post_qc_sweep_features,
            sweep_states=sweep_states,
            cell_features=cell_features,
            cell_state=cell_state
        )

        # todo add qc features
        full_sweep_qc_info = [{
            'sweep_number': sweep['sweep_number'],
            'stimulus_code': sweep['stimulus_code'],
            'stimulus_name': sweep['stimulus_name'],
            'auto_qc_state': "n/a",
            'manual_qc_state': "default",
            'passed': None,
            'qc_tags': [],
            'feature_tags': []
        } for sweep in self.sweep_data_tuple]

        full_sweep_qc_info = self.get_full_sweep_qc_info(
            full_sweep_qc_info=full_sweep_qc_info,
            pre_qc_sweep_features=pre_qc_sweep_features,
            post_qc_sweep_features=post_qc_sweep_features,
            sweep_states=sweep_states,
            nuc_vc_features=nuc_vc_features
        )

        qc_results = QCResults(
            cell_features=cell_features,
            cell_tags=cell_tags,
            cell_state=cell_state,
            sweep_features=post_qc_sweep_features,
            sweep_states=sweep_states
        )

        return qc_results, full_sweep_qc_info, sweep_types

    def get_sweep_types(self):
        sweep_keys = (
            'v_clamp', 'i_clamp', 'blowout', 'in_bath', 'seal', 'break_in',
            'ramp', 'long_square', 'coarse_long_square', 'short_square_triple',
            'short_square', 'search', 'test', 'ex_tp', 'nuc_vc'
        )

        sweep_types = {key: set() for key in sweep_keys}

        ontology = self.ontology

        for sweep in self.sweep_data_tuple:
            sweep_num = sweep['sweep_number']
            stim_unit = sweep['stimulus_unit']

            if stim_unit == "Volts":
                sweep_types['v_clamp'].add(sweep_num)
            if stim_unit == "Amps":
                sweep_types['i_clamp'].add(sweep_num)

            stim_code = sweep['stimulus_code']
            stim_name = sweep['stimulus_name']

            # check ontology for a stim name match or check if stim code
            # contains a partial match with first ontology name
            if stim_name in ontology.extp_names or ontology.extp_names[0] in stim_code:
                sweep_types['ex_tp'].add(sweep_num)
            if stim_name in ontology.test_names:
                sweep_types['test'].add(sweep_num)
            if stim_name in ontology.blowout_names or ontology.blowout_names[0] in stim_code:
                sweep_types['blowout'].add(sweep_num)
            if stim_name in ontology.bath_names or ontology.bath_names[0] in stim_code:
                sweep_types['in_bath'].add(sweep_num)
            if stim_name in ontology.seal_names or ontology.seal_names[0] in stim_code:
                sweep_types['seal'].add(sweep_num)
            if stim_name in ontology.breakin_names or ontology.breakin_names[0] in stim_code:
                sweep_types['break_in'].add(sweep_num)

            # check ontology for exact match for these names
            if stim_name in ontology.ramp_names:
                sweep_types['ramp'].add(sweep_num)
            if stim_name in ontology.long_square_names:
                sweep_types['long_square'].add(sweep_num)
            if stim_name in ontology.coarse_long_square_names:
                sweep_types['coarse_long_square'].add(sweep_num)
            if stim_name in ontology.short_square_triple_names:
                sweep_types['short_square_triple'].add(sweep_num)
            if stim_name in ontology.short_square_names:
                sweep_types['short_square'].add(sweep_num)
            if stim_name in ontology.search_names:
                sweep_types['search'].add(sweep_num)

            # manual entry for channel recording 'NucVC' sweeps
            if "NucVC" in stim_code:
                sweep_types['nuc_vc'].add(sweep_num)

        return sweep_types

    def nuc_vc_sweep_qc(self, nuc_vc_sweeps):
        if not nuc_vc_sweeps:
            return None
        nuc_vc_list = sorted(nuc_vc_sweeps)
        nuc_vc_gen = (self.sweep_data_tuple[idx] for idx in nuc_vc_list)
        nuc_vc_qc_results = [
            {
                'sweep_number': sweep['sweep_number'],
                'seal_value': self.get_seal_from_test_pulse(
                    sweep['stimulus'], sweep['response'],   # voltage and current
                    np.arange(len(sweep['stimulus'])) / sweep['sampling_rate']  # time vector
                )
            } for sweep in nuc_vc_gen
        ]
        return nuc_vc_qc_results

    def fast_extract_blowout(self, blowout_sweeps, tags):
        if blowout_sweeps:
            last_blowout_sweep = self.sweep_data_tuple[max(blowout_sweeps)]
            blowout_mv = qcf.measure_blowout(
                last_blowout_sweep['response'], last_blowout_sweep['epochs']['test'][1]
            )
        else:
            tags.append("Blowout is not available")
            blowout_mv = None
        return blowout_mv

    def fast_extract_electrode_0(self, bath_sweeps: list, tags):
        if bath_sweeps:
            last_bath_sweep = self.sweep_data_tuple[max(bath_sweeps)]
            e0 = qcf.measure_electrode_0(last_bath_sweep['response'], last_bath_sweep['sampling_rate'])
        else:
            tags.append("Electrode 0 is not available")
            e0 = None
        return e0

    def fast_extract_clamp_seal(self, seal_sweeps: list, tags, manual_values=None):
        if seal_sweeps:
            last_seal_sweep = self.sweep_data_tuple[max(seal_sweeps)]

            time = np.arange(
                len(last_seal_sweep['stimulus'])
            ) / last_seal_sweep['sampling_rate']

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
            last_breakin_sweep = self.sweep_data_tuple[max(breakin_sweeps)]

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
        if len(sweep_types['i_clamp']) == 0:
            logging.warning("No current clamp sweeps available to compute QC features")

        qc_sweeps = sorted(
            sweep_types['i_clamp'].difference(
                sweep_types['test'], sweep_types['search']
            )
        )

        sweep_gen = (self.sweep_data_tuple[idx] for idx in qc_sweeps)
        sweep_qc_results = []

        for sweep in sweep_gen:
            sweep_num = sweep['sweep_number']
            sweep_features = {
                'sweep_number': sweep_num, 'stimulus_code': sweep['stimulus_code'],
                'stimulus_name': sweep['stimulus_name']
            }
            is_ramp = False

            if sweep_num in sweep_types['ramp']:
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
        # do not check for ramp, because they do not have enough time to recover

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
    def get_full_sweep_qc_info(
            full_sweep_qc_info: List[dict],
            pre_qc_sweep_features: List[dict],
            post_qc_sweep_features: List[dict],
            sweep_states: List[dict],
            nuc_vc_features: List[dict]
    ):
        # TODO write docstring
        """ foo
        """

        # loop through sweep states and assign auto qc states to sweep_table_data
        for idx, state in enumerate(sweep_states):
            # cache sweep number
            sweep_num = state['sweep_number']
            # it state is auto passed, update table data appropriately
            if state['passed']:
                full_sweep_qc_info[sweep_num]['passed'] = True
                full_sweep_qc_info[sweep_num]['auto_qc_state'] = "passed"
            else:
                full_sweep_qc_info[sweep_num]['passed'] = False
                full_sweep_qc_info[sweep_num]['auto_qc_state'] = "failed"
            # update tags
            full_sweep_qc_info[sweep_num]['qc_tags'] += post_qc_sweep_features[idx]['tags']
            full_sweep_qc_info[sweep_num]['qc_tags'] += state['reasons']

        # loop through sweeps that got dropped due to having fail tags update table data
        for feature in pre_qc_sweep_features:
            # cache sweep number
            sweep_num = feature['sweep_number']
            # skip over sweeps that were processed in above step
            if full_sweep_qc_info[sweep_num]['passed'] not in {True, False}:
                full_sweep_qc_info[sweep_num]['passed'] = False
                full_sweep_qc_info[sweep_num]['auto_qc_state'] = "failed"
            # update tags
            full_sweep_qc_info[sweep_num]['qc_tags'] += feature['tags']

        # loop through all the other sweeps not included in auto qc
        # for sweep_num in range(len(full_sweep_qc_info)):
            # only handle sweeps that were not processed in previous two steps
            # if full_sweep_qc_info[sweep_num]['passed'] not in {True, False}:
                # these sweeps have no auto qc, so update tags appropriately
                # full_sweep_qc_info[sweep_num]['qc_tags'] += ["no auto qc"]

        if nuc_vc_features:
            for feature in nuc_vc_features:
                sweep_num = feature['sweep_number']
                seal_value = feature['seal_value']  # extract np.float seal value
                seal_str = str(np.rint(seal_value))  # round it to nearest value
                # full_sweep_qc_info[sweep_num]['feature_tags'] += [
                full_sweep_qc_info[sweep_num]['qc_tags'] += [
                    f"Test pulse resistance: {seal_str} MOhm"
                ]   # append string to feature tags

        return full_sweep_qc_info

    @staticmethod
    def get_seal_from_test_pulse(voltage: np.ndarray, current: np.ndarray, time: np.ndarray):
        """Compute input resistance from the stable pulse response

        Parameters
        ----------
        voltage : np.ndarray
            membrane voltage (mV)
        current : np.ndarray
            input current (pA)
        time : np.ndarray

        time : np.ndarray
            time vector (s)
        up_idx : int
            index of start of square pulse
        down_idx : int
            index of end of square pulse

        Returns
        -------
        input resistance : np.float
            seal resistance (MOhm)

        """
        dv = np.diff(voltage)
        # find index of first upstroke and downstroke in stimulus voltage
        try:
            # first upstroke index
            up_idx = np.flatnonzero(dv > 0)[0]
            # first downstroke index
            down_idx = np.flatnonzero(dv < 0)[0]
        except IndexError:
            logging.warning("Could not find full test pulse.")
            return np.nan

        dt = time[1] - time[0]
        one_ms = int(0.001 / dt)

        # take average v and i one ms before start
        end = up_idx - 1
        start = end - one_ms

        avg_v_base = np.mean(voltage[start:end])
        avg_i_base = np.mean(current[start:end])

        # take average v and i one ms before end
        end = down_idx - 1
        start = end - one_ms

        avg_v_steady = np.mean(voltage[start:end])
        avg_i_steady = np.mean(current[start:end])

        seal_resistance = (avg_v_steady - avg_v_base) / (avg_i_steady - avg_i_base)

        return 1e3 * np.mean(seal_resistance)  # multiply by 1000 to convert GOhm to MOhm


def run_auto_qc(sweep_data_tuple: tuple, ontology: StimulusOntology,
                qc_criteria: list, recording_date: str, qc_output: Connection):

    qc_operator = QCOperator(
        sweep_data_tuple, ontology, qc_criteria, recording_date
    )
    qc_results, full_sweep_qc_info, sweep_types = qc_operator.fast_experiment_qc()

    qc_out = qc_output
    qc_out.send((qc_results, full_sweep_qc_info, sweep_types))
    qc_out.close()
