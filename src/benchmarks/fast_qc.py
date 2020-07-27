from pathlib import Path
from timeit import default_timer
from warnings import filterwarnings
import datetime as dt
import logging
from copy import copy, deepcopy
import json
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


class QCOperator(object):
    __slots__ = [
        'nwb_file', '_data_set', 'ontology_file',
        '_series_iter', '_data_iter', '_qc_criteria', '_ontology'
    ]

    def __init__(self, nwb_file: str, ontology_file=None, qc_criteria=None):
        super().__init__( )
        self.nwb_file = nwb_file
        self._qc_criteria = qc_criteria
        self.ontology_file = ontology_file
        self.ontology = ontology_file
        self._data_set = None
        self._series_iter = None
        self._data_iter = None

    @property
    def qc_criteria(self):
        return self._qc_criteria

    @qc_criteria.setter
    def qc_criteria(self, filename=None):
        if filename:
            with open(filename, "r") as path:
                self._qc_criteria = json.load(path)
        else:
            with open(DEFAULT_QC_CRITERIA_FILE, "r") as path:
                self._qc_criteria = json.load(path)

    @property
    def ontology(self):
        return self._ontology

    @ontology.setter
    def ontology(self, filename=None):
        if filename:
            with open(filename, "r") as path:
                self._ontology = StimulusOntology(json.load(path))
        else:
            with open(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE, "r")\
                    as path:
                self._ontology = StimulusOntology(json.load(path))

    @property
    def data_set(self):
        return self._data_set

    @data_set.setter
    def data_set(self, nwb_file):
        if nwb_file:
            self._data_set = create_ephys_data_set(
                nwb_file=nwb_file, ontology=self.ontology_file
            )
        else:
            self._data_set = create_ephys_data_set(
                nwb_file=self.nwb_file, ontology=self.ontology_file
            )

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

        epochs = get_epochs(
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

        features['blowout_mv'] = fast_extract_blowout(sweep_types['blowouts'], tags)

        features['electrode_0_pa'] = fast_extract_electrode_0(sweep_types['baths'], tags)

        features['recording_date'] = extract_recording_date(self.data_set, tags)

        features["seal_gohm"] = fast_extract_clamp_seal(sweep_types['seals'], tags,
                                                             manual_values)

        input_resistance, access_resistance = \
            fast_extract_input_and_acess_resistance(sweep_types['breakins'], tags)

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

    def _get_iterators(self):
        data_set = create_ephys_data_set(self.nwb_file)
        self._data_set = data_set
        sweep_table = data_set._data.nwb.sweep_table

        # number of sweeps is half the shape of the sweep table here because
        # each sweep has two series associated with it (stimulus and response)
        num_sweeps = sweep_table.series.shape[0] // 2
        self._series_iter = map(sweep_table.get_series, range(num_sweeps))
        # iterator with all the necessary sweep data
        self._data_iter = map(self._extract_series_data, self._series_iter)
        return self._series_iter, self._data_iter

    def fast_experiment_qc(self):
        self._get_iterators()

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

        # sweep_table_data = self.populate_qc_info(
        #     sweep_table_data=sweep_table_data,
        #     pre_qc_sweep_features=pre_qc_sweep_features,
        #     post_qc_sweep_features=post_qc_sweep_features,
        #     sweep_states=sweep_states
        # )

        qc_results = (
            cell_features, cell_tags, pre_qc_sweep_features, cell_state, sweep_states
        )

        return qc_results, #sweep_table_data

    def get_sweep_types(self):
        sweep_keys = (
            'v_clamps', 'i_clamps', 'blowouts', 'baths', 'seals', 'breakins',
            'ramps', 'long_squares', 'coarse_long_squares', 'short_square_triples',
            'short_squares', 'searches', 'tests', 'extps'
        )

        sweep_types = {key: [] for key in sweep_keys}

        ontology = self.data_set.ontology

        for sweep in self._data_iter:

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


def slow_qc(nwb_file: str, return_data_set = False):
    """ Does Auto QC and makes plots using single process.

    Parameters:
        nwb_file (str): string of nwb path

    Returns:
        thumbs (list[QByteArray]): a list of plot thumbnails

        qc_results (tuple): cell_features, cell_tags,
            sweep_features, cell_state, sweep_states

    """

    with open(DEFAULT_QC_CRITERIA_FILE, "r") as path:
        qc_criteria = json.load(path)

    with open(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE, "r") as path:
        stimulus_ontology = StimulusOntology(json.load(path))

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
        ontology=stimulus_ontology,
        cell_features=cell_features,
        sweep_features=sweep_features,
        qc_criteria=qc_criteria
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

    profile_dir = base_dir.joinpath(f'fast_qc_profiles/{today}_{now}')
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
            qc_operator = QCOperator(nwb_file=nwb_file)
            fast_qc_results = qc_operator.fast_experiment_qc()
            fast_qc_time = default_timer()-start_time
            print(f'Fast QC: {file} took {fast_qc_time} to load')
            times[trial][str(files[index])]['fast_qc'] = fast_qc_time

            start_time = default_timer()
            slow_qc_results = slow_qc(nwb_file=nwb_file)
            slow_qc_time = default_timer()-start_time
            print(f'Slow QC: {file} took {slow_qc_time} to load')
            times[trial][str(files[index])]['slow_qc'] = slow_qc_time

            print(f"Cell features difference? "
                  f"{set(slow_qc_results[0]).symmetric_difference(fast_qc_results[0])}")
            print(f"Cell tags difference? "
                  f"{set(slow_qc_results[1]).symmetric_difference(fast_qc_results[1])}")
            print(f"Cell state difference?"
                  f" {set(slow_qc_results[3]).symmetric_difference(fast_qc_results[3])}")
            print(f"Sweep_states equal? "
                  f"{slow_qc_results[4] == fast_qc_results[4]}")
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


class DataExtractorLite(object):
    __slots__ = [
        'nwb_file', '_ontology', '_data_set', '_recording_date',
        '_series_iter', '_data_iter', '_series_table', '_num_sweeps'
    ]

    def __init__(self, nwb_file, ontology):
        self.nwb_file = nwb_file
        self._ontology = ontology
        self._data_set = None
        self._recording_date = None
        self._series_table = None
        self._num_sweeps = None
        self._series_iter = None
        self._data_iter = None

    @property
    def data_set(self):
        if not self._data_set:
            self._data_set = create_ephys_data_set(
                nwb_file=self.nwb_file, ontology=self._ontology
            )
        return self._data_set

    @property
    def recording_date(self):
        if not self._recording_date:
            try:
                self._recording_date = self.data_set.get_recording_date()
            except KeyError:
                logging.warning('recording_date is missing')
        return self._recording_date

    @property
    def ontology(self):
        return self._ontology

    # @property
    # def series_table(self):
    #     if not self._series_table:
    #         self._series_table = self.data_set._data.nwb.sweep_table
    #     return self._series_table
    #
    # @property
    # def _num_sweeps(self):
    #     if not self._num_sweeps:
    #         self._num_sweeps = self._series_table.series.shape[0] // 2
    #     return self._num_sweeps

    @property
    def series_iter(self):
        if not self._series_iter:
            series_table = self.data_set._data.nwb.sweep_table
            # number of sweeps is half the shape of the sweep table here because
            # each sweep has two series associated with it (stimulus and response)
            num_sweeps = series_table.series.shape[0] // 2
            self._series_iter = map(series_table.get_series, range(num_sweeps))
        return self._series_iter

    @property
    def data_iter(self):
        if not self._data_iter:
            self._data_iter = map(self._extract_series_data, self.series_iter)
        return self._data_iter

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

        epochs = get_epochs(
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


class QCOperatorLite(object):
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

        # sweep_table_data = self.populate_qc_info(
        #     sweep_table_data=sweep_table_data,
        #     pre_qc_sweep_features=pre_qc_sweep_features,
        #     post_qc_sweep_features=post_qc_sweep_features,
        #     sweep_states=sweep_states
        # )

        qc_results = (
            cell_features, cell_tags, pre_qc_sweep_features, cell_state, sweep_states
        )

        return qc_results,  # sweep_table_data

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