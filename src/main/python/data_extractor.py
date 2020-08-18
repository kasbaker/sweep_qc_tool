import logging
from copy import copy
from typing import Tuple

import numpy as np
from pynwb.icephys import (
    CurrentClampSeries, CurrentClampStimulusSeries,
    VoltageClampSeries, VoltageClampStimulusSeries,
    IZeroClampSeries
)
from ipfx.dataset.create import create_ephys_data_set
from ipfx.epochs import (
    get_test_epoch, get_sweep_epoch, get_recording_epoch,
    get_stim_epoch, get_experiment_epoch
)


class DataExtractor(object):

    # __slots__ = [
    #     'nwb_file', '_ontology', '_data_set', '_recording_date',
    #     '_series_iter', '_data_iter', '_series_table', '_num_sweeps',
    #     'v_clamp_settings_dict', 'i_clamp_settings_dict'
    # ]

    def __init__(
            self, nwb_file, ontology,
            v_clamp_settings_keys: Tuple[str] = 'default',
            i_clamp_settings_keys: Tuple[str] = 'default'
    ):

        # todo make settings keys 'default', False, or tuple-like
        self.nwb_file = nwb_file
        self._ontology = ontology
        self._data_set = None
        self._recording_date = None
        self._series_table = None
        self._num_sweeps = None
        self._series_iter = None
        self._data_iter = None

        self.prev_experiment_epoch = None
        self.prev_stim_epoch = None

        # initialize dictionary of amplifier settings for voltage clamp
        if v_clamp_settings_keys != 'default':
            self.v_clamp_settings_dict = dict.fromkeys(v_clamp_settings_keys)
        else:
            # tuple of strings to use as keys for v-clamp settings
            default_v_clamp_settings_keys = (
                "V-Clamp Holding Enable", "V-Clamp Holding Level",
                "RsComp Enable", "RsComp Correction", "RsComp Prediction",
                "Whole Cell Comp Enable", "Whole Cell Comp Cap", "Whole Cell Comp Resist",
                "Pipette Offset"
            )
            self.v_clamp_settings_dict = dict.fromkeys(default_v_clamp_settings_keys)

        # initialize dictionary of amplifier settings for current clamp
        if i_clamp_settings_keys != 'default':
            self.i_clamp_settings_dict = dict.fromkeys(i_clamp_settings_keys)
        else:
            # tuple of keys to use for v-clamp settings
            default_i_clamp_settings_keys = (
                "I-Clamp Holding Enable", "I-Clamp Holding Level",
                "Neut Cap Enabled", "Neut Cap Value",
                "Bridge Bal Enable", "Bridge Bal Value",
                "Autobias", "Autobias Vcom", "Autobias Vcom variance",
                "Pipette Offset"
            )
            self.i_clamp_settings_dict = dict.fromkeys(default_i_clamp_settings_keys)

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
        # initialize empty variables
        stimulus_code = ""
        stimulus_name = ""
        response = None
        stimulus = None
        stimulus_unit = None

        # grab sampling rate and sweep number from first series
        sampling_rate = float(series[0].rate)
        sweep_number = int(series[0].sweep_number)

        # initialize dictionary of amplifier settings to extract
        patch_clamp_settings_dict = None

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
                comment_str = s.comments
                if not unit:
                    stimulus_unit = "Unknown"
                # do this for current clamp sweeps
                elif unit in {"Amps", "A", "amps", "amperes"}:
                    stimulus_unit = "Amps"
                    for key in self.i_clamp_settings_dict.keys():
                        # partition comments by key and grab the last partition
                        value_str = comment_str.partition(f"{key}: ")[2]
                        if value_str:  # if this is not an empty string extract the value
                            # grab value and chop off everything after newline
                            value = value_str.splitlines()[0]
                            self.i_clamp_settings_dict[key] = value  # add value to dict
                    # store dictionary as temporary value
                    patch_clamp_settings_dict = copy(self.i_clamp_settings_dict)
                # do this for voltage clamp sweeps
                elif unit in {"Volts", "V", "volts"}:
                    stimulus_unit = "Volts"
                    # loop through keys in v_clamp settings and extract strings
                    for key in self.v_clamp_settings_dict.keys():
                        # partition comments by key and grab the part with value we want
                        value_str = comment_str.partition(f"{key}: ")[2]
                        if value_str:  # if this is not an empty string extract the value
                            # chop off everything after newline and ": " at the start
                            value = value_str.splitlines()[0]
                            self.v_clamp_settings_dict[key] = value  # add value to dict
                    # store dictionary as temporary value
                    patch_clamp_settings_dict = copy(self.v_clamp_settings_dict)
                else:
                    stimulus_unit = unit

        if stimulus_unit == "Volts":
            stimulus = stimulus * 1.0e3   # should use stim_series.conversion ** -1 here?
            response = response * 1.0e12  # should use series.conversion ** -1 here?
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
            sampling_rate=sampling_rate, stimulus=stimulus, response=response,
            stimulus_code=stimulus_code
        )

        # print(patch_clamp_settings_dict)

        return {
            'sweep_number': sweep_number,
            'stimulus_code': stimulus_code,
            'stimulus_name': stimulus_name,
            'stimulus': stimulus,
            'response': response,
            'stimulus_unit': stimulus_unit,
            'sampling_rate': sampling_rate,
            'epochs': epochs,
            'amp_settings': patch_clamp_settings_dict
        }

    def get_epochs(self, sampling_rate, stimulus, response, stimulus_code):

        test_epoch = get_test_epoch(stimulus, sampling_rate)
        if test_epoch:
            test_pulse = True
        else:
            test_pulse = False

        sweep_epoch = get_sweep_epoch(response)
        recording_epoch = get_recording_epoch(response)
        stimulus_epoch = get_stim_epoch(stimulus, test_pulse=test_pulse)
        experiment_epoch = get_experiment_epoch(
            stimulus, sampling_rate, test_pulse=test_pulse
        )
        # handle instance where a certain NucVC sweep has a stimulus of zero
        if stimulus_code == 'NucVCSus0':
            # if there is no experiment epoch handle things accordingly
            if experiment_epoch is None:
                # if the end of the experiment epoch is less than the sweep length
                if self.prev_experiment_epoch[1] < len(stimulus):
                    # use the previous experiment epoch instead of None
                    experiment_epoch = self.prev_experiment_epoch
            else:
                # otherwise, assign this to previous experiment epoch
                self.prev_experiment_epoch = experiment_epoch

            # if there is no stimulus epoch handle things accordingly
            if stimulus_epoch is None:
                # if the end of the stim epoch is less than the sweep length
                if self.prev_stim_epoch[1] < len(stimulus):
                    # use the previous stimulus epoch instead of None
                    stimulus_epoch = self.prev_stim_epoch
            else:
                # otherwise, assign this to previous stimulus epoch
                self.prev_stim_epoch = stimulus_epoch

        return {
            'test': test_epoch,
            'sweep': sweep_epoch,
            'recording': recording_epoch,
            'experiment': experiment_epoch,
            'stim': stimulus_epoch
        }
