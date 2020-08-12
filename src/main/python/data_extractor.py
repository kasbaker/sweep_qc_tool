import logging
from copy import copy

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

    __slots__ = [
        'nwb_file', '_ontology', '_data_set', '_recording_date',
        '_series_iter', '_data_iter', '_series_table', '_num_sweeps',
        'v_clamp_settings_dict'
    ]

    def __init__(self, nwb_file, ontology, v_clamp_settings_keys='default'):

        # todo make settings keys 'default', False, or tuple-like
        self.nwb_file = nwb_file
        self._ontology = ontology
        self._data_set = None
        self._recording_date = None
        self._series_table = None
        self._num_sweeps = None
        self._series_iter = None
        self._data_iter = None
        if v_clamp_settings_keys != 'default':
            self.v_clamp_settings_dict = dict.fromkeys(v_clamp_settings_keys)
        else:
            # tuple of keys to use for v-clamp settings
            default_v_clamp_settings_keys = (
                "V-Clamp Holding Enable: ", "V-Clamp Holding Level: ",
                "RsComp Enable: ", "RsComp Correction: ", "RsComp Prediction: ",
                "Whole Cell Comp Enable: ", "Whole Cell Comp Cap: ", "Whole Cell Comp Resist: "
            )
            self.v_clamp_settings_dict = dict.fromkeys(default_v_clamp_settings_keys)

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


        # v_clamp_settings_keys = (
        #     "V-Clamp Holding Enable: ", "V-Clamp Holding Level: ",
        #     "RsComp Enable: ", "RsComp Correction: ", "RsComp Prediction: ",
        #     "Whole Cell Comp Enable: ", "Whole Cell Comp Cap: ", "Whole Cell Comp Resist: "
        # )
        #
        # v_clamp_settings_dict = dict.fromkeys(v_clamp_settings_keys)

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
                elif unit in {"Amps", "A", "amps", "amperes"}:
                    stimulus_unit = "Amps"
                elif unit in {"Volts", "V", "volts"}:
                    stimulus_unit = "Volts"
                    # loop through keys in v_clamp settings and extract strings
                    for key in self.v_clamp_settings_dict.keys():
                        # partition comments by key and grab the part with value we want
                        value_str = comment_str.partition(key)[2]
                        if value_str:  # if this is not an empty string extract value
                            value = value_str.splitlines()[0]  # chop off everything after newline
                            self.v_clamp_settings_dict[key] = value  # add value to dict
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

        epochs = get_epochs(
            sampling_rate=sampling_rate, stimulus=stimulus, response=response
        )

        print(self.v_clamp_settings_dict)

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


def get_epochs(sampling_rate, stimulus, response):
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

    return {
        'test': test_epoch,
        'sweep': sweep_epoch,
        'recording': recording_epoch,
        'experiment': experiment_epoch,
        'stim': stimulus_epoch
    }
