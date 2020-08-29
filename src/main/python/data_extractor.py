import logging
from copy import copy
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
from pynwb.icephys import (
    CurrentClampSeries, CurrentClampStimulusSeries,
    VoltageClampSeries, VoltageClampStimulusSeries,
    IZeroClampSeries
)
from ipfx.dataset.ephys_data_set import EphysDataSet
from ipfx.dataset.create import create_ephys_data_set
from ipfx.epochs import (
    get_test_epoch, get_sweep_epoch, get_recording_epoch,
    get_stim_epoch, get_experiment_epoch
)


class DataExtractor(object):

    __slots__ = [
        'nwb_file', '_ontology', '_data_set', '_recording_date',
        '_sweep_data_tuple', 'prev_experiment_epoch', 'prev_stim_epoch',
        'v_clamp_settings_dict', 'i_clamp_settings_dict'
    ]

    def __init__(
            self, nwb_file: str, ontology: Optional[Union[str, object]] = None,
            v_clamp_settings_keys: Tuple[str] = 'default',
            i_clamp_settings_keys: Tuple[str] = 'default'
    ):
        """ An object that extracts data from an nwb file and returns an
        iterator containing all the sweep data and metdata associated with
        each sweep.

        Parameters
        ----------
        nwb_file : str
            String containing the location of the nwb file to extract data from
        ontology : Optional[Union[str, StimulusOntology]]
            String containing location of stimulus ontology or StimulusOntology
            object from ipfx, which contains information about stimulus sets in
            nwb file. If left as None, extraction uses default ontology file
        v_clamp_settings_keys : Tuple[str]
            Tuple of dictionary keys to extract metadata strings from the
            comments section for each voltage clamp sweep. If settings are
            'default' these are settings from the path clamp amplifier
        i_clamp_settings_keys : Tuple[str]
            Same as v_clamp_settings_keys, but for current clamp instead

        """
        # initialize parameters passed in to __init__
        self.nwb_file = nwb_file
        self._ontology = ontology

        # initialize dictionary of keys to extract data from comments strings
        if v_clamp_settings_keys != 'default':
            # pass in user defined keys if parameter is 'default'
            self.v_clamp_settings_dict = dict.fromkeys(v_clamp_settings_keys)
        else:
            # tuple of default strings to use as keys for v-clamp settings
            default_v_clamp_settings_keys = (
                "V-Clamp Holding Enable", "V-Clamp Holding Level",
                "RsComp Enable", "RsComp Correction", "RsComp Prediction",
                "Whole Cell Comp Enable", "Whole Cell Comp Cap", "Whole Cell Comp Resist",
                "Pipette Offset"
            )
            # initialize empty dictionary with default keys
            self.v_clamp_settings_dict = dict.fromkeys(default_v_clamp_settings_keys)

        # same as above, but for voltage clamp
        if i_clamp_settings_keys != 'default':
            # pass in user defined keys if parameter is 'default'
            self.i_clamp_settings_dict = dict.fromkeys(i_clamp_settings_keys)
        else:
            # tuple of default strings to use as keys for i-clamp settings
            default_i_clamp_settings_keys = (
                "I-Clamp Holding Enable", "I-Clamp Holding Level",
                "Neut Cap Enabled", "Neut Cap Value",
                "Bridge Bal Enable", "Bridge Bal Value",
                "Autobias", "Autobias Vcom", "Autobias Vcom variance",
                "Pipette Offset"
            )
            # initialize empty dictionary with default keys
            self.i_clamp_settings_dict = dict.fromkeys(default_i_clamp_settings_keys)

        # initialize protected members of this class involved in data extraction
        self._data_set = None
        self._recording_date = None
        self._sweep_data_tuple = None

        # initialize empty members of this class to use to get epochs for a
        # particular sweep type that has a stimulus epoch of zero amplitude
        self.prev_experiment_epoch = None
        self.prev_stim_epoch = None

    @property
    def data_set(self) -> EphysDataSet:
        """ Creates an ipfx EphysDataSet if it does not exist then returns it.

        Returns
        -------
        self._data_set : EphysDataSet
            EphysDataset object created in ipfx

        """
        if not self._data_set:
            # create the data set if it doesn't exist yet
            self._data_set = create_ephys_data_set(
                nwb_file=self.nwb_file, ontology=self._ontology
            )
        return self._data_set

    @property
    def recording_date(self) -> str:
        """ Extracts the recording date if it does not exist and returns it.

        Returns
        -------
        self._recording_date : str
            Recording date of the experiment extracted from the data set

        """
        # extract the recording date if it doesn't exist yet
        if not self._recording_date:
            try:
                self._recording_date = self.data_set.get_recording_date()
            except KeyError:
                # return a warning if the date is missing
                logging.warning('recording_date is missing')
        return self._recording_date

    @property
    def sweep_data_tuple(self) -> tuple:
        """ Extracts sweep data if it hasn't already been extracted and then
        returns a tuple of that extracted data.

        Returns
        -------
        self._sweep_data_tuple : Tuple[dict, Any]


        """
        # extract the sweep data tuple if it doesn't exist yet
        if not self._sweep_data_tuple:
            # TODO pull request to ipfx so this doens't access protected member
            #    of EphysDataSet class
            series_table = self.data_set._data.nwb.sweep_table
            # number of sweeps is half the shape of the sweep table here because
            # each sweep has two series associated with it (stimulus and response)
            num_sweeps = series_table.series.shape[0] // 2
            # map get_series onto all of the sweeps
            series_iter = map(series_table.get_series, range(num_sweeps))
            # map series data extraction on to series_iter
            sweep_data_iter = map(self._extract_series_data, series_iter)
            # create a tuple out of the data iterator
            self._sweep_data_tuple = tuple(sweep_data_iter)
        return self._sweep_data_tuple

    def _extract_series_data(self, series: List[Union[
        VoltageClampSeries, CurrentClampSeries, IZeroClampSeries,
        VoltageClampStimulusSeries, CurrentClampStimulusSeries
    ]]) -> dict:
        """ Extracts the data that we want from a list containing two patch
        clamp series and returns a dictionary containg data from this sweep.

        Parameters
        ----------
        series : List[PatchClampSeries]
            A list containing a pair of patch clamp series for this sweep

        Returns
        -------
        Dict[str, Any]
            A dictionary of data extracted from this sweep

        """
        # initialize some empty variables for this sweep
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

        # loop through list containg a pair of series and extract data from it
        for s in series:
            # check to see if this is a time series of the sweep response
            if isinstance(s, (VoltageClampSeries, CurrentClampSeries, IZeroClampSeries)):
                # grab raw time series data
                response = copy(s.data[:] * float(s.conversion))
                # grab stimulus code from description and chop off '_DA_0' and '['
                stimulus_code = s.stimulus_description.partition("_DA_0")[0].partition("[")[0]
                # get stimulus name from StimulusOntology owned by the data set
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

    def get_epochs(self, sampling_rate: float, stimulus: np.ndarray,
                   response: np.ndarray, stimulus_code: str) -> Dict[str, tuple]:
        """ Detects the indexes of various epochs for a sweep, given the
        sampling rate, stimulus, and response of a given sweep. Stimulus code
        is used as a workaround for a specific voltage clamp sweep that has
        a zero-amplitude stimulus.

        Parameters
        ----------
        sampling_rate : float
            The sampling rate for the stimulus and response arrays in Hz
        stimulus : np.ndarray

        response : np.ndarray
            foo
        stimulus_code : np.ndarray
            foo

        Returns
        -------
        Dict[str, (start_idx, end_idx)]
            A dictionary of tuples containing the epoch start and end indices

        """
        # grab test pulse epoch if it exists
        test_epoch = get_test_epoch(stimulus, sampling_rate)

        # set test pulse bool based on whether or not there is a test epoch
        if test_epoch:
            test_pulse = True
        else:
            test_pulse = False

        # grab sweep, recording, stimulus, and experiment epochs
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
