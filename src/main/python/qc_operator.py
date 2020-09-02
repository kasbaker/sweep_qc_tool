"""TODO module level docstring goes here

"""
import logging
from copy import deepcopy
from multiprocessing.connection import Connection
from typing import List, Tuple, Dict, Union, NamedTuple, Any, Set, Optional

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
    nuc_vc_features: List[dict]
    full_sweep_qc_info: List[dict]


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
            sweep_data_tuple: Tuple[Dict[str, Any]],
            ontology: StimulusOntology, qc_criteria: dict, recording_date: str
    ):
        """Takes in extracted nwb data and performs auto qc operations on it

        An object that takes in data extracted from nwb file and performs
        QC operations on it. Can be used with multiprocessing when using the run
        auto QC function at the end of this file.

        Parameters
        ----------
        sweep_data_tuple : Tuple[Dict[str, Any]]
            A tuple of dictionaries, where each dictionary is the extracted data
            from one sweep.
        ontology : StimulusOntology
            Stimulus ontology object from ipfx, which contains information about
            the stimulus sets associated with this data set.
        qc_criteria : dict
            A dictionary containing the criteria used for auto qc
        recording_date : str
            A string representation of the recording date for this experiment

        """
        self._sweep_data_tuple = sweep_data_tuple
        self._ontology = ontology
        self._qc_criteria = qc_criteria
        self._recording_date = recording_date

    @property
    def sweep_data_tuple(self) -> Tuple[Dict[str, Any]]:
        """Returns tuple of sweep data."""
        return self._sweep_data_tuple

    @property
    def ontology(self) -> StimulusOntology:
        """Returns StimulusOntology object."""
        return self._ontology

    @property
    def qc_criteria(self) -> Dict[str, Union[float, int, str]]:
        """Returns qc criteria dictionary."""
        return self._qc_criteria

    @property
    def recording_date(self) -> str:
        """Returns string representation of experiment recording date."""
        return self._recording_date

    def fast_cell_qc(
            self, sweep_types: Dict[str, set], manual_values: Optional[dict] = None
    ) -> Tuple[Dict[str, Optional[Any]], List[str]]:
        """Performs auto QC at the cell level and returns a dictionary of cell
        qc features and cell qc tags.

        Parameters
        ----------
        sweep_types : Dict[str, Set[int]
            A dictionary of sets indicating which sweeps belong to which
            specific stimulus types
        manual_values : dict
            A dictionary of manual values for cell qc info to fall back on.
            Used to manually input a clamp seal if the sweep wasn't recorded.

        Returns
        -------
        features : Dict[str, Optional[Any]]
            A dictionary of cell QC features with string keys. Values are either
            float, np.float or str in the case of recording date
        tags : List[str]
            A list of strings for cell qc feature tags for missing sweeps
            (e.g. 'Seal is not available)

        """
        # intiailize manual values as empty dictionary if it is not provided
        if manual_values is None:
            manual_values = {}

        # intialize empty dictionary for features and empty list for tags
        features = {}
        tags = []

        # grab blowout voltage
        features['blowout_mv'] = self.extract_blowout_mv(sweep_types, tags)
        # grab electrode offset picoamp value from last offset sweep
        features['electrode_0_pa'] = self.extract_electrode_offset_pa(sweep_types, tags)
        # grab recording date
        if self.recording_date is None:
            tags.append("Recording date is missing")
        features['recording_date'] = self.recording_date
        # grab gigaseal from last cell attached sweep
        features["seal_gohm"] = self.fast_extract_clamp_seal(
            sweep_types, tags, manual_values
        )
        # compute input and access resistance from last breakin sweep
        input_resistance, access_resistance = \
            self.fast_extract_input_and_acess_resistance(sweep_types, tags)
        # add input and access resistance as well as ratio to features dict
        features['input_resistance_mohm'] = input_resistance
        features["initial_access_resistance_mohm"] = access_resistance
        features['input_access_resistance_ratio'] = \
            compute_input_access_resistance_ratio(input_resistance, access_resistance)

        return features, tags

    def fast_experiment_qc(self) -> Tuple[
        QCResults, Dict[str, Set[Optional[int]]]
    ]:
        """Runs cell and sweep qc, then compiles a list of full qc info.

        Gets the sweep types for all sweeps in the data set, runs cell qc and
        sweep qc. Evaluates those cell and sweep qc features then displays a qc
        summary. Compiles a list of full sweep qc info. Packages all of this
        information into a NamedTuple called qc results and a dictionary of
        sweep types. Returns the qc results and  sweep types

        Returns
        -------
        qc_results : NamedTuple[Union[list, dict]]
            A named tuple, which packages up all the qc information gathered
            during computer qc features and evaluating them
        sweep_types : Dict[Set[Optional[int]]]
            A dictionary defining sets of integers, which correspond to sweep
            numbers for sweeps of a particular 'type'

        """
        # gets a dictionary of sets of integers of sweeps of a given type
        sweep_types = self.get_sweep_types()

        # give sweep types to fast cell qc so it knows which sweeps to use
        cell_features, cell_tags = self.fast_cell_qc(sweep_types)
        # deepcopy of these features TODO why do we need to deepcopy here?
        cell_features = deepcopy(cell_features)

        # get sweep qc features for i clamp and channel recordings
        pre_qc_i_clamp_sweep_features, nuc_vc_sweep_features = self.run_sweep_qc(sweep_types)
        # deepcopy i clamp features (needed because we are about to drop bad sweeps)
        i_clamp_sweep_features = deepcopy(pre_qc_i_clamp_sweep_features)
        drop_tagged_sweeps(i_clamp_sweep_features)

        # evaluate the cell and sweep qc features based on given qc criteria
        cell_state, sweep_states = qc_experiment(
            ontology=self.ontology,
            cell_features=cell_features,
            sweep_features=i_clamp_sweep_features,
            qc_criteria=self.qc_criteria
        )

        # grab set of sweeps that made it through auto qc and add to sweep types
        # pipeline sweeps are to be displayed by default in sweep qc tool
        sweep_types['pipeline'] = {state['sweep_number'] for state in sweep_states}

        # print out a summary of the qc results to the log
        qc_summary(
            sweep_features=i_clamp_sweep_features,
            sweep_states=sweep_states,
            cell_features=cell_features,
            cell_state=cell_state
        )

        # Initialize a list of dictionaries of full qc information
        full_sweep_qc_info = [{
            'sweep_number': sweep['sweep_number'],
            'stimulus_code': sweep['stimulus_code'],
            'stimulus_name': sweep['stimulus_name'],
            'passed': None
        } for sweep in self.sweep_data_tuple]

        # populate list of full qc information based on information we have now
        full_sweep_qc_info = self.get_full_sweep_qc_info(
            full_sweep_qc_info=full_sweep_qc_info,
            pre_qc_sweep_features=pre_qc_i_clamp_sweep_features,
            post_qc_sweep_features=i_clamp_sweep_features,
            nuc_vc_features=nuc_vc_sweep_features,
            sweep_states=sweep_states
        )

        # package up all our qc information into one big NamedTuple
        qc_results = QCResults(
            cell_features=cell_features,
            cell_tags=cell_tags,
            cell_state=cell_state,
            sweep_features=i_clamp_sweep_features,
            sweep_states=sweep_states,
            nuc_vc_features=nuc_vc_sweep_features,
            full_sweep_qc_info=full_sweep_qc_info
        )

        return qc_results, sweep_types

    def get_sweep_types(self) -> Dict[str, Set[Optional[int]]]:
        """Loops through sweep data tuple and extracts sweep types from it.

        Creates a dictionary of sets of sweep types using the following keys,
        then loops through the sweep data tuple and searches for clamp modes,
        stimulus codes, and stimulus names that correspond to these keys. When
        a given sweep type is detected it is added to a set of integer sweep
        numbers that are values of the sweep types dictionary.

        Returns
        -------
        sweep_types : Dict[str, Set[Optional[int]]]
            A dictionary of sets of sweep types, where the key is the sweep type
            and the set contains integer sweep numbers of that type

        """
        # keys to use for sweep types dictionary
        sweep_keys = (
            'v_clamp', 'i_clamp', 'blowout', 'in_bath', 'seal', 'break_in',
            'ramp', 'long_square', 'coarse_long_square', 'short_square_triple',
            'short_square', 'search', 'test', 'ex_tp', 'nuc_vc'
        )

        # initialize dictionary of sweep types
        sweep_types = {key: set() for key in sweep_keys}

        # cache ontology to be referenced while looking up stimulus names
        ontology = self.ontology

        # loop through sweep data and add sweep numbers to sweep types dict
        for sweep in self.sweep_data_tuple:
            # cache necessary values from this sweep
            sweep_num = sweep['sweep_number']
            stim_unit = sweep['stimulus_unit']
            stim_code = sweep['stimulus_code']
            stim_name = sweep['stimulus_name']

            # assign sweep to either voltage clamp or current clamp set
            if stim_unit == "Volts":
                sweep_types['v_clamp'].add(sweep_num)
            if stim_unit == "Amps":
                sweep_types['i_clamp'].add(sweep_num)

            # TODO make sure the correct clamp mode is used for these sweeps
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

    def run_sweep_qc(
            self, sweep_types: Dict[str, Set[Optional[int]]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Runs sweep-wise qc feature extraction given sweep types dictionary.

        Compiles a set of sweeps to calculate qc features from, then puts it in
        to a sorted list. Loops through a generator of data for sweeps from
        which to calculate QC features. Appends QC features to one of two lists,
        i_clamp_qc_features, for pipeline IVSCC current clamp auto QC, or
        nuc_vc_qc_features, for nucleated patch voltage clamp recordings.

        Parameters
        ----------
        sweep_types : Dict[str, Set[Optional[int]]]
            A dictionary of sets of sweep numbers for certain sweep types

        Returns
        -------
        i_clamp_qc_features : List[Dict[str, Any]]
            A list of QC features and metadata associated with current clamp
            sweeps on which to perform auto QC.
        nuc_vc_qc_features : List[Dict[str, Any]]
            A list of QC features and metadata associated with nucleated patch
            voltage clamp sweeps on which to perform auto QC.

        """
        # set of current clamp pipeline sweeps to calculate features from
        i_clamp_qc_sweeps = sweep_types['i_clamp'].difference(
            sweep_types['test'], sweep_types['search']
        )
        # send out a logging warning if there are no i clamp sweeps to qc
        if not i_clamp_qc_sweeps:
            logging.warning("No current clamp sweeps available to compute QC features")

        # nuc vc sweeps to qc are intersection of nuc vc and voltage clamp clamp
        nuc_vc_qc_sweeps = sweep_types['v_clamp'].intersection(
            sweep_types['nuc_vc']
        )
        # send out a logging warning if there are no nuc vc sweeps to qc
        if not nuc_vc_qc_sweeps:
            logging.warning("No channel recording sweeps available to compute QC features")

        # sweeps to qc union of i clamp and nuc vc sweeps put in sorted list
        qc_sweeps = sorted(i_clamp_qc_sweeps.union(nuc_vc_qc_sweeps))
        # generator of sweeps to calculate qc features from
        sweep_gen = (self.sweep_data_tuple[idx] for idx in qc_sweeps)

        # initialize empty lists of i clamp and nuc vc qc features
        i_clamp_qc_features = []
        nuc_vc_qc_features = []

        # loop through sweep gen and grab qc features for each sweep
        for sweep in sweep_gen:
            # cache sweep number
            sweep_num = sweep['sweep_number']

            # check if this is a ramp sweep since they can be slow to return
            # to baseline and therefore fail some auto qc checks
            is_ramp = False
            if sweep_num in sweep_types['ramp']:
                is_ramp = True

            # check for sweep integrity and grab those tags
            tags = self.check_sweep_integrity(sweep, is_ramp)

            # initialize dictionary of sweep features
            sweep_features = {
                'sweep_number': sweep_num, 'stimulus_code': sweep['stimulus_code'],
                'stimulus_name': sweep['stimulus_name'], 'tags': tags
            }

            # grab stimulus features and update sweep features dictionary
            stim_features = self.get_stimulus_features(sweep)
            sweep_features.update(stim_features)

            # get qc features if there is no early termination or missing epochs
            if not tags:
                # get generic sweep qc features for both v clamp or i clamp
                qc_features = self.get_sweep_qc_features(sweep, is_ramp)
                if sweep_num in i_clamp_qc_sweeps:
                    # update sweep qc features with more specific keys
                    # TODO pre_vm_mv (fast) and slow_vm_mv are the same - why?
                    sweep_features.update({
                        'pre_vm_mv': qc_features['pre_baseline'],
                        'pre_noise_rms_mv': qc_features['pre_rms_fast'],
                        'slow_vm_mv': qc_features['pre_baseline'],
                        'slow_noise_rms_mv': qc_features['pre_rms'],
                        'post_vm_mv': qc_features['post_baseline'],
                        'post_noise_rms_mv': qc_features['post_rms'],
                        'vm_delta_mv': qc_features['baseline_delta']
                    })
                    i_clamp_qc_features.append(sweep_features)
                elif sweep_num in nuc_vc_qc_sweeps:
                    # update with new generic qc features
                    sweep_features.update(qc_features)
                    # get the seal value from the test pulse for this sweep
                    sweep_features['seal_value'] = self.get_seal_from_test_pulse(
                        sweep['stimulus'], sweep['response'],  # voltage and current
                        np.arange(len(sweep['stimulus'])) / sweep['sampling_rate'],  # time vector
                    )
                    nuc_vc_qc_features.append(sweep_features)
            else:
                # if there are tags for early termination or missing epochs,
                # then skip getting sweep qc features and move on to next sweep
                logging.warning("sweep {}: {}".format(sweep_num, tags))
                # set sweep passed to False due to early term / missing epochs
                sweep_features['passed'] = False
                # append incomplete features to qc features lists
                if sweep_num in i_clamp_qc_sweeps:
                    i_clamp_qc_features.append(sweep_features)
                else:
                    nuc_vc_qc_features.append(sweep_features)

        return i_clamp_qc_features, nuc_vc_qc_features

    def extract_blowout_mv(
            self, sweep_types: Dict[str, Set[Optional[int]]], tags: List[Optional[str]]
    ) -> Optional[np.ndarray]:
        """Finds the last blowout sweep and returns the average response.

        qcf.measure_blowout computes the average response from the end index of
        the test pulse to the end index of the sweep. Note that the list of tags
        passed in is modified directly and therefore this is an inplace
        operation with respect to the tags.

        Parameters
        ----------
        sweep_types : Set[Optional[int]]
            A set of sweep numbers for the blowout stimulus, blowout voltage
            is computed from the last sweep in this list.

        tags : List[Optional[str]]
            A list of strings

        Returns
        -------
        blowout_mv : np.ndarray or None
            Blowout voltage in millivolts of last current clamp blowout sweep

        """
        # blowout sweeps are an intersection of i clamp and blowout stimulus codes
        blowout_sweeps = sweep_types['i_clamp'].intersection(sweep_types['blowout'])

        if blowout_sweeps:
            # get the last blowout sweep and extract the ending voltage from it
            last_blowout_sweep = self.sweep_data_tuple[max(blowout_sweeps)]
            blowout_mv = qcf.measure_blowout(
                last_blowout_sweep['response'], last_blowout_sweep['epochs']['test'][1]
            )
        else:
            # update tags and return None if there are no blowout sweeps
            tags.append("Blowout is not available")
            blowout_mv = None

        return blowout_mv

    def extract_electrode_offset_pa(
            self, sweep_types: Dict[str, Set[Optional[int]]],
            tags: List[Optional[str]]
    ) -> Optional[np.ndarray]:
        """TODO docstring"""
        # bath sweeps are an intersection of i clamp and bath stimulus codes
        bath_sweeps = sweep_types['v_clamp'].intersection(sweep_types['in_bath'])
        if bath_sweeps:
            # get the last bath sweep and extract the ending voltage from it
            last_bath_sweep = self.sweep_data_tuple[max(bath_sweeps)]
            e0 = qcf.measure_electrode_0(last_bath_sweep['response'], last_bath_sweep['sampling_rate'])
        else:
            # update tags and return None if there are no bath sweeps
            tags.append("Electrode 0 is not available")
            e0 = None
        return e0

    def fast_extract_clamp_seal(
            self, sweep_types: Dict[str, Set[Optional[int]]],
            tags: List[Optional[str]], manual_values: dict
    ) -> Optional[np.ndarray]:
        """"TODO docstring"""
        # bath sweeps are an intersection of v clamp and bath stimulus codes
        seal_sweeps = sweep_types['v_clamp'].intersection(sweep_types['seal'])
        if seal_sweeps:
            # get the last seal sweep and measure the resistance
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
            # update tags and grab manual value if it is available
            tags.append("Seal is not available")
            seal_gohm = manual_values.get('manual_seal_gohm', None)
            if seal_gohm is not None:
                tags.append("Using manual seal value: %f" % seal_gohm)

        return seal_gohm

    def fast_extract_input_and_acess_resistance(
            self, sweep_types: Dict[str, Set[Optional[int]]],
            tags: List[Optional[str]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """TODO docstring"""
        # breakin sweeps are intersection of v clamp and breakin stimulus codes
        breakin_sweeps = sweep_types['v_clamp'].intersection(sweep_types['break_in'])
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

    @staticmethod
    def get_stimulus_features(sweep: Dict[str, Any]) -> Dict[str, Any]:
        """TODO docstring"""
        stim_features = {}

        stimulus = sweep['stimulus']
        hz = sweep['sampling_rate']

        time = np.arange(len(sweep['stimulus'])) / hz

        start_time, dur, amp, start_idx, end_idx = stf.get_stim_characteristics(stimulus, time)

        stim_features['stimulus_start_time'] = start_time
        stim_features['stimulus_amplitude'] = amp
        stim_features['stimulus_duration'] = dur

        if sweep['epochs']['experiment']:
            expt_start_idx, _ = sweep['epochs']['experiment']
            interval = stf.find_stim_interval(expt_start_idx, stimulus, hz)
        else:
            interval = None

        stim_features['stimulus_interval'] = interval

        return stim_features

    @staticmethod
    def check_sweep_integrity(sweep: Dict[str, Any], is_ramp: bool) -> List[str]:
        """TODO docstring"""

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
    def get_sweep_qc_features(sweep: Dict[str, Any], is_ramp: bool) -> Dict[str, Any]:
        qc_features = {}

        response = sweep['response']
        hz = sweep['sampling_rate']
        # measure noise before stimulus
        idx0, idx1 = ep.get_first_noise_epoch(sweep['epochs']['experiment'][0], hz)
        # count from the beginning of the experiment
        qc_features['pre_baseline_fast'], qc_features['pre_rms_fast'] = qcf.measure_vm(response[idx0:idx1])

        # measure mean and rms of Vm at end of recording
        # do not check for ramp, because they do not have enough time to recover

        rec_end_idx = sweep['epochs']['recording'][1]
        if not is_ramp:
            idx0, idx1 = ep.get_last_stability_epoch(rec_end_idx, hz)
            mean_last_stability_epoch, _ = qcf.measure_vm(response[idx0:idx1])

            idx0, idx1 = ep.get_last_noise_epoch(rec_end_idx, hz)
            _, rms_last_noise_epoch = qcf.measure_vm(response[idx0:idx1])
        else:
            rms_last_noise_epoch = None
            mean_last_stability_epoch = None

        qc_features['post_baseline'] = mean_last_stability_epoch
        qc_features["post_rms"] = rms_last_noise_epoch

        # measure mean and rms of Vm and over extended interval before stimulus, to check stability

        stim_start_idx = sweep['epochs']['stim'][0]

        idx0, idx1 = ep.get_first_stability_epoch(stim_start_idx, hz)
        mean_first_stability_epoch, rms_first_stability_epoch = qcf.measure_vm(response[idx0:idx1])

        qc_features['pre_baseline'] = mean_first_stability_epoch
        qc_features['pre_rms'] = rms_first_stability_epoch

        qc_features['baseline_delta'] = qcf.measure_vm_delta(mean_first_stability_epoch, mean_last_stability_epoch)

        return qc_features

    # @staticmethod
    # def get_epochs(
    #         sampling_rate: np.float, stimulus: np.ndarray, response: np.ndarray
    # ) -> Dict[str, Tuple[int, int]]:
    #     test_epoch = ep.get_test_epoch(stimulus, sampling_rate)
    #     if test_epoch:
    #         test_pulse = True
    #     else:
    #         test_pulse = False
    #
    #     sweep_epoch = ep.get_sweep_epoch(response)
    #     recording_epoch = ep.get_recording_epoch(response)
    #     stimulus_epoch = ep.get_stim_epoch(stimulus, test_pulse=test_pulse)
    #     experiment_epoch = ep.get_experiment_epoch(
    #         stimulus, sampling_rate, test_pulse=test_pulse
    #     )
    #
    #     return {
    #         'test': test_epoch,
    #         'sweep': sweep_epoch,
    #         'recording': recording_epoch,
    #         'experiment': experiment_epoch,
    #         'stim': stimulus_epoch
    #     }

    @staticmethod
    def get_full_sweep_qc_info(
            full_sweep_qc_info: List[dict],
            pre_qc_sweep_features: List[dict],
            post_qc_sweep_features: List[dict],
            nuc_vc_features: List[dict],
            sweep_states: List[dict]
    ) -> List[dict]:
        # TODO write docstring
        """ foo
        """
        feature_lists = [
            pre_qc_sweep_features, post_qc_sweep_features, nuc_vc_features,
            sweep_states
        ]

        for feature_list in feature_lists:
            for sweep in feature_list:
                full_sweep_qc_info[sweep['sweep_number']].update(sweep)

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


def run_auto_qc(
        sweep_data_tuple: tuple, ontology: StimulusOntology,
        qc_criteria: dict, recording_date: str, qc_output: Connection
):
    """TODO docstring"""
    qc_operator = QCOperator(
        sweep_data_tuple, ontology, qc_criteria, recording_date
    )
    qc_results, sweep_types = qc_operator.fast_experiment_qc()

    qc_out = qc_output
    qc_out.send((qc_results, sweep_types))
    qc_out.close()
