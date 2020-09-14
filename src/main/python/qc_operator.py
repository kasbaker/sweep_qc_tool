"""A module containing a class to perform QC operations and function to call it.

This module is intended to be used with the multiprocessing module in order to
quickly run auto QC in parallel with other operations. The function run_auto_qc
should be the target of a multiprocessing Process object.

Examples TODO get to work with doctest '>>>'
--------
>> from multiprocessing import Pipe, Process
>> from qc_operator import run_auto_qc, QCResults


>> qc_pipe = Pipe(duplex=False)
>> qc_worker = Process(
..     name='qc_worker', target=run_auto_qc, args=(
..     sweep_data_tuple, stimulus_ontology, qc_criteria, recording_date, qc_pipe[1]
..     )
.. )
>> qc_worker.start()
# Do other stuff in parallel here
>> qc_pipe[1].close()
>> qc_results, sweep_types = qc_pipe[0].recv()
>> qc_worker.join()
>> qc_worker.terminate()

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
    """Namedtuple to store the output QC information.

    Parameters
    ----------
    cell_features : dict
        Dictionary of cell QC features calculated from auto QC
    cell_tags : list
        List of QC tags from cell QC
    cell_state : dict
        Dictionary of results form cell QC evaluation
    sweep_features : List[dict]
        List of dictionaries of sweep QC features calculated from auto QC
    sweep_states : List[dict]
        List of dictionaries with sweep numbers and sweep 'passed' bool states
    nuc_vc_features : List[dict]
        Sweep QC features for channel recording 'NucVC' sweeps
    full_sweep_qc_info : List[dict]
        All of the sweep features lists compiled into one big list

    """
    cell_features: dict
    cell_tags: list
    cell_state: dict
    sweep_features: List[dict]
    sweep_states: List[dict]
    nuc_vc_features: List[dict]
    full_sweep_qc_info: List[dict]


class QCOperator(object):
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
        features['electrode_0_pa'] = self.extract_pipette_offset_pa(sweep_types, tags)
        # grab recording date
        if self.recording_date is None:
            tags.append("Recording date is missing")
        features['recording_date'] = self.recording_date
        # grab gigaseal from last cell attached sweep
        features["seal_gohm"] = self.extract_clamp_seal_gohm(
            sweep_types, tags, manual_values
        )
        # compute input and access resistance from last breakin sweep
        input_resistance, access_resistance = \
            self.extract_input_and_acess_resistance_mohm(sweep_types, tags)
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

        # populate list of full qc information based on information we have now
        full_sweep_qc_info = self.update_full_sweep_qc_info(
            pre_qc_i_clamp_sweep_features, i_clamp_sweep_features,
            nuc_vc_sweep_features, sweep_states
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
            stim_features = self.extract_stimulus_features(sweep)
            sweep_features.update(stim_features)

            # get qc features if there is no early termination or missing epochs
            if not tags:
                # get generic sweep qc features for both v clamp or i clamp
                qc_features = self.extract_sweep_qc_features(sweep, is_ramp)
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
                    # todo check full sweep
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
            A dictionary of sets of sweep numbers for certain sweep types.
            Used to find the last current-clamp blowout sweep.
        tags : List[Optional[str]]
            A list of strings for cell qc tags indicating missing sweeps etc.

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
                last_blowout_sweep['response'],
                last_blowout_sweep['epochs']['test'][1]
            )

        else:
            # update tags and return None if there are no blowout sweeps or the
            # sweep is incomplete for some reason
            tags.append("Blowout is not available")
            blowout_mv = None

        # TODO exceptions?

        return blowout_mv

    def extract_pipette_offset_pa(
            self, sweep_types: Dict[str, Set[Optional[int]]],
            tags: List[Optional[str]]
    ) -> Optional[np.ndarray]:
        """Finds the last pipette offset sweep and returns the average response.

        Computes the average response of the first 5 milliseconds of the
        pipette offset (in bath) sweep. Returns a tag saying the offset is not
        available if there are no voltage clamp bath sweeps.

        Parameters
        ----------
        sweep_types : Set[Optional[int]]
            A dictionary of sets of sweep numbers for certain sweep types.
            Used to find the last voltage-clamp in-bath sweep.
        tags : List[Optional[str]]
            A list of strings for cell qc tags indicating missing sweeps etc.

        Returns
        -------
        offset_pa : np.ndarray or None
            Offset current in pA of last current clamp blowout sweep

        """
        # bath sweeps are an intersection of i clamp and bath stimulus codes
        bath_sweeps = sweep_types['v_clamp'].intersection(sweep_types['in_bath'])
        if bath_sweeps:
            # get the last bath sweep and extract the ending voltage from it
            last_bath_sweep = self.sweep_data_tuple[max(bath_sweeps)]
            offset_pa = qcf.measure_electrode_0(
                last_bath_sweep['response'], last_bath_sweep['sampling_rate']
            )
        else:
            # update tags and return None if there are no bath sweeps
            tags.append("Electrode 0 is not available")
            offset_pa = None
        # TODO exceptions?
        return offset_pa

    def extract_clamp_seal_gohm(
            self, sweep_types: Dict[str, Set[Optional[int]]],
            tags: List[Optional[str]], manual_values: dict
    ) -> Optional[np.ndarray]:
        """Finds the last clamp seal sweep and returns the resistance in GOhm.

        Finds the last cell-attached 'seal' sweep and computes the average
        resistance of that seal. The function qcf.measure_seal() takes the last
        1 ms before the square stimulus pulse and the last 1 ms of the end of
        the square pulse. It then computes the average of the two and uses that
        to calculate the resistance. The seal sweep has multiple pulses in it so
        it averages all of those averages in order to compute the GOhm seal.
        If we can't compute the seal or there is no seal sweep, then we try to
        grab a value from the manual_values dict.

        Parameters
        ----------
        sweep_types : Set[Optional[int]]
            A dictionary of sets of sweep numbers for certain sweep types.
            Used to find the last voltage-clamp cell-attached seal sweep.
        tags : List[Optional[str]]
            A list of strings for cell qc tags indicating missing sweeps etc.
        manual_values : dict
            A dictionary of manual values to fall back on. The value for
            manual_values['manual_seal_gohm'] is used if we can't get the seal.

        Returns
        -------
        seal_gohm : np.ndarray or None
            Resistance of the clamp seal in gigaohms of the last seal sweep

        Raises
        ------
        FeatureError
            If we can't compute the seal or the seal is infinite

        """
        # seal sweeps are an intersection of v clamp and bath stimulus codes
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
        # TODO exceptions?
        return seal_gohm

    def extract_input_and_acess_resistance_mohm(
            self, sweep_types: Dict[str, Set[Optional[int]]],
            tags: List[Optional[str]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Computes the input and access resistance of the last break-in sweep.

        Finds the last break-in sweep and computes the average input and access
        resistance of that sweep. The function qcf.measure_input_resistance()
        takes the last 1 ms before the square stimulus pulse and the last 1 ms
        of the end of the square pulse. It then computes the average of the two
        and uses that to calculate the resistance. The break-in has multiple
        pulses in it so it averages all of those averages in order to compute
        the input resistance in MOhms. Access resistance is measured in a
        similar way, but instead of finding the average steady state
        qcf.measure_initial_access_resistance() takes the maximum value of the
        response in pA.


        Parameters
        ----------
        sweep_types : Set[Optional[int]]
            A dictionary of sets of sweep numbers for certain sweep types.
            Used to find the last voltage-clamp break-in sweep.
        tags : List[Optional[str]]
            A list of strings for cell qc tags indicating missing sweeps etc.

        Returns
        -------
        input_resistance : np.ndarray or None
            Input resistance in MOhm of the last voltage-clamp break-in sweep
        access_resistance : np.ndarray or None
            Access resistance in MOhm of the last voltage-clamp break-in sweep

        Raises
        ------
        Exception
            Raise any errors we encountered while trying to compute resistances

        """

        # breakin sweeps are intersection of v clamp and breakin stimulus codes
        breakin_sweeps = sweep_types['v_clamp'].intersection(sweep_types['break_in'])

        if breakin_sweeps:
            # find the last break-in sweep
            last_breakin_sweep = self.sweep_data_tuple[max(breakin_sweeps)]
            # generate time vector
            time = np.arange(
                len(last_breakin_sweep['stimulus'])
            ) / last_breakin_sweep['sampling_rate']

            try:
                # try to calculate input resistance
                input_resistance = qcf.measure_input_resistance(
                    last_breakin_sweep['stimulus'], last_breakin_sweep['response'], time
                )
            except Exception:
                # log and raise any errors we encounter if this doesn't work
                logging.warning("Error reading input resistance.")
                raise

            try:
                # try to calculate access resistance
                access_resistance = qcf.measure_initial_access_resistance(
                    last_breakin_sweep['stimulus'], last_breakin_sweep['response'], time
                )
            except Exception:
                # log and raise any errors we encounter if this doesn't work
                logging.warning("Error reading initial access resistance.")
                raise

        else:
            # if we can't find the break-in sweep, then note it in tags and
            # return None for both input and access resistance
            tags.append("Breakin sweep not found")
            input_resistance = None
            access_resistance = None
        # TODO exceptions?
        return input_resistance, access_resistance

    def update_full_sweep_qc_info(self, *args: List[dict]) -> List[dict]:
        """Compiles full list of qc information from various QC feature lists.

        Takes in any number of lists of dictionaries of QC information. Those
        dictionaries must contain the key 'sweep_number'. Loops through all of
        the dictionaries in those lists and updates the full_sweep_qc_info
        list appropriately.

        Parameters
        ----------
        *args : List[dict]
            List of dictionaries of qc information. Dictionaries must contain
            the key 'sweep_number' in order for this to work.

        Returns
        -------
        full_sweep_qc_info : List[dict]
            A list of dictionaries with all the compiled QC information

        """
        # Initialize a list of dictionaries of full qc information
        full_sweep_qc_info = [{
            'sweep_number': sweep['sweep_number'],
            'stimulus_code': sweep['stimulus_code'],
            'stimulus_name': sweep['stimulus_name'],
            'passed': None
        } for sweep in self.sweep_data_tuple]

        # generator of lists of qc features
        feature_lists = (feature_list for feature_list in args)

        # loop through feature lists and sweeps and update full_sweep_qc_info
        for feature_list in feature_lists:
            for sweep in feature_list:
                try:
                    full_sweep_qc_info[sweep['sweep_number']].update(sweep)
                except KeyError:
                    logging.warning(f"Could not find sweep number of {sweep}")

        return full_sweep_qc_info

    @staticmethod
    def extract_stimulus_features(sweep: Dict[str, Any]) -> Dict[str, Any]:
        """Takes in a sweep and computes stimulus features from it.

        This function takes in a dictionary of extracted data from one sweep,
        then uses the stimulus time series, the sampling rate, and the
        experiment epoch to calculate stimulus features.


        Parameters
        ----------
        sweep : Dict[str, Any]
            A dictionary of extracted data for one sweep. The stimulus time
            series, the sampling rate, and the epoch indices are needed here.

        Returns
        -------
        stimulus_features : Dict[str, Any]
            A dictionary of stimulus features indicating things like stimulus
            start time, amplitude, and duration.

        """
        # initialize empty dictionary of stimulus features
        stimulus_features = {}

        # cache stimulus array and sampling rate from sweep
        stimulus = sweep['stimulus']
        hz = sweep['sampling_rate']

        # generate time vector
        time = np.arange(len(sweep['stimulus'])) / hz

        # calculate stimulus features from time series data
        start_time, dur, amp, start_idx, end_idx = stf.get_stim_characteristics(stimulus, time)

        # update stimuls features dictionary with calculated values
        stimulus_features['stimulus_start_time'] = start_time
        stimulus_features['stimulus_amplitude'] = amp
        stimulus_features['stimulus_duration'] = dur

        # find the stimulus interval if the experiment epoch exists
        if sweep['epochs']['experiment']:
            expt_start_idx, _ = sweep['epochs']['experiment']
            interval = stf.find_stim_interval(expt_start_idx, stimulus, hz)
        else:
            interval = None
        # update stimulus features with stimulus interval
        stimulus_features['stimulus_interval'] = interval

        return stimulus_features

    @staticmethod
    def check_sweep_integrity(sweep: Dict[str, Any], is_ramp: bool) -> List[str]:
        """Checks sweep for missing epochs and early recording termination.

        Loops through the sweeps epochs and checks for any missing epochs. Then
        checks to see if the end of the recording epoch is less than the end of
        the experiment epoch, given that this is not a 'Ramp' sweep. Appends
        strings to list of tags for missing epochs or early recording
        termination and returns a list of these tags.

        Parameters
        ----------
        sweep : Dict[str, Any]
            A dictionary of extracted data for one sweep. Only the epochs are
            needed for this function.
        is_ramp : bool
            A boolean value indicating whether or not a sweep is a 'Ramp' sweep
            since sometimes these sweeps terminate early, but are still good.

        Returns
        -------
        tags : List[str]
            A list of qc tags for this sweep. Missing epochs or early recording
            termination are indicated here.

        """
        # initialize empty list of tags
        tags = []

        # loop through sweep epochs and check for missing ones
        for k, v in sweep['epochs'].items():
            if not v:
                # append missing sweep tags as appropriate
                tags.append(f"{k} epoch is missing")

        # check to see if the recording ends before the end of experiment epoch
        if not is_ramp:
            if sweep['epochs']['recording'] and sweep['epochs']['experiment']:
                if sweep['epochs']['recording'][1] < sweep['epochs']['experiment'][1]:
                    # append early termination tag as appropriate
                    tags.append("Recording stopped before completing the experiment epoch")

        return tags

    @staticmethod
    def extract_sweep_qc_features(
            sweep: Dict[str, Any], is_ramp: bool
    ) -> Dict[str, Any]:
        """Calculates QC features from the response of one sweep.

        Uses the sweep response, sampling rate, and epochs to calculate QC
        features such as baseline mean, fast rms noise, slow rms noise, and
        change in baseline from pre to post stimulus.

        Parameters
        ----------
        sweep : Dict[str, Any]
            A dictionary of extracted data for one sweep. The sweep response,
            sampling rate, and epochs are needed here.
        is_ramp : bool
            A boolean value indicating whether or not a sweep is a 'Ramp' sweep
            since sometimes these don't return to baseline, but are still good.

        Returns
        -------
        qc_features : Dict[str, Any]
            A dictionary of QC features calculated from this sweep's response

        """
        # initialize empty dictionary of qc features
        qc_features = {}
        # cache response and sampling rate from sweep
        response = sweep['response']
        hz = sweep['sampling_rate']

        # measure noise before stimulus
        idx0, idx1 = ep.get_first_noise_epoch(sweep['epochs']['experiment'][0], hz)
        # count from the beginning of the experiment
        qc_features['pre_baseline_fast'], qc_features['pre_rms_fast'] = qcf.measure_vm(response[idx0:idx1])

        # measure mean and rms of response at end of recording
        # do not check for ramp, because they do not have enough time to recover
        rec_end_idx = sweep['epochs']['recording'][1]
        if not is_ramp:
            # get index of last stability epoch and measure it's mean response
            idx0, idx1 = ep.get_last_stability_epoch(rec_end_idx, hz)
            mean_last_stability_epoch, _ = qcf.measure_vm(response[idx0:idx1])

            # get the index of the last noise epoch and measure its rms
            idx0, idx1 = ep.get_last_noise_epoch(rec_end_idx, hz)
            _, rms_last_noise_epoch = qcf.measure_vm(response[idx0:idx1])
        else:
            # don't measure these for ramp
            rms_last_noise_epoch = None
            mean_last_stability_epoch = None

        # store calculated values in qc features dictionary
        qc_features['post_baseline'] = mean_last_stability_epoch
        qc_features["post_rms"] = rms_last_noise_epoch

        # measure mean and rms of Vm and over extended interval before stimulus, to check stability
        stim_start_idx = sweep['epochs']['stim'][0]

        # get first stability epoch, then measure mean and rms noise for it.
        idx0, idx1 = ep.get_first_stability_epoch(stim_start_idx, hz)
        mean_first_stability_epoch, rms_first_stability_epoch = qcf.measure_vm(response[idx0:idx1])

        # store more calculated values
        qc_features['pre_baseline'] = mean_first_stability_epoch
        qc_features['pre_rms'] = rms_first_stability_epoch

        # store absolute value difference of pre and post baseline means
        qc_features['baseline_delta'] = qcf.measure_vm_delta(
            mean_first_stability_epoch, mean_last_stability_epoch
        )

        return qc_features

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
            time vector (s)

        Returns
        -------
        np.float
            Resistance of test pulse seal in MOhm

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
        # todo
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
        sweep_data_tuple: Tuple[Dict[str, Any]], ontology: StimulusOntology,
        qc_criteria: dict, recording_date: str, qc_output: Connection
):
    """Runs auto QC on EPhys data and pipes results through output connection.

    Initializes QCOperator using provided parameters. Runs auto QC on the
    experiment, which returns the qc_results NamedTuple and sweep_types
    dictionary. Pipes out the data via provided qc_output Connection and finally
    closes the pipe. This function is intended to be the target of a Process
    from the multiprocessing module.

    Parameters
    ----------
    sweep_data_tuple : Tuple[Dict[str, Any]]
        A tuple of dictionaries containing extracted data from each sweep.
    ontology : StimulusOntology
        An ipfx stimulus ontology object, used for identifying sweep types.
    qc_criteria : dict
        A dictionary of auto QC criteria to use when evaluating QC features.
    recording_date : str
        A string with the recording date in it, used in cell QC features.
    qc_output : multiprocessing.Connection
        The output end of a multiprocessing Pipe used to transmit QC results.

    """
    qc_operator = QCOperator(
        sweep_data_tuple, ontology, qc_criteria, recording_date
    )
    qc_results, sweep_types = qc_operator.fast_experiment_qc()

    qc_out = qc_output
    qc_out.send((qc_results, sweep_types))
    qc_out.close()
