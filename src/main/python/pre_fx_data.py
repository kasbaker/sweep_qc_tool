import json
import logging
import os
import copy
from typing import Optional, Dict, Any
import ipfx
from PyQt5.QtCore import QObject, pyqtSignal

from ipfx.ephys_data_set import EphysDataSet
from ipfx.qc_feature_extractor import cell_qc_features, sweep_qc_features
from ipfx.qc_feature_evaluator import qc_experiment, DEFAULT_QC_CRITERIA_FILE
from ipfx.bin.run_qc import qc_summary
from ipfx.stimulus import StimulusOntology, Stimulus
from ipfx.data_set_utils import create_data_set
from ipfx.sweep_props import drop_tagged_sweeps
import ipfx.sweep_props as sweep_props
from error_handling import exception_message
from marshmallow import ValidationError
from schemas import PipelineParameters


class PreFxData(QObject):

    # TODO move to NWBData
    stimulus_ontology_set = pyqtSignal(StimulusOntology, name="stimulus_ontology_set")
    stimulus_ontology_unset = pyqtSignal(name="stimulus_ontology_unset")

    # TODO move to QCData
    qc_criteria_set = pyqtSignal(dict, name="qc_criteria_set")
    qc_criteria_unset = pyqtSignal(name="qc_criteria_unset")

    # TODO break into QCData: begin/end_auto_qc
    end_commit_calculated = pyqtSignal(list, list, dict, EphysDataSet, name="end_commit_calculated")

    # TODO replace with NWBData begin/end_load_nwb
    new_data = pyqtSignal(EphysDataSet, name="new_data")

    data_changed = pyqtSignal(str, StimulusOntology, list, dict, name="data_changed")

    # TODO move to DataManager
    status_message = pyqtSignal(str, name="status_message")

    def __init__(self):
        """ Main data store for all data upstream of feature extraction. This
        includes:
            - the EphysDataSet
            - the StimulusOntology
            - the qc criteria
            - the sweep extraction results
            - the qc results
        """
        super(PreFxData, self).__init__()



        # TODO QCData
        self._qc_criteria: Optional[Dict] = None
        self.manual_qc_states: Dict[int, str] = {}
        # TODO cell_features -> cell_qc_info
        self.cell_features: dict = {}
        self.cell_tags: list = []
        self.cell_state: dict = {}
        # TODO sweep_features -> sweep_qc_info
        self.sweep_features: list = []
        self.sweep_states: list = []

        # TODO NWBData
        self._stimulus_ontology: Optional[StimulusOntology] = None
        self.data_set: Optional[EphysDataSet] = None
        self.nwb_path: Optional[str] = None
        self.ontology_file = None

    # TODO move to DataManager
    def _notifying_setter(
        self, 
        attr_name: str, 
        value: Any, 
        on_set: pyqtSignal, 
        on_unset: pyqtSignal,
        send_value: bool = False
    ):
        """ Utility for a setter that emits Qt signals when the attribute in 
        question changes state.

        Parameters
        ----------
        attr_name :
            identifies attribute to be set
        value : 
            set attribute to this value
        on_set : 
            emitted when the new value is not None
        on_unset :
            emitted when the new value is None
        send_value : 
            if True, the new value will be included in the emitted signal

        """
        setattr(self, attr_name, value)

        if value is None:
            on_unset.emit()
        else:
            if send_value:
                on_set.emit(value)
            else:
                on_set.emit()

    # TODO move to NWBData
    @property
    def stimulus_ontology(self) -> Optional[StimulusOntology]:
        return self._stimulus_ontology

    @stimulus_ontology.setter
    def stimulus_ontology(self, value: Optional[StimulusOntology]):
        self._notifying_setter(
            "_stimulus_ontology", 
            value,
            self.stimulus_ontology_set, 
            self.stimulus_ontology_unset,
            send_value=True
        )

    # TODO move to QCData
    @property
    def qc_criteria(self) -> Optional[Dict]:
        return self._qc_criteria

    # TODO move to QCData
    @qc_criteria.setter
    def qc_criteria(self, value: Optional[Dict]):
        self._notifying_setter(
            "_qc_criteria", 
            value,
            self.qc_criteria_set, 
            self.qc_criteria_unset,
            send_value=True
        )

    # TODO NWBData
    def set_default_stimulus_ontology(self):
        self.load_stimulus_ontology_from_json(
            StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE
        )

    # TODO move to QCData
    def set_default_qc_criteria(self):
        self.load_qc_criteria_from_json(DEFAULT_QC_CRITERIA_FILE)

    # TODO move to NWBData
    def load_stimulus_ontology_from_json(self, path: str):
        """ Attempts to read a stimulus ontology file from a JSON. If 
        successful (and other required data are already set), attempts to 
        run the pre-fx pipeline

        Parameters
        ----------
        path : str
            load ontology from here

        """

        try:
            with open(path, "r") as ontology_file:
                ontology_data = json.load(ontology_file)
            self.stimulus_ontology = StimulusOntology(ontology_data)
            self.ontology_file = path

            if self.nwb_path is not None and self.qc_criteria is not None:
                self.run_extraction_and_auto_qc(commit=True)

        except Exception as err:
            exception_message(
                "StimulusOntology load failed",
                f"failed to load stimulus ontology file from {path}",
                err
            )

    # TODO move to QCData
    def load_qc_criteria_from_json(self, path: str):
        """ Attempts to read qc criteria from a JSON. If successful (and other 
        required data are already set), attempts to run the pre-fx pipeline

        Parameters
        ----------
        path : str
            load criteria from here

        """

        try:
            with open(path, "r") as criteria_file:
                self.qc_criteria = json.load(criteria_file)
            
            if self.nwb_path is not None and self.stimulus_ontology is not None:
                self.run_extraction_and_auto_qc(commit=True)

        except Exception as err:
            exception_message(
                "QC criteria load failure",
                f"failed to load qc criteria file from {path}",
                err
            )

    # TODO move to NWBData
    def load_data_set_from_nwb(self, path: str):
        """ Attempts to read an NWB file describing an experiment. Fails if 
        qc criteria or stimulus ontology not already present. Emits new_data
        signal, which calls SweepTableModel.build_sweep_table

        Parameters
        ----------
        path: str
            Load the dataset from this location
        """
        self.nwb_path = path
        try:
            if self.stimulus_ontology is None:
                raise ValueError("must set stimulus ontology before loading a data set!")
            elif self.qc_criteria is None:
                raise ValueError("must set qc criteria before loading a data set!")

            self.status_message.emit("Creating data set from .nwb...")
            self.data_set = create_data_set(
                sweep_info=None,
                nwb_file=self.nwb_path,
                ontology=self.stimulus_ontology,
                api_sweeps=True,
                h5_file=None,
                validate_stim=True
            )
            # Builds an abstract sweep table model with the new data
            self.status_message.emit("Building new sweep table...")
            self.new_data.emit(self.data_set)
            self.status_message.emit("Done building sweep table...")
            # self.status_message.emit("Running extraction and auto qc...")
            # self.run_extraction_and_auto_qc(commit=True)
            #
            # self.status_message.emit("Done running extraction and auto qc")

        except Exception as err:
            exception_message(
                "Unable to load NWB",
                f"failed to load NWB file from {path}",
                err
            )

    # TODO move to QCData
    def extract_manual_sweep_states(self):
        """ Extract manual sweep states in the format schemas.ManualSweepStates
        from PreFxData
        """
        # TODO change this to all sweeps, not just sweep_features
        return [
            {
                "sweep_number": sweep["sweep_number"],
                "sweep_state": self.manual_qc_states[sweep["sweep_number"]]
            }
            for sweep in self.sweep_features
        ]

    # TODO move to QCData
    def save_manual_states_to_json(self, filepath: str):

        json_data = {
            "input_nwb_file": self.nwb_path,
            "stimulus_ontology_file": self.ontology_file,
            "manual_sweep_states": self.extract_manual_sweep_states(),
            "qc_criteria": self._qc_criteria,
            "ipfx_version": ipfx.__version__
        }

        try:
            PipelineParameters().load(json_data)
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=4)

        except ValidationError as valerr:
            exception_message(
                "Unable to save manual states to JSON",
                f"Manual states data failed schema validation",
                valerr
            )

        except IOError as ioerr:
            exception_message(
                "Unable to write file",
                f'Unable to write to file {filepath}',
                ioerr
            )

    # TODO move auto QC portion to QCData
    def run_extraction_and_auto_qc(self, commit=True):
        """ Creates a data set from the nwb path;
        calculates cell features, tags, and sweep features using ipfx;
        and runs auto qc on the experiment. If commit=True (default setting),
        it creates a dictionary of default manual qc states and calls
        SweepTableModel.on_new_data(), which builds the sweep table and
        generates all the thumbnail plots.

        Parameters
        ----------
        commit : bool
            indicates whether or not to build new sweep table model
        """
        # TODO make this work with pre-loaded dataset
        # cell_features: overall features for the cell
        # cell_tags: details about the cell (e.g. 'Blowout is not available'
        # sweep_features: list of dictionaries containing sweep features for
        #   sweeps that have been filtered and have passed auto-qc
        self.cell_features, self.cell_tags, self.sweep_features = \
            extract_qc_features(self.data_set)

        sweep_props.drop_tagged_sweeps(self.sweep_features)
        # cell_state: list of dictionaries containing sweep pass/fail states
        self.cell_state, self.cell_features, \
            self.sweep_states, self.sweep_features = \
            run_qc(self.stimulus_ontology, self.cell_features,
                   self.sweep_features, self.qc_criteria)

        if commit:
            self.begin_commit_calculated.emit()

            # creates dictionary of manual qc states from sweep features
            self.manual_qc_states = {
                sweep["sweep_number"]: "default"
                for sweep in self.sweep_features
            }

            # Calls SweepTableModel.on_new_data(), which builds the sweep table
            # and PreFxController.on_data_set_set()
            self.end_commit_calculated.emit(
                self.sweep_features, self.sweep_states,
                self.manual_qc_states, self.data_set
            )

        # calls FxData.set_fx_parameters()
        self.data_changed.emit(
            self.nwb_path, self.stimulus_ontology,
            self.sweep_features, self.cell_features
        )

    # TODO move to QCData
    def on_manual_qc_state_updated(self, sweep_number: int, new_state: str):
        self.manual_qc_states[sweep_number] = new_state
        self.update_sweep_states()
        self.data_changed.emit(self.nwb_path,
                               self.stimulus_ontology,
                               self.sweep_features,
                               self.cell_features)

    # TODO move to QCData
    def get_non_default_manual_sweep_states(self):
        manual_sweep_states = []

        for k, v in self.manual_qc_states.items():
            if v not in ["default"]:
                manual_sweep_states.append(
                    {"sweep_number": k,
                     "passed": v == "passed"
                     }
                )
        return manual_sweep_states

    # TODO move to QCData
    def update_sweep_states(self):
        manual_sweep_states = self.get_non_default_manual_sweep_states()
        sweep_states = copy.deepcopy(self.sweep_states)
        sweep_props.override_auto_sweep_states(manual_sweep_states, sweep_states)
        sweep_props.assign_sweep_states(sweep_states, self.sweep_features)


# TODO move to QCData
def extract_qc_features(data_set):
    # gets QC features at the cell level (input / access / seal / etc.)
    cell_features, cell_tags = cell_qc_features(
        data_set,
        # manual_values=cell_qc_manual_values
    )
    # extracts features at the sweep level, filtering out current clamp,
    # test sweeps and search sweeps
    # TODO implement this function: data_set.filtered_sweep_table(
    #  clamp_mode=data_set.CURRENT_CLAMP, stimuli_exclude=["Test", "Search"])
    sweep_features = sweep_qc_features(data_set)    # gets features from ipfx
    # drops any sweeps that have tags
    drop_tagged_sweeps(sweep_features)
    return cell_features, cell_tags, sweep_features


# TODO move to QCData
def run_qc(stimulus_ontology, cell_features, sweep_features, qc_criteria):
    """Adding qc status to sweep features
    Outputs qc summary on a screen
    """
    # making a deep copy of features
    cell_features = copy.deepcopy(cell_features)
    sweep_features = copy.deepcopy(sweep_features)

    # runs auto QC on the data set
    cell_state, sweep_states = qc_experiment(
        ontology=stimulus_ontology,
        cell_features=cell_features,
        sweep_features=sweep_features,
        qc_criteria=qc_criteria
    )
    # Outputs a QC Summary log to the terminal
    qc_summary(
        sweep_features=sweep_features, 
        sweep_states=sweep_states, 
        cell_features=cell_features, 
        cell_state=cell_state
    )

    return cell_state, cell_features, sweep_states, sweep_features 
