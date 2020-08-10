import json
from typing import Optional, List, Dict, Any
from multiprocessing import Pipe, Process

from PyQt5.QtCore import QObject, pyqtSignal
from error_handling import exception_message
from marshmallow import ValidationError

from ipfx import __version__ as ipfx_version
from ipfx.stimulus import StimulusOntology
from ipfx.ephys_data_set import EphysDataSet
from ipfx.qc_feature_evaluator import DEFAULT_QC_CRITERIA_FILE

from schemas import PipelineParameters
from sweep_plotter import SweepPlotConfig
from qc_operator import run_auto_qc, QCResults
from data_extractor import DataExtractor
from sweep_plotter import SweepPlotter


class PreFxData(QObject):

    stimulus_ontology_set = pyqtSignal(StimulusOntology, name="stimulus_ontology_set")
    stimulus_ontology_unset = pyqtSignal(name="stimulus_ontology_unset")

    qc_criteria_set = pyqtSignal(dict, name="qc_criteria_set")
    qc_criteria_unset = pyqtSignal(name="qc_criteria_unset")

    begin_commit_calculated = pyqtSignal(name="begin_commit_calculated")
    end_commit_calculated = pyqtSignal(list, list, dict, EphysDataSet, name="end_commit_calculated")

    # signal to send data to the sweep table once auto qc and plotting is done
    model_data_ready = pyqtSignal(list, name="data_ready")

    update_fx_sweep_info = pyqtSignal(
        str, StimulusOntology, list, name="update_fx_sweep_info"
    )

    status_message = pyqtSignal(str, name="status_message")

    def __init__(self, plot_config: SweepPlotConfig):
        """ Main data store for all data upstream of feature extraction. This
        includes:
            - the EphysDataSet
            - the StimulusOntology
            - the qc criteria
            - the sweep extraction results
            - the qc results
        """
        super(PreFxData, self).__init__()

        self.plot_config = plot_config

        # Nwb related data
        self.data_set: Optional[EphysDataSet] = None
        self.nwb_path: Optional[str] = None
        # Ontology path and data
        self.ontology_file: Optional[str] = None
        self._stimulus_ontology: Optional[StimulusOntology] = None

        # QC related data
        # criteria used with auto QC
        self._qc_criteria: Optional[Dict] = None
        # full list of sweep qc info used for qc-ing sweeps and feature extraction
        self._full_sweep_qc_info: Optional[list] = None
        # dictionary of sweep types, which can be used for filtering sweeps
        self._sweep_types: Optional[dict] = None
        # named tuple containing qc results obtained from auto-qc
        self._qc_results: Optional[QCResults] = None

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

    @property
    def qc_criteria(self) -> Optional[Dict]:
        return self._qc_criteria

    @qc_criteria.setter
    def qc_criteria(self, value: Optional[Dict]):
        self._notifying_setter(
            "_qc_criteria", 
            value,
            self.qc_criteria_set, 
            self.qc_criteria_unset,
            send_value=True
        )

    def set_default_stimulus_ontology(self):
        self.load_stimulus_ontology_from_json(
            StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE
        )

    def set_default_qc_criteria(self):
        self.load_qc_criteria_from_json(DEFAULT_QC_CRITERIA_FILE)

    def load_stimulus_ontology_from_json(self, path: str):
        """ Attempts to read a stimulus ontology file from a JSON. If 
        successful (and other required data are already set), attempts to 
        run the pre-fx pipeline

        Parameters
        ----------
        path : 
            load ontology from here

        """

        try:
            with open(path, "r") as ontology_file:
                ontology_data = json.load(ontology_file)
            ontology = StimulusOntology(ontology_data)
            self.ontology_file = path

            if self.nwb_path is not None and self.qc_criteria is not None:
                self.run_extraction_and_auto_qc(
                    self.nwb_path, 
                    ontology, 
                    self.qc_criteria, 
                    commit=True
                )
            else:
                self.stimulus_ontology = ontology

        except Exception as err:
            exception_message(
                "StimulusOntology load failed",
                f"failed to load stimulus ontology file from {path}",
                err
            )

    def load_qc_criteria_from_json(self, path: str):
        """ Attempts to read qc criteria from a JSON. If successful (and other 
        required data are already set), attempts to run the pre-fx pipeline

        Parameters
        ----------
        path : 
            load criteria from here

        """

        try:
            with open(path, "r") as criteria_file:
                criteria = json.load(criteria_file)
            
            if self.nwb_path is not None and self.stimulus_ontology is not None:
                self.run_extraction_and_auto_qc(
                    self.nwb_path, 
                    self.stimulus_ontology, 
                    criteria, 
                    commit=True
                )
            else:
                self.qc_criteria = criteria

        except Exception as err:
            exception_message(
                "QC criteria load failure",
                f"failed to load qc criteria file from {path}",
                err
            )

    def load_data_set_from_nwb(self, path: str):
        """ Attempts to read an NWB file describing an experiment. Fails if 
        qc criteria or stimulus ontology not already present. Otherwise, 
        attempts to run the pre-fx pipeline.

        Parameters
        ----------
        path : 
            load data set from here

        """
        try:
            if self.stimulus_ontology is None:
                raise ValueError("must set stimulus ontology before loading a data set!")
            elif self.qc_criteria is None:
                raise ValueError("must set qc criteria before loading a data set!")

            self.status_message.emit("Running extraction and auto qc...")
            # self.run_extraction_and_auto_qc(
            #     path, self.stimulus_ontology, self.qc_criteria, commit=True
            # )
            self.run_auto_qc_and_make_plots(
                path, self.stimulus_ontology, self.qc_criteria
            )
            self.status_message.emit("Done running extraction and auto qc")
        except Exception as err:
            exception_message(
                "Unable to load NWB",
                f"failed to load NWB file from {path}",
                err
            )

    def extract_manual_sweep_states(self):
        """ Extract manual sweep states in the format schemas.ManualSweepStates
        from PreFxData
        """

        return [
            {
                "sweep_number": sweep['sweep_number'],
                "sweep_state": sweep['manual_qc_state']
            }
            for sweep in self._full_sweep_qc_info
        ]

    def save_manual_states_to_json(self, filepath: str):

        json_data = {
            "input_nwb_file": self.nwb_path,
            "stimulus_ontology_file": self.ontology_file,
            "manual_sweep_states": self.extract_manual_sweep_states(),
            "qc_criteria": self._qc_criteria,
            "ipfx_version": ipfx_version
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

    def run_auto_qc_and_make_plots(
            self, nwb_path: str,
            stimulus_ontology: StimulusOntology, qc_criteria: list
    ):
        """" Extract ephys data and obtain sweep data iterator. Generate a list
        from this sweep data iterator, pickle it, then send it through a pipe
        to a QC_worker process. Simultaneously generate thumbnail-popup plot
        pairs. Close the pipe, receive the qc results, join, and terminate the
        QC worker process. Send the new sweep table data to the sweep table
        model.

        Parameters
        ----------
        nwb_path : str
            path of the .nwb file to extract the data from
        stimulus_ontology : StimulusOntology
            stimulus ontology object used in data extraction
        qc_criteria : list
            list containing the criteria for auto-qc

        """
        self.status_message.emit("Extracting EPhys Data...")
        data_extractor = DataExtractor(nwb_file=nwb_path, ontology=stimulus_ontology)
        sweep_data_tuple = tuple(data_extractor.data_iter)
        recording_date = data_extractor.recording_date

        qc_pipe = Pipe(duplex=False)
        qc_worker = Process(
            name="qc_worker", target=run_auto_qc, args=(
                sweep_data_tuple, stimulus_ontology, qc_criteria, recording_date, qc_pipe[1]
            )
        )
        qc_worker.daemon = True
        self.status_message.emit("Starting auto-QC...")
        qc_worker.start()

        # initialize sweep table and generate plots for sweep table
        plotter = SweepPlotter(sweep_data_tuple=sweep_data_tuple, config=self.plot_config)

        self.status_message.emit("Generating plots...")
        sweep_plots = tuple(plotter.gen_plots())

        # close the qc pipe output and receive qc operator's output
        self.status_message.emit("Waiting on QC results...")
        qc_pipe[1].close()
        qc_results, full_sweep_qc_info, sweep_types = qc_pipe[0].recv()
        # join and terminate qc worker
        qc_worker.join()
        qc_worker.terminate()

        # create list of data to send to sweep table model, exclude 'Search' sweeps
        self.status_message.emit("Preparing data for sweep page...")

        new_data = [[
            sweep_num,
            full_sweep_qc_info[sweep_num]['stimulus_code'],
            full_sweep_qc_info[sweep_num]['stimulus_name'],
            full_sweep_qc_info[sweep_num]['auto_qc_state'],
            full_sweep_qc_info[sweep_num]['manual_qc_state'],
            format_fail_tags(full_sweep_qc_info[sweep_num]['tags']),  # fail tags
            tp_plot,    # test pulse plot
            exp_plot    # experiment plot
        ] for sweep_num, tp_plot, exp_plot in sweep_plots
            if full_sweep_qc_info[sweep_num]['stimulus_name'] != "Search"]

        self.status_message.emit("Finalizing results")

        # finalize nwb path, stimulus ontology, and qc criteria
        self.nwb_path = nwb_path
        self.stimulus_ontology = stimulus_ontology
        self.qc_criteria = qc_criteria
        self.data_set = data_extractor.data_set

        # update self with qc results, full sweep info, and sweep types
        self._full_sweep_qc_info = full_sweep_qc_info
        self._qc_results = qc_results
        self._sweep_types = sweep_types

        # update feature extractor with new info
        self.update_fx_sweep_info.emit(
            self.nwb_path, self.stimulus_ontology, self._full_sweep_qc_info
        )
        # send new sweep table data to
        self.model_data_ready.emit(new_data)

    def on_manual_qc_state_updated(self, index: int, new_state: str):
        """ Takes in new manual QC state and updates sweep_states and
        sweep features appropriately. Note that sweep features that do not get
        passed the first round of auto qc are left as none in order to avoid
        breaking feature extraction.

        Parameters:
            index : int
                Sweep number that is being updated. Used as an index when
                    addressing sweep_States and sweep_features
            new_state : str
                String specifying manual QC state "default", "passed", or "failed"
        """
        # assign new manual qc state
        self._full_sweep_qc_info[index]['manual_qc_state'] = new_state

        # cache this row of the full sweep qc info list
        sweep = self._full_sweep_qc_info[index]
        # only change sweep['passed'] if this sweep is auto-qc-able
        if sweep['auto_qc_state'] == "passed":
            self._full_sweep_qc_info[index]['passed'] = True
        elif sweep['auto_qc_state'] == "failed":
            # sweeps that fail auto qc will break feature extraction if we set
            # 'passed' to True, so leave this false for now
            self._full_sweep_qc_info[index]['passed'] = False
        else:
            self._full_sweep_qc_info[index]['passed'] = None

        # revert to original auto qc value if manual value set back to 'default'
        if new_state == "default":
            if sweep['auto_qc_state'] == "passed":
                self._full_sweep_qc_info[index]['passed'] = True
            elif sweep['auto_qc_state'] == "failed":
                self._full_sweep_qc_info[index]['passed'] = False
            else:
                # 'auto_qc_state' should be "n/a" here, so set to 'passed' to None
                self._full_sweep_qc_info[index]['passed'] = None

        # send updated sweep qc info to fx_data for feature extraction
        self.update_fx_sweep_info.emit(
            self.nwb_path, self.stimulus_ontology, self._full_sweep_qc_info
        )


def format_fail_tags(tags: List[str]) -> str:
    """ Joins lists of strings containing information about the qc state
    for each sweep and joins them together in a nice readable format.

    Parameters
    ----------
    tags: List[str]
        a list of strings containing tags related to qc states

    Returns
    -------
    formatted_tags : str
        a single string containing the tags passed into this function

    """
    return "\n\n".join(tags)
