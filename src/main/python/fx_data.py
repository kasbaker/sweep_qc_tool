from PyQt5.QtCore import QObject, pyqtSignal
from ipfx.sweep_props import drop_failed_sweeps
from ipfx.data_set_utils import create_data_set
from ipfx.error import FeatureError
from ipfx.data_set_features import extract_data_set_features
from error_handling import exception_message


class FxData(QObject):

    state_outdated = pyqtSignal(name="state_outdated")
    new_state_set = pyqtSignal(dict, name="new_state_set")

    status_message = pyqtSignal(str, name="status_message")

    def __init__(self):
        super().__init__()
        self._state_out_of_date: bool = False


        self.input_nwb_file = None
        self.ontology = None
        self.sweep_info = None
        self.cell_info = None

    def out_of_date(self):
        self.state_outdated.emit()
        self._state_out_of_date = True

    def new_state(self):
        self.new_state_set.emit(self.feature_data)
        self._state_out_of_date = False

    def set_fx_parameters(self, nwb_path, ontology, sweep_info, cell_info):
        self.out_of_date()
        self.input_nwb_file = nwb_path
        self.ontology = ontology
        self.sweep_info = sweep_info
        self.cell_info = cell_info

    def connect(self, pre_fx_data):
        pre_fx_data.data_changed.connect(self.set_fx_parameters)

    # this does the same thing as the
    def run_auto_qc(self, dataset):
        ...
        # """ Creates a data set from the nwb path;
        #     calculates cell features, tags, and sweep features using ipfx;
        #     and runs auto qc on the experiment. If commit=True (default setting),
        #     it creates a dictionary of default manual qc states and calls
        #     SweepTableModel.on_new_data(), which builds the sweep table and
        #     generates all the thumbnail plots.
        #
        #     Parameters
        #     ----------
        #     commit : bool
        #         indicates whether or not to build new sweep table model
        #     """
        #
        # self.cell_features, self.cell_tags, self.sweep_features = \
        #     extract_qc_features(self.data_set)
        #
        # sweep_props.drop_tagged_sweeps(self.sweep_features)
        # # cell_state: list of dictionaries containing sweep pass/fail states
        # self.cell_state, self.cell_features, \
        # self.sweep_states, self.sweep_features = \
        #     run_qc(self.stimulus_ontology, self.cellfeatures,
        #            self.sweep_features, self.qc_criteria)
        #
        # if commit:
        #     self.begin_commit_calculated.emit()
        #
        #     # creates dictionary of manual qc states from sweep features
        #     self.manual_qc_states = {
        #         sweep["sweep_number"]: "default"
        #         for sweep in self.sweep_features
        #     }
        #
        #     # Calls SweepTableModel.on_new_data(), which builds the sweep table
        #     # and PreFxController.on_data_set_set()
        #     self.end_commit_calculated.emit(
        #         self.sweep_features, self.sweep_states,
        #         self.manual_qc_states, self.data_set
        #     )
        #
        # # calls FxData.set_fx_parameters()
        # self.data_changed.emit(
        #     self.nwb_path, self.stimulus_ontology,
        #     self.sweep_features, self.cell_features
        # )

    def run_feature_extraction(self, dataset):
        self.status_message.emit("Computing features, please wait.")
        drop_failed_sweeps(self.sweep_info)
        # Creates a data set only containing sweeps that have passed qc
        # sweep_info is an extended sweep table contianing only good sweeps
        # api_sweeps=False (normally True in PreFxData), which tells
        # AibsDataSet to keep the modified sweep table, here named sweep_info
        data_set = create_data_set(sweep_info=self.sweep_info,
                                   nwb_file=self.input_nwb_file,
                                   ontology=self.ontology,
                                   api_sweeps=False)
        try:
            cell_features, sweep_features, cell_record, sweep_records = \
                extract_data_set_features(data_set)

            cell_state = {"failed_fx": False, "fail_fx_message": None}

            self.feature_data = {'cell_features': cell_features,
                                 'sweep_features': sweep_features,
                                 'cell_record': cell_record,
                                 'sweep_records': sweep_records,
                                 'cell_state': cell_state
                                }

            self.new_state()
            self.status_message.emit("Done computing features!")

        except (FeatureError, IndexError) as ferr:
            exception_message("Feature extraction error",
                              f"failed feature extraction",
                              ferr
                              )



