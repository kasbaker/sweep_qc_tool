from pathlib import Path
import json
from typing import Optional, Dict

from PyQt5.QtWidgets import (
    QWidget,
    QAction,
    QFileDialog,
    QDialog,
    QGridLayout,
    QTextEdit
)
from PyQt5.QtCore import pyqtSignal

from pre_fx_data import PreFxData


class PreFxController(QWidget):

    selected_stimulus_ontology_path = pyqtSignal(str, name="selected_stimulus_ontology_path")
    selected_qc_criteria_path = pyqtSignal(str, name="selected_qc_criteria_path")
    selected_data_set_path = pyqtSignal(str, name="selected_data_set_path")


    def __init__(self, *args, **kwargs):
        """PreFxController provides an interface between GUI elements, such as 
        menus and the application's underlying PreFxData. It does so mainly by 
        exposing a set of QActions, which open dialogs and emit signals 
        containing the user's selections.
        """

        super(PreFxController, self).__init__()

        self._stimulus_ontology: Optional[Dict] = None
        self._qc_criteria: Optional[Dict] = None
        self._has_data_set: bool = False

        self.init_actions()

    def init_actions(self):
        """PreFxController exposes several actions, suitable for a menu or 
        toolbar. These are:
            - load_stimulus_ontology_action
            - load_qc_criteria_action
            - load_data_set_action
            - show_qc_criteria_action
            - show_stimulus_ontology_action
        """

        self.load_stimulus_ontology_action = QAction("Load stimulus ontology from JSON", self)
        self.load_stimulus_ontology_action.triggered.connect(self.load_stimulus_ontology)

        self.load_qc_criteria_action = QAction("Load qc criteria from JSON", self)
        self.load_qc_criteria_action.triggered.connect(self.load_qc_criteria)

        self.load_data_set_action = QAction("Load data set from NWB file", self)
        self.load_data_set_action.triggered.connect(self.load_data_set)

        self.show_qc_criteria_action = QAction("Display QC criteria", self)
        self.show_qc_criteria_action.triggered.connect(self.show_qc_criteria)

        self.show_stimulus_ontology_action = QAction("Display stimulus ontology", self)
        self.show_stimulus_ontology_action.triggered.connect(self.show_stimulus_ontology)

        self.on_stimulus_ontology_unset()
        self.on_qc_criteria_unset()
        self.on_data_set_unset()

    def connect(self, pre_fx_data: PreFxData):
        """ Sets up communication between this controller and a PreFxData 
        instance. This object sends signals describing user inputs, while the 
        PreFxData sends signals reporting the status of its data and 
        operations.

        Parameters
        ----------
        pre_fx_data : 
            the object with which to communicate

        """

        # controller -> data
        self.selected_stimulus_ontology_path.connect(pre_fx_data.load_stimulus_ontology_from_json)
        self.selected_qc_criteria_path.connect(pre_fx_data.load_qc_criteria_from_json)
        self.selected_data_set_path.connect(pre_fx_data.load_data_set_from_nwb)

        # data -> controller
        pre_fx_data.stimulus_ontology_set.connect(self.on_stimulus_ontology_set)
        pre_fx_data.stimulus_ontology_unset.connect(self.on_stimulus_ontology_unset)

        pre_fx_data.qc_criteria_set.connect(self.on_qc_criteria_set)
        pre_fx_data.qc_criteria_unset.connect(self.on_qc_criteria_unset)

    def on_stimulus_ontology_set(self, ontology):
        """ Triggered when the PreFxData's stimulus_ontology becomes not None
        """

        self._stimulus_ontology = [stim.tag_sets for stim in ontology.stimuli]
        if self._qc_criteria is not None:
            self.load_data_set_action.setEnabled(True)

    def on_stimulus_ontology_unset(self):
        """ Triggered when the PreFxData's stimulus_ontology becomes None
        """
        self._stimulus_ontology = None
        self.load_data_set_action.setEnabled(False)

    def on_qc_criteria_set(self, criteria):
        """ Triggered when the PreFxData's qc_criteria becomes not None
        """
        self._qc_criteria = criteria
        if self._stimulus_ontology is not None:
            self.load_data_set_action.setEnabled(True)

    def on_qc_criteria_unset(self):
        """ Triggered when the PreFxData's qc criteria becomes None
        """
        self._qc_criteria = None
        self.load_data_set_action.setEnabled(False)

    def on_data_set_set(self):
        """ Triggered when the PreFxData's data set becomes not None
        """
        self._has_data_set = True

    def on_data_set_unset(self):
        """ Triggered when the PreFxData's data set becomes None
        """
        self._has_data_set = False

    def load_stimulus_ontology(self):
        """ Prompts the user to select a JSON file containing a serialized ipfx 
        stimulus ontology.
        """

        path = QFileDialog.getOpenFileName(
            self, 
            "load stimulus ontology file", 
            str(Path.cwd()), 
            "JSON files (*.json)"
        )[0]

        if path == "":
            return

        self.selected_stimulus_ontology_path.emit(path)

    def load_qc_criteria(self):
        """ Prompts the user to select a JSON file containing serialized 
        ipfx qc criteria settings.
        """

        path = QFileDialog.getOpenFileName(
            self, 
            "load qc criteria file", 
            str(Path.cwd()), 
            "JSON files (*.json)"
        )[0]

        if path == "":
            return

        self.selected_qc_criteria_path.emit(path)

    def load_data_set(self):
        """ Prompts the user to select an NWB file containing data for one 
        experiment.
        """
        path = QFileDialog.getOpenFileName(
            self, "load NWB file", str(Path.cwd()), "NWB files (*.nwb)"
        )[0]

        if path == "":
            return

        self.selected_data_set_path.emit(path)

    def show_qc_criteria(self):
        simple_ro_text_dialog(
            json.dumps(self._qc_criteria, indent=2),
            "QC criteria",
            700, 800,
            self
        )


    def show_stimulus_ontology(self):
        simple_ro_text_dialog(
            json.dumps(self._stimulus_ontology, indent=2),
            "Stimulus Ontology",
            700, 800,
            self
        )


def simple_ro_text_dialog(text, title, width=None, height=None, parent=None):
        dialog = QDialog(parent)
        layout = QGridLayout()

        dialog.setWindowTitle(title)
        if width or height:
            dialog.resize(width, height)  # TODO nicer to resize to content
        
        view = QTextEdit()
        view.setReadOnly(True)
        view.setText(text)

        layout.addWidget(view, 1, 1, 1, 1)
        dialog.setLayout(layout)
        dialog.exec()