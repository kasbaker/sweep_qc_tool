from typing import Optional, Dict, List, Any, Set, Sequence

from PyQt5.QtCore import (
    QAbstractTableModel, QModelIndex, pyqtSignal
)
from PyQt5.QtGui import QColor
from PyQt5 import QtCore

from ipfx.ephys_data_set import EphysDataSet

from pre_fx_data import PreFxData
from sweep_plotter import SweepPlotter, SweepPlotConfig


class SweepTableModel(QAbstractTableModel):
    """ Abstract table model holding the raw data for the sweep page.

    Attributes
    ----------
    qc_state_updated : pyqtSignal
        Signal that is emitted with the user updates the manual qc state.
    new_data : pyqtSignal
        Signal that is emitted when the user loads a new data set.
    FAIL_BGCOLOR : QColor
        Color that is used to pain the auto qc column when a sweep auto-fails

    """
    qc_state_updated = pyqtSignal(int, str, name="qc_state_updated")
    sweep_types_ready = pyqtSignal(name="table_model_data_loaded")

    FAIL_BGCOLOR = QColor(255, 225, 225)

    def __init__(
            self,
            colnames: Sequence[str],
            plot_config: SweepPlotConfig
    ):
        """ Initializes and configures abstract table model

        Parameters
        ----------
        colnames : Sequence[str]
            list of column names for the sweep table model
        plot_config : SweepPlotConfig
            named tuple with constants used for plotting sweeps

        """
        super().__init__()

        self.colnames = colnames
        self.column_map = {colname: idx for idx, colname in enumerate(colnames)}
        self._data: List[List[Any]] = []

        # self.plot_config = plot_config
        # self.sweep_features: Optional[list] = None
        # self.sweep_states: Optional[list] = None
        # self.manual_qc_states: Optional[list] = None

        # dictionary translating between sweep numbers and table model row indexes
        self.sweep_num_to_idx_key: Optional[Dict[int, int]] = None
        # dictionary of sets of sweep numbers classified in various ways
        self.sweep_types: Optional[Dict[str, set]] = None

    def connect(self, data: PreFxData):
        """ Set up signals and slots for communication with the underlying data store.

        Parameters
        ----------
        data : 
            Will be used as the underlying data store. Will emit notifications when 
            data has been updated. Will receive notifications when users update
            QC states for individual sweeps.

        """
        data.table_model_data_ready.connect(self.build_sweep_table)
        self.qc_state_updated.connect(data.on_manual_qc_state_updated)

    def build_sweep_table(self, table_model_data: List[list], sweep_types: Dict[str, Set[int]]):
        """ foobar """

        if self.rowCount() > 0:
            # reset the model if it is not already empty
            self.beginResetModel()
            self._data = []
            self.endResetModel()

        self.beginInsertRows(QModelIndex(), 0, len(table_model_data) - 1)
        self._data = table_model_data
        self.endInsertRows()

        # key for translating sweep numbers to table model indexes
        sweep_num_to_idx_key = {
            sweep[0]: idx for idx, sweep in enumerate(table_model_data)
        }
        # add set of all sweeps contained in the sweep table to sweep types
        sweep_types['all_sweeps'] = set(sweep_num_to_idx_key.keys())

        # implement sweep types and key, then emit signal saying they are ready
        self.sweep_num_to_idx_key = sweep_num_to_idx_key
        self.sweep_types = sweep_types
        self.sweep_types_ready.emit()

    def rowCount(self, *args, **kwargs):
        """ The number of rows in the sweep table model, which should be the
        same as the number of sweeps currently loaded in the table model

        Returns
        -------
        num_rows : int
            number of rows in the sweep table model

        """
        return len(self._data)

    def columnCount(self, *args, **kwargs) -> int:
        """ The number of columns in the sweep table model. The last two
        columns contain thumbnails for popup plots and the rest contain
        sweep characteristics and qc states.

        Returns
        -------
        num_cols : int
            number of columns in the sweep table model

        """
        return len(self.colnames)

    def data(self,
             index: QModelIndex,
             role: int = QtCore.Qt.DisplayRole
             ):

        """ The data stored at a given index.

        Parameters
        ----------
        index : QModelIndex
            Which table cell to read.
        role : QtCore.Qt.ItemDataRole
            the role for the data being accessed

        Returns
        -------
        None if
            - the index is invalid (e.g. out of bounds)
            - the role is not supported
        otherwise whatever data is stored at the requested index.

        """

        if not index.isValid():
            return

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            return self._data[index.row()][index.column()]

        if role == QtCore.Qt.BackgroundRole and index.column() == 3:
            if self._data[index.row()][3] == "failed":
                return self.FAIL_BGCOLOR

    def headerData(
            self,
            section: int,
            orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
            role: QtCore.Qt.ItemDataRole = QtCore.Qt.DisplayRole
    ):
        """ Returns the name of the 'section'th column

        Parameters
        ----------
        section : int
            integer index of the column to return the name for
        orientation : QtCore.Qt.Orientation
            the orientation for the data being accessed
        role : QtCore.Qt.ItemDataRole
            the display role for the data being accessed

        Returns
        -------
        colname : str
            the name of the column that is being accessed

        """

        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.colnames[section]

    def flags(
            self,
            index: QModelIndex
    ) -> QtCore.Qt.ItemFlag:
        """ Returns integer flags for the item at a supplied index.

        Parameters
        ----------
        index : QModelIndex
            index used to locate data in the model

        Returns
        -------
        flags: QtCore.Qt.ItemFlag
            describes the properties of the item being accessed

        """

        flags = super(SweepTableModel, self).flags(index)

        if index.column() == self.colnames.index(" Manual QC State "):
            flags |= QtCore.Qt.ItemIsEditable

        return flags

    def setData(
            self,
            index: QModelIndex,
            value: str,
            role: QtCore.Qt.ItemDataRole = QtCore.Qt.EditRole
    ) -> bool:
        """ Updates the data at the supplied index.

        Parameters
        ----------
        index : QModelIndex
            index used to locate data in the model
        value : str
            if this value is an entry in the manual QC state column and it is
            different than the current one this updates it to the new value
        role : QtCore.Qt.ItemDataRole
            the display role for the data being accessed

        Returns
        -------
        state : bool
            returns True if data was successfully updated
            returns False if data was not updated

        """

        current = self._data[index.row()][index.column()]

        if index.isValid() \
                and isinstance(value, str) \
                and index.column() == self.column_map[" Manual QC State "] \
                and role == QtCore.Qt.EditRole \
                and value != current:
            self._data[index.row()][index.column()] = value
            self.qc_state_updated.emit(
                self._data[index.row()][self.column_map[" Sweep "]], value
            )
            return True

        return False


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
