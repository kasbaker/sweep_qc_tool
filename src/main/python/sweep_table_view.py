from typing import Optional

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QTableView, QWidget, QAction, QHeaderView
from PyQt5.QtCore import QModelIndex

from delegates import SvgDelegate, ComboBoxDelegate
from sweep_table_model import SweepTableModel


class SweepTableView(QTableView):

    @property
    def colnames(self):
        return self._colnames

    @colnames.setter
    def colnames(self, names):
        self._colnames = names
        self._idx_colname_map = {}
        self._colname_idx_map = {}
        
        for idx, name in enumerate(self._colnames):
            self._idx_colname_map[idx] = name
            self._colname_idx_map[name] = idx

    def __init__(self, colnames: tuple):
        super().__init__()
        # self.setParent()
        # initialize parent object to use for relative popup positions
        self.colnames = colnames
        self.setFont(QFont("Monospace"))
        # A list to store active popup plots
        self.popup_plots = []

        self.svg_delegate = SvgDelegate()
        manual_qc_choices = ["default", "failed", "passed"]
        self.cb_delegate = ComboBoxDelegate(self, manual_qc_choices)

        self.setItemDelegateForColumn(self.colnames.index(" Test Pulse Epoch "), self.svg_delegate)
        self.setItemDelegateForColumn(self.colnames.index(" Experiment Epoch "), self.svg_delegate)
        self.setItemDelegateForColumn(self.colnames.index(" Manual QC State "), self.cb_delegate)

        self.verticalHeader().setMinimumSectionSize(120)

        # self.horizontalHeader().setSectionResizeMode()

        self.clicked.connect(self.on_clicked)

        self.setWordWrap(True)

        # sweep view filter actions
        self.view_all_sweeps = QAction("All sweeps")
        # current / voltage clamp
        self.view_v_clamp = QAction("Voltage clamp")
        self.view_i_clamp = QAction("Current clamp")
        # stimulus codes
        self.view_pipeline = QAction("QC Pipeline")
        self.view_ex_tp = QAction("EXTP - Test sweeps")
        self.view_nuc_vc = QAction("NucVC - Channel recordings")
        self.view_core_one = QAction("Core 1")
        self.view_core_two = QAction("Core 2")
        # qc status
        self.view_auto_pass = QAction("Auto passed")
        self.view_auto_fail = QAction("Auto failed")
        self.view_no_auto_qc = QAction("No auto QC")
        # initialize these actions
        self.init_actions()

    def init_actions(self):
        """ Initializes menu actions which are responsible for filtering sweeps
        """
        # initialize view all sweeps action
        self.view_all_sweeps.setCheckable(True)
        self.view_all_sweeps.triggered.connect(self.filter_sweeps)
        self.view_all_sweeps.setEnabled(False)

        # initialize filter down to auto qc action
        self.view_pipeline.setCheckable(True)
        self.view_pipeline.triggered.connect(self.filter_sweeps)
        self.view_pipeline.setEnabled(False)

        # initialize filter down to channel sweeps action
        self.view_nuc_vc.setCheckable(True)
        self.view_nuc_vc.triggered.connect(self.filter_sweeps)
        self.view_nuc_vc.setEnabled(False)

    def get_column_index(self, name: str) -> Optional[int]:
        return self._colname_idx_map.get(name, None)
    
    def get_index_column(self, index: int) -> Optional[str]:
        return self._idx_colname_map.get(index, None)

    def setModel(self, model: SweepTableModel):
        """ Attach a SweepTableModel to this view. The model will provide data for 
        this view to display.
        """
        super(SweepTableView, self).setModel(model)
        model.rowsInserted.connect(self.persist_qc_editor)
        model.rowsInserted.connect(self.resize_to_content)

    def resize_to_content(self, *args, **kwargs):
        """ This function just exists so that we can connect signals with 
        extraneous data to resizeRowsToContents
        """

        self.resizeRowsToContents()

        header = self.horizontalHeader()

        # TODO make it so long fail tags don't take up too much column width
        for column in range(header.count()-2):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            width = header.sectionSize(column)
            # width = int(width*1.2 // 1)  # scale width up by 20% for prettiness
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            header.resizeSection(column, width)

        # set the last two thumbnail columns so they stretch
        tp_col = header.count()-2   # column of test pulse thumbnail
        header.setSectionResizeMode(tp_col, QHeaderView.Stretch)
        width = header.sectionSize(tp_col)
        header.setSectionResizeMode(tp_col, QHeaderView.Interactive)
        header.resizeSection(tp_col, width)

        # experiment epoch column
        exp_col = header.count()-1  # column of experiment plot thumbnail
        header.setSectionResizeMode(exp_col, QHeaderView.Stretch)
        width = header.sectionSize(exp_col)
        header.setSectionResizeMode(exp_col, QHeaderView.Interactive)
        header.resizeSection(exp_col, width)
        # set last section so it that it stretches takes up the whole page
        header.setStretchLastSection(True)

    def resizeEvent(self, *args, **kwargs):
        """ Makes sure that we resize the rows to their contents when the user
        resizes the window
        """

        super(SweepTableView, self).resizeEvent(*args, **kwargs)
        self.resize_to_content()

    def persist_qc_editor(self, *args, **kwargs):
        """ Ensure that the QC state editor can be opened with a single click.

        Parameters
        ----------
        All are ignored. They are present because this method is triggered
        by a data-carrying signal.

        """

        column = self.colnames.index(" Manual QC State ")

        for row in range(self.model().rowCount()):
            self.openPersistentEditor(self.model().index(row, column))

    def on_clicked(self, index: QModelIndex):
        """ When plot thumbnails are clicked, open a larger plot in a popup.

        Parameters
        ----------
        index : 
            Which plot to open. The popup will be mopved to this item's location.

        """

        test_column = self.get_column_index(" Test Pulse Epoch ")
        exp_column = self.get_column_index(" Experiment Epoch ")

        if not index.column() in {test_column, exp_column}:
            return

        # display popup plot at (100, 100) for user convenience
        self.popup_plot(self.model().data(index).full(), left=100, top=100)

    def popup_plot(self, graph: QWidget, left: int = 0, top: int = 0):
        """ Make a popup with a single widget, which ought to be a plotter for 
        the full experiment or test pulse plots.

        Parameters
        ----------
        graph : a widget to be displayed in the popup
        left : left position at which the popup will be placed (px)
        top : top position at which the popup will be placed (px)

        """
        # add the graph to popup_plots
        self.popup_plots.append(graph)
        # remove old popup plots if there are more than 5
        while len(self.popup_plots) > 5:
            self.popup_plots.pop(0)

        # the great-great-grandparent of this widget should be the main window
        main_window = self.parent().parent().parent().parent()

        # move popup plot to nice location relative to main window
        graph.window().move(
            left + main_window.pos().x(),
            top + main_window.pos().y()
        )
        graph.show()

    def filter_sweeps(self):
        """ Filters the table down to sweeps based on the checkboxes that are
        check in the view menu. If 'Auto QC sweeps' is checked then it will
        only show sweeps that have gone through the auto QC pipeline. If
        'Channel recording sweeps' is checked then it will only show channel
        recording sweeps with the 'NucVC' prefix. If both are checked then
        it will only show auto QC pipeline sweeps and channe
        l recording sweeps.
        If neither are checked it will show everything except 'Search' sweeps.

        """
        # temporary variable of visible sweeps
        sweep_nums_to_show = set()
        # cache sweep types
        sweep_types = self.model().sweep_types
        # add checked view options to set of visible sweeps
        # all sweeps
        if self.view_all_sweeps.isChecked():
            sweep_nums_to_show.update(sweep_types['all_sweeps'])
        # pipeline sweeps
        if self.view_pipeline.isChecked():
            sweep_nums_to_show.update(sweep_types['pipeline'])
        # channel recording sweeps
        if self.view_nuc_vc.isChecked():
            sweep_nums_to_show.update(sweep_types['nuc_vc'])

        # set view pipeline to checked if it is a subset of visible sweeps
        self.view_pipeline.setChecked(
            sweep_types['pipeline'].issubset(sweep_nums_to_show)
        )
        # set view nuc vc to checked if it is a subset of visible sweeps
        self.view_nuc_vc.setChecked(
            sweep_types['nuc_vc'].issubset(sweep_nums_to_show)
        )

        # remove any skipped sweeps that are not present in all sweeps
        sweep_nums_to_show.intersection_update(sweep_types['all_sweeps'])

        # cache key to translate sweep numbers to table model indexes
        swp_idx_key = self.model().sweep_num_to_idx_key
        # translate sweep numbers to show into table model indexes to show
        sweep_idx_to_show = {
            swp_idx_key[sweep_num] for sweep_num in sweep_nums_to_show
        }

        # loop through rows of table model and show only visible sweep indexes
        for index in range(self.model().rowCount()):
            if index in sweep_idx_to_show:
                self.showRow(index)
            else:
                self.hideRow(index)
