import json
from pathlib import Path
import sys
from multiprocessing import Pool

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QApplication, QHeaderView
from ipfx.qc_feature_evaluator import DEFAULT_QC_CRITERIA_FILE
from ipfx.stimulus import StimulusOntology

from main import SweepPage
from sweep_plotter import SweepPlotConfig
from sweep_table_model import SweepTableModel
from sweep_table_view import SweepTableView
from optimization.sweep_plotter_lite import SweepPlotterLite
from data_extractor import DataExtractor


with open(DEFAULT_QC_CRITERIA_FILE, "r") as path:
    QC_CRITERIA = json.load(path)

with open(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE, "r") \
        as path:
    ONTOLOGY = StimulusOntology(json.load(path))


def initialize_sweep_table_and_model():
    config = SweepPlotConfig(
        test_pulse_plot_start=.04,
        test_pulse_plot_end=0.1,
        test_pulse_baseline_samples=100,
        backup_experiment_start_index=5000,
        experiment_baseline_start_index=5000,
        experiment_baseline_end_index=9000,
        thumbnail_step=20
    )

    model = SweepTableModel(
        SweepPage.colnames,
        config
    )

    view = SweepTableView(
        SweepPage.colnames
    )
    return model, view, config


def extract_data(nwb_path):
    data_extractor = DataExtractor(nwb_file=nwb_path, ontology=ONTOLOGY)
    return tuple(data_extractor.data_iter)


def populate_model_data(model, model_data):
    if model.rowCount() > 0:
        # reset the model if it is not already empty
        model.beginResetModel()
        model._data = []
        model.endResetModel()

    model.beginInsertRows(QModelIndex(), 0, len(model_data) - 1)
    model._data = model_data
    model.endInsertRows()


def make_plot_page(nwb_path):

    model, view, config = initialize_sweep_table_and_model()

    view.setWindowTitle(nwb_path)

    view.resize(2000, 2000)
    view.move(2500, 100)

    sweep_data_tuple = extract_data(nwb_path=nwb_path)

    plotter = SweepPlotterLite(sweep_data_tuple=sweep_data_tuple, config=config)
    sweep_plots = tuple(plotter.gen_plots())

    # model_data = [[
    #     index,
    #     row['stimulus_code'],
    #     row['stimulus_name'],
    #     'auto_qc_state',
    #     'manual_qc_state',
    #     'tags',  # fail tags
    #     sweep_plots[index][0],
    #     sweep_plots[index][1]
    # ] for index, row in enumerate(sweep_data_tuple)]

    model_data = [[
        swp_num,
        sweep_data_tuple[swp_num]['stimulus_code'],
        sweep_data_tuple[swp_num]['stimulus_name'],
        'auto_qc_state',
        'manual_qc_state',
        'tags',  # fail tags
        tp_plot,
        exp_plot
    ] for swp_num, tp_plot, exp_plot in sweep_plots
        if sweep_data_tuple[swp_num]['stimulus_name'] != "Search"]

    populate_model_data(model, model_data)

    view.setModel(model)
    view.setColumnHidden(3, True)
    view.setColumnHidden(4, True)
    view.setColumnHidden(5, True)

    view.resize_to_content()
    view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    view.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

    return view


def main(nwb_file):
    app = QApplication(sys.argv)

    view = make_plot_page(nwb_file)
    view.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    files = list(Path("data/nwb").glob("*.nwb"))
    base_dir = Path(__file__).parent

    file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Ctgf-T2A-dgCre;Ai14-495723.05.02.01.nwb"
    # file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Vip-IRES-Cre;Ai14-331294.04.01.01.nwb"

    # channel recording with no rs comp / cap comp for this cell? --- series is broken?
    # file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Esr2-IRES2-Cre;Ai14-494673.04.02.03.nwb"

    # file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Sncg-IRES2-FlpO-neo;Ai65F-499191.03.02.01.nwb"

    # file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Pvalb-IRES-Cre;Ai14(IVSCC)-165172.05.02.nwb"

    main(file)

    # with Pool(processes=len(files)) as pool:
    #     pool.map(main, tuple(map(str, files)))
