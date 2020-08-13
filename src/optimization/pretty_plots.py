import json
from pathlib import Path
import sys
from typing import List
from multiprocessing import Pool

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QApplication, QHeaderView
from PyQt5.QtGui import QFont
from ipfx.qc_feature_evaluator import DEFAULT_QC_CRITERIA_FILE
from ipfx.stimulus import StimulusOntology

from main import SweepPage
from sweep_plotter import SweepPlotConfig, SweepPlotter
from sweep_table_model import SweepTableModel
from sweep_table_view import SweepTableView
# from optimization.sweep_plotter_lite import SweepPlotterLite
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

    plotter = SweepPlotter(sweep_data_tuple=sweep_data_tuple, config=config)
    sweep_plots = tuple(plotter.gen_plots())

    model_data = [[
        swp_num,
        sweep_data_tuple[swp_num]['stimulus_code'],
        sweep_data_tuple[swp_num]['stimulus_name'],
        'auto_qc_state',
        'manual_qc_state',
        'fail tag',
        format_amp_setting_strings(
            sweep_data_tuple[swp_num]['stimulus_unit'],
            sweep_data_tuple[swp_num]['amp_settings'],
        ),  # mcc settings from amplifier
        tp_plot,
        exp_plot
    ] for swp_num, tp_plot, exp_plot in sweep_plots]

    populate_model_data(model, model_data)

    view.setModel(model)
    # view.setColumnHidden(3, True)
    # view.setColumnHidden(4, True)
    # view.setColumnHidden(5, True)

    view.resize_to_content()
    view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    view.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

    return view


def format_amp_setting_strings(stimulus_unit, amp_settings, line_length=30):
    str_list = []
    # justify_key_value_strings("Sweep", str(sweep['sweep_number']), line_length, indent=0)

    # print this out for voltage clamp stuff
    if stimulus_unit == "Volts":
        # holding voltage print out
        holding_v_key = "V-Clamp Holding"
        if amp_settings['V-Clamp Holding Enable'] == "On":
            holding_v = amp_settings['V-Clamp Holding Level']

            str_list.append(justify_key_value_strings(holding_v_key, holding_v, line_length))
        else:
            str_list.append(justify_key_value_strings(holding_v_key, "Off", line_length))

        # rs comp print out
        rs_comp_key = "RsComp"
        if amp_settings['RsComp Enable'] == "On":
            corr = amp_settings['RsComp Correction']
            pred = amp_settings['RsComp Prediction']
            str_list.append(justify_key_value_strings(rs_comp_key, f"{corr}, {pred}", line_length))
        else:
            str_list.append(justify_key_value_strings(rs_comp_key, "Off", line_length))

        # whole cell comp print out
        whole_cell_key = "WcComp"
        if amp_settings['Whole Cell Comp Enable'] == "On":
            wc_cap = amp_settings['Whole Cell Comp Cap']
            wc_res = amp_settings['Whole Cell Comp Resist']
            str_list.append(justify_key_value_strings(
                whole_cell_key, f"{wc_cap}, {wc_res}", line_length)
            )
        else:
            str_list.append(
                justify_key_value_strings(whole_cell_key, "Off", line_length)
            )

    # print out for current clamps sweeps
    elif stimulus_unit == "Amps":
        # holding current print out
        holding_i_key = "I-Clamp Holding"
        if amp_settings['I-Clamp Holding Enable'] == "On":
            holding_i = amp_settings['I-Clamp Holding Level']
            str_list.append(
                justify_key_value_strings(holding_i_key, holding_i, line_length)
            )
        else:
            str_list.append(justify_key_value_strings(holding_i_key, "Off", line_length))

        # cap neutralization print out
        cap_neut_key = "Cap Neutralization"
        if amp_settings['Neut Cap Enabled'] == "On":
            cap = amp_settings['Neut Cap Value']
            str_list.append(
                justify_key_value_strings(cap_neut_key, cap, line_length)
            )
        else:
            str_list.append(justify_key_value_strings(cap_neut_key, "Off", line_length))

        # bridge balance print out
        bb_key = "Bridge Balance"
        if amp_settings['Bridge Bal Enable'] == "On":
            bb = amp_settings['Bridge Bal Value']
            str_list.append(justify_key_value_strings(bb_key, bb, line_length))
        else:
            str_list.append(justify_key_value_strings(bb_key, "Off", line_length))

    # pipette offset
    offset = amp_settings['Pipette Offset']
    justify_key_value_strings("Pipette Offset", offset, line_length)

    return str_list_to_rows(str_list)

def justify_key_value_strings(key: str, value: str, line_length: int, indent: int = 0):
    justify_len = line_length - len(key) - indent
    return f"{' '*indent}{key}: {value.rjust(justify_len)}"


def str_list_to_rows(str_list: List[str], num_newlines: int = 2) -> str:
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
    join_char = "\n"*num_newlines

    output_str = join_char.join(str_list)
    print(output_str)
    return join_char.join(str_list)

def main(nwb_file):
    app = QApplication(sys.argv)

    view = make_plot_page(nwb_file)
    table_font = QFont("Monospace")
    view.setFont(table_font)
    view.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    files = list(Path("data/nwb").glob("*.nwb"))
    base_dir = Path(__file__).parent

    # file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Ctgf-T2A-dgCre;Ai14-495723.05.02.01.nwb"
    # file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Vip-IRES-Cre;Ai14-331294.04.01.01.nwb"

    # channel recording with no rs comp / cap comp for this cell? --- series is broken?
    # file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Esr2-IRES2-Cre;Ai14-494673.04.02.03.nwb"

    file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Sncg-IRES2-FlpO-neo;Ai65F-499191.03.02.01.nwb"

    # file = "/home/katie/GitHub/sweep_qc_tool/src/optimization/data/nwb/Pvalb-IRES-Cre;Ai14(IVSCC)-165172.05.02.nwb"

    main(file)

    # with Pool(processes=len(files)) as pool:
    #     pool.map(main, tuple(map(str, files)))
