from pathlib import Path
from timeit import default_timer
from warnings import filterwarnings
import datetime as dt
from multiprocessing import Pool, cpu_count
import cProfile
import pstats
import logging

import numpy as np
from pynwb.icephys import (
    CurrentClampSeries, CurrentClampStimulusSeries,
    VoltageClampSeries, VoltageClampStimulusSeries,
    IZeroClampSeries, PatchClampSeries
)

from ipfx.qc_feature_extractor import cell_qc_features, sweep_qc_features
from ipfx.qc_feature_evaluator import qc_experiment, DEFAULT_QC_CRITERIA_FILE
from ipfx.dataset.create import create_ephys_data_set
import ipfx.epochs as ep
import ipfx.qc_features as qcf
from ipfx import error as er
from ipfx.qc_feature_extractor \
    import extract_recording_date, compute_input_access_resistance_ratio


def fast_extract_blowout(blowout_sweeps, tags):
    if blowout_sweeps:
        blowout_mv = qcf.measure_blowout(
            blowout_sweeps[-1]['response'], blowout_sweeps[-1]['test_epoch'][1]
        )
    else:
        tags.append("Blowout is not available")
        blowout_mv = None
    return blowout_mv


def fast_extract_electrode_0(bath_sweeps, tags):
    if bath_sweeps:
        e0 = qcf.measure_electrode_0(
            bath_sweeps[-1]['response'], bath_sweeps[-1]['sampling_rate']
        )
    else:
        tags.append("Electrode 0 is not available")
        e0 = None
    return e0


def fast_extract_clamp_seal(seal_sweeps, tags):
    if seal_sweeps:
        num_pts = len(seal_sweeps[-1]['stimulus'])
        time = np.linspace(0, num_pts/seal_sweeps[-1]['sampling_rate'], num_pts)
        seal_gohm = qcf.measure_seal(
            seal_sweeps[-1]['stimulus'], seal_sweeps[-1]['response'], time
        )
        if seal_gohm is None or not np.isfinite(seal_gohm):
            raise er.FeatureError("Could not compute seal")
    else:
        tags.append("Seal is not available")
        seal_gohm = None

    return seal_gohm


def fast_extract_input_and_acess_resistance(breakin_sweeps, tags):
    if breakin_sweeps:
        num_pts = len(breakin_sweeps[-1]['stimulus'])
        time = np.linspace(0, num_pts/breakin_sweeps[-1]['sampling_rate'], num_pts)
        try:
            input_resistance = qcf.measure_input_resistance(
                breakin_sweeps[-1]['stimulus'], breakin_sweeps[-1]['response'], time
            )
        except Exception as e:
            logging.warning("Error reading input resistance.")
            raise

        try:
            access_resistance = qcf.measure_initial_access_resistance(
                breakin_sweeps[-1]['stimulus'], breakin_sweeps[-1]['response'], time
            )
        except Exception as e:
            logging.warning("Error reading initial access resistance.")
            raise

    else:
        tags.append("Breakin sweep not found")
        input_resistance = None
        access_resistance = None

    return input_resistance, access_resistance


def fast_experiment_qc(nwb_file):
    data_set = create_ephys_data_set(nwb_file=nwb_file)

    sweep_table = data_set._data.nwb.sweep_table

    # number of sweeps is half the shape of the sweep table here because
    # each sweep has two series associated with it (stimulus and response)
    num_sweeps = sweep_table.series.shape[0]//2

    # reverse the sweep list so when we grab the last sweep when we iterate
    # through to find the appropriate sweep
    sweep_range = range(num_sweeps)
    series_iter = map(sweep_table.get_series, sweep_range)

    # iterator with all the necessary sweep data
    data_iter = map(extract_series_data, series_iter)

    blowout_sweeps, bath_sweeps, seal_sweeps, breakin_sweeps = get_sweep_types(
        data_iter, data_set.ontology
    )

    features = {}
    tags = []

    features['blowout_mv'] = fast_extract_blowout(blowout_sweeps, tags)

    features['electrode_0_pa'] = fast_extract_electrode_0(bath_sweeps, tags)

    features['recording_date'] = extract_recording_date(data_set, tags)

    features["seal_gohm"] = fast_extract_clamp_seal(seal_sweeps, tags)

    input_resistance, access_resistance = \
        fast_extract_input_and_acess_resistance(breakin_sweeps, tags)

    features['input_resistance_mohm'] = input_resistance
    features["initial_access_resistance_mohm"] = access_resistance

    features['input_access_resistance_ratio'] = \
        compute_input_access_resistance_ratio(input_resistance, access_resistance)

    return features, tags


def get_sweep_types(data_iter, ontology):

    blowout_sweeps = []
    bath_sweeps = []
    seal_sweeps = []
    breakin_sweeps = []

    for sweep in data_iter:
        stim_code = sweep['stimulus_code']
        if ontology.blowout_names[0] in stim_code:
            blowout_sweeps.append(sweep)
        if ontology.bath_names[0] in stim_code:
            bath_sweeps.append(sweep)
        if ontology.seal_names[0] in stim_code:
            seal_sweeps.append(sweep)
        if ontology.breakin_names[0] in stim_code:
            breakin_sweeps.append(sweep)

    return blowout_sweeps, bath_sweeps, seal_sweeps, breakin_sweeps


def extract_series_data(series):

    sweep_number = series[0].sweep_number

    stimulus_code = ""

    response = None
    stimulus = None
    stimulus_unit = None
    sampling_rate = float(series[0].rate)

    for s in series:
        if isinstance(s, (VoltageClampSeries, CurrentClampSeries, IZeroClampSeries)):
            response = s.data[:] * float(s.conversion)

            stim_code = s.stimulus_description
            if stim_code[-5:] == "_DA_0":
                stim_code = stim_code[:-5]

            stimulus_code = stim_code.split("[")[0]
            # TODO use EphysDataInterface.get_stimulus_name(stimulus_code) here

        elif isinstance(s, (VoltageClampStimulusSeries, CurrentClampStimulusSeries)):
            stimulus = s.data[:] * float(s.conversion)
            unit = s.unit
            if not unit:
                stimulus_unit = "Unknown"
            elif unit in {"Amps", "A", "amps", "amperes"}:
                stimulus_unit = "Amps"
            elif unit in {"Volts", "V", "volts"}:
                stimulus_unit = "Volts"
            else:
                stimulus_unit = unit

    if stimulus_unit == "Volts":
        stimulus = stimulus * 1.0e3
        response = response * 1.0e12
    elif stimulus_unit == "Amps":
        stimulus = stimulus * 1.0e12
        response = response * 1.0e3

    epochs = get_epochs(sampling_rate,stimulus,response)

    return {
        'sweep_number': sweep_number,
        'stimulus_code': stimulus_code,
        'stimulus': stimulus,
        'response': response,
        'stimulus_unit': stimulus_unit,
        'sampling_rate': sampling_rate,
        'recording_epoch': epochs[0],
        'test_epoch': epochs[1],
        'stimulus_epoch': epochs[2],
        'experiment_epoch': epochs[3],
        'first_stability_epoch': epochs[4],
        'first_noise_epoch': epochs[5],
        'last_stability_epoch': epochs[6],
        'last_noise_epoch': epochs[7]
    }

        # get long unit name
        # unit = s.unit
        # if not unit:
        #     stimulus_unit = "Unknown"
        # elif unit in ["Amps", "A", "amps", "amperes"]:
        #     stimulus_unit = "Amps"
        # elif unit in ["Volts", "V", "volts"]:
        #     stimulus_unit = "Volts"
        # else:
        #     stimulus_unit = unit





    # print(series[0][0].stimulus_description)
    # print(series[0][1].stimulus_description)
    # print(series[0][0].data[:]*float(series[0][0].conversion))
    #     print('---------
    #     # for sweep in series:--------------------')
    #     for field in sweep:
    #         print(field)
    #     print(sweep)
    #     print('-----------------------------')

    # stim_codes = tuple(map(data_set.get_stimulus_code, sweep_range))
    #
    # sweep_datas = tuple(map(data_set.get_sweep_data, sweep_range))
    # # test epoch for blowout - skip this?
    # # expt epoch, stim epoch, recording epoch
    # # first/last noise/stability epochs
    #
    # epochs = tuple(map(get_epochs, sweep_datas))
    # # for epoch in epochs:
    # #     print(epoch)
    #
    # # sweep_info = [{'sweep_number': x, 'stimulus_code': stim_codes[x]}.update()]
    #
    # sweep_info = zip(stim_codes, sweep_datas, epochs)

    # for sweep in sweep_info:
    #     print('------------')
    #     for char in sweep:
    #         print(char)
    #     print('------------')


def get_epochs(sampling_rate, stimulus, response):
    recording_epoch = ep.get_recording_epoch(response)
    test_epoch = ep.get_test_epoch(stimulus, sampling_rate)

    if test_epoch:
        stimulus_epoch = ep.get_stim_epoch(stimulus)
        experiment_epoch = ep.get_experiment_epoch(stimulus, sampling_rate)
    else:
        stimulus_epoch = ep.get_stim_epoch(stimulus, test_pulse=False)
        experiment_epoch = ep.get_experiment_epoch(
            stimulus, sampling_rate, test_pulse=False
        )

    if stimulus_epoch:
        first_stability_epoch = ep.get_first_stability_epoch(
            stimulus_epoch[0], sampling_rate
        )
        first_noise_epoch = ep.get_first_noise_epoch(
            experiment_epoch[0], sampling_rate
        )

        last_stability_epoch = ep.get_last_stability_epoch(
            recording_epoch[1], sampling_rate
        )
        last_noise_epoch = ep.get_last_noise_epoch(
            recording_epoch[1], sampling_rate
        )

    else:
        first_stability_epoch = None
        first_noise_epoch = None

        last_stability_epoch = None
        last_noise_epoch = None

    return (
        recording_epoch, test_epoch,stimulus_epoch, experiment_epoch,
        first_stability_epoch, first_noise_epoch,
        last_stability_epoch, last_noise_epoch
    )


if __name__ == "__main__":

    # ignore warnings during loading .nwb files
    filterwarnings('ignore')

    files = list(Path("data/nwb").glob("*.nwb"))
    base_dir = Path(__file__).parent

    today = dt.datetime.now().strftime('%y%m%d')
    now = dt.datetime.now().strftime('%H.%M.%S')

    profile_dir = base_dir.joinpath(f'fast_qc_profiles/{today}_{now}')
    # profile_dir.mkdir(parents=True)

    # fast_experiment_qc(nwb_file=str(base_dir.joinpath(files[1])))

    for file in files:
        start_time = default_timer()
        features, tags = fast_experiment_qc(nwb_file=str(base_dir.joinpath(file)))

        print(f'{file} took {default_timer()-start_time} to load')
        for feature in features.items():
            print(feature)
        print('--------------------------------------------------------------')


        # profile_file = str(profile_dir.joinpath(f'{file.stem}.prof'))
        # cProfile.run(
        #     'fast_experiment_qc(nwb_file=str(base_dir.joinpath(file)))',
        #     filename=profile_file
        # )
        # p = pstats.Stats(profile_file)
        # p.sort_stats('cumtime').print_stats(10)

    # # cell_qc_features
    # if manual_values is None:
    #     manual_values = {}
    #
    # features = {}
    # tags = []
    #
    # features['blowout_mv'] = extract_blowout(data_set, tags)
    #
    # features['electrode_0_pa'] = extract_electrode_0(data_set, tags)
    #
    # features['recording_date'] = extract_recording_date(data_set, tags)
    #
    # features["seal_gohm"] = extract_clamp_seal(data_set, tags, manual_values)
    #
    # ir, sr = extract_input_and_access_resistance(data_set, tags)
    #
    # features['input_resistance_mohm'] = ir
    # features["initial_access_resistance_mohm"] = sr
    #
    # features['input_access_resistance_ratio'] = compute_input_access_resistance_ratio(ir, sr)
    #
    # # return features, tags
    # # sweep_qc_features
    # ontology = data_set.ontology
    # sweeps_features = []
    # iclamp_sweeps = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP,
    #                                               s
    # timuli_exclude = ["Test", "Search"],
    # )
    # if len(iclamp_sweeps.index) == 0:
    #     logging.warning("No current clamp sweeps available to compute QC features")
    #
    # for sweep_info in iclamp_sweeps.to_dict(orient='records'):
    #     sweep_features = {}
    #     sweep_features.update(sweep_info)
    #
    #     sweep_num = sweep_info['sweep_number']
    #     sweep = data_set.sweep(sweep_num)
    #     is_ramp = sweep_info['stimulus_name'] in ontology.ramp_names
    #     tags = check_sweep_integrity(sweep, is_ramp)
    #     sweep_features["tags"] = tags
    #
    #     stim_features = current_clamp_sweep_stim_features(sweep)
    #     sweep_features.update(stim_features)
    #
    #     if not tags:
    #         qc_features = current_clamp_sweep_qc_features(sweep, is_ramp)
    #         sweep_features.update(qc_features)
    #     else:
    #         logging.warning("sweep {}: {}".format(sweep_num, tags))
    #
    #     sweeps_features.append(sweep_features)
    #
    # return sweeps_features