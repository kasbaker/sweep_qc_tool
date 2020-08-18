import io
from typing import NamedTuple, Optional

from PyQt5.QtCore import QByteArray
from pyqtgraph import PlotWidget, mkPen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from ipfx.ephys_data_set import EphysDataSet
from ipfx.epochs import get_first_stability_epoch

PLOT_FONTSIZE = 24
DEFAULT_FIGSIZE = (8, 8)

TEST_PULSE_CURRENT_COLOR = "#000000"
# hex color code with transparency added for previous and initial test pulses
TEST_PULSE_PREV_COLOR = "#0000ff70"
TEST_PULSE_INIT_COLOR = "#ff000070"

EXP_PULSE_CURRENT_COLOR = "#000000"
EXP_PULSE_BASELINE_COLOR = "#0000ff70"


class SweepPlotConfig(NamedTuple):
    test_pulse_plot_start: float
    test_pulse_plot_end: float
    test_pulse_baseline_samples: int
    backup_experiment_start_index: int
    experiment_baseline_start_index: int
    experiment_baseline_end_index: int
    thumbnail_step: int


# class PlotConfigLite(NamedTuple):
#     first_downsample_factor: int
#     second_downsample_factor: int


class PlotData(NamedTuple):
    """ Contains numpy arrays for each type of data: stimulus, response, time.
    This is needed due to ensure plotting doesn't break when different sampling
    rates are used in previous and initial test pulses.

    """
    # stimulus for part of a sweep; current for I-clamp or voltage for V-clamp
    stimulus: np.ndarray
    # response for part of a sweep; voltage for I-clamp or current for V-clamp
    response: np.ndarray
    # time vector for part of a sweep
    time: np.ndarray


class PopupPlotter:
    """ Stores data needed to make an interactive plot and generates it on
    __call__()

    """

    __slots__ = ['plot_data', 'sweep_number', 'y_label', 'stimulus_code']

    def __init__(self, plot_data: PlotData, sweep_number: int, y_label: str, stimulus_code: str):
        """ Displays an interactive plot of a sweep

        Parameters
        ----------
        plot_data : PlotData
            named tuple with raw data for plotting
        sweep_number : int
            sweep number used in naming the plot
        y_label : str
            label for the y-axis (mV or pA)
        stimulus_code : str
            stimulus code to use when labeling the popup window

        """
        self.plot_data = plot_data
        self.sweep_number = sweep_number
        self.y_label = y_label
        self.stimulus_code = stimulus_code

    def initialize_plot(self, graph: PlotWidget):
        """ Generates an interactive plot widget from this plotter's data. This
        function is used for easy implementation of __call__() in child classes

        Returns
        -------
        plot : PlotWidget PlotItem
            a plot item that can be plotted upon

        """
        plot = graph.getPlotItem()
        plot.setLabel("left", self.y_label)
        plot.setLabel("bottom", "time (s)")
        graph.setWindowTitle(f"Sweep: {self.sweep_number} - {self.stimulus_code}")

        return plot


class ExperimentPopupPlotter(PopupPlotter):
    """ Subclass of PopupPlotter used for the experiment epoch. """

    __slots__ = ['plot_data', 'baseline', 'sweep_number', 'y_label', 'stimulus_code']

    def __init__(
            self, plot_data: PlotData, baseline: Optional[np.ndarray],
            sweep_number: int, y_label: str, stimulus_code: str
    ):
        """ Displays an interactive plot of a sweep's experiment epoch, along
        with a horizontal line at the baseline.

        Parameters
        ----------
        plot_data : PlotData
            named tuple with raw data for plotting
        baseline: float
            baseline mean of the initial response in mV or pA
        sweep_number : int
            sweep number used in naming the plot
        y_label: str
            label for the y-axis (mV or pA)
        stimulus_code : str
            stimulus code to use when labeling the popup window

        """
        super().__init__(
            plot_data=plot_data, sweep_number=sweep_number,
            y_label=y_label, stimulus_code=stimulus_code
        )

        self.baseline = baseline

    def __call__(self) -> PlotWidget:
        """ Generates an interactive plot widget from this plotter's data.

        Returns
        -------
        graph : PlotWidget
            a pyqtgraph interactive PlotWidget that pops up when user clicks
            on a thumbnail of the graph

        """
        graph = PlotWidget()
        plot = self.initialize_plot(graph)
        # plot baseline if it exists
        if self.baseline is not None:
            plot.addLine(
                y=self.baseline,
                pen=mkPen(color=EXP_PULSE_BASELINE_COLOR, width=2),
                # label="baseline"
            )
        # plot experiment data
        experiment_plot = plot.plot(
            self.plot_data.time, self.plot_data.response,
            pen=mkPen(color=EXP_PULSE_CURRENT_COLOR, width=2),
        )
        # set downsampling and clip to view for performance reasons
        experiment_plot.setDownsampling(auto=True)
        experiment_plot.setClipToView(True)

        return graph


class PulsePopupPlotter(PopupPlotter):
    """ Subclass of PopupPlotter used for the test pulse epoch. """

    __slots__ = ['plot_data', 'previous_plot_data', 'initial_plot_data',
                 'sweep_number', 'y_label', 'stimulus_code']

    def __init__(
            self,
            plot_data: PlotData,
            previous_plot_data: PlotData,
            initial_plot_data: PlotData,
            sweep_number: int,
            y_label: str,
            stimulus_code: str
    ):
        """ Plots the test pulse reponse, along with responses to the previous
        and first test pulse.

        Parameters
        ----------
        plot_data : : PlotData
            named tuple with raw data for plotting
        previous_plot_data : PlotData
            named tuple with previous test pulse data
        initial_plot_data : PlotData
            named tuple with initial test pulse data
        sweep_number : int
            sweep number used in naming the plot
        y_label: str
            label for the y-axis (mV or pA)
        stimulus_code : str
            stimulus code to use when labeling the popup window
        """

        super().__init__(
            plot_data=plot_data, sweep_number=sweep_number,
            y_label=y_label, stimulus_code=stimulus_code
        )

        self.previous_plot_data = previous_plot_data
        self.initial_plot_data = initial_plot_data

    def __call__(self) -> PlotWidget:
        """ Generates an interactive plot widget from this plotter's data.

        Returns
        -------
        graph : PlotWidget
            a pyqtgraph interactive PlotWidget that pops up when user clicks
            on a thumbnail of the graph

        """

        graph = PlotWidget()
        plot = self.initialize_plot(graph)
        plot.addLegend()

        if self.initial_plot_data is not None:
            plot.plot(
                self.initial_plot_data.time, self.initial_plot_data.response,
                pen=mkPen(color=TEST_PULSE_INIT_COLOR, width=2),
                name="initial"
            )

        if self.previous_plot_data is not None:
            plot.plot(
                self.previous_plot_data.time, self.previous_plot_data.response,
                pen=mkPen(color=TEST_PULSE_PREV_COLOR, width=2),
                name="previous"
            )

        plot.plot(
            self.plot_data.time, self.plot_data.response,
            pen=mkPen(color=TEST_PULSE_CURRENT_COLOR, width=2),
            name=f"sweep {self.sweep_number}"
        )

        return graph


class FixedPlots(NamedTuple):
    """ Each plot displayed in the sweep table comes in a thumbnail-full plot
    pair.

    """
    thumbnail: QByteArray
    full: PopupPlotter


class SweepPlotter(object):

    DEFAULT_TP_EXCLUDE = {
        "EXTPBLWOUT", "EXTPBREAKN", "EXTPCllATT", "EXTPEXPEND", "EXTPINBATH",
        "EXTPRSCHEK", "EXTPSAFETY", "EXTPSMOKET", "EXTPGGAEND", "Search"  # save GGAEND to test seal?
    }

    def __init__(self, sweep_data_tuple: tuple, config: SweepPlotConfig,):
        """ Generate plots for each sweep in an experiment

        Parameters
        ----------
        data_set : EphysDataSet
            plots will be generated from these experimental data
        config : SweepPlotConfig
            parameters tweaking the generated plots

        """
        self.fig, self.ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        self.ax.set_xlabel("time (s)", fontsize=PLOT_FONTSIZE)

        self._sweep_data_tuple = sweep_data_tuple
        self.config = config
        self.ds1_factor = 4
        self.ds2_factor = 4

        self.tp_exclude_codes = self.DEFAULT_TP_EXCLUDE

        self.tp_baseline_samples = config.test_pulse_baseline_samples
        self.exp_baseline_samples = config.experiment_baseline_start_index - \
                                    config.experiment_baseline_end_index

        # self._plot_data_iter = map(self.get_plot_data, self._sweep_data_iter)

        # initial and previous test pulse data for current clamp
        self.initial_vclamp_data: Optional[PlotData] = None
        self.previous_vclamp_data: Optional[PlotData] = None

        # initial and previous test pulse data for voltage clamp
        self.initial_iclamp_data: Optional[PlotData] = None
        self.previous_iclamp_data: Optional[PlotData] = None

    def get_plot_data(self, sweep_data: dict,):
        """ Split sweep data into test pulse and experiment epochs then return
        PlotData for both of them."""
        # grab experiment epoch
        # exp_epoch = sweep_data['epochs']['experiment']
        # test_epoch = sweep_data['epochs']['test']
        # sampling rate in hz
        hz = sweep_data['sampling_rate']
        # number of points in the full sweep
        # num_pts = len(sweep_data['stimulus'])
        time = np.arange(len(sweep_data['stimulus'])) / hz

        # set test pulse start index to zero
        # tp_start_idx = 0
        # if the test epoch exists grab that number, otherwise set it to backup
        if sweep_data['epochs']['test']:
            tp_end_idx = sweep_data['epochs']['test'][1]
        else:
            tp_end_idx = self.config.backup_experiment_start_index

        # start experiment epoch epoch at end of test pulse
        exp_start_idx = tp_end_idx
        # set
        # exp_end_idx = num_pts - 1

        # if the experiment epoch exists, split up the epochs like this
        # if exp_epoch:
        #     # set test pulse end index to 5000
        #     tp_end_idx = sweep_data['epochs']['test'][1]
        #     # set experiment start index to be the and of the test pulse end index
        #     exp_start_idx = tp_end_idx
        #     # set the experiment end index to be the end of the sweep
        #     exp_end_idx = len(sweep_data['stimulus']) - 1
        #
        # # if the experiment epoch doesn't exist, then take the whole sweep
        # else:
        #     # set the end index to be the end of the sweep
        #     tp_end_idx = len(sweep_data['stimulus']) - 1
        #     # set the experiment start and end to be the same as the test pulse
        #     exp_start_idx = tp_start_idx
        #     exp_end_idx = tp_end_idx

        # calculate baseline mean for test pulse
        tp_baseline_mean = np.nanmean(
            sweep_data['response'][:self.tp_baseline_samples]
        )
        # plot data for baseline subtracted test pulse epoch
        tp_plot_data = PlotData(
            stimulus=sweep_data['stimulus'][:tp_end_idx],
            response=sweep_data['response'][:tp_end_idx] - tp_baseline_mean,
            time=time[:tp_end_idx]
        )

        # calculate baseline mean for experiment
        if sweep_data['epochs']['stim']:
            # get the stability epoch for calculating baseline
            stability_start_idx, stability_end_idx = get_first_stability_epoch(
                sweep_data['epochs']['stim'][0], hz
            )
            # if start of stability epoch is less than start of test epoch then
            # take the end of the test epoch as the start of the stability epoch
            if stability_start_idx < tp_end_idx:
                stability_start_idx = tp_end_idx
            # take the mean of the stability epoch as experiment baseline
            exp_baseline_mean = np.nanmean(
                    sweep_data['response'][stability_start_idx:stability_end_idx]
            )
        else:
            exp_baseline_mean = None

        # plot data for baseline subtracted experiment epoch
        exp_plot_data = PlotData(
            stimulus=sweep_data['stimulus'][exp_start_idx:],
            response=sweep_data['response'][exp_start_idx:],
            time=time[exp_start_idx:]
        )

        return tp_plot_data, exp_plot_data, exp_baseline_mean

    def gen_plots(self):
        """ Generate a pair of fixed plots for sweeps in sweep data iterator. """
        for sweep in self._sweep_data_tuple:
            if sweep['stimulus_name'] == "Search":
                continue

            # split up test pulse and experiment epochs
            tp_plot_data, exp_plot_data, exp_baseline_mean = self.get_plot_data(sweep)

            sweep_num = sweep['sweep_number']
            # initialize previous and initial test pulses to None
            previous = None
            initial = None
            # store the test pulse if this is true
            store_tp = True

            # cache stimulus code
            stim_code = sweep['stimulus_code']
            # only save test pulse if it is not in excluded test pulse codes
            if any(substring in stim_code for substring in self.tp_exclude_codes):
                store_tp = False

            # handle voltage clamp previous and initial test pulses
            if sweep['stimulus_unit'] == "Volts":
                # set label for sweep response
                y_label = "holding current (pA)"
                if store_tp:
                    # grab previous and initial tp's if they exist
                    previous = self.previous_vclamp_data
                    initial = self.initial_vclamp_data
                    if self.initial_vclamp_data is None:
                        # set test pulse data to initial if it doesn't exist yet
                        self.initial_vclamp_data = tp_plot_data
                    else:
                        # set test pulse data to previous if initial already exists
                        self.previous_vclamp_data = tp_plot_data

            # handle current clamp previous and initial test pulses
            elif sweep['stimulus_unit'] == "Amps":
                # set label for sweep response
                y_label = "membrane potential (mV)"
                if store_tp:
                    # grab previous and initial tp's if they exist
                    previous = self.previous_iclamp_data
                    initial = self.initial_iclamp_data
                    if self.initial_iclamp_data is None:
                        # set test pulse data to initial if it doesn't exist yet
                        self.initial_iclamp_data = tp_plot_data
                    else:
                        # set test pulse data to previous if initial already exists
                        self.previous_iclamp_data = tp_plot_data
            else:
                y_label = "unknown"

            thumbnail_step = self.config.thumbnail_step
            # test pulse thumb-popup pair
            tp_plots = FixedPlots(
                thumbnail=self.make_test_pulse_plot(
                    sweep_number=sweep_num, plot_data=tp_plot_data,
                    previous=previous, initial=initial, y_label=y_label,
                    step=thumbnail_step, labels=False
                ),
                full=PulsePopupPlotter(
                    plot_data=tp_plot_data,
                    previous_plot_data=previous, initial_plot_data=initial,
                    sweep_number=sweep['sweep_number'], y_label=y_label,
                    stimulus_code=stim_code
                )
            )

            # experiment thumb-popup pair
            exp_plots = FixedPlots(
                thumbnail=self.make_experiment_plot(
                    sweep_number=sweep_num, plot_data=exp_plot_data,
                    exp_baseline=exp_baseline_mean, y_label=y_label,
                    step=thumbnail_step, labels=False
                ),
                full=ExperimentPopupPlotter(
                    plot_data=exp_plot_data, baseline=exp_baseline_mean,
                    sweep_number=sweep['sweep_number'], y_label=y_label,
                    stimulus_code=stim_code
                )
            )

            yield int(sweep_num), tp_plots, exp_plots

    def make_test_pulse_plot(
            self,
            sweep_number: int,
            plot_data: PlotData,
            previous: Optional[PlotData] = None,
            initial: Optional[PlotData] = None,
            y_label: str = "",
            step: int = 1,
            labels: bool = True
    ) -> mpl.figure.Figure:
        """ Make a (static) plot of the response to a single sweep's test pulse,
        optionally comparing to other sweeps from this experiment.

        Parameters
        ----------
        sweep_number : int
            Identifier for this sweep. Used for labeling.
        plot_data : PlotData
            named tuple with raw data used for plotting
        previous : Optional[PlotData]
           named tuple with raw data used for the previous sweep of the same
           clamp mode
        initial : Optional[PlotData]
            named tuple with raw data used to plot the first sweep for a given
            clamp mode or stimulus code
        y_label: str
            label for the y-axis (mV or pA)
        step : int
            stepsize applied to each array. Can be used to generate decimated
            thumbnails
        labels : bool
            If False, labels will not be generated (useful for thumbnails).

        Returns
        -------
        fig : mpl.figure.Figure
            a matplotlib figure containing the plot to be turned into a thumbnail

        """
        # from scipy.signal import decimate
        # ds1 = 4
        # ds2 = 4
        # step = ds1 * ds2 // 1
        if initial is not None:
            # ds_initial = decimate(decimate(initial.response, ds1), ds2)
            self.ax.plot(initial.time[::step], initial.response[::step], linewidth=1,
                         label=f"initial",
                         color=TEST_PULSE_INIT_COLOR)

        if previous is not None:
            # ds_previous = decimate(decimate(previous.response, ds1), ds2)
            self.ax.plot(previous.time[::step], previous.response[::step], linewidth=1,
                         label=f"previous",
                         color=TEST_PULSE_PREV_COLOR)

        # ds_response = decimate(decimate(plot_data.response, ds1), ds2)
        self.ax.plot(plot_data.time[::step], plot_data.response[::step], linewidth=1,
                     label=f"sweep {sweep_number}", color=TEST_PULSE_CURRENT_COLOR)

        time_lim = (plot_data.time[0], plot_data.time[-1])
        self.ax.set_xlim(time_lim)

        # self.ax.set_ylabel(y_label, fontsize=PLOT_FONTSIZE)

        if labels:
            self.ax.legend()
        else:
            self.ax.xaxis.set_major_locator(plt.NullLocator())
            self.ax.yaxis.set_major_locator(plt.NullLocator())

        thumbnail = svg_from_mpl_axes(self.fig)

        self.ax.clear()

        return thumbnail

    def make_experiment_plot(
            self,
            sweep_number: int,
            plot_data: PlotData,
            exp_baseline: Optional[np.ndarray],
            y_label: str,
            step: int = 1,
            labels: bool = True,
    ) -> mpl.figure.Figure:
        """ Make a (static) plot of the response to a single sweep's stimulus

        Parameters
        ----------
        sweep_number : int
            Identifier for this sweep. Used for labeling.
        plot_data : PlotData
            named tuple with raw data for plotting
        exp_baseline : float
            the average response (mV or pA) during a period just before stimulation
        y_label: str
            label for the y-axis (mV or pA)
        step : int
            stepsize applied to each array. Can be used to generate decimated
            thumbnails
        labels : bool
            If False, labels will not be generated (useful for thumbnails).

        Returns
        -------
        fig : mpl.figure.Figure
            a matplotlib figure containing the plot to be turned into a thumbnail

        """
        # from scipy.signal import decimate
        # ds1 = 4
        # ds2 = 4
        # step = ds1 * ds2 // 1
        # ds_response = decimate(decimate(plot_data.response, ds1), ds2)

        time_lim = [plot_data.time[0], plot_data.time[-1]]
        # y_lim = [min(ds_response), max(ds_response)]

        self.ax.plot(plot_data.time[::step], plot_data.response[::step], linewidth=1,
                     color=EXP_PULSE_CURRENT_COLOR,
                     label=f"sweep {sweep_number}")

        if exp_baseline:
            self.ax.hlines(exp_baseline, *time_lim, linewidth=1,
                           color=EXP_PULSE_BASELINE_COLOR,
                           label="baseline")

        self.ax.set_xlim(time_lim)

        # self.ax.set_ylabel(y_label, fontsize=PLOT_FONTSIZE)

        if labels:
            self.ax.legend()
        else:
            self.ax.xaxis.set_major_locator(plt.NullLocator())
            self.ax.yaxis.set_major_locator(plt.NullLocator())

        thumbnail = svg_from_mpl_axes(self.fig)

        self.ax.clear()

        return thumbnail


def svg_from_mpl_axes(fig: mpl.figure.Figure) -> QByteArray:
    """ Convert a matplotlib figure to SVG and store it in a Qt byte array.

    Parameters
    ----------
    fig: mpl.figure.Figure
        a matplotlib figure containing the plot to be turned into a thumbnail

    Returns
    -------
    thumbnail : QByteArray
        a QByteArray used as a thumbnail for the given plot

    """

    data = io.BytesIO()
    fig.savefig(data, format="svg")
    plt.close(fig)

    return QByteArray(data.getvalue())
