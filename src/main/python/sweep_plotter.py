import io
from multiprocessing.connection import Connection
from typing import NamedTuple, List, Tuple, Optional

from PyQt5.QtCore import QByteArray

from pyqtgraph import PlotWidget, mkPen

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# from ipfx.ephys_data_set import EphysDataSet
from ipfx.sweep import Sweep
from ipfx.epochs import (
    get_experiment_epoch, get_test_epoch,
    get_stim_epoch, get_first_stability_epoch
)


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


class SweepTuple(NamedTuple):
    """ NamedTuple with minimum info needed for plotting."""
    sweep_number: int
    stimulus_code: str
    sweep_data: dict


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

    __slots__ = ['plot_data', 'sweep_number', 'y_label']

    def __init__(self, plot_data: PlotData, sweep_number: int, y_label: str):
        """ Displays an interactive plot of a sweep

        Parameters
        ----------
        plot_data : PlotData
            named tuple with raw data for plotting
        sweep_number : int
            sweep number used in naming the plot
        y_label: str
            label for the y-axis (mV or pA)

        """
        self.plot_data = plot_data
        self.sweep_number = sweep_number
        self.y_label = y_label

    def initialize_plot(self, graph: PlotWidget):
        """ Generates an interactive plot widget from this plotter's data. This
        function is used for easy implementation of __call__() in child classes

        Returns
        -------
        plot : PlotWidget PlotItem
            a plot item that can be plotted upon

        """
        plot = graph.getPlotItem()
        plot.addLegend()
        plot.setLabel("left", self.y_label)
        plot.setLabel("bottom", "time (s)")

        return plot


class ExperimentPopupPlotter(PopupPlotter):
    """ Subclass of PopupPlotter used for the experiment epoch. """

    __slots__ = ['plot_data', 'baseline', 'sweep_number', 'y_label']

    def __init__(
            self, plot_data: PlotData, baseline: float,
            sweep_number: int, y_label: str
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

        """
        super().__init__(plot_data=plot_data, sweep_number=sweep_number, y_label=y_label)

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

        plot.addLine(
            y=self.baseline,
            pen=mkPen(color=EXP_PULSE_BASELINE_COLOR, width=2),
            label="baseline"
        )

        plot.plot(
            self.plot_data.time, self.plot_data.response,
            pen=mkPen(color=EXP_PULSE_CURRENT_COLOR, width=2),
            name=f"sweep {self.sweep_number}"
        )

        return graph


class PulsePopupPlotter(PopupPlotter):
    """ Subclass of PopupPlotter used for the test pulse epoch. """

    __slots__ = ['plot_data', 'previous_plot_data', 'initial_plot_data',
                 'sweep_number', 'y_label']

    def __init__(
        self,
        plot_data: PlotData,
        previous_plot_data: PlotData,
        initial_plot_data: PlotData,
        sweep_number: int,
        y_label: str
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

        """

        super().__init__(plot_data=plot_data, sweep_number=sweep_number, y_label=y_label)

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


class SweepPlotter:
    # a set of stimulus codes to skip saving the test pulses from
    tp_exclude_codes = {
        "EXTPBLWOUT", "EXTPBREAKN", "EXTPCllATT", "EXTPEXPEND", "EXTPINBATH",
        "EXTPRSCHEK", "EXTPSAFETY", "EXTPSMOKET", "EXTPGGAEND", "Search"
    }

    def __init__(self, sweep_list: List[SweepTuple], config: SweepPlotConfig):
        """ Generate plots for each sweep in an experiment

        Parameters
        ----------
        sweep_list : List[Tuple[int, str, dict]
            A list containing tuples with the index, stimulus code, and
            sweep data dictionary for each sweep
        config : SweepPlotConfig
            parameters tweaking the generated plots

        """
        # input parameters
        self.data_list = sweep_list
        self.config = config

        # initialize a single figure and axes to use for fast plotting
        self.fig, self.ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        self.ax.set_xlabel("time (s)", fontsize=PLOT_FONTSIZE)

        # initial and previous test pulse data for current clamp
        self.initial_vclamp_data: Optional[PlotData] = None
        self.previous_vclamp_data: Optional[PlotData] = None

        # initial and previous test pulse data for voltage clamp
        self.initial_iclamp_data: Optional[PlotData] = None
        self.previous_iclamp_data: Optional[PlotData] = None

    # def get_plot_data(self, sweep_tuple: SweepTuple):
    #     """ Split the sweep data into test pulse and experiment epochs, then
    #     return a PlotData object for both of them as well as a baseline mean
    #     for the experiment epoch.
    #
    #     Parameters
    #     ----------
    #     sweep_tuple : SweepTuple
    #         NamedTuple object containing the neccessary data for plotting
    #
    #     Returns
    #     -------
    #     tp_plot_data : PlotData
    #         data to plot for the test pulse epoch
    #     exp_plot_data : PlotData
    #         data to plot for the experiment epoch
    #     exp_baseline_mean : np.ndarray
    #         baseline mean for the experiment epoch
    #
    #     """
    #     # use the number of points and the sampling rate to generate time vector
    #     hz = sweep_tuple.sweep_data['sampling_rate']
    #     stimulus = sweep_tuple.sweep_data['stimulus']
    #     response = sweep_tuple.sweep_data['response']
    #     num_pts = len(stimulus)
    #     time = np.arange(num_pts) / hz
    #
    #     # TODO make sure this works as expected
    #     # define indexes for splitting up experiment and test pulse
    #     test_epoch = get_test_epoch(sweep_tuple.sweep_data['stimulus'], hz)
    #     if test_epoch:
    #         # stimulus is not flat and length is greater than TP max time
    #         tp_end_idx = test_epoch[1]
    #         exp_start_idx = tp_end_idx
    #         test_pulse = True
    #     # these last two conditions might not be necessary
    #     elif self.config.backup_experiment_start_index < num_pts-1:
    #         # stimulus terminated before end of test pulse
    #         tp_end_idx = self.config.backup_experiment_start_index
    #         exp_start_idx = tp_end_idx
    #         test_pulse = False
    #     else:
    #         # no test pulse like a 'Search' sweep
    #         tp_end_idx = num_pts-1
    #         exp_start_idx = 0
    #         test_pulse = False
    #
    #     # use baseline mean to set test pulse to start at zero
    #     tp_baseline_mean = np.nanmean(
    #         response[:self.config.test_pulse_baseline_samples]
    #     )
    #     tp_plot_data = PlotData(
    #         stimulus=stimulus[:tp_end_idx],
    #         response=response[:tp_end_idx] - tp_baseline_mean,
    #         time=time[:tp_end_idx]
    #     )
    #
    #     # this function is also called in get_experiment_epoch
    #     stim_epoch = get_stim_epoch(stimulus, test_pulse)
    #     if stim_epoch:
    #         # use this to calculate baseline mean
    #         stability_start_idx, stability_end_idx = get_first_stability_epoch(
    #             stim_epoch[0], hz
    #         )
    #         # take end of tp as start of stability epoch as a backup
    #         if stability_start_idx < tp_end_idx:
    #             stability_start_idx = tp_end_idx
    #         exp_baseline_mean = np.nanmean(
    #             response[stability_start_idx:stability_end_idx]
    #         )
    #     else:
    #         exp_baseline_mean = None
    #
    #     exp_plot_data = PlotData(
    #         stimulus=stimulus[exp_start_idx:],
    #         response=response[exp_start_idx:],
    #         time=time[exp_start_idx:]
    #     )
    #
    #     return tp_plot_data, exp_plot_data, exp_baseline_mean

    def make_test_pulse_plots(
        self, 
        sweep_number: int, 
        sweep: Sweep,
        y_label: str = "",
        store_test_pulse: bool = True
    ) -> FixedPlots:
        """ Generate test pulse response plots for a single sweep

        Parameters
        ----------
        sweep_number : int
            used to generate meaningful labels
        sweep : Sweep
            holds timestamps and response values for this sweep
        y_label: str
            label for the y-axis (mV or pA)
        store_test_pulse : bool
            if True, store this sweep's response for use in later plots

        Returns
        -------
        fixed_plots : FixedPlots
            a named tuple containing a thumbnail-popup plot pair

        """

        # defining initial and previous test response
        initial = None
        previous = None

        # grabbing data for test pulse
        plot_data = test_response_plot_data(
            sweep,
            self.config.test_pulse_plot_start,
            self.config.test_pulse_plot_end,
            self.config.test_pulse_baseline_samples
        )

        # called for sweeps that will save initial / previous test pulses
        if store_test_pulse:
            if sweep.clamp_mode == "CurrentClamp":
                previous = self.previous_iclamp_data
                initial = self.initial_iclamp_data
                if self.initial_iclamp_data is None:
                    self.initial_iclamp_data = plot_data
                else:
                    self.previous_iclamp_data = plot_data

            else:
                previous = self.previous_vclamp_data
                initial = self.initial_vclamp_data
                if self.initial_vclamp_data is None:
                    self.initial_vclamp_data = plot_data
                else:
                    self.previous_vclamp_data = plot_data

        thumbnail = make_test_pulse_plot(
            sweep_number=sweep_number, plot_data=plot_data,
            previous=previous, initial=initial, y_label=y_label,
            step=self.config.thumbnail_step, labels=False
        )

        return FixedPlots(
            thumbnail=svg_from_mpl_axes(thumbnail),
            full=PulsePopupPlotter(
                plot_data=plot_data,
                previous_plot_data=previous,
                initial_plot_data=initial,
                sweep_number=sweep_number,
                y_label=y_label
            )
        )

    def make_experiment_plots(
        self, 
        sweep_number: int, 
        sweep_data: Sweep,
        y_label: str = ""
    ) -> FixedPlots:
        """ Generate experiment response plots for a single sweep

        Parameters
        ----------
        sweep_number : used to generate meaningful labels
        sweep_data : holds timestamps and voltage values for this sweep
        y_label: label for the y-axis (mV or pA)

        """

        plot_data, exp_baseline = experiment_plot_data(
            sweep=sweep_data,
            backup_start_index=self.config.backup_experiment_start_index,
            baseline_start_index=self.config.experiment_baseline_start_index,
            baseline_end_index=self.config.experiment_baseline_end_index
        )

        thumbnail = make_experiment_plot(
            sweep_number=sweep_number, plot_data=plot_data,
            exp_baseline=exp_baseline, y_label=y_label,
            step=self.config.thumbnail_step, labels=False
        )

        return FixedPlots(
            thumbnail=svg_from_mpl_axes(thumbnail),
            full=ExperimentPopupPlotter(
                plot_data=plot_data,
                baseline=exp_baseline,
                sweep_number=sweep_number,
                y_label=y_label
            )
        )

    # def generate_plots(self):
    #     """ Generate a pair of fixed plots of all sweeps in the list"""
    #     sweep_plots = []
    #     for sweep in self.data_list:
    #         if any(substring in sweep.stimulus_code
    #                for substring in self.tp_exclude_codes):
    #             # don't store test pulse for defined sweeps
    #             store_test_pulse = False
    #         else:
    #             store_test_pulse = True
    #
    #         if sweep.sweep_data['stimulus_unit'] == "Amps":
    #             y_label = "membrane potential (mV)"
    #         else:
    #             y_label = "holding current (pA)"
    #         sweep_plots.append([
    #             self.make_test_pulse_plots(
    #                 sweep_number=sweep.sweep_number,
    #                 sweep=sweep.sweep_data, y_label=y_label,
    #                 store_test_pulse=store_test_pulse
    #             ),
    #             self.make_experiment_plots(
    #                 sweep_number=sweep.sweep_number,
    #                 sweep=sweep.sweep_data, y_label=y_label
    #             )
    #         ])


    def advance(self, sweep_number: int):
        """ Determines what the y-label for the plots should be based on the
        clamp mode and then generates two fixed plots: one for the test pulse
        epoch and another for the experiment epoch.

        Parameters
        ----------
        sweep_number : sweep number for the sweep to be plotted

        Returns
        -------
        Tuple[FixedPlots, FixedPlots] : two thumbnail and popup plot pairs
             for the test pulse and experiment epoch of the sweep to be plotted

        """
        # grab sweep object and stimulus code for this sweep number
        sweep_data = self.data_set.sweep(sweep_number)
        stimulus_code = self.data_set.sweep_table['stimulus_code'][sweep_number]

        # determine y-axis label based on clamp mode and which tp's to store
        if sweep_data.clamp_mode == "CurrentClamp":
            # don't store test pulse for 'Search' in current clamp
            if stimulus_code[-6:] == "Search":
                return None, None
            else:
                store_test_pulse = True
            y_label = "membrane potential (mV)"
        else:
            # only store test pulse for 'NucVC' sweeps in voltage clamp
            if stimulus_code[0:5] == "NucVC":
                store_test_pulse = True
            else:
                store_test_pulse = False
            y_label = "holding current (pA)"

        return (
            self.make_test_pulse_plots(
                sweep_number=sweep_number,
                sweep=sweep_data, y_label=y_label,
                store_test_pulse=store_test_pulse
            ),
            self.make_experiment_plots(sweep_number, sweep_data, y_label)
        )


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


def test_response_plot_data(
    sweep: Sweep, 
    test_pulse_plot_start: float = 0.0,
    test_pulse_plot_end: float = 0.1, 
    num_baseline_samples: int = 100
) -> PlotData:
    """ Generate time and response arrays for the test pulse plots.

    Parameters
    ----------
    sweep :
        data source for one sweep
    test_pulse_plot_start :
        The start point of the plot (s)
    test_pulse_plot_end :
        The endpoint of the plot (s)
    num_baseline_samples :
        How many samples (from time 0) to use when calculating the baseline 
        mean.

    Returns
    -------
    plot_data : PlotData
        A named tuple with the sweep's stimulus, response, and time

    """
    time = np.arange(len(sweep['stimulus'])/sweep['sampling_rate'])

    start_index, end_index = (
        np.searchsorted(time, [test_pulse_plot_start, test_pulse_plot_end])
        .astype(int)
    )

    return PlotData(
        stimulus=sweep['stimulus'][start_index:end_index],
        response=sweep['response'][start_index: end_index] - np.mean(sweep.response[0: num_baseline_samples]),
        time=time[start_index:end_index]
    )


def make_test_pulse_plot(
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
    
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if initial is not None:
        ax.plot(initial.time[::step], initial.response[::step], linewidth=1, label=f"initial",
            color=TEST_PULSE_INIT_COLOR)

    if previous is not None:
        ax.plot(previous.time[::step], previous.response[::step], linewidth=1, label=f"previous",
            color=TEST_PULSE_PREV_COLOR)
    
    ax.plot(plot_data.time[::step], plot_data.response[::step], linewidth=1,
            label=f"sweep {sweep_number}", color=TEST_PULSE_CURRENT_COLOR)

    ax.set_xlabel("time (s)", fontsize=PLOT_FONTSIZE)

    ax.set_ylabel(y_label, fontsize=PLOT_FONTSIZE)

    if labels:
        ax.legend()
    else:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    return fig

    
def experiment_plot_data(
    sweep: Sweep,
    backup_start_index: int = 5000,
    baseline_start_index: int = 5000,
    baseline_end_index: int = 9000
) -> Tuple[PlotData, float]:
    """ Extract the data required for plotting a single sweep's experiment 
    epoch.

    Parameters
    ----------
    sweep : Sweep
        contains raw data that the experiment epoch will be extracted from
    backup_start_index : int
        Fall back on this if the experiment epoch start index cannot be
        programmatically assessed
    baseline_start_index : int
        Start accumulating baseline samples from this index
    baseline_end_index : int
        Stop accumulating baseline samples at this index

    Returns
    -------
    plot_data : PlotData
        A named tuple with the sweep's stimulus, response, and time

    baseline_mean : float
        The average response (mV) during the baseline epoch for this sweep

    """

    # might want to grab this from sweep.epochs instead
    start_index, end_index = \
        get_experiment_epoch(sweep.i, sweep.sampling_rate) \
            or (backup_start_index, len(sweep.i))

    if start_index <= 0:
        start_index = backup_start_index
    
    stimulus = sweep.stimulus[start_index:end_index]
    response = sweep.response[start_index:end_index]
    time = sweep.t[start_index:end_index]

    if len(response) > baseline_end_index:
        baseline_mean = float(np.nanmean(response[baseline_start_index: baseline_end_index]))
    else:
        baseline_mean = float(np.nanmean(response))

    return PlotData(stimulus, response, time), baseline_mean


def make_experiment_plot(
    sweep_number: int,
    plot_data: PlotData,
    exp_baseline: float,
    y_label: str,
    step: int = 1,
    labels: bool = True
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

    time_lim = [plot_data.time[0], plot_data.time[-1]]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    ax.plot(plot_data.time[::step], plot_data.response[::step], linewidth=1,
            color=EXP_PULSE_CURRENT_COLOR,
            label=f"sweep {sweep_number}")
    ax.hlines(exp_baseline, *time_lim, linewidth=1, 
        color=EXP_PULSE_BASELINE_COLOR,
        label="baseline")
    ax.set_xlim(time_lim)

    ax.set_xlabel("time (s)", fontsize=PLOT_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=PLOT_FONTSIZE)

    if labels:
        ax.legend()
    else:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    return fig

def make_plots(sweep_plot_list: list, plot_output: Connection):
    # dummy return for testing
    plot_output.send(['foo' for _ in sweep_plot_list])
    plot_output.close()
