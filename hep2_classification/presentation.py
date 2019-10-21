""" Module containing all presentation related functions. """
import math
import itertools
from abc import ABC, abstractmethod
from typing import List, Any, Callable, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes


# ==========================================================
#  GRIDS
# ==========================================================
def display_grid(data: List[Any], plotter: Callable[[Any, Axes], None],
                 titles: List[str] = None,
                 cols: int = 3, rows: int = None,
                 width: int = 20, height: float = None,
                 **kwargs):
    """
    Helper function that creates matplotlib grid and passes each created axes with corresponding data to provided
    plotter.

    :param data: list of data to be displayed in grid
    :param plotter: callable that accepts data and axis and plots given data on given axis
    :param titles: list of titles for each cell in grid
    :param cols: number of columns in grid
    :param rows: number of rows in grid, if not provided the value is calculated using size of `data` and `cols` so
                that all data will be displayed
    :param width: width in inches of grid
    :param height: height of grid in inches, if not provided the value is calculated using first available method:
                1. using aspect ratio of `shape` attribute of the first object from `data`
                2. using default height of figure
    :param kwargs: any additional keyword arguments are passed to `ply.subplots`
    """
    # calculate rows number if not provided
    if rows is None:
        rows = math.ceil(len(data) / cols)

    # calculate height if not provided
    if height is None:
        if hasattr(data[0], 'shape'):
            # use aspect ario of data
            height = (width / cols) * (data[0].shape[0] / data[0].shape[1])
            # add margin for title in case of small cells
            height += 0.4
            # multiply by rows
            height *= rows
        else:
            height = plt.rcParams["figure.figsize"][1] * rows

    # create figure
    f, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, **kwargs)

    # setup each subplot
    for d, ax, title in itertools.zip_longest(data, itertools.chain.from_iterable(axes), titles or []):
        # in case data or titles is longer then axes
        if ax is None:
            break
        # data available => call plotter
        if d is not None:
            ax.set_title(title or '')
            plotter(d, ax)
        # no data (probably empty cells in last row) => clear this cell
        else:
            f.delaxes(ax)


# ==========================================================
#  PLOTTERS
# ==========================================================
class Plotter(ABC):
    """ Base class for plotters. """

    @abstractmethod
    def __call__(self, data: Any, ax: Axes):
        pass


class LayerPlotter(Plotter):
    """
    Plotter that merges different plotters.

    If this plotter is called with data that is a instance of tuple, then each subplotter will be called with next
    element from data. Otherwise each subplotter will receive the same data.
    """

    def __init__(self, *plotters: Plotter):
        """ Creates plotter that applies each of the given plotters. """
        self.plotters = plotters

    def __call__(self, data: Any, ax: Axes):
        if isinstance(data, tuple):
            for plotter, d in zip(self.plotters, data):
                plotter(d, ax)
        else:
            for plotter in self.plotters:
                plotter(data, ax)


class ImagePlotter(Plotter):
    """ Plotter that plots images. """
    DEFAULTS = dict(
        cmap='gray'
    )

    def __init__(self, show_axis: bool = True, **kwargs):
        """
        Creates plotter with default settings that can be overridden by kwargs.

        :param show_axis: whether or not show axis of image
        :param kwargs: arguments to be passed to `Axes.imshow()`
        """
        self.show_axis = show_axis
        self.kwargs = self.DEFAULTS.copy()
        self.kwargs.update(kwargs)

    def __call__(self, data: np.ndarray, ax: Axes):
        ax.imshow(data, **self.kwargs)

        if not self.show_axis:
            ax.axis('off')


class HistogramPlotter(Plotter):
    """ Plotter that plots histograms. """
    DEFAULTS = dict(
        log=True,
        bins=64,
        range=(0, 255)
    )

    def __init__(self, **kwargs):
        """
        Creates plotter with default settings that can be overridden by kwargs.

        :param kwargs: arguments to be passed to `Axes.hist()`
        """
        self.kwargs = self.DEFAULTS.copy()
        self.kwargs.update(kwargs)

    def __call__(self, data: np.ndarray, ax: Axes):
        ax.hist(data.ravel(), **self.kwargs)


class RectanglePlotter(Plotter):
    """
    Plotter that plots rectangles.

    Expects to be called with data that's either tuple, or list of tuples.
    Each tuple should be in format (x, y, width, height, [rotation]).
    """
    DEFAULTS = dict(
        color='r',
        fill=False
    )

    def __init__(self, **kwargs):
        """
        Creates plotter with default settings that can be overridden by kwargs.

        :param kwargs: arguments to be passed to `patches.Rectangle()`
        """
        self.kwargs = self.DEFAULTS.copy()
        self.kwargs.update(kwargs)

    def __call__(self, data: Union[List[tuple], tuple], ax: Axes):
        if not isinstance(data, list):
            data = [data]

        for d in data:
            ax.add_patch(patches.Rectangle(d[:2], *d[2:], **self.kwargs))