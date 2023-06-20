"""
Methods for plotting meshes and surfaces
"""

import bokeh
import bokeh.palettes as palettes

from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import Column, gridplot, Row

palette = palettes.Category10[10]


def plot_points(points):
    """Return Bokeh plot of points plotted on xz and yz axes
    
    Parameters
    ----------
    points : (n_points, 3) ndarray
        array of point coordinates

    Returns
    -------
    Bokeh.GridPlot
        row of xz and yz plots
    """
    ps = []

    labels = 'xyz'
    axes_idxs_set = [(0, 2), (1, 2)]

    for (i1, i2) in axes_idxs_set:
        p = figure(
            width=400,
            height=400,
            margin=20,
            x_axis_label=labels[i1],
            y_axis_label=labels[i2]
        )

        p.scatter(points[:, i1], points[:, i2])
        p.xaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_size = '14pt'

        ps.append(p)
    
    #ps[1].y_range = ps[0].y_range
    
    layout = gridplot([ps])
    layout.toolbar.logo = None
    return layout