#!/usr/bin/env python
""" A selection of  plotting routines for Granule Explorer output data.

    Outline
    -------
    We provide a number of routines for visualing the data stored in
    "aggregate_hittings.h5". These include various 1 and 2D histograms,
    quartile plots and error estimates.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as c
from matplotlib.cm import ScalarMappable

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re
from collections import OrderedDict
from scipy.stats import sem, gmean, gstd,norm
from scipy.stats import gmean as _gmean
from matplotlib import rc

from matplotlib.ticker import EngFormatter
# import granule_explorer_core.tools.plot_tools as pt   
from .plot_tools import create_axes,annotate_axis,cart2polar,despine_axis,force_integer_ticks,format_si,hide_axis_lables 

# Regex capturing --N group termiated with '--' or '.'
re_experiment_name = re.compile("--N(?P<exp>[\d\w_-]+?)(?:--|\.)")
re_time_stamp = re.compile("_(.+)--N")


def split_hist(
    plot_column,
    split_column,
    plot_label,
    granule_data: pd.DataFrame,
    bins=None,
    units="",
    summary=False,
    title=None,
    density=False,
    save_png=True,
    out_dir="/tmp/",
):
    """Split the data by some column [split_column], then plot a row of histograms of [plot_column] side-by-side.

    Outline
    -------

    Used to compare paramters across experiments or conditions, without overlaying data. 

    Parameters
    ----------

    plot_column: str
        The name of the column in [granule_data] whose histogram is to be plotted

    split_column: str
        The name of the column in [granule_data] to split the data by. This column should
        not have more than about 10 unique values, or else you will run out of memory and the plot will be 
        illegible anyway.

    plot_label: str
        The label for the x-axis of the histograms

    granule_data: Pandas dataframe
        The granule data for the plot, see the section on "aggregate_fitting.h5"
        for the format required of the dataframe

    bins: int or sequence or str or None
        If None, then there will be 10 bins for each histogram
        If int n, then each histogram will have n bins.
        If bins is a sequence, it defines the bin edges, from the left edge of the 
        first bin to the right edge of the last.
        If bins is a str it is one of the binning strategies supported by
        numpy.histogram_bin_edges: 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.

    units: str
        The units of [plot_column], as they should be displayed

    summary: bool
        Add summary statistics to each plot if True

    title: str
        The title for each plot

    density: bool
        if True, draw a probability density, so that the area under the histogram
        is 1.

    save_png: bool
        Saves the figure to a png in [out_dir] if true

    out_dir: str
        The path that the output figure should be saved to

    Outputs
    -------
    Figure to [out_dir] if [save_file] is true

    Returns
    -------
    A matplotlib figure

    """

    g = sns.FacetGrid(col=split_column, data=granule_data, height=8.5 / 2.5)

    g.map(plt.hist, plot_column, bins=bins, density=density)
    g.map(_plot_geometric_mean, plot_column)
    g.set_xlabels(plot_label)
    g.set_ylabels("Count")

    if summary:
        g.map(_add_summary, plot_column, units=units)
    if title:
        g.set_titles(title)

    if bins is not None and not isinstance(bins, int):
        bin_ratio = bins[-1] / bins[0]
        if bin_ratio > 20:
            g.set(xscale="log")

    # if save_png:
    #     out_dir = Path(out_dir)
    #     pt.save_figure_and_trim(
    #         out_dir / f"split-{plot_column}-{split_column}.png",
    #     )
    return plt.gcf()


def binned_plot (
    plot_column,
    plot_title,
    bin_column,
    bin_title,
    granule_data,
    group_by="experiment",
    bin_type="quantile",
    n_bins=4,
    agg=gmean,
    legend=True,
    x_log_scale=False,
    y_log_scale = False,
    errors=True,
    save_png=True,
    out_dir="/tmp/",
):
    """Split the granules into [q-bins]-tiles (generalised quartiles) based on [bin_column]
       plot the mean value of [plot_column] for each n-tile.

    Outline
    -------

    Used to see (crudely) if a particular vaiable is correlated with another.
    Use 2D histograms instead if sufficient data points are available.

    Each n-tile will contain the same number of granules. If multiple experiments are in 
    [granule_data] they will be represented by different colours and plotted on top of one
    another.

    Parameters
    ----------

    plot_column: str
        The name of the column in [granule_data] to be averaged. Sets y-axis

    plot_title: str
        The label for the y-axis

    bin_column: str
        The name of the column in [granule_data] to split the data by

    bin_name: str
        The label for the x-axis

    granule_data: Pandas dataframe
        The granule data for the plot, see the section on "aggregate_fitting.h5"
        for the format required of the dataframe

    group_by: str or None
        The name of the column in [granule_data] which will be used to group the data before
        plotting. Different groups will be plotted on top of each other

    bin_type: str
        Set the binning mode:
        If "linear", then even width bins across the full data range
        If "log", then bin width scales exponentially 
        If "quantile" the split the spance into bins containing equal numbers of condensates

    n_bins: int or array
        If int then the number of bins. If [bin_type] is "log" then n_bins must be an int.
        If array and [bin_type] if linear then array of bin edges.
        If array and [bin_type] is quantile then quantiles to split the data into eg. [0, .25, .5, .75, 1.] for quartiles

    agg: function Pandas dataseries -> float
        The function used to calculate the bin heights. Usually some type of mean.

    legend: bool
        Add a legend to the plot or not

    x_log_scale: bool
        Set x axis to a log scale if true

    y_log_scale: bool
        Set y axis to a log scale if true

    errors: bool
        If true, plot error bars based on the standard-error-on-the-mean for each
        point.

    save_png: bool
        Saves the figure to a png in [out_dir] if true

    out_dir: str
        The path that the output figure should be saved to

    Outputs
    -------
    Figure to [out_dir] if [save_file] is true

    Returns
    -------
    A matplotlib figure

    """

    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)

    get_colour = colour_gen()

    for num, (hue_name, group_) in enumerate(granule_data.groupby(group_by)):
        group = group_.copy()
        
        if bin_type == "quantile":
            bins = pd.qcut(group[bin_column], n_bins)
        elif bin_type == "linear":
            bins = pd.cut(group[bin_column], n_bins)
        elif bin_type == "log":
            bin_max = group[bin_column].max()
            bin_min = group[bin_column].min()
            bin_groups = np.geomspace(bin_min, bin_max, n_bins + 1)
            bins = pd.cut(group[bin_column], bin_groups, right=True) 
        else:
            raise ValueError("bin_type must be quantile, linear, or log")
        # Ensure that the granule size is saved as a float
        group["bin"] = bins

        mean_values = (
            group.groupby("bin",observed=False)[plot_column].agg([agg, sem])
        ).reset_index()
        mean_values["mid_point"] = pd.to_numeric(mean_values["bin"].apply(lambda i: float(i.mid)))
        mean_values["bin_width"] = pd.to_numeric(mean_values["bin"].apply(lambda i: float(i.right)-float(i.left)))

        plot_colour = get_colour(hue_name)
        error_colour = _adjust_lightness(plot_colour,0.6)

        plt_kwargs = dict(
            label=hue_name, data=mean_values, alpha = 0.8
        )
        if errors:
            ax.errorbar(
                "mid_point", agg.__name__, yerr="sem", elinewidth=0.8,fmt="o",color=error_colour, **plt_kwargs,
            )
        
        ax.bar("mid_point", agg.__name__, width="bin_width", edgecolor="black",  color=plot_colour, **plt_kwargs)

    ax.set_ylabel(plot_title)
    ax.set_xlabel(bin_title)
    if legend:
        ax.legend(fontsize=10)
    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")

    print(f"plot_column = {plot_column}")

    # if save_png:
    #     pt.save(
    #         Path(out_dir) / f"binned-{bin_type}-{bin_column}-{plot_column}.png",
    #         padding=0.05,
    #     )
    return fig

def binned2D_plot(
    plot_average,
    average_label,
    bin_column,
    column_label,
    bin_row,
    row_label,
    granule_data,
    bin_type = "linear",
    column_nbins = 5,  
    row_nbins = 5,
    group_by = "experiment",
    plot_group = "As",
    agg = gmean,
    x_log_scale = False,
    y_log_scale = False,
    average_log_scale= False,
    save_png=True,
    out_dir="/tmp/",
    ):

    """Bins the granules into a grid based on [bin_row] and [bin_column]. Makes
        a heatmap, where the colour is the average of [plot_height] for granules that
        fall within a given bin-square.

    Outline
    -------

    Used to see if a particular vaiable is correlated with two other variables.
    Each grid square will contain the same number of granules.

    Parameters
    ----------

    plot_average: str
        The name of the column in [granule_data] to be averaged. Sets colour-axis

    average_label:
        The label for the colour-axis

    bin_column: str
        The name of the first column in [granule_data] to split the data by

    column_label: str
        The label for the first column

    plot_row: str
        The name of the second column in [granule_data] to split the data by

    row_label: str
        The label for the second column

    granule_data: Pandas dataframe
        The granule data for the plot, see the section on "aggregate_fitting.h5"
        for the format required of the dataframe

    bin_type: str
        Set the binning mode:
        If "linear", then even width bins across the full data range
        If "log", then bin width scales exponentially 
        If "quantile" the split the spance into bins containing equal numbers of condensates

    column_nbins: int or array
        If int then the number of bins. If [bin_type] is "log" then n_bins must be an int.
        If array and [bin_type] if linear then array of bin edges.
        If array and [bin_type] is quantile then quantiles to split the data into eg. [0, .25, .5, .75, 1.] for quartiles

    row_nbins: int or array
        If int then the number of bins. If [bin_type] is "log" then n_bins must be an int.
        If array and [bin_type] if linear then array of bin edges.
        If array and [bin_type] is quantile then quantiles to split the data into eg. [0, .25, .5, .75, 1.] for quartiles

    group_by: str
        The name of the column in [granule_data] which will be used to group the data before
        plotting. Only granules with a value of [plot_group] in this column will be plotted

    plot_group: anything
        The value in [group_by] of granules that should be plotted

    agg: function Pandas dataseries -> float
        The function used to calculate the colour values. Usually some type of mean.

    x_log_scale: bool
        Set x axis to a log scale if true

    y_log_scale: bool
        Set y axis to a log scale if true

    average_log_scale: bool
        Set the colour axis to a log scale
        
    save_png: bool
        Saves the figure to a png in [out_dir] if true

    out_dir: str
        The path that the output figure should be saved to

    Outputs
    -------
    Figure to [out_dir] if [save_file] is true

    Returns
    -------
    A matplotlib figure

    """

    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)

    my_filter = f'{group_by} == "{plot_group}"'
    group = granule_data.query(my_filter).copy()

    if bin_type == "quantile":
        column_bins = pd.qcut(group[bin_column], column_nbins)
        row_bins = pd.qcut(group[bin_row], row_nbins)
    elif bin_type == "linear":
        column_bins = pd.cut(group[bin_column], column_nbins)
        row_bins = pd.cut(group[bin_row], row_nbins)
    elif bin_type == "log":
        bin_max = group[bin_column].max()
        bin_min = group[bin_column].min()
        bin_groups = np.geomspace(bin_min, bin_max, column_nbins + 1)
        column_bins = pd.cut(group[bin_column], bin_groups, right=True)

        bin_max = group[bin_row].max()
        bin_min = group[bin_row].min()
        bin_groups = np.geomspace(bin_min, bin_max, row_nbins + 1)
        row_bins = pd.cut(group[bin_row], bin_groups, right=True)            
    else:
        raise ValueError("bin_type must be quantile, linear, or log")

    group["bin_column"] = column_bins
    group["bin_row"] = row_bins
    mean_values = (
        group.groupby(["bin_column","bin_row"],observed=False)[plot_average].agg([agg, sem])
        ).reset_index()

    mean_values["column_origin"] = pd.to_numeric(mean_values["bin_column"].apply(lambda i: float(i.left)))
    mean_values["column_width"] = pd.to_numeric(mean_values["bin_column"].apply(lambda i: float(i.right)-float(i.left)))

    mean_values["row_origin"] = pd.to_numeric(mean_values["bin_row"].apply(lambda i: float(i.left)))
    mean_values["row_width"] = pd.to_numeric(mean_values["bin_row"].apply(lambda i: float(i.right)-float(i.left)))

    cmap = plt.get_cmap('inferno')

    if average_log_scale:
        norm = c.LogNorm(mean_values[agg.__name__].min(),mean_values[agg.__name__].max())
    else:
        norm = c.Normalize(mean_values[agg.__name__].min(),mean_values[agg.__name__].max())

    for index, row in mean_values.iterrows():
        origin = row["column_origin"], row["row_origin"]
        x_step = row["column_width"]
        y_step = row["row_width"]
        colour = cmap(norm(row[agg.__name__]))
        ax.add_artist(plt.Rectangle(origin,x_step,y_step,color=colour))

    ax.set_xlim( group[bin_column].min(), 
                    group[bin_column].max()  )

    ax.set_ylim( group[bin_row].min(), 
                    group[bin_row].max()  )

    ax.set_ylabel(row_label)
    ax.set_xlabel(column_label)

    cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm),label = average_label,ax=ax)

    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("symlog")

    # if save_png:
    #     pt.save(
    #         Path(out_dir) / f"Binned-2D-{bin_type}-{bin_column}-{bin_row}-{plot_average}.png",
    #         padding=0.05,
    #     )
    return fig
        

def filter_plot(
    plot_column,
    plot_title,
    bin_column,
    bin_title,
    granule_data,
    group_by="experiment",
    plot_group="As",
    filter_list = [("all filters","green","sigma > 1e-10 and pass_rate > 0.6 and fitting_error < 0.6"),
                     ("outline filter only","red" ,"sigma > 1e-10 and pass_rate > 0.6"),
                     ("no filters","blue", "sigma > 1e-10")
                    ],
    n_bins=5,
    bin_type="count",
    agg=gmean,
    legend=True,
    x_log_scale=False,
    y_log_scale=False,
    errors=False,
    save_png=True,
    out_dir="/tmp/",
):
    """
    Used to see how changing the filters effects the distrubution of certain parameters.

    Outline
    -------

    Bin the data by [bin_column] using the [bin_type] method. The plot the average
    of [plot_column] for each bin.Do this for each filter given in [filters]

    Parameters
    ----------

    plot_column: str
        The name of the column in [granule_data] to be averaged. Sets y-axis

    plot_title: str
        The label for the y-axis

    bin_column: str
        The name of the column in [granule_data] to split the data by

    bin_name: str
        The label for the x-axis

    granule_data: Pandas dataframe
        The granule data for the plot, see the section in the Docs on "aggregate_fitting.h5"
        for the format required of the dataframe

    group_by: str
        The name of the column in [granule_data] which will be used to group the data before
        plotting. Only granules with a value of [plot_group] in this column will be plotted

    plot_group: anything
        The value in [group_by] of granules that should be plotted  

    filter_list: [(str,color,str)]
        The list of filters to be applied to the data to plot each line.
        It is a list of tuples of three elements:
            1: The title of the filter
            2: The colour of the line associated with the folter
            3: The filter itself, passed to DataTable.query(...)

    n_bins: int
        The number of bins

    bin_type: str
        If "count" then split the data into n-tiles.
        If "range" then split the data into equally sized bins
        If "log" the splint into logarithmically sized bins.

    agg: function Pandas dataseries -> float
        The function used to calculate the colour values. Usually some type of mean.

    legend: bool
        Add a legend to the plot or not

    x_log_scale: bool
        Set x axis to a log scale if true

    y_log_scale: bool
        Set y axis to a log scale if true

    errors: bool
        If true, plot error bars based on the standard-error-on-the-mean for each
        point.

    save_png: bool
        Saves the figure to a png in [out_dir] if true

    out_dir: str
        The path that the output figure should be saved to

    Outputs
    -------
    Figure to [out_dir] if [save_file] is true

    Returns
    -------
    A matplotlib figure
    """

    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)
    my_filter = f'{group_by} == "{plot_group}"'
    granule_data = granule_data.query(my_filter).copy()   

    for title,colour, query_string in filter_list:
        group = granule_data.query(query_string).copy()
        if bin_type == "count":
            bins = pd.qcut(group[bin_column], n_bins)
        elif bin_type == "range":
            bins = pd.cut(group[bin_column], n_bins)
        elif bin_type == "log":
            bin_max = group[bin_column].max()
            bin_min = group[bin_column].min()
            bin_groups = np.geomspace(bin_min, bin_max, n_bins + 1)
            bins = pd.cut(group[bin_column], bin_groups, right=True)

        # Ensure that the granule size is saved as a float
        bin_mid_point = bins.apply(lambda i: float(i.mid))
        group["bin_mid_point"] = pd.to_numeric(bin_mid_point)

        mean_values = (
            group.groupby("bin_mid_point")[plot_column].agg([agg, sem])
        ).reset_index()

        plt_kwargs = dict(
            label=title, data=mean_values, color=colour
        )
        if errors:
            ax.errorbar(
                "bin_mid_point", agg.__name__, yerr="sem", elinewidth=0.8, **plt_kwargs,
            )
        else:
            ax.plot("bin_mid_point", agg.__name__, **plt_kwargs)

        agg_mean = agg(np.abs(group[plot_column]))
        ax.axhline(agg_mean,0,1,color=colour,ls="--",lw=0.8,alpha=1.0)

    ax.set_ylabel(plot_title)
    ax.set_xlabel(bin_title)
    if legend:
        ax.legend(fontsize=10)
    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")

    # if save_png:
    #     pt.save(
    #         Path(out_dir) / f"filters-{bin_column}-{plot_column}-{bin_type}.png",
    #         padding=0.05,
    #     )
    return fig

def scatter_plot(
    plot_column,
    plot_title,
    plot_row,
    row_title,
    granule_data,
    group_by = "experiment",
    plot_group = "As",
    legend = True,
    log_scaleX = True,
    log_scaleY = True,
    save_png = True,
    out_dir = "/tmp/",
):
    """
    A 2D scatter plot to visuale correlations between parameters. If it looks to busy,
    use histogram2D

    Outline
    -------

    Scatter the data along the [plot_column] and [plot_row] axis. The colour is set by the fitting 
    error for each particle.

    Parameters
    ----------

    plot_column: str
        The name of the column in [granule_data] to be scattered along the y-axis

    plot_title: str
        The label for the y-axis 

    plot_row: str
        The name of the column in [granule_data] to be scattered along the x-axis

    plot_row: str
        The label for the y-axis

    granule_data: Pandas dataframe
        The granule data for the plot, see the section in the Docs on "aggregate_fitting.h5"
        for the format required of the dataframe

    group_by: str
        The name of the column in [granule_data] which will be used to group the data before
        plotting. Only granules with a value of [plot_group] in this column will be plotted

    plot_group: anything
        The value in [group_by] of granules that should be plotted

    legend: bool
        Add a legend to the plot or not

    x_log_scale: bool
        Set x axis to a log scale if true

    y_log_scale: bool
        Set y axis to a log scale if true

    save_png: bool
        Saves the figure to a png in [out_dir] if true

    out_dir: str
        The path that the output figure should be saved to

    Outputs
    -------
    Figure to [out_dir] if [save_file] is true

    Returns
    -------
    A matplotlib figure
    """

    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)
    # print(granule_data.columns)
    # my_filter = f'{group_by} == "{plot_group}"'
    # granule_data = granule_data.query(my_filter).copy()   


    sorted_granules = granule_data.sort_values(by=['fitting_error'],ascending=False)
    ax.scatter(plot_row,plot_column,c="fitting_error",
                label=plot_group, data=sorted_granules, linewidths = 0.0, 
                s = 3)

    #radius_units = "μm" if draft else "\si{\micro m}"
    ax.set_ylabel(plot_title)
    ax.set_xlabel(row_title)
    if legend:
        ax.legend(fontsize=10)
    if log_scaleY:
        ax.set_yscale("log")
    if log_scaleX:
        ax.set_xscale("log")

    # if save_png:
    #     pt.save(
    #         Path(out_dir) / f"scatter-{plot_group}-{plot_row}-{plot_column}.png",
    #         padding=0.05,
    #     )
    return fig


def histogram2D (
    plot_column,
    plot_title,
    plot_row,
    row_title,
    granule_data,
    group_by = "experiment",
    plot_group = "As",
    column_nbins = 20,
    row_nbins = 20,
    legend = True,
    log_scaleX = True,
    log_scaleY = True,
    save_png = True,
    out_dir = "/tmp/", 
):
    """
    A 2D histogram plot to visuale correlations between parameters. If it looks too
    sparse, (not enough points per bin) use scatter_plot instead.

    Outline
    -------

    histogram of the data along the [plot_column] and [plot_row] axis.

    Parameters
    ----------

    plot_column: str
        The name of the column in [granule_data] to be binned along the y-axis

    plot_title: str
        The label for the y-axis 

    plot_row: str
        The name of the column in [granule_data] to be binned along the x-axis

    plot_row: str
        The label for the y-axis

    granule_data: Pandas dataframe
        The granule data for the plot, see the section in the Docs on "aggregate_fitting.h5"
        for the format required of the dataframe

    group_by: str
        The name of the column in [granule_data] which will be used to group the data before
        plotting. Only granules with a value of [plot_group] in this column will be plotted

    plot_group: anything
        The value in [group_by] of granules that should be plotted

    column_nbins: int
        The number of bins along the column axis

    row_nbins: int
        The number of bins along the row axis

    legend: bool
        Add a legend to the plot or not

    x_log_scale: bool
        Set x axis to a log scale if true

    y_log_scale: bool
        Set y axis to a log scale if true

    save_png: bool
        Saves the figure to a png in [out_dir] if true

    out_dir: str
        The path that the output figure should be saved to

    Outputs
    -------
    Figure to [out_dir] if [save_file] is true

    Returns
    -------
    A matplotlib figure
    """

    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)
    my_filter = f'{group_by} == "{plot_group}"'
    group = granule_data.query(my_filter).copy()

    if not log_scaleX:
        bin_max = group[plot_row].max()
        bin_min = group[plot_row].min()
        binsX = np.linspace(bin_min,bin_max,row_nbins + 1,endpoint=True)
    else:
        bin_max = group[plot_row].max()
        bin_min = group[plot_row].min()
        binsX = np.geomspace(bin_min, bin_max, row_nbins + 1)

    if not log_scaleY:
        bin_max = group[plot_column].max()
        bin_min = group[plot_column].min()
        binsY = np.linspace(bin_min,bin_max,column_nbins + 1,endpoint=True)
    else:
        bin_max = group[plot_column].max()
        bin_min = group[plot_column].min()
        binsY = np.geomspace(bin_min, bin_max, column_nbins + 1)

    h = ax.hist2d(plot_row,plot_column, bins = [binsX,binsY],
                label=plot_group, data=granule_data,norm=c.LogNorm(clip=True)) 
    fig.colorbar(h[3])

    ax.set_ylabel(plot_title)
    ax.set_xlabel(row_title)
    if legend:
        ax.legend(fontsize=10)
    if log_scaleY:
        ax.set_yscale("log")
    if log_scaleX:
        ax.set_xscale("log")

    # if save_png:
    #     pt.save(
    #         Path(out_dir) / f"2D-hist-{plot_group}-{plot_row}-{plot_column}.png",
    #         padding=0.05,
    #     )

    return fig


def pair_plot(granule_data: pd.DataFrame, save_png = True, out_dir: Path = "/tmp/"):

    """
    Uses seaborn's pairplot to draw 1 and 2D histograms of the Surface Tension,
    Bending Rigidity and Mean Radius for granules in [granule_data].

    Outline
    -------

    Data is split by the "experiment" field. Each experiment is assigned a colour,

    Parameters
    ----------

    granule_data: Pandas dataframe
        The granule data for the plot, see the section in the Docs on "aggregate_fitting.h5"
        for the format required of the dataframe

    save_png: bool
        Saves the figure to a png in [out_dir] if true

    out_dir: str
        The path that the output figure should be saved to

    Outputs
    -------
    Figure to [out_dir] if [save_file] is true

    Returns
    -------
    A matplotlib figure
   
    Outputs
    -------
    Figure to [out_dir] 
    """

    #make colour dict
    experiments = set(granule_data["experiment"].to_list())
    get_colour = colour_gen()
    colour_dict = {}
    for experiment in experiments:
        colour_dict[experiment] = get_colour(experiment)

    # pair plot does not work well with log axes, so we set these values directly
    granule_data["log_sigma"] = np.log10(granule_data["sigma"])
    granule_data["log_kappa"] = np.log10(granule_data["kappa_scale"])

    # Create the corner plot
    g = sns.pairplot(
        data=granule_data,
        hue="experiment",
        vars=["log_sigma", "log_kappa", "mean_radius"],
        markers="x",
        plot_kws=dict(linewidths=0.8, levels=4, alpha=0.8),
        kind="kde",
        corner=False,
        palette=colour_dict,
        height=4,
    )

    # Make the names more human readable
    rename_dict = {
        "mean_radius": "Mean Radius",
        "log_kappa": "Log(Bending Rigidity)",
        "log_sigma": "Log(Surface Tension)",
    }
    for ax in g.axes.flat:
        if ax is None:
            continue
        y_label = ax.get_ylabel()
        if y_label in rename_dict:
            ax.set_ylabel(rename_dict[y_label])
        x_label = ax.get_xlabel()
        if x_label in rename_dict:
            ax.set_xlabel(rename_dict[x_label])

    if save_png:
        save_path = Path(out_dir) / "pair_plot.png"
        plt.savefig(save_path)
    return plt.gcf()

def overlap_hist(
    plot_column,
    plot_label,
    granule_data: pd.DataFrame,
    plot_errors=None,
    group_by = "experiment",
    n_bins=20,
    agg = gmean,
    density=False,
    legend=False,
    log_scale = True,
    benchling_format: bool = False,
    save_png = True,
    out_dir = "/tmp/", 
):
    """
    Draw overlapping histograms of [plot_column], split by [group_by].

    Outline
    -------

    Plots a histogram of a variable with the 67% of points cloest to the medium shown in a darker colour,
    and the average (as determined by agg) shown with a verticle line.
    Also prints a summary of the mean and error. 

    Parameters
    ----------

    plot_column: str
        The name of the column in [granule_data] to be plotted as a histogram

    plot_label: str
        The label for the x-axis

    granule_data: Pandas dataframe
        The granule data for the plot, see the section in the Docs on "aggregate_fitting.h5"
        for the format required of the dataframe

    plot_errors: str or None
        If None, errorbars are not plotted
        The column in [granule_data] containing the error estimates for the values in [plot_column],
        used to estimate the error bars on the histogram bars.

    group_by: str
        The name of the column in [granule_data] which will be used to group the data before
        plotting. The graphs for each group will be plotted one on top of the other.

    n_bins: int or array
        If int, then the number of bins.
        If array, then the bin edges.

    agg: function Pandas dataseries -> float
        The function used to calculate the colour values. Usually some type of mean.

    out_dir: str
        The path that the output figure should be saved to

    density: bool
        If true, plot a probability density so the area under the graph is 1.

    legend: bool
        Add a legend to the plot or not

    log_scale: bool
        Set x axis to a log scale if true

    benchling_format: bool
        If true, print summary to the screen, optimized for cutting and pasting into tables.

    save_png: bool
        Saves the figure to a png in [out_dir] if true

    out_dir: str
        The path that the output figure should be saved to

    Outputs
    -------
    Figure to [out_dir] if [save_file] is true

    Returns
    -------
    A matplotlib figure

    """

    fig, ax = create_axes(1, axes_height= 8.3 / 2.5,aspect = 1)

    chunks = granule_data.groupby(group_by)
    get_colour = colour_gen()
    #fix n_bins
    if type(n_bins) == int and log_scale == True:
        bin_max = granule_data[plot_column].max()
        bin_min = granule_data[plot_column].min()
        n_bins = np.geomspace(bin_min, bin_max, n_bins + 1)        

    for num, (label, chunk) in enumerate(chunks):
        colour = get_colour(label)
        hist_vals, bin_edges = np.histogram(
            chunk[plot_column], bins=n_bins, density=density
        )

        widths = bin_edges[1:] - bin_edges[:-1]
        if plot_errors is None:
            hist_err = None
        else:
            hist_err = _get_hist_err(chunk[plot_column], chunk[plot_errors], bin_edges)

        hist_vals, hist_err = _get_normalised(hist_vals, hist_err)

        low_index, low_limit, high_index, high_limit = _calculate_limits(hist_vals,chunk,plot_column)

        ax.bar(
            bin_edges[:low_index],
            hist_vals[:low_index],
            label="_",
            alpha=0.410,
            color=colour,
            width=widths[:low_index],
            align="edge",
        )

        ax.bar(
            bin_edges[low_index:high_index],
            hist_vals[low_index:high_index],
            label=label,
            alpha=0.8,
            color=colour,
            width=widths[low_index:high_index],
            align="edge",
        )

        ax.bar(
            bin_edges[high_index:-1],
            hist_vals[high_index:],
            label="_",
            alpha=0.410,
            color=colour,
            width=widths[high_index:],
            align="edge",
        )

        if log_scale:
            bar_centers = np.exp(0.5 * (np.log(bin_edges[1:]) + np.log(bin_edges[:-1])))
        else:
            bar_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        ax.errorbar(
            bar_centers,
            hist_vals,
            yerr=hist_err,
            lw=0.0,
            elinewidth=0.5,
            ecolor=colour,
            capsize=2.0,
            capthick=0.5,
        )

        n_granules = len(chunk)
        gmean = agg(np.abs(chunk[plot_column]))
        plt.axvline(gmean, 0, 1, color=colour, ls="--", lw=0.8, alpha=1.0)

        # If this is being run as part of treatment_comparison.py to write data to a file for transfer to benchling, format the data appropriately.
        # if benchling_format:
        #     if plot_label == "Surface Tension (N/m)":
        #         low_limit*=10**9
        #         gmean*=10**9
        #         high_limit*=10**9
        #         plot_label = "Surface Tension (nN/m)"
        #     elif plot_label == "q2":
        #         low_limit*=10**6
        #         gmean*=10**6
        #         high_limit*=10**6
        #         plot_label = "q2 (micro)"
        #     if label == 'unknown':
        #         print(f"{float('%.3g' % low_limit)}-{float('%.3g' % gmean)}-{float('%.3g' % high_limit)} ; {plot_label}")
        #     else:
        #         print(f"{float('%.3g' % low_limit)}-{float('%.3g' % gmean)}-{float('%.3g' % high_limit)} ; {plot_label}; {label}")
        # else:
        #     print(
        #         f"For {n_granules} granules is {format_si(gmean)}"
        #         f" for {label} - {plot_label}",
        #     )

        #     print(
        #         f"Lower bound {format_si(low_limit)}. Upper bound {format_si(high_limit)}"
        #     )

    ax.set_xlabel(plot_label)
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0.0)
    if legend:
        ax.legend(fontsize=10)

    if log_scale:
        ax.set_xscale("log")

    if save_png:
        out_dir = Path(out_dir)
        save_path = out_dir / f"overlap-{plot_column}-{group_by}.png"
        sns.despine()
        plt.tight_layout()
        plt.savefig(save_path, dpi=330)
    return fig


def read_data(input_file,comp_file = None,data_file_name="aggregate_fittings.h5"):

    """Reads in one or more aggregate_fitting.h5 files and concatenates each one
    
        Parameters
        ----------
        input_file: str
            The path to either a [data_file_name] file or a folder containing data files.
            If a file, it will open that file as a data frame and return it.
            If a folder, it will recursivly search subfolders for files named
            [data_file_name], open all the files and concatenate the result
            into a single data frame.

        comp_file: str
            This is for backwards compatability only! If you have data from before
            May 2022, it may come with a separate "comparision" file containing additional
            information. This parameter should be a path to that file, otherwise None.

        data_file_name: str
            the name of the .h5 file to open. Default: aggregate_fittings.h5

        Returns
        -------
        a pandas data frame containing all the data from the .h5 files opened
    
    """

    if input_file.is_dir():
        input_file_list = input_file.rglob(data_file_name)
        granule_data = pd.concat(map(_load_terms, input_file_list), ignore_index=True)
    else:
        granule_data = _load_terms(input_file)

    if "fitting_diff" in granule_data.columns:
        #new style output, no additional processing needed
        return granule_data

    elif comp_file != None:
        #resolve comparison
        if comp_file.is_dir():
            comp_file_list = comp_file.rglob(data_file_name)
            comp_data = pd.concat(map(_load_terms, comp_file_list), ignore_index=True)
        else:
            comp_data = _load_terms(comp_file)

        sigma_diffs = []
        for granule, comp in zip(granule_data.itertuples(),comp_data.itertuples()):
            sigma_diffs.append(abs(granule.sigma - 4.0 * comp.sigma))    
        granule_data = granule_data.assign(sigma_diff = sigma_diffs)

        fitting_diffs = []
        for granule, comp in zip(granule_data.itertuples(),comp_data.itertuples()):
            fitting_diffs.append(comp.fitting_error - granule.fitting_error)
    
        granule_data = granule_data.assign(fitting_diff = fitting_diffs)
        return granule_data

    else:
        #fallback, set nonsense values in missing columns
        granule_data["sigma_diff"] = 10000000.0
        granule_data["fitting_diff"] = 10000000.0
        return granule_data

def _load_terms(aggregate_fittings_path: Path) -> pd.DataFrame:
    """Load the spectrum fitting terms and physical values from disk."""
    # Container for the physical properties of the granules
    print(aggregate_fittings_path)
    aggregate_fittings = pd.read_hdf(
        aggregate_fittings_path, key="aggregate_data", mode="r"
    )

    if "experiment" not in aggregate_fittings.columns:
        #old style output files need to have the experiment column infered.
        aggregate_fittings["experiment"] = _get_treament_type(
            aggregate_fittings["figure_path"].iloc[0]
        )

    try: #TODO fix times frame_gen so this is handled more elegently
        times = [_convert_to_sec(path) for path in aggregate_fittings["figure_path"]]
        start = min(times)
        times = [time - start for time in times]
    except:
        times = 1.0

    aggregate_fittings["times"] = times

    return aggregate_fittings


def _get_treament_type(im_path):
    """Get the experiment name from the image path."""
    path_name = Path(im_path).name
    experiment_group = re_experiment_name.search(path_name)

    if experiment_group is None:
        return "unknown"

    experiment_name = experiment_group.groupdict()["exp"]
    print(path_name, " ", experiment_name)
    if experiment_name.startswith("Control") or experiment_name.startswith("As"):
        return "As"
    if experiment_name.startswith("Cz"):
        return "Cz"
    if experiment_name.startswith("FXR1"):
        if experiment_name.endswith("mCh"):
            return "FXR1-G3BP1"
        elif experiment_name.endswith("GFP"):
            return "FXR1-FXR1"
        else:
            return "NaAs+FXR1"
    if experiment_name.startswith("Caprin"):
        return "NaAs+Caprin1"
    raise ValueError("Unable to get experiment name.")


def _convert_to_sec(path):
    time = re_time_stamp.findall(path)
    t = time[0].split(".")
    return int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])


def _get_hist_err(vals, errors, bin_edges):

    hist_errors = np.zeros(len(bin_edges) - 1)

    for mean, sd in zip(vals, errors):
        for i in range(len(bin_edges) - 1):
            p = norm.cdf(bin_edges[i + 1], loc=mean, scale=np.abs(sd)) - norm.cdf(
                bin_edges[i], loc=mean, scale=np.abs(sd)
            )
            hist_errors[i] += p * (1 - p)
            # print(mean,sd,p)
    return np.sqrt(hist_errors)


def _get_dist_vals(vals, errors, bin_edges):
    hist_vals = np.zeros(len(bin_edges) - 1)

    for mean, sd in zip(vals, errors):
        for i in range(len(bin_edges) - 1):
            p = norm.cdf(bin_edges[i + 1], loc=mean, scale=np.abs(sd)) - norm.cdf(
                bin_edges[i], loc=mean, scale=np.abs(sd)
            )
            hist_vals[i] += p
    return hist_vals, bin_edges


def _get_normalised(hist_vals, hist_errors):

    hist_vals_norm = hist_vals / sum(hist_vals)

    if isinstance(hist_errors, np.ndarray):
        hist_errors_norm = [
            norm * err / val if val > 0.0 else 0.0
            for err, val, norm in zip(hist_errors, hist_vals, hist_vals_norm)
        ]
    else:
        hist_errors_norm = None

    return hist_vals_norm, hist_errors_norm


def colour_gen():
    treatments = {}
    colours = ["#7fc97f","#beaed4","#4da6ff","#ff0000","#fdc086","#cc7700"]
    num = 0
    def get_colour(treatment):
        nonlocal treatments
        nonlocal colours
        nonlocal num
        if treatment in treatments:
            return treatments[treatment]
        else:
            if colours != []:
                colour = colours[0]
                colours = colours[1:]
            else:
                colour = list(c.CSS4_COLORS.values())[num]
                num +=1

            treatments[treatment] = colour
            return colour
    return get_colour

def _plot_geometric_mean(data, color=None, label=None, c="black"):
    """ Plot a vertical line at the geometric mean. """
    geometric_mean = gmean(np.abs(data)) #TODO not abs
    plt.axvline(geometric_mean, 0, 1, color=c, ls="--", lw=0.8, alpha=1.0)

    return geometric_mean


def _add_summary(data, color=None, label=None, units=""):
    """ Print some text about the properties of the granules. """
    geometric_mean = gmean(data)
    count = len(data)
    label = f"Mean: {format_si(geometric_mean)}{units} \nCount: {count}"

    plt.annotate(
        label,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=8,
    )

def _adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def _calculate_limits(vals,chunk,plot_column):

    tot = 0
    low_index = 0
    for index, val in enumerate(vals):
        tot += val
        if tot > 0.165:
            low_index = index
            break

    tot = 0
    high_index = len(vals)-1
    for index,val in enumerate(vals):
        tot += val
        if tot > 1.0 - 0.165:
            high_index = index
            break

    chunk_sorted = chunk.sort_values(by=plot_column)
    tot = 0
    target = 0.165 * len(chunk_sorted)
    for _,entry in chunk_sorted.iterrows():
        tot += 1
        if tot > target:
            low_limit = entry[plot_column]
            break
   
    tot = 0
    target = (1.0 - 0.165) * len(chunk_sorted)
    for _,entry in chunk_sorted.iterrows():
        tot += 1
        if tot > target:
            high_limit = entry[plot_column]
            break

    return low_index, low_limit, high_index, high_limit