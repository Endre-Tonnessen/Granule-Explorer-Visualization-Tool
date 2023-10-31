#!/usr/bin/env python
""" Plot histograms of the results split by some column. """

import matplotlib.pyplot as plt
import matplotlib.colors as c

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
from .plot_tools import create_axes,annotate_axis,cart2polar,despine_axis,force_integer_ticks,format_si,hide_axis_lables 

# Regex capturing --N group termiated with '--' or '.'
re_experiment_name = re.compile("--N(?P<exp>[\d\w_-]+?)(?:--|\.)")
re_time_stamp = re.compile("_(.+)--N")


def split_hist(
    plot_column,
    split_column,
    plot_label,
    split_label,
    granule_data: pd.DataFrame,
    bins=None,
    units="",
    out_dir="/tmp/",
    summary=False,
    title=None,
    error=False,
    density=False,
):
    """ Split a histogram on the given variable. """

    # plt.style.use("seaborn-paper")
    g = sns.FacetGrid(col=split_column, data=granule_data, height=8.5 / 2.5)

    g.map(plt.hist, plot_column, bins=bins, density=density)
    g.map(_plot_geometric_mean, plot_column)
    g.set_xlabels(plot_label)
    g.set_ylabels("Count")
    # if error:
    #     g.map(_get_error, plot_column)
    if summary:
        g.map(_add_summary, plot_column, units=units)
    if title:
        # g.set_titles("{col_name} with smoothing {row_name} σ")
        g.set_titles(title)

    if bins is not None and not isinstance(bins, int):
        bin_ratio = bins[-1] / bins[0]
        if bin_ratio > 20:
            g.set(xscale="log")

    # for ax in g.axes:
    #     ax[0].xaxis.set_major_formatter(EngFormatter())

    additional_metadata = dict(func="split_hist", file="split_histograms.py")
    out_dir = Path(out_dir)
    # save_figure_and_trim(
    #     out_dir / f"split-{plot_column}-{split_column}.png",
    #     # additional_metadata=additional_metadata,
    # )


def quartile_plot (
    plot_column,
    plot_title,
    plot_row,
    row_title,
    granule_data,
    hue="treatment",
    qbins=5,
    agg=gmean,
    legend=True,
    log_scale=True,
    out_dir="/tmp/",
    draft=True,
    errors=True,
):
    """Plot the variable as a mean of binned values.

    The bins are chosen so that there is an even number of points in each bin.
    """

    binColumn = plot_row

    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)
    for num, (hue_name, group_) in enumerate(granule_data.groupby(hue)):
        group = group_.copy()
        
        bins = pd.qcut(group[binColumn], qbins)

        # Ensure that the granule size is saved as a float
        bin_mid_point = bins.apply(lambda i: float(i.mid))
        group["bin_mid_point"] = pd.to_numeric(bin_mid_point)

        mean_values = (
            group.groupby("bin_mid_point")[plot_column].agg([agg, sem])
        ).reset_index()
        qmids = [(qbins[i]+qbins[i+1])*0.5 for i in range(len(qbins)-1)]
        mean_values["quartiles"] = qmids

        plt_kwargs = dict(
            label=hue_name, data=mean_values, color=_get_treatment_colour(hue_name)
        )
        if errors:
            ax.errorbar(
                "quartiles", agg.__name__, yerr="sem", elinewidth=0.8, **plt_kwargs,
            )
        else:
            ax.plot("quartiles", agg.__name__, **plt_kwargs)

    log = False

    ax.set_ylabel(plot_title)
    ax.set_xlabel(row_title)
    if legend:
        ax.legend(fontsize=10)
    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")

    print(f"plot_column = {plot_column}")

    # save(
    #     Path(out_dir) / f"quartile-{plot_row}-{plot_column}.png",
    #     padding=0.05,
    # )


def filter_plot(
    plot_column,
    plot_title,
    plot_row,
    row_title,
    granule_data,
    hue="treatment",
    n_bins=5,
    agg=gmean,
    bin_type="count",
    legend=True,
    log_scale=True,
    out_dir="/tmp/",
    draft=True,
    errors=False,
):
    """Plot the variable as a mean of binned values, for three filtering steps.

    The bins are chosen so that there is an even number of points in each bin.
    """

    binColumn = plot_row

    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)
    query_strings = [("all filters","green","sigma > 1e-10 and pass_rate > 0.6 and fitting_error < 0.6"),
                     ("outline filter only","red" ,"sigma > 1e-10 and pass_rate > 0.6"),
                     ("no filters","blue", "sigma > 1e-10")
                    ]
    

    for title,colour, query_string in query_strings:
        group = granule_data.query(query_string).copy()
        if bin_type == "count":
            bins = pd.qcut(group[binColumn], n_bins)
        elif bin_type == "radius":
            bins = pd.cut(group[binColumn], n_bins)
        elif bin_type == "log":
            bin_max = group[binColumn].max()
            bin_min = group[binColumn].min()
            bin_groups = np.geomspace(bin_min, bin_max, n_bins + 1)
            bins = pd.cut(group[binColumn], bin_groups, right=True)

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
    ax.set_xlabel(row_title)
    if legend:
        ax.legend(fontsize=10)
    if log_scale:
        ax.set_yscale("log")
        ax.set_xscale("log")
    return fig
    # save(
    #     Path(out_dir) / f"filters-{plot_row}-{plot_column}-{bin_type}.png",
    #     padding=0.05,
    # )


def variable_bin_plot(
    plot_column,
    plot_title,
    plot_row,
    row_title,
    granule_data,
    hue="treatment",
    n_bins=5,
    agg=gmean,
    bin_type="count",
    legend=True,
    log_scale=False,
    out_dir="/tmp/",
    draft=True,
    errors=False,
):
    """Plot the variable as a mean of binned values.

    The bins are chosen so that there is an even number of points in each bin.
    """

    binColumn = plot_row

    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)
    for num, (hue_name, group_) in enumerate(granule_data.groupby(hue)):
        group = group_.copy()
        if bin_type == "count":
            bins = pd.qcut(group[binColumn], n_bins)
        elif bin_type == "radius":
            bins = pd.cut(group[binColumn], n_bins)
        elif bin_type == "log":
            bin_max = group[binColumn].max()
            bin_min = group[binColumn].min()
            bin_groups = np.geomspace(bin_min, bin_max, n_bins + 1)
            bins = pd.cut(group[binColumn], bin_groups, right=True)

        # Ensure that the granule size is saved as a float
        bin_mid_point = bins.apply(lambda i: float(i.mid))
        group["bin_mid_point"] = pd.to_numeric(bin_mid_point)

        mean_values = (
            group.groupby("bin_mid_point")[plot_column].agg([agg, sem])
        ).reset_index()
        plt_kwargs = dict(
            label=hue_name, data=mean_values, color=_get_treatment_colour(hue_name)
        )
        if errors:
            ax.errorbar(
                "bin_mid_point", agg.__name__, yerr="sem", elinewidth=0.8, **plt_kwargs,
            )
        else:
            ax.plot("bin_mid_point", agg.__name__, **plt_kwargs)

    log = log_scale
    #radius_units = "μm" if draft else "\si{\micro m}"
    ax.set_ylabel(plot_title)
    ax.set_xlabel(row_title)
    if legend:
        ax.legend(fontsize=10)
    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")
    
    agg_mean = agg(np.abs(group[plot_column]))
    # print(f"plot_column = {plot_column}")
    #ax.axhline(agg_mean,0,1,color="black",ls="--",lw=0.8,alpha=1.0)

    # cat,qbins = pd.qcut(group[plot_column],[0.0, 0.165, 0.835, 1.0],retbins=True)
    
    # ax.axhspan(qbins[1],qbins[2],0,1,color="lightblue",alpha=0.3)

    # print(f"plot_column = {plot_column}")

    # save(
    #     Path(out_dir) / f"binned-{plot_row}-{plot_column}-{bin_type}.png",
    #     padding=0.05,
    # )


def scatter_plot(
    plot_column,
    plot_title,
    plot_row,
    row_title,
    granule_data,
    hue = "treatment",
    legend = True,
    log_scaleX = True,
    log_scaleY = True,
    out_dir = "/tmp/"
):
    """ scatter plot to look for correlations"""
    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)
    for num, (hue_name, group_) in enumerate(granule_data.groupby(hue)):
        sorted_granules = granule_data.sort_values(by=['fitting_error'],ascending=False)
        ax.scatter(plot_row,plot_column,c="fitting_error",
                   label=hue_name, data=sorted_granules, linewidths = 0.0, 
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
    return fig

    # save(
    #     Path(out_dir) / f"scatter-{plot_row}-{plot_column}.png",
    #     padding=0.05,
    # )


def histogram2D (
    plot_column,
    plot_title,
    plot_row,
    row_title,
    granule_data,
    hue = "treatment",
    legend = True,
    log_scaleX = True,
    log_scaleY = True,
    out_dir = "/tmp/"   
):
    n_bins_x = 20
    n_bins_y = 20
    fig, ax = create_axes(1, fig_width=8.3 / 2.5, aspect=1)
    for num, (hue_name, group_) in enumerate(granule_data.groupby(hue)):
        group = group_.copy()
        if not log_scaleX:
            bin_max = group[plot_row].max()
            bin_min = group[plot_row].min()
            binsX = np.linspace(bin_min,bin_max,n_bins_x + 1,endpoint=True)
        else:
            bin_max = group[plot_row].max()
            bin_min = group[plot_row].min()
            binsX = np.geomspace(bin_min, bin_max, n_bins_x + 1)

        if not log_scaleY:
            bin_max = group[plot_column].max()
            bin_min = group[plot_column].min()
            binsY = np.linspace(bin_min,bin_max,n_bins_y + 1,endpoint=True)
        else:
            bin_max = group[plot_column].max()
            bin_min = group[plot_column].min()
            binsY = np.geomspace(bin_min, bin_max, n_bins_y + 1)

        h = ax.hist2d(plot_row,plot_column, bins = [binsX,binsY],
                   label=hue_name, data=granule_data,norm=c.LogNorm(clip=True)) 
        fig.colorbar(h[3])

    ax.set_ylabel(plot_title)
    ax.set_xlabel(row_title)
    if legend:
        ax.legend(fontsize=10)
    if log_scaleY:
        ax.set_yscale("log")
    if log_scaleX:
        ax.set_xscale("log")
    return fig

    # save(
    #     Path(out_dir) / f"2D-hist-{plot_row}-{plot_column}.png",
    #     padding=0.05,
    # )


def pair_plot(granule_data: pd.DataFrame, out_dir: Path):
    """Add a pair plot distribution of the granule properties."""
    colour_dict = {
        "FXR1-FXR1": "green",
        "FXR1-G3BP1": "red",
        "NaAs": "gray",
        "NaAs+Caprin1": "blue",
        "unknown": "black",
    }

    # pair plot does not work well with log axes, so we set these values directly
    granule_data["log_sigma"] = np.log10(granule_data["sigma"])
    granule_data["log_kappa"] = np.log10(granule_data["kappa"])

    # Create the corner plot
    g = sns.pairplot(
        data=granule_data,  # .query("treatment != 'NaAs'"),
        hue="treatment",
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

    save_path = Path(out_dir) / "pair_plot.png"
    plt.savefig(save_path)


def overlap_hist(
    plot_column,
    split_column, # Treatment
    plot_label,
    split_label,
    granule_data: pd.DataFrame,
    bins=None,
    out_dir="/tmp/",
    density=True,
    plot_errors=None,
    legend=False,
    is_log = True
):
    """Split a histogram on the given variable."""

    fig, ax = create_axes(1, axes_height= 8.3 / 2.5,aspect = 1)

    chunks = granule_data.groupby(split_column)

    for num, (label, chunk) in enumerate(chunks):
        colour = _get_colour(label, split_column)
        hist_vals, bin_edges = np.histogram(
            chunk[plot_column], bins=bins, density=density
        )

        widths = bin_edges[1:] - bin_edges[:-1]
        if plot_errors is None:
            hist_err = None
        else:
            hist_err = _get_hist_err(chunk[plot_column], chunk[plot_errors], bin_edges)

        hist_vals, hist_err = _get_normalised(hist_vals, hist_err)

        # mean and SD
        gmean = _plot_geometric_mean(chunk[plot_column], c=colour)
        error = sem(chunk[plot_column])

        tot = 0
        low_index = 0
        for index, val in enumerate(hist_vals):
            tot += val
            if tot > 0.165:
                low_index = index
                break

        tot = 0
        high_index = len(hist_vals)-1
        for index,val in enumerate(hist_vals):
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

        if is_log:
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

        if is_log:
            gmean = _plot_geometric_mean(chunk[plot_column], c=colour)
            gstdev = gstd(chunk[plot_column], ddof=1)
            gsem = np.exp(np.log(gstdev) / np.sqrt(n_granules))
            bound_up = gmean * (gstdev**1)
            bound_low = gmean * (gstdev**-1)
        else:
            gmean = _plot_geometric_mean(chunk[plot_column], c=colour)
            gsem = sem(chunk[plot_column])
            gstdev = gsem * np.sqrt(n_granules)
            bound_up = gmean + (gstdev*1)
            bound_low = gmean + (gstdev*-1)

        number_low = sum(chunk[plot_column] < bound_low)
        number_up = sum(chunk[plot_column] > bound_up)
        num_total = len(chunk[plot_column])

        

        print(
            f"For {n_granules} granules is {format_si(gmean)} */"
            f" {gsem} for {label} - {plot_label}",
        )

        print(
            f"Lower bound {format_si(low_limit)}. Upper bound {format_si(high_limit)}"
        )

    ax.set_xlabel(plot_label)
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0.0)

    if legend:
        ax.legend(title=split_label, fontsize=10)

    if bins is not None and not isinstance(bins, int):
        bin_ratio = bins[-1] / bins[0]
        if bin_ratio > 20:
            ax.set_xscale("log")

    additional_metadata = dict(func="split_hist", file="split_histograms.py")
    out_dir = Path(out_dir)
    save_path = out_dir / f"overlap-{plot_column}-{split_column}.png"
    sns.despine()
    plt.tight_layout()
    return fig
    # plt.savefig(save_path, dpi=330)


def overlap_dist(
    plot_column,
    split_column,
    plot_label,
    split_label,
    granule_data: pd.DataFrame,
    bins=None,
    out_dir="/tmp/",
    density=True,
    plot_errors=None,
):
    """Split a distribution on the given variable. calculcated with mean and sd"""

    fig, ax = create_axes(1, axes_height=8.3 / 2.5)
    if bins is None:
        raise ValueError("Must specify bin edges")

    chunks = granule_data.groupby(split_column)
    for num, (label, chunk) in enumerate(chunks):
        colour = _get_colour(label,split_column)
        hist_vals, bin_edges = _get_dist_vals(
            chunk[plot_column], chunk[plot_errors], bins
        )

        widths = bin_edges[1:] - bin_edges[:-1]
        if plot_errors is None:
            hist_err = None
        else:
            hist_err = _get_hist_err(chunk[plot_column], chunk[plot_errors], bin_edges)

        hist_vals, hist_err = _get_normalised(hist_vals, hist_err)

        ax.bar(
            bin_edges[:-1],
            hist_vals,
            label=label,
            alpha=0.6,
            color=colour,
            width=widths,
            align="edge",
        )

        bar_centers = np.exp(0.5 * (np.log(bin_edges[1:]) + np.log(bin_edges[:-1])))
        ax.errorbar(
            bar_centers,
            hist_vals,
            yerr=hist_err,
            lw=0.0,
            elinewidth=0.5,
            ecolor="black",
            capsize=2.0,
            capthick=0.5,
        )

    ax.set_xlabel(plot_label)
    ax.set_ylabel("Denisity")
    ax.set_ylim(bottom=0.0)
    ax.legend(title=split_label, fontsize=10)

    ax.set_xscale("log")

    additional_metadata = dict(func="split_hist", file="split_histograms.py")
    out_dir = Path(out_dir)
    save_path = out_dir / f"overlap-dist-{plot_column}-{split_column}.png"
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=330)


def overlap_hist_filter(
    plot_column,
    split_column,
    plot_label,
    split_label,
    granule_data: pd.DataFrame,
    bins=None,
    out_dir="/tmp/",
    density=True,
    plot_errors=None,
    legend=False
):
    """Split a histogram on the given variable."""

    fig, ax = create_axes(1, axes_height=8.3 / 2.5)

    query_strings = [("early","#7fc97f","times <= 2000"),
                     ("mid","#85C0F9" ,"times <= 4000 and times > 2000"),
                     ("late","#0F2080", "times <= 6000 and times > 4000"),
                     ("end","#A95AA1", "times > 6000"),
                    ]
    

    for label,colour, query_string in query_strings:
        chunk = granule_data.query(query_string).copy()

        hist_vals, bin_edges = np.histogram(
            chunk[plot_column], bins=bins, density=density
        )

        widths = bin_edges[1:] - bin_edges[:-1]
        if plot_errors is None:
            hist_err = None
        else:
            hist_err = _get_hist_err(chunk[plot_column], chunk[plot_errors], bin_edges)

        hist_vals, hist_err = _get_normalised(hist_vals, hist_err)

        # mean and SD
        gmean = _plot_geometric_mean(chunk[plot_column], c=colour)
        error = sem(chunk[plot_column])

        ax.bar(
            bin_edges[:-1],
            hist_vals,
            label=label,
            alpha=0.8,
            color=colour,
            width=widths,
            align="edge",
        )

        bar_centers = np.exp(0.5 * (np.log(bin_edges[1:]) + np.log(bin_edges[:-1])))
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

        gmean = _plot_geometric_mean(chunk[plot_column], c=colour)
        error = sem(chunk[plot_column])
        n_granules = len(chunk)

        gstdev = gstd(chunk[plot_column], ddof=1)
        gsem = np.exp(np.log(gstdev) / np.sqrt(n_granules))

        bound_up = gmean * (gstdev**1)
        bound_low = gmean * (gstdev**-1)

        number_low = sum(chunk[plot_column] < bound_low)
        number_up = sum(chunk[plot_column] > bound_up)
        num_total = len(chunk[plot_column])

        print(
            f"For {n_granules} granules is {format_si(gmean)} */"
            f" {gsem} for {label} - {plot_label}",
        )

        print(
            f"Mid: {num_total - number_low - number_up} granules, {number_up} above GSD and {number_low} below 3SD"
        )

    ax.set_xlabel(plot_label)
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0.0)

    if legend:
        ax.legend(title=split_label, fontsize=10)

    if bins is not None and not isinstance(bins, int):
        bin_ratio = bins[-1] / bins[0]
        if bin_ratio > 20:
            ax.set_xscale("log")

    additional_metadata = dict(func="split_hist", file="split_histograms.py")
    out_dir = Path(out_dir)
    save_path = out_dir / f"overlap-filter-{plot_column}-{split_column}.png"
    sns.despine()
    plt.tight_layout()
    # plt.savefig(save_path, dpi=330)
    return fig


def read_data(input_file,comp_file,data_file_name="aggregate_fittings.h5"):

    if input_file.is_dir():
        input_file_list = input_file.rglob(data_file_name)
        granule_data = pd.concat(map(_load_terms, input_file_list), ignore_index=True)
    else:
        granule_data = _load_terms(input_file)

    if comp_file != None:
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

    else:
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

    aggregate_fittings["treatment"] = _get_treament_type(
        aggregate_fittings["figure_path"].iloc[0]
    )

    try:
        times = [_convert_to_sec(path) for path in aggregate_fittings["figure_path"]]

        start = min(times)
        times = [time - start for time in times]
    except:
        times = 1.0

    aggregate_fittings["times"] = times

    return aggregate_fittings


def _get_treament_type(im_path):
    """Get the treatment name from the image path."""
    path_name = Path(im_path).name
    experiment_group = re_experiment_name.search(path_name)

    if experiment_group is None:
        # print(path_name)
        # raise ValueError("No regex match")
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


def _get_colour(label,split_label):
    if split_label == "treatment":
        return _get_treatment_colour(label)
    elif split_label =="correction":
        return _get_correction_colour(label)
    else:
        return "#7fc97f"


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


def _get_error(data, plot_column):
    """ Calculate the SEM for the given result.  """
    print(f"data = {data}")
    print(f"plot_column = {plot_column}")


def _get_treatment_colour(treatment_name: str):
    """ Get a consistent colour for each experiment. """
    # colours = dict(naas="#cc7700", cz="#5cdac5", fxr1="#116966", caprin="#95c858")
    colours = dict(naas="#cc7700", cz="#8de4d3", fxr1="#8f0f12", caprin="#88dc40",unknown="#88dc40")
    colours = OrderedDict(
        fxr1="#fdc086", caprin="#4da6ff", cz="#beaed4", naas="#7fc97f"
    )

    colours["as"] = "#7fc97f"
    colours["cz"] = "#beaed4"
    colours["fxr1-g3bp1"] = "#ff0000"
    colours["fxr1-fxr1"] = "#00ff00"
    colours["unknown"] = "#00ff00"
    colours.move_to_end("fxr1-g3bp1", last=False)
    colours.move_to_end("fxr1-fxr1", last=False)

    lower_treatment_name = treatment_name.lower()
    for treatment, colour in colours.items():
        if treatment == lower_treatment_name:
            return colour

    raise ValueError(f"Can't match treatment {treatment_name} in colour dict.")


def _get_correction_colour(label):
    if label == "spherical":
        return "#7fc97f"
    else:
        return "#beaed4"